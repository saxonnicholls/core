//
//  distill.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` knowledge distillation: the forward and reverse KL between a teacher's and a student's distributions at a temperature, fused and stable, with the student's gradient; and the variant that keeps only the teacher's top-k logits.
/// \ingroup core
///
/// **Overview.** Distillation trains a student language model to match a teacher's next-token distribution at every
/// response position (TRAINING.md T11). With temperature `T`, `p = softmax(teacher / T)` and `q = softmax(student /
/// T)` (Hinton et al. 2015, eq. 1):
///
/// - `forward_kl{T}`: `KL(p || q) = Σ p (log p - log q)`, mean-seeking, Hinton's soft-target cross-entropy less the
///   teacher's entropy (the same gradient). `∂/∂student_i = (q_i - p_i) / T` (Hinton's eq. 2).
/// - `reverse_kl{T}`: `KL(q || p) = Σ q (log q - log p)`, mode-seeking, as on-policy distillation (MiniLLM, GKD) uses.
///   `∂/∂student_i = q_i (log q_i - log p_i - KL) / T`.
///
/// Hinton multiplies the soft loss by `T²` so its gradient keeps its size as `T` changes; here that factor is the
/// caller's `scale`. Every log-probability is `(x_i - max) · (1/T) - log Σ e^((x - max)/T)`, never a logarithm of a
/// probability, so the loss is finite whatever the logits; a zero probability contributes nothing.
///
/// - `distill_loss(params, student, teacher, dstudent, scale)`: one row, the loss returned and the gradient written
///   (`dstudent` may be `student`).
/// - `distill_rows(backend, params, students, teachers, mask, losses, dstudents, scale)`: every row of a batch on the
///   backend (TRAINING.md T9), the masked rows (the prompt) 0 in loss and gradient.
/// - `teacher_topk(logits, ids, values)`, `distill_loss_topk` and `distill_topk_rows`: the forward KL against the
///   teacher's `k` largest logits only, renormalised over them, `p̃ = softmax(values / T)`: `Σ_j p̃_j (log p̃_j - log
///   q_{id_j})`, gradient `(q - p̃) / T` with `p̃` 0 off the top-k. A teacher's logits stored this way take `k` numbers
///   and `k` indices a position instead of the vocabulary. It is an approximation (the tail's mass is dropped), exact
///   when the teacher puts no mass outside its top-k.
///
/// \snippet ml/distill_demo.cpp distill
///
/// **Relationships.** `ml/softmax.hpp` (the same shifted sums), `ml/logprobs.hpp` (the mask's layout), `ml/lm_head.hpp`
/// (the logits). Group: \ref core.
///
/// **Implementations.** One each, on the host; the backend-first functions are the seam for a device.
///
/// **Guarantees.** `constexpr`, no allocation, `noexcept` for the standard floating-point types. Two passes over each
/// row and the teacher's, three with the gradient; `teacher_topk` is O(`V · k`).
///
/// **Errors.** Preconditions, asserted: `temperature > 0`; the rows' sizes agree; `k` at most `V`; top-k indices
/// below `V`.
///
/// **Definitions.** The softened softmax, the gradient `(q - p) / T` and the `T²` factor: checked against Hinton et al.
/// 2015 (arXiv 1503.02531, eqs. 1-2, §2). The reverse KL and its gradient are derived here and checked by central
/// differences; the top-k renormalisation is the usual sparse-logit form, recalled.
///
/// **Performance.** `distill_rows` with the forward KL against the same loop by hand
/// (`benchmarks/ml/distill_benchmark.cpp`): see `BENCHMARKS.md`.
///
/// **Tests and demos.** `tests/ml/distill_test.cpp`, `demos/ml/distill_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`.
///
/// **Known issues.** A mix of forward and reverse (the generalised JSD of GKD) is two calls and a weighted sum.
///
/// **Status.** Experimental, new in Unreleased.

#ifndef snicholls_core_ml_distill_hpp
#define snicholls_core_ml_distill_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief Forward KL distillation, `KL(teacher || student)` at a temperature.
/// \ingroup core
struct forward_kl
{
    double temperature = 1.0;   ///< `T > 0`.
};

/// \brief Reverse KL distillation, `KL(student || teacher)` at a temperature.
/// \ingroup core
struct reverse_kl
{
    double temperature = 1.0;   ///< `T > 0`.
};

namespace detail::distill
{
    // max, 1/T and log Σ e^((x - max)/T): log p_i = (x_i - max)·(1/T) - log_sum
    template <typename T>
    struct tempered
    {
        T max;
        T inv_t;
        T log_sum;

        constexpr T log_p(T x) const { return (x - max) * inv_t - log_sum; }
    };

    template <typename T>
    constexpr tempered<T> tempered_of(std::span<const T> x, T inv_t)
    {
        T m = -std::numeric_limits<T>::infinity();
        for (const T v : x)
            m = v > m ? v : m;
        T s = T(0);
        for (const T v : x)
            s += math::exp((v - m) * inv_t);
        return {m, inv_t, math::log(s)};
    }
} // namespace detail::distill

/// \brief The forward KL of one row, `Σ p (log p - log q)`, and the student's gradient `scale · (q - p) / T`.
/// \param c The temperature.
/// \param student The student's logits.
/// \param teacher The teacher's logits, the same size.
/// \param dstudent Written, the same size; may be `student`.
/// \param scale The factor on the gradient (`T²` for Hinton's weighting, `1/n` for a mean).
/// \return The KL, at least 0 up to rounding.
template <row S, row Tc, mutable_row G>
    requires std::same_as<row_value_t<S>, row_value_t<Tc>> && std::same_as<row_value_t<S>, row_value_t<G>>
constexpr row_value_t<S> distill_loss(const forward_kl& c, const S& student, const Tc& teacher, G&& dstudent,
                                      std::type_identity_t<row_value_t<S>> scale = 1) noexcept(std::is_floating_point_v<row_value_t<S>>)
{
    using T = row_value_t<S>;
    const auto s = as_row(student);
    const auto t = as_row(teacher);
    const auto g = as_mutable_row(dstudent);
    assert(c.temperature > 0.0 && t.size() == s.size() && g.size() == s.size());
    const T inv = T(1) / T(c.temperature);
    const auto ps = detail::distill::tempered_of(s, inv), pt = detail::distill::tempered_of(t, inv);
    T kl = T(0);
    for (std::size_t i = 0; i < s.size(); ++i) {
        const T lp = pt.log_p(t[i]), lq = ps.log_p(s[i]);
        const T p = math::exp(lp), q = math::exp(lq);
        if (p != T(0))
            kl += p * (lp - lq);
        g[i] = scale * inv * (q - p);
    }
    return kl;
}

/// \brief The reverse KL of one row, `Σ q (log q - log p)`, and the student's gradient `scale · q (log q - log p - KL) / T`.
/// \param c The temperature.
/// \param student The student's logits.
/// \param teacher The teacher's logits.
/// \param dstudent Written; may be `student`.
/// \param scale The factor on the gradient.
/// \return The KL.
template <row S, row Tc, mutable_row G>
    requires std::same_as<row_value_t<S>, row_value_t<Tc>> && std::same_as<row_value_t<S>, row_value_t<G>>
constexpr row_value_t<S> distill_loss(const reverse_kl& c, const S& student, const Tc& teacher, G&& dstudent,
                                      std::type_identity_t<row_value_t<S>> scale = 1) noexcept(std::is_floating_point_v<row_value_t<S>>)
{
    using T = row_value_t<S>;
    const auto s = as_row(student);
    const auto t = as_row(teacher);
    const auto g = as_mutable_row(dstudent);
    assert(c.temperature > 0.0 && t.size() == s.size() && g.size() == s.size());
    const T inv = T(1) / T(c.temperature);
    const auto ps = detail::distill::tempered_of(s, inv), pt = detail::distill::tempered_of(t, inv);
    T kl = T(0);
    for (std::size_t i = 0; i < s.size(); ++i) {
        const T lq = ps.log_p(s[i]);
        const T q = math::exp(lq);
        if (q != T(0))
            kl += q * (lq - pt.log_p(t[i]));
    }
    for (std::size_t i = 0; i < s.size(); ++i) {
        const T lq = ps.log_p(s[i]);
        const T q = math::exp(lq);
        g[i] = q == T(0) ? T(0) : scale * inv * q * ((lq - pt.log_p(t[i])) - kl);
    }
    return kl;
}

/// \brief Every row's distillation loss on the backend: the masked rows' losses and gradients 0.
/// \param params `forward_kl` or `reverse_kl`.
/// \param students `n` rows of `V`. \param teachers `n` rows of `V`. \param mask `n` (1 for a response position), or
/// empty for every row. \param losses Receives `n`. \param dstudents Written, `n · V`; may be `students`.
/// \param scale The factor on every row's gradient (`T² / count` for Hinton's mean).
template <math::math_number T, typename B, typename P>
constexpr void distill_rows(const B&, const P& params, std::span<const T> students, std::type_identity_t<std::span<const T>> teachers,
                            std::span<const std::uint8_t> mask, std::type_identity_t<std::span<T>> losses,
                            std::type_identity_t<std::span<T>> dstudents, std::type_identity_t<T> scale) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = losses.size();
    assert(teachers.size() == students.size() && dstudents.size() == students.size() && (mask.empty() || mask.size() == n)
           && (n == 0 || students.size() % n == 0));
    if (n == 0)
        return;
    const std::size_t v = students.size() / n;
    for (std::size_t r = 0; r < n; ++r) {
        const auto g = dstudents.subspan(r * v, v);
        if (!mask.empty() && mask[r] == 0) {
            losses[r] = T(0);
            for (T& x : g)
                x = T(0);
            continue;
        }
        losses[r] = distill_loss(params, students.subspan(r * v, v), teachers.subspan(r * v, v), g, scale);
    }
}

/// \brief The teacher's `k` largest logits and their indices, largest first, ties to the lower index.
/// \param logits The teacher's row, `V`. \param ids Receives `k` indices. \param values Receives their `k` logits.
template <row In>
constexpr void teacher_topk(const In& logits, std::span<std::uint32_t> ids, std::span<row_value_t<In>> values) noexcept(std::is_floating_point_v<row_value_t<In>>)
{
    using T = row_value_t<In>;
    const auto x = as_row(logits);
    assert(ids.size() == values.size() && ids.size() <= x.size());
    // Each slot takes the largest (value, -index) strictly below the previous slot's, so no element is taken twice
    for (std::size_t j = 0; j < ids.size(); ++j) {
        std::size_t best = x.size();
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (j > 0 && !(x[i] < values[j - 1] || (x[i] == values[j - 1] && i > ids[j - 1])))
                continue;
            if (best == x.size() || x[i] > x[best])
                best = i;
        }
        assert(best < x.size());
        ids[j] = static_cast<std::uint32_t>(best);
        values[j] = T(x[best]);
    }
}

/// \brief The forward KL against the teacher's top-k logits, renormalised over them, and the student's gradient
/// `scale · (q - p̃) / T`.
/// \param c The temperature. \param student The student's logits, `V`. \param ids The teacher's top-k indices, below
/// `V`, distinct. \param values Their logits. \param dstudent Written, `V`; may be `student`. \param scale The factor.
/// \return `Σ_j p̃_j (log p̃_j - log q_{id_j})`.
template <row S, mutable_row G>
    requires std::same_as<row_value_t<S>, row_value_t<G>>
constexpr row_value_t<S> distill_loss_topk(const forward_kl& c, const S& student, std::span<const std::uint32_t> ids,
                                           std::type_identity_t<std::span<const row_value_t<S>>> values, G&& dstudent,
                                           std::type_identity_t<row_value_t<S>> scale = 1) noexcept(std::is_floating_point_v<row_value_t<S>>)
{
    using T = row_value_t<S>;
    const auto s = as_row(student);
    const auto g = as_mutable_row(dstudent);
    assert(c.temperature > 0.0 && ids.size() == values.size() && !ids.empty() && g.size() == s.size());
    const T inv = T(1) / T(c.temperature);
    const auto ps = detail::distill::tempered_of(s, inv), pt = detail::distill::tempered_of(values, inv);
    T kl = T(0);
    for (std::size_t j = 0; j < ids.size(); ++j) {
        assert(ids[j] < s.size());
        const T lp = pt.log_p(values[j]);
        const T p = math::exp(lp);
        if (p != T(0))
            kl += p * (lp - ps.log_p(s[ids[j]]));
    }
    for (std::size_t i = 0; i < s.size(); ++i)
        g[i] = scale * inv * math::exp(ps.log_p(s[i]));
    for (std::size_t j = 0; j < ids.size(); ++j)
        g[ids[j]] -= scale * inv * math::exp(pt.log_p(values[j]));
    return kl;
}

/// \brief `distill_loss_topk` on every row of a batch, on the backend; the masked rows 0.
/// \param c The temperature. \param students `n · V`. \param ids `n · k`. \param values `n · k`. \param mask `n`, or
/// empty. \param losses Receives `n`. \param dstudents Written, `n · V`. \param scale The factor.
template <math::math_number T, typename B>
constexpr void distill_topk_rows(const B&, const forward_kl& c, std::span<const T> students, std::span<const std::uint32_t> ids,
                                 std::type_identity_t<std::span<const T>> values, std::span<const std::uint8_t> mask,
                                 std::type_identity_t<std::span<T>> losses, std::type_identity_t<std::span<T>> dstudents,
                                 std::type_identity_t<T> scale) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = losses.size();
    assert(ids.size() == values.size() && dstudents.size() == students.size() && (mask.empty() || mask.size() == n)
           && (n == 0 || (students.size() % n == 0 && ids.size() % n == 0)));
    if (n == 0)
        return;
    const std::size_t v = students.size() / n, k = ids.size() / n;
    for (std::size_t r = 0; r < n; ++r) {
        const auto g = dstudents.subspan(r * v, v);
        if (!mask.empty() && mask[r] == 0) {
            losses[r] = T(0);
            for (T& x : g)
                x = T(0);
            continue;
        }
        losses[r] = distill_loss_topk(c, students.subspan(r * v, v), ids.subspan(r * k, k), values.subspan(r * k, k), g, scale);
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_distill_hpp
