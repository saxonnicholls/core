//
//  logprobs.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` log-probabilities of given tokens: per token from logits (fused and stable), with the entropy, summed or averaged over a sequence's response tokens, and their backward passes.
/// \ingroup core
///
/// **Overview.** Every post-training objective (preference losses, reinforcement learning, distillation) starts from
/// the log-probability the model gives the tokens of a response: `log π(y_t | x, y_<t)` at each position, then summed
/// (or averaged) over the response's tokens, never the prompt's (TRAINING.md T11). This header is that first step:
///
/// - `token_logprob(logits, target)`: `log softmax(logits)[target]`, computed as `(x_target - max) - log(sum e^(x -
///   max))`, as `cross_entropy` does, of which it is the negation to the bit.
/// - `token_logprob_with_gradient(logits, target, dlogits, dlogp)`: the same and, in the pass, the gradient
///   `dlogp · (one_hot - softmax)`.
/// - `token_entropy(logits)`: `H = -Σ p log p`, for the entropy bonus of reinforcement learning.
/// - `token_logprobs_rows(backend, logits, targets, logps, entropies)`: every row of a batch, on the backend
///   (TRAINING.md T9), the entropies too when asked.
/// - `token_logprobs_rows_backward(backend, logits, targets, dlogps, dentropies)`: each row replaced by its gradient,
///   `dlogp · (one_hot - p) - dH · p (log p + H)`, fused as the classifier of `cross_entropy_rows` is, so the `n x V`
///   logits are their own gradient buffer.
/// - `sequence_logprobs(backend, token_logps, mask, out, reduction)` and `sequence_logprobs_backward`: the masked sum
///   (DPO, IPO, GRPO's ratios) or mean (SimPO, ORPO: length-normalised) of each row's token log-probabilities over its
///   response tokens; a row of no response tokens gives 0 and a zero gradient.
///
/// The layout is a batch of `rows` sequences padded to one `length`: position `t` of row `r` is element `r · length +
/// t` of the token log-probabilities, the targets and the mask. The target at a position is the token that follows
/// it (the caller shifts), and the mask is 1 where that target is a response token, 0 for the prompt and padding.
///
/// \snippet ml/logprobs_demo.cpp logprobs
///
/// **Relationships.** `ml/softmax.hpp` (`cross_entropy_with_gradient`, the same arithmetic), `ml/lm_head.hpp` (the
/// logits), `ml/preference.hpp`, `ml/rl.hpp`, `ml/distill.hpp` (the users). Group: \ref core.
///
/// **Implementations.** One each, on the host. A device library overloads the backend-first functions for its own
/// backend type, found by argument-dependent lookup (TRAINING.md T9), and is checked against these.
///
/// **Guarantees.** `constexpr`, no allocation. `noexcept` for the standard floating-point types. Linear in the logits:
/// two passes over each row forward, three backward. The backward writes (it does not add): a sequence's gradient
/// in its logits is complete after one call.
///
/// **Errors.** Preconditions, asserted: the targets are below the row's width; the spans' sizes agree (`logits` is
/// `n · V` for `n` targets; the token log-probabilities and the mask are `rows · length`). A mask is 0 or 1.
///
/// **Performance.** `token_logprobs_rows` and its backward against the same loops by hand
/// (`benchmarks/ml/logprobs_benchmark.cpp`): see `BENCHMARKS.md`; level within noise.
///
/// **Tests and demos.** `tests/ml/logprobs_test.cpp`, `demos/ml/logprobs_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`,
/// `ml/row.hpp`, `ml/softmax.hpp`.
///
/// **Known issues.** A vocabulary padded to a hardware multiple (`cross_entropy_rows`' `classes`) is not handled
/// here: pass the unpadded rows. A soft cap on the logits is the caller's, before this.
///
/// **Status.** Experimental, new in Unreleased.

#ifndef snicholls_core_ml_logprobs_hpp
#define snicholls_core_ml_logprobs_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/softmax.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief How a sequence's token log-probabilities become one number: their sum or their mean over the response tokens.
/// \ingroup core
enum class logprob_reduction : std::uint8_t
{
    sum,    ///< `Σ_t mask_t · logp_t`: the sequence's log-probability (DPO, IPO, GRPO).
    mean,   ///< The sum divided by the response's length (SimPO, ORPO); 0 for an empty response.
};

namespace detail::logprobs
{
    // The row's maximum, log of the shifted sum, and 1 / sum: everything the forward and backward need
    template <typename T>
    struct row_stats
    {
        T max;
        T log_sum;
        T inv_sum;
    };

    template <typename T>
    constexpr row_stats<T> stats_of(std::span<const T> x)
    {
        const auto [m, s] = detail::softmax::shifted_sum_of(x);
        return {m, math::log(s), T(1) / s};
    }

    template <typename T>
    constexpr T entropy(std::span<const T> x, const row_stats<T>& st)
    {
        T h = T(0);
        for (const T v : x) {
            const T lp = (v - st.max) - st.log_sum;
            h -= math::exp(v - st.max) * st.inv_sum * lp;
        }
        return h;
    }
} // namespace detail::logprobs

/// \brief The log-probability of one target under a row of logits: `(x_target - max) - log Σ e^(x - max)`, `-cross_entropy` to the bit.
/// \tparam In A `row`.
/// \param logits The logits.
/// \param target The token, below the row's size.
/// \return `log softmax(logits)[target]`, at most 0.
template <row In>
constexpr row_value_t<In> token_logprob(const In& logits, std::size_t target) noexcept(std::is_floating_point_v<row_value_t<In>>)
{
    const auto x = as_row(logits);
    assert(target < x.size());
    const auto [m, s] = detail::softmax::shifted_sum_of(x);
    return (x[target] - m) - math::log(s);
}

/// \brief The log-probability of one target and, in the same pass, its gradient `dlogits = dlogp · (one_hot(target) - softmax(logits))`.
/// \tparam In A `row`.
/// \tparam Out A `mutable_row` of the same number type.
/// \param logits The logits.
/// \param target The token, below the row's size.
/// \param dlogits The gradient, written: the row's size; may be `logits`.
/// \param dlogp The gradient of the loss in this log-probability.
/// \return The log-probability.
template <row In, mutable_row Out>
    requires std::same_as<row_value_t<In>, row_value_t<Out>>
constexpr row_value_t<In> token_logprob_with_gradient(const In& logits, std::size_t target, Out&& dlogits,
                                                      std::type_identity_t<row_value_t<In>> dlogp) noexcept(std::is_floating_point_v<row_value_t<In>>)
{
    using T = row_value_t<In>;
    const auto x = as_row(logits);
    const auto g = as_mutable_row(dlogits);
    assert(target < x.size() && g.size() == x.size());
    const auto st = detail::logprobs::stats_of(x);
    const T lp = (x[target] - st.max) - st.log_sum;
    for (std::size_t i = 0; i < x.size(); ++i)
        g[i] = dlogp * ((i == target ? T(1) : T(0)) - math::exp(x[i] - st.max) * st.inv_sum);
    return lp;
}

/// \brief The entropy of the softmax of a row of logits: `H = -Σ p_i log p_i`, in nats, with `log p` taken as `(x - max) - log(sum)`.
/// \tparam In A `row`.
/// \param logits The logits.
/// \return The entropy, between 0 and `log V`.
template <row In>
constexpr row_value_t<In> token_entropy(const In& logits) noexcept(std::is_floating_point_v<row_value_t<In>>)
{
    const auto x = as_row(logits);
    return detail::logprobs::entropy(x, detail::logprobs::stats_of(x));
}

/// \brief Each row's log-probability of its target, and optionally its entropy, on the backend (TRAINING.md T9, T11).
/// \param logits `n` rows of `V`, end to end.
/// \param targets `n` targets, each below `V`.
/// \param logps Receives `n` log-probabilities.
/// \param entropies Receives `n` entropies, or empty to skip them.
template <math::math_number T, typename B>
constexpr void token_logprobs_rows(const B&, std::type_identity_t<std::span<const T>> logits, std::span<const std::uint32_t> targets,
                                   std::span<T> logps, std::type_identity_t<std::span<T>> entropies = {}) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = targets.size();
    assert(logps.size() == n && (entropies.empty() || entropies.size() == n) && (n == 0 || logits.size() % n == 0));
    if (n == 0)
        return;
    const std::size_t v = logits.size() / n;
    for (std::size_t t = 0; t < n; ++t) {
        const auto x = logits.subspan(t * v, v);
        assert(targets[t] < v);
        const auto st = detail::logprobs::stats_of(x);
        logps[t] = (x[targets[t]] - st.max) - st.log_sum;
        if (!entropies.empty())
            entropies[t] = detail::logprobs::entropy(x, st);
    }
}

/// \brief The backward pass of `token_logprobs_rows`: each row of logits replaced by its gradient,
/// `dlogp · (one_hot - p) - dH · p (log p + H)`, in one fused pass per row.
/// \param logits `n` rows of `V`: replaced by the gradient.
/// \param targets `n` targets.
/// \param dlogps The gradient of the loss in each row's log-probability, `n` (0 for a masked position).
/// \param dentropies The gradient of the loss in each row's entropy, `n`, or empty when no entropy term is trained.
template <math::math_number T, typename B>
constexpr void token_logprobs_rows_backward(const B&, std::span<T> logits, std::span<const std::uint32_t> targets,
                                            std::type_identity_t<std::span<const T>> dlogps,
                                            std::type_identity_t<std::span<const T>> dentropies = {}) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = targets.size();
    assert(dlogps.size() == n && (dentropies.empty() || dentropies.size() == n) && (n == 0 || logits.size() % n == 0));
    if (n == 0)
        return;
    const std::size_t v = logits.size() / n;
    for (std::size_t t = 0; t < n; ++t) {
        const auto x = logits.subspan(t * v, v);
        assert(targets[t] < v);
        const auto st = detail::logprobs::stats_of(std::span<const T>(x));
        const T dh = dentropies.empty() ? T(0) : dentropies[t];
        const T h = dh == T(0) ? T(0) : detail::logprobs::entropy(std::span<const T>(x), st);
        const T dl = dlogps[t];
        for (std::size_t i = 0; i < v; ++i) {
            const T lp = (x[i] - st.max) - st.log_sum;
            const T p = math::exp(x[i] - st.max) * st.inv_sum;
            const T g = dl * ((i == targets[t] ? T(1) : T(0)) - p);
            x[i] = dh == T(0) ? g : g - dh * p * (lp + h);
        }
    }
}

/// \brief The number of response tokens in a mask. \param mask 0 or 1 per position. \return The count.
constexpr std::size_t response_length(std::span<const std::uint8_t> mask) noexcept
{
    std::size_t c = 0;
    for (const auto m : mask)
        c += m != 0 ? 1 : 0;
    return c;
}

/// \brief Each row's token log-probabilities summed, or averaged, over its response tokens (the mask's 1s), on the backend.
/// \param token_logps `rows · length` per-token log-probabilities.
/// \param mask `rows · length`: 1 for a response token, 0 for the prompt and padding.
/// \param out Receives `rows` values: the sum, or the mean (0 for an empty response).
/// \param reduction Sum or mean.
template <math::math_number T, typename B>
constexpr void sequence_logprobs(const B&, std::type_identity_t<std::span<const T>> token_logps, std::span<const std::uint8_t> mask,
                                 std::span<T> out, logprob_reduction reduction) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t rows = out.size();
    assert(mask.size() == token_logps.size() && (rows == 0 || token_logps.size() % rows == 0));
    if (rows == 0)
        return;
    const std::size_t len = token_logps.size() / rows;
    for (std::size_t r = 0; r < rows; ++r) {
        T s = T(0);
        std::size_t c = 0;
        for (std::size_t t = r * len; t < (r + 1) * len; ++t)
            if (mask[t] != 0) {
                s += token_logps[t];
                ++c;
            }
        out[r] = reduction == logprob_reduction::sum ? s : (c == 0 ? T(0) : s / T(c));
    }
}

/// \brief The backward pass of `sequence_logprobs`: `dtoken_t = mask_t · dseq_r`, divided by the response's length for a mean.
/// \param mask `rows · length`.
/// \param dseq The gradient in each row's value, `rows`.
/// \param dtoken Written, `rows · length`: 0 on the prompt and padding.
/// \param reduction The forward's reduction.
template <math::math_number T, typename B>
constexpr void sequence_logprobs_backward(const B&, std::span<const std::uint8_t> mask, std::type_identity_t<std::span<const T>> dseq,
                                          std::span<T> dtoken, logprob_reduction reduction) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t rows = dseq.size();
    assert(mask.size() == dtoken.size() && (rows == 0 || dtoken.size() % rows == 0));
    if (rows == 0)
        return;
    const std::size_t len = dtoken.size() / rows;
    for (std::size_t r = 0; r < rows; ++r) {
        const auto m = mask.subspan(r * len, len);
        const std::size_t c = response_length(m);
        const T g = reduction == logprob_reduction::sum ? dseq[r] : (c == 0 ? T(0) : dseq[r] / T(c));
        for (std::size_t t = 0; t < len; ++t)
            dtoken[r * len + t] = m[t] != 0 ? g : T(0);
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_logprobs_hpp
