//
//  softmax.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` softmax and losses: `softmax`, `log_softmax`, `logsumexp`, cross-entropy with its fused gradient, bits per byte, and `online_softmax`, a running softmax that merges across workers.
/// \ingroup core
///
/// **Overview.** The normalisation at the end of every language model and inside every attention head. Each
/// function subtracts the maximum before exponentiating, so no logit overflows `exp`, and the losses are computed
/// from `logsumexp` rather than from a probability, so an improbable target gives a large finite loss instead of
/// `log(0)`.
///
/// - `softmax(in, out)`, `log_softmax(in, out)`, `logsumexp(in)` over one row of logits (any `row`: a vector, an array,
///   a span). `out` may be `in`.
/// - `softmax_backward(probs, dprobs, dlogits)`: the Jacobian-vector product of the backward pass.
/// - `cross_entropy(logits, target)` and `cross_entropy_with_gradient(logits, target, dlogits, scale)`: the loss of
///   one row and, in the same pass, its gradient `scale · (softmax - one_hot)`, which may overwrite the logits
///   (llm.c's fused classifier).
/// - `bits_per_byte(nats, bytes)`: a loss independent of the tokenizer.
/// - `online_softmax<T>`: the maximum and the sum of `e^(x - max)` seen so far, updated one value at a time and
///   merged with another worker's. Splitting a row of a million logits across threads, or across the blocks of a
///   GPU kernel, and merging the partials gives the same `logsumexp` as one pass: this is the reduction under
///   flash attention.
///
/// \snippet ml/softmax_demo.cpp softmax
///
/// **Relationships.** `ml/activations.hpp`; `ml/attention.hpp` extends `online_softmax` with the weighted sum of the
/// values. Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`. No allocation. `online_softmax` is trivially copyable, and `merge` is commutative
/// and associative up to rounding, so partials can be reduced in any order. Complexity: linear in the row, two
/// passes for `softmax` and `logsumexp`, one for `online_softmax`.
///
/// **Errors.** Preconditions, asserted: `in` and `out` have equal sizes; `target` is an index of the row. An empty
/// row has `logsumexp` -infinity. A row of only -infinity has `logsumexp` -infinity and a softmax of NaN, as in
/// PyTorch; NaN propagates.
///
/// **Performance.** Benchmarked against the same loops written by hand (`benchmarks/ml/softmax_benchmark.cpp`).
///
/// **Tests and demos.** `tests/ml/softmax_test.cpp`, `demos/ml/softmax_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<limits>`, `<span>`, `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`,
/// `ml/row.hpp`.
///
/// **Known issues.** Sums accumulate in `T`; a row of a million `float` logits loses about three digits to
/// rounding, which is within what the loss needs.
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_ml_softmax_hpp
#define snicholls_core_ml_softmax_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief A running softmax: the maximum and `sum e^(x - max)` of the values seen, updated one at a time and merged across workers.
/// \ingroup core
/// \tparam T A `math::math_number`.
template <math::math_number T>
struct online_softmax
{
    T max = -std::numeric_limits<T>::infinity();   ///< The largest value seen; -infinity before any.
    T sum = T(0);                                   ///< `sum e^(x - max)` over the values seen.

    /// \brief Adds one value.
    /// \param x The value; -infinity adds nothing.
    constexpr void add(T x)
    {
        if (x == -std::numeric_limits<T>::infinity())
            return;
        if (x > max) {
            sum = sum * math::exp(max - x) + T(1);
            max = x;
        } else {
            sum += math::exp(x - max);
        }
    }

    /// \brief Merges another worker's partial: the result is as if this one had seen both sets of values.
    /// \param other The other partial.
    constexpr void merge(const online_softmax& other)
    {
        if (other.sum == T(0))
            return;
        if (sum == T(0)) {
            *this = other;
            return;
        }
        if (other.max > max) {
            sum = sum * math::exp(max - other.max) + other.sum;
            max = other.max;
        } else {
            sum += other.sum * math::exp(other.max - max);
        }
    }

    /// \brief `log sum e^x` of the values seen: -infinity before any.
    /// \return The result.
    [[nodiscard]] constexpr T log_sum_exp() const
    {
        return sum == T(0) ? -std::numeric_limits<T>::infinity() : max + math::log(sum);
    }

    /// \brief The softmax weight of a value `x` among those seen: `e^(x - max) / sum`.
    /// \param x The value.
    /// \return The weight.
    [[nodiscard]] constexpr T probability(T x) const { return math::exp(x - max) / sum; }
};

namespace detail::softmax
{
    template <typename T>
    constexpr T max_of(std::span<const T> in)
    {
        T m = -std::numeric_limits<T>::infinity();
        for (const T x : in) {
            if (x != x)
                return x;   // NaN
            if (x > m)
                m = x;
        }
        return m;
    }

    template <typename T>
    constexpr T logsumexp(std::span<const T> in)
    {
        const T m = max_of(in);
        if (m == -std::numeric_limits<T>::infinity() || m != m)
            return m;
        T s = T(0);
        for (const T x : in)
            s += math::exp(x - m);
        return m + math::log(s);
    }
} // namespace detail::softmax

/// \brief `log sum e^x` of a row, with the maximum subtracted first: -infinity for an empty row.
/// \tparam In A `row`.
/// \param in The row.
/// \return The result.
template <row In>
constexpr row_value_t<In> logsumexp(const In& in)
{
    return detail::softmax::logsumexp(as_row(in));
}

/// \brief The softmax of a row: `out_i = e^(in_i - max) / sum_j e^(in_j - max)`.
/// \tparam In A `row`.
/// \tparam Out A `mutable_row` of the same number type.
/// \param in The logits.
/// \param out The probabilities: the same size as `in`, and may be `in`.
template <row In, mutable_row Out>
    requires std::same_as<row_value_t<In>, row_value_t<Out>>
constexpr void softmax(const In& in, Out&& out)
{
    using T = row_value_t<In>;
    const auto x = as_row(in);
    const auto y = as_mutable_row(out);
    assert(x.size() == y.size());
    const T m = detail::softmax::max_of(x);
    T s = T(0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = math::exp(x[i] - m);
        s += y[i];
    }
    const T inv = T(1) / s;
    for (T& p : y)
        p *= inv;
}

/// \brief The log-softmax of a row: `out_i = in_i - logsumexp(in)`, exact where `log(softmax)` would underflow.
/// \tparam In A `row`.
/// \tparam Out A `mutable_row` of the same number type.
/// \param in The logits.
/// \param out The log-probabilities: the same size as `in`, and may be `in`.
template <row In, mutable_row Out>
    requires std::same_as<row_value_t<In>, row_value_t<Out>>
constexpr void log_softmax(const In& in, Out&& out)
{
    const auto x = as_row(in);
    const auto y = as_mutable_row(out);
    assert(x.size() == y.size());
    const auto lse = detail::softmax::logsumexp(x);
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] - lse;
}

/// \brief The backward pass of `softmax`: `dlogits_i = p_i (dp_i - sum_j p_j dp_j)`.
/// \tparam P A `row`.
/// \tparam D A `row` of the same number type.
/// \tparam Out A `mutable_row` of the same number type.
/// \param probs The softmax output of the forward pass.
/// \param dprobs The gradient of the loss in the probabilities.
/// \param dlogits The gradient in the logits: the same size, written; may be `dprobs`.
template <row P, row D, mutable_row Out>
    requires std::same_as<row_value_t<P>, row_value_t<D>> && std::same_as<row_value_t<P>, row_value_t<Out>>
constexpr void softmax_backward(const P& probs, const D& dprobs, Out&& dlogits)
{
    using T = row_value_t<P>;
    const auto p = as_row(probs);
    const auto d = as_row(dprobs);
    const auto g = as_mutable_row(dlogits);
    assert(p.size() == d.size() && p.size() == g.size());
    T dot = T(0);
    for (std::size_t i = 0; i < p.size(); ++i)
        dot += p[i] * d[i];
    for (std::size_t i = 0; i < p.size(); ++i)
        g[i] = p[i] * (d[i] - dot);
}

/// \brief The cross-entropy of one row against a target class: `logsumexp(logits) - logits[target]`, in nats.
/// \tparam In A `row`.
/// \param logits The logits.
/// \param target The index of the correct class: less than the row's size.
/// \return The loss, at least 0.
template <row In>
constexpr row_value_t<In> cross_entropy(const In& logits, std::size_t target)
{
    const auto x = as_row(logits);
    assert(target < x.size());
    return detail::softmax::logsumexp(x) - x[target];
}

/// \brief The cross-entropy of one row and its gradient in one pass: `dlogits = scale · (softmax(logits) - one_hot(target))`.
///
/// `scale` is the gradient of the total loss in this row's loss: `1 / (B T)` for a mean over a batch.
/// \tparam In A `row`.
/// \tparam Out A `mutable_row` of the same number type.
/// \param logits The logits.
/// \param target The index of the correct class: less than the row's size.
/// \param dlogits The gradient: the same size, written; may be `logits`, to compute the gradient in place.
/// \param scale The factor on the gradient.
/// \return The loss, in nats.
template <row In, mutable_row Out>
    requires std::same_as<row_value_t<In>, row_value_t<Out>>
constexpr row_value_t<In> cross_entropy_with_gradient(const In& logits, std::size_t target, Out&& dlogits,
                                                      std::type_identity_t<row_value_t<In>> scale = 1)
{
    using T = row_value_t<In>;
    const auto x = as_row(logits);
    const auto g = as_mutable_row(dlogits);
    assert(target < x.size() && x.size() == g.size());
    const T lse = detail::softmax::logsumexp(x);
    const T loss = lse - x[target];
    for (std::size_t i = 0; i < x.size(); ++i)
        g[i] = scale * (math::exp(x[i] - lse) - (i == target ? T(1) : T(0)));
    return loss;
}

/// \brief Bits per byte: a loss in nats over tokens, divided by `ln 2` and by the bytes those tokens encode, so models with different tokenizers compare.
/// \param total_nats The sum of the per-token losses, in nats.
/// \param total_bytes The bytes of text the target tokens encode (special tokens count 0).
/// \return The bits per byte: infinity when `total_bytes` is 0.
constexpr double bits_per_byte(double total_nats, std::size_t total_bytes)
{
    if (total_bytes == 0)
        return std::numeric_limits<double>::infinity();
    return total_nats / (0.693147180559945309417232121458176568 * static_cast<double>(total_bytes));
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_softmax_hpp
