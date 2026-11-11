//
//  attention.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` attention for one query: `attend` over a block of keys and values in one pass, its backward pass, `attention_accumulator`, the partial that merges across workers, and `attention_window` for causal and sliding-window masks.
/// \ingroup core
///
/// **Overview.** Scaled dot-product attention, `out = softmax(scale · q·K) · V`, for one query of one head: the
/// inner loop of llm.c's `attention_forward` and of nanochat's attention. It is computed in one pass over the keys
/// with a running softmax (flash attention's recurrence), so there is no row of scores to store and the keys can be
/// split into blocks, each attended to by a different worker, a thread or a GPU block, and the partials merged.
///
/// - `attend(q, keys, values, out, scale)`: the keys and values are `n` rows of `q.size()` channels, laid end to end
///   in one row each (a slice of a KV cache). Writes `out` and returns `logsumexp` of the scores, which the backward
///   pass needs. With no keys, `out` is zero and the result -infinity.
/// - `attend_backward(q, keys, values, out, dout, lse, scale, dq, dkeys, dvalues)`: writes `dq`, adds into `dkeys`
///   and `dvalues` (which sum over queries), recomputing each score from `lse` as flash attention does.
/// - `attention_accumulator<T, D>`: a fixed-size partial, the running softmax plus the weighted sum of the values.
///   `add` a score and a value, `merge` another worker's, `result` into the output. Trivially copyable; `merge` is
///   commutative and associative up to rounding.
/// - `attention_window(pos, window)`: the keys a query at `pos` may see, causal and optionally limited to the last
///   `window` positions (nanochat's sliding window). Pass that slice of the keys and values to `attend`.
///
/// Multi-head and grouped-query attention are loops over heads: each query head attends to the key and value head it
/// shares. Rotate queries and keys with `ml/rope.hpp` first.
///
/// \snippet ml/attention_demo.cpp attention
///
/// **Relationships.** Extends `online_softmax` (`ml/softmax.hpp`) with the values; `ml/rope.hpp`; `ml/row.hpp`.
/// Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** `constexpr`, no allocation. `attend` reads each key and value once: O(n · D). The result equals
/// `softmax` of the scores times the values up to rounding, and a sum of merged partials equals one pass up to
/// rounding.
///
/// **Errors.** Preconditions, asserted: `keys` and `values` have the same size, a multiple of `q.size()`; `out`, `dq`
/// and `dout` have `q`'s size; `dkeys` and `dvalues` have `keys`' size. A score of -infinity (a masked key) has
/// weight zero.
///
/// **Performance.** Against the same recurrence written by hand (`benchmarks/ml/attention_benchmark.cpp`), one query
/// of 64 float channels: level at 256 keys, within 5% at 4,096, about 10.5 million keys per second (Xeon W-3245, Release, Apple clang 17, 2026-11-11, one
/// thread).
///
/// **Tests and demos.** `tests/ml/attention_test.cpp`, `demos/ml/attention_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<limits>`, `<span>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`, `ml/softmax.hpp`.
///
/// **Known issues.** Sums accumulate in `T`, as llm.c's do in `float`.
///
/// **Status.** Experimental, new in 0.7.0.

#ifndef snicholls_core_ml_attention_hpp
#define snicholls_core_ml_attention_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/softmax.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>

namespace snicholls::ml
{

/// \brief The keys a query may attend to: positions `[first, last)`.
/// \ingroup core
struct key_range
{
    std::size_t first;   ///< The first visible key.
    std::size_t last;    ///< One past the last visible key.

    /// \brief The number of visible keys.
    /// \return `last - first`.
    [[nodiscard]] constexpr std::size_t size() const noexcept { return last - first; }
};

/// \brief The keys a query at `pos` may see: causal, and within the last `window` positions if `window` is not 0.
/// \param pos The query's position.
/// \param window The sliding window, counting the query's own position; 0 for none.
/// \return `[max(0, pos + 1 - window), pos + 1)`.
constexpr key_range attention_window(std::size_t pos, std::size_t window = 0) noexcept
{
    return {window == 0 || window > pos ? 0 : pos + 1 - window, pos + 1};
}

namespace detail::attention
{
    template <typename T>
    constexpr T dot(std::span<const T> a, std::span<const T> b)
    {
        T s = T(0);
        for (std::size_t i = 0; i < a.size(); ++i)
            s += a[i] * b[i];
        return s;
    }

    // One step of the recurrence: fold value v with score s into (max, sum, acc)
    template <typename T>
    constexpr void fold(T& max, T& sum, std::span<T> acc, T s, std::span<const T> v)
    {
        if (s == -std::numeric_limits<T>::infinity())
            return;
        if (s > max) {
            const T scale = math::exp(max - s);
            for (std::size_t i = 0; i < acc.size(); ++i)
                acc[i] = acc[i] * scale + v[i];
            sum = sum * scale + T(1);
            max = s;
        } else {
            const T w = math::exp(s - max);
            for (std::size_t i = 0; i < acc.size(); ++i)
                acc[i] += w * v[i];
            sum += w;
        }
    }
} // namespace detail::attention

/// \brief A partial of attention for one query: the running softmax of the scores and the weighted sum of the values.
/// \ingroup core
/// \tparam T A `math::math_number`.
/// \tparam D The channels of a head.
template <math::math_number T, std::size_t D>
struct attention_accumulator
{
    online_softmax<T> weights{};   ///< The maximum score and `sum e^(s - max)`.
    std::array<T, D> acc{};        ///< `sum e^(s - max) · v`.

    /// \brief Adds one key's score and value.
    /// \param score The scaled score `scale · q·k`; -infinity adds nothing.
    /// \param value The key's value, `D` channels.
    template <row V>
        requires std::same_as<row_value_t<V>, T>
    constexpr void add(T score, const V& value)
    {
        const auto v = as_row(value);
        assert(v.size() == D);
        detail::attention::fold(weights.max, weights.sum, std::span<T>(acc), score, v);
    }

    /// \brief Merges another worker's partial: the result is as if this one had seen both sets of keys.
    /// \param other The other partial.
    constexpr void merge(const attention_accumulator& other)
    {
        if (other.weights.sum == T(0))
            return;
        if (weights.sum == T(0)) {
            *this = other;
            return;
        }
        const T m = std::max(weights.max, other.weights.max);
        const T a = math::exp(weights.max - m), b = math::exp(other.weights.max - m);
        for (std::size_t i = 0; i < D; ++i)
            acc[i] = acc[i] * a + other.acc[i] * b;
        weights.sum = weights.sum * a + other.weights.sum * b;
        weights.max = m;
    }

    /// \brief Writes the attention output, `acc / sum`: zero if no key was added.
    /// \param out `D` channels.
    template <mutable_row Out>
        requires std::same_as<row_value_t<Out>, T>
    constexpr void result(Out&& out) const
    {
        const auto y = as_mutable_row(out);
        assert(y.size() == D);
        for (std::size_t i = 0; i < D; ++i)
            y[i] = weights.sum == T(0) ? T(0) : acc[i] / weights.sum;
    }

    /// \brief `logsumexp` of the scores added, for the backward pass: -infinity before any.
    /// \return The result.
    [[nodiscard]] constexpr T log_sum_exp() const { return weights.log_sum_exp(); }
};

/// \brief Attention for one query over a block of keys and values, in one pass: `out = softmax(scale · q·K) · V`.
/// \param q The query, `D` channels.
/// \param keys `n` keys of `D` channels, end to end.
/// \param values `n` values of `D` channels, end to end.
/// \param out The output, `D` channels; may not alias the inputs.
/// \param scale The score scale, usually `1 / sqrt(D)`.
/// \return `logsumexp` of the scaled scores, for `attend_backward`; -infinity with no keys.
template <row Q, row K, row V, mutable_row Out>
    requires std::same_as<row_value_t<Q>, row_value_t<K>> && std::same_as<row_value_t<Q>, row_value_t<V>>
             && std::same_as<row_value_t<Q>, row_value_t<Out>>
constexpr row_value_t<Q> attend(const Q& q, const K& keys, const V& values, Out&& out, std::type_identity_t<row_value_t<Q>> scale)
{
    using T = row_value_t<Q>;
    const auto qs = as_row(q);
    const auto ks = as_row(keys);
    const auto vs = as_row(values);
    const auto y = as_mutable_row(out);
    const std::size_t d = qs.size();
    assert(y.size() == d && ks.size() == vs.size() && (d == 0 || ks.size() % d == 0));
    std::fill(y.begin(), y.end(), T(0));
    T max = -std::numeric_limits<T>::infinity(), sum = T(0);
    for (std::size_t j = 0; d != 0 && j < ks.size() / d; ++j)
        detail::attention::fold(max, sum, y, scale * detail::attention::dot(qs, ks.subspan(j * d, d)), vs.subspan(j * d, d));
    for (T& v : y)
        v = sum == T(0) ? T(0) : v / sum;
    return sum == T(0) ? -std::numeric_limits<T>::infinity() : max + math::log(sum);
}

/// \brief The backward pass of `attend`: writes `dq`, adds into `dkeys` and `dvalues`.
///
/// With `p_j = e^(scale · q·k_j - lse)` and `D = dout·out`: `dv_j += p_j · dout`,
/// `ds_j = p_j · (dout·v_j - D)`, `dq = scale · sum ds_j · k_j`, `dk_j += scale · ds_j · q`.
/// \param q The query.
/// \param keys The keys.
/// \param values The values.
/// \param out What `attend` wrote.
/// \param dout The gradient in `out`.
/// \param lse What `attend` returned.
/// \param scale The score scale.
/// \param dq The gradient in `q`: written.
/// \param dkeys Accumulated into; `keys`' size.
/// \param dvalues Accumulated into; `values`' size.
template <row Q, row K, row V, row O, row DO, mutable_row DQ, mutable_row DK, mutable_row DV>
    requires std::same_as<row_value_t<Q>, row_value_t<K>> && std::same_as<row_value_t<Q>, row_value_t<V>>
             && std::same_as<row_value_t<Q>, row_value_t<O>> && std::same_as<row_value_t<Q>, row_value_t<DO>>
             && std::same_as<row_value_t<Q>, row_value_t<DQ>> && std::same_as<row_value_t<Q>, row_value_t<DK>>
             && std::same_as<row_value_t<Q>, row_value_t<DV>>
constexpr void attend_backward(const Q& q, const K& keys, const V& values, const O& out, const DO& dout,
                               std::type_identity_t<row_value_t<Q>> lse, std::type_identity_t<row_value_t<Q>> scale, DQ&& dq,
                               DK&& dkeys, DV&& dvalues)
{
    using T = row_value_t<Q>;
    const auto qs = as_row(q);
    const auto ks = as_row(keys);
    const auto vs = as_row(values);
    const auto o = as_row(out);
    const auto g = as_row(dout);
    const auto gq = as_mutable_row(dq);
    const auto gk = as_mutable_row(dkeys);
    const auto gv = as_mutable_row(dvalues);
    const std::size_t d = qs.size();
    assert(o.size() == d && g.size() == d && gq.size() == d && ks.size() == vs.size() && gk.size() == ks.size()
           && gv.size() == vs.size() && (d == 0 || ks.size() % d == 0));
    std::fill(gq.begin(), gq.end(), T(0));
    if (d == 0 || lse == -std::numeric_limits<T>::infinity())
        return;
    const T big_d = detail::attention::dot(g, o);
    for (std::size_t j = 0; j < ks.size() / d; ++j) {
        const auto k = ks.subspan(j * d, d);
        const auto v = vs.subspan(j * d, d);
        const T p = math::exp(scale * detail::attention::dot(qs, k) - lse);
        const T ds = p * (detail::attention::dot(g, v) - big_d);
        for (std::size_t i = 0; i < d; ++i) {
            gv[j * d + i] += p * g[i];
            gq[i] += scale * ds * k[i];
            gk[j * d + i] += scale * ds * qs[i];
        }
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_attention_hpp
