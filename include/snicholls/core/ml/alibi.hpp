//
//  alibi.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` ALiBi: each head's slope as ggml computes it, a compile-time table of slopes, the linear bias of a query and a key, and attention with that bias, forward and backward.
/// \ingroup core
///
/// **Overview.** ALiBi (Press et al., 2022; BLOOM, MPT, Baichuan 13B, Falcon's early versions) has no positional
/// rotation: each head adds `-m_h · |i - j|` to the score of query `i` against key `j`, with slopes `m_h` a geometric
/// series over the heads. ggml computes them (`ggml_compute_forward_soft_max_f32` and
/// `ggml_compute_forward_flash_attn_ext_f16_one_chunk`, `ggml/src/ggml-cpu/ops.cpp`, llama.cpp 39039aa) as
/// `n = 2^floor(log2 H)`, `m0 = 2^(-max_bias / n)`, `m1 = 2^(-max_bias / 2 / n)`, and
/// `m_h = m0^(h + 1)` for `h < n`, `m1^(2(h - n) + 1)` above, so a head count that is not a power of two takes the
/// odd powers of the next series (the paper's rule); the cache's mask holds `-|p_q - p_k|`
/// (`llama_kv_cache::set_input_kq_mask` with `use_alibi`), multiplied by the slope and added after the scale (and any
/// soft cap).
///
/// - `alibi_slope<T>(h, heads, max_bias)` and `alibi_slopes<T, Heads>(max_bias)`, a `std::array` the compiler can
///   build (`max_bias` 8 in every ALiBi model; 0 gives slopes of 1, ggml's "no ALiBi").
/// - `alibi_bias(slope, query_pos, key_pos)`: `-slope · |query_pos - key_pos|`.
/// - `attend_alibi(q, keys, values, out, scale, slope, query_pos, first_key_pos)`: `attend` over keys at consecutive
///   positions from `first_key_pos`, with the bias added to each score; `attend_alibi_backward` its gradients (the bias
///   is a constant of the inputs).
///
/// \snippet ml/alibi_demo.cpp alibi
///
/// **Relationships.** `ml/attention.hpp` (`attend` and its recurrence; with a slope of 0 this is `attend` to the bit);
/// `ml/kv_cells.hpp` (`attend_cells` takes a slope for keys at scattered positions). Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the keys; the slopes are exact powers where `max_bias / n` is
/// an integer exponent of 2. The recurrence and its order are `attend`'s.
///
/// **Errors.** Preconditions, asserted: `heads > 0`, `h < heads`, `max_bias >= 0`; the rows' sizes as `attend`'s.
///
/// **Performance.** Not benchmarked: one multiplication and one addition per score over `attend`, which is.
///
/// **Tests and demos.** `tests/ml/alibi_test.cpp`, `demos/ml/alibi_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`,
/// `ml/attention.hpp`, `ml/row.hpp`, `snicholls/core/math/constexpr_math.hpp`.
///
/// **Known issues.** ggml computes the slopes in `float` with `powf`; core's `math::pow` agrees to the last place where
/// both are exact (every head of an 8-, 16-, 32- or 64-head model at `max_bias` 8) and within an ulp otherwise.
/// ggml stores the mask in `float16` for flash attention: positions beyond 2048 apart are then rounded; core keeps the
/// distance exact.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_alibi_hpp
#define snicholls_core_ml_alibi_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/row.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::ml
{

/// \brief Head `h`'s ALiBi slope, as ggml computes it.
/// \tparam T A `math::math_number`.
/// \param h The head, below `heads`.
/// \param heads The heads of the layer.
/// \param max_bias The bias of the last head of a power-of-two series: 8 in BLOOM, MPT and the paper.
/// \return `m_h`.
template <math::math_number T>
[[nodiscard]] constexpr T alibi_slope(std::size_t h, std::size_t heads, T max_bias = T(8))
{
    assert(heads > 0 && h < heads && max_bias >= T(0));
    if (max_bias == T(0))
        return T(1);
    std::size_t n = 1;
    while (2 * n <= heads)
        n *= 2;   // 2^floor(log2 heads)
    const T m0 = math::pow(T(2), -max_bias / T(n)), m1 = math::pow(T(2), -(max_bias / T(2)) / T(n));
    return h < n ? math::pow(m0, T(h + 1)) : math::pow(m1, T(2 * (h - n) + 1));
}

/// \brief Every head's ALiBi slope, in a `std::array` the compiler can build.
/// \tparam T A `math::math_number`.
/// \tparam Heads The heads.
/// \param max_bias As `alibi_slope`.
/// \return `m_0` to `m_{Heads - 1}`.
template <math::math_number T, std::size_t Heads>
    requires(Heads > 0)
[[nodiscard]] constexpr std::array<T, Heads> alibi_slopes(T max_bias = T(8))
{
    std::array<T, Heads> out{};
    for (std::size_t h = 0; h < Heads; ++h)
        out[h] = alibi_slope<T>(h, Heads, max_bias);
    return out;
}

/// \brief The bias ALiBi adds to a score: `-slope · |query_pos - key_pos|`.
/// \tparam T A `math::math_number`.
/// \param slope The head's slope.
/// \param query_pos The query's position.
/// \param key_pos The key's position.
/// \return The bias, at most 0.
template <math::math_number T>
[[nodiscard]] constexpr T alibi_bias(T slope, std::int64_t query_pos, std::int64_t key_pos) noexcept
{
    const std::int64_t d = query_pos - key_pos;
    return slope * -T(d < 0 ? -d : d);
}

/// \brief Attention of one query with ALiBi over keys at consecutive positions: score `j` is
/// `scale · q · k_j + alibi_bias(slope, query_pos, first_key_pos + j)`, then `attend`'s recurrence.
/// \param q The query, `d` channels.
/// \param keys The keys, `d` each, end to end.
/// \param values The values, as the keys.
/// \param out Receives the weighted values, `d`.
/// \param scale The score scale, usually `1/sqrt(d)`.
/// \param slope The head's slope.
/// \param query_pos The query's position.
/// \param first_key_pos The first key's position.
/// \return The log-sum-exp of the biased scores, for the backward pass; -infinity with no keys.
template <row Q, row K, row V, mutable_row Out>
    requires std::same_as<row_value_t<Q>, row_value_t<K>> && std::same_as<row_value_t<Q>, row_value_t<V>>
             && std::same_as<row_value_t<Q>, row_value_t<Out>>
constexpr row_value_t<Q> attend_alibi(const Q& q, const K& keys, const V& values, Out&& out, std::type_identity_t<row_value_t<Q>> scale,
                                      std::type_identity_t<row_value_t<Q>> slope, std::int64_t query_pos, std::int64_t first_key_pos)
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
    for (std::size_t j = 0; d != 0 && j < ks.size() / d; ++j) {
        const T s = scale * detail::attention::dot(qs, ks.subspan(j * d, d));
        detail::attention::fold(max, sum, y, s + alibi_bias(slope, query_pos, first_key_pos + std::int64_t(j)), vs.subspan(j * d, d));
    }
    for (T& v : y)
        v = sum == T(0) ? T(0) : v / sum;
    return sum == T(0) ? -std::numeric_limits<T>::infinity() : max + math::log(sum);
}

/// \brief The backward pass of `attend_alibi`: writes `dq`, adds into `dkeys` and `dvalues` (the bias depends on no
/// input, so these are `attend_backward`'s with the biased probabilities).
/// \param q The query.
/// \param keys The keys.
/// \param values The values.
/// \param out The forward pass's output.
/// \param dout The gradient in the output.
/// \param lse The forward pass's log-sum-exp.
/// \param scale The score scale.
/// \param slope The head's slope.
/// \param query_pos The query's position.
/// \param first_key_pos The first key's position.
/// \param dq Receives the gradient in the query.
/// \param dkeys Accumulated into: the gradients in the keys.
/// \param dvalues Accumulated into: the gradients in the values.
template <row Q, row K, row V, row O, row DO, mutable_row DQ, mutable_row DK, mutable_row DV>
    requires std::same_as<row_value_t<Q>, row_value_t<K>> && std::same_as<row_value_t<Q>, row_value_t<V>>
             && std::same_as<row_value_t<Q>, row_value_t<O>> && std::same_as<row_value_t<Q>, row_value_t<DO>>
             && std::same_as<row_value_t<Q>, row_value_t<DQ>> && std::same_as<row_value_t<Q>, row_value_t<DK>>
             && std::same_as<row_value_t<Q>, row_value_t<DV>>
constexpr void attend_alibi_backward(const Q& q, const K& keys, const V& values, const O& out, const DO& dout,
                                     std::type_identity_t<row_value_t<Q>> lse, std::type_identity_t<row_value_t<Q>> scale,
                                     std::type_identity_t<row_value_t<Q>> slope, std::int64_t query_pos, std::int64_t first_key_pos,
                                     DQ&& dq, DK&& dkeys, DV&& dvalues)
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
        const T s = scale * detail::attention::dot(qs, k) + alibi_bias(slope, query_pos, first_key_pos + std::int64_t(j));
        const T p = math::exp(s - lse);
        const T ds = p * (detail::attention::dot(g, v) - big_d);
        for (std::size_t i = 0; i < d; ++i) {
            gv[j * d + i] += p * g[i];
            gq[i] += scale * ds * k[i];
            gk[j * d + i] += scale * ds * qs[i];
        }
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_alibi_hpp
