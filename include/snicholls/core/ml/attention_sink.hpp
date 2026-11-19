//
//  attention_sink.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::attend_with_sink` and its backward pass: attention for one query with a learnable sink logit, a virtual key every query attends to that carries no value (gpt-oss), with keys and values of different widths.
/// \ingroup core
///
/// **Overview.** A softmax must give its whole mass to the keys, so a head with nothing to attend to still has to
/// attend somewhere; trained models dump that mass on the first tokens ("attention sinks", Xiao et al., 2023). A
/// learnable sink gives it a place of its own: one logit `σ` per head, appended to the scaled scores before the softmax
/// and dropped after, so the weights sum to `1 - e^(σ - lse)` (gpt-oss's `sdpa`, `QK = cat([QK, S]); W = softmax(QK)[...,
/// :-1]`; the sink is not scaled):
///
///     out = Σ_j e^(s_j - lse) v_j,   lse = log(Σ_j e^(s_j) + e^σ),   s_j = scale · q·k_j
///
/// - `attend_with_sink(q, keys, values, out, scale, sink)`: the keys are `n` rows of `q.size()` channels, the values
///   `n` rows of `out.size()`, so a head whose keys and values differ in width (multi-head latent attention's 192 and
///   128) is one call. Returns `lse`, the sink included. A sink of -infinity is plain attention: `attend`'s bits when
///   the widths agree.
/// - `attend_with_sink_backward(...)`: writes `dq`, adds into `dkeys` and `dvalues`, and returns the gradient in the
///   sink, `-e^(σ - lse) · (dout·out)`, which the caller adds into the head's sink gradient.
/// - `attention_sink_weight(lse, sink)`: the mass the sink took, `e^(σ - lse)`.
///
/// The recurrence is `attend`'s (one pass, flash attention's running softmax, the sink folded last), and it is written
/// once, over a score and a value per key given by the caller (`detail::attention_sink::forward` and `backward`), so
/// the latent and sparse attentions (`ml/latent_attention.hpp`, `ml/sparse_attention.hpp`) are the same code.
///
/// **StreamingLLM's kept tokens** (Xiao et al., 2023, section 3.2) are the other form of a sink: a cache of the first
/// four tokens plus a rolling window, positions counted within the cache. Here that is the caller's choice of keys: the
/// first tokens' rows and the window's (`attention_window`) passed as one block, or two `attention_accumulator`s merged;
/// re-rotating cached keys to their positions in the cache needs the keys cached before RoPE. It is not a function of
/// its own.
///
/// \snippet ml/attention_sink_demo.cpp attention_sink
///
/// **Relationships.** `ml/attention.hpp` (`attend`, whose recurrence this is), `ml/latent_attention.hpp` and
/// `ml/sparse_attention.hpp` (the users), `transformer_config::attention_sinks`. Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** `constexpr`, no allocation, O(`n · (Dk + Dv)`). With a sink of -infinity the result is `attend`'s
/// and `attend_backward`'s to the bit (the widths equal). The weights sum to `1 - attention_sink_weight`.
///
/// **Errors.** Preconditions, asserted: `keys` is a whole number of rows of `q.size()`, `values` the same number of rows
/// of `out.size()`; the gradients have their inputs' sizes. A sink of +infinity is not allowed.
///
/// **Performance.** Not benchmarked on its own: `attend`'s loop plus one fold.
///
/// **Tests and demos.** `tests/ml/attention_sink_test.cpp`, `demos/ml/attention_sink_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<limits>`, `<span>`, `<type_traits>`,
/// `ml/attention.hpp`, `ml/row.hpp`.
///
/// **Known issues.** gpt-oss keeps its sinks in bfloat16; here they are the caller's `T`.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_attention_sink_hpp
#define snicholls_core_ml_attention_sink_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/row.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

namespace detail::attention_sink
{
    // One query over n keys: score(j) is the scaled score of key j, value(j) its value (out.size() channels). The
    // sink, unless -infinity, is folded last as a key whose value is zero. Writes out, returns logsumexp.
    template <typename T, typename Score, typename Value>
    constexpr T forward(std::size_t n, Score&& score, Value&& value, std::span<T> out, T sink)
    {
        constexpr T none = -std::numeric_limits<T>::infinity();
        std::fill(out.begin(), out.end(), T(0));
        T max = none, sum = T(0);
        for (std::size_t j = 0; j < n; ++j)
            detail::attention::fold(max, sum, out, score(j), value(j));
        if (sink != none) {
            if (sink > max) {
                const T s = math::exp(max - sink);
                for (T& a : out)
                    a = a * s;
                sum = sum * s + T(1);
                max = sink;
            } else {
                sum += math::exp(sink - max);
            }
        }
        for (T& v : out)
            v = sum == T(0) ? T(0) : v / sum;
        return sum == T(0) ? none : max + math::log(sum);
    }

    // The backward pass from the saved lse: grad(j, p, ds) is called for each key with its weight p and the gradient
    // ds in its scaled score. Returns the gradient in the sink.
    template <typename T, typename Score, typename Value, typename Grad>
    constexpr T backward(std::size_t n, Score&& score, Value&& value, std::span<const T> out, std::span<const T> dout, T lse,
                         T sink, Grad&& grad)
    {
        constexpr T none = -std::numeric_limits<T>::infinity();
        if (lse == none)
            return T(0);
        const T big_d = detail::attention::dot(dout, out);
        for (std::size_t j = 0; j < n; ++j) {
            const T p = math::exp(score(j) - lse);
            const T ds = p * (detail::attention::dot(dout, value(j)) - big_d);
            grad(j, p, ds);
        }
        return sink == none ? T(0) : -math::exp(sink - lse) * big_d;
    }
} // namespace detail::attention_sink

/// \brief The share of attention a sink took: `e^(sink - lse)`; 0 with no sink.
/// \param lse What `attend_with_sink` returned.
/// \param sink The sink logit, or -infinity.
/// \return The weight in [0, 1].
template <math::math_number T>
constexpr T attention_sink_weight(T lse, T sink)
{
    return sink == -std::numeric_limits<T>::infinity() ? T(0) : math::exp(sink - lse);
}

/// \brief Attention for one query with a sink logit: `out = Σ_j e^(scale · q·k_j - lse) v_j`, `lse` counting `e^sink`.
/// \param q The query, `Dk` channels.
/// \param keys `n` keys of `Dk`, end to end.
/// \param values `n` values of `Dv = out.size()`, end to end.
/// \param out The output, `Dv`; may not alias the inputs.
/// \param scale The score scale; the sink is not scaled.
/// \param sink The sink logit; -infinity for none.
/// \return `logsumexp` of the scaled scores and the sink; -infinity with no keys and no sink.
template <row Q, row K, row V, mutable_row Out>
    requires std::same_as<row_value_t<Q>, row_value_t<K>> && std::same_as<row_value_t<Q>, row_value_t<V>>
             && std::same_as<row_value_t<Q>, row_value_t<Out>>
constexpr row_value_t<Q> attend_with_sink(const Q& q, const K& keys, const V& values, Out&& out,
                                          std::type_identity_t<row_value_t<Q>> scale,
                                          std::type_identity_t<row_value_t<Q>> sink = -std::numeric_limits<row_value_t<Q>>::infinity())
{
    const auto qs = as_row(q), ks = as_row(keys), vs = as_row(values);
    const auto y = as_mutable_row(out);
    const std::size_t dk = qs.size(), dv = y.size(), n = dk == 0 ? 0 : ks.size() / dk;
    assert(dk > 0 && ks.size() == n * dk && vs.size() == n * dv);
    return detail::attention_sink::forward(
        n, [&](std::size_t j) { return scale * detail::attention::dot(qs, ks.subspan(j * dk, dk)); },
        [&](std::size_t j) { return vs.subspan(j * dv, dv); }, y, sink);
}

/// \brief The backward pass of `attend_with_sink`: writes `dq`, adds into `dkeys` and `dvalues`.
///
/// With `p_j = e^(s_j - lse)` and `D = dout·out`: `dv_j += p_j dout`, `ds_j = p_j (dout·v_j - D)`,
/// `dq = scale Σ ds_j k_j`, `dk_j += scale ds_j q`, and the sink's gradient `-e^(σ - lse) D`.
/// \param q The query. \param keys The keys. \param values The values. \param out What the forward wrote.
/// \param dout The gradient in `out`. \param lse What the forward returned. \param scale The score scale.
/// \param sink The sink logit. \param dq Written, `q`'s size. \param dkeys Added into, `keys`' size.
/// \param dvalues Added into, `values`' size.
/// \return The gradient in the sink logit (0 without one).
template <row Q, row K, row V, row O, row DO, mutable_row DQ, mutable_row DK, mutable_row DV>
    requires std::same_as<row_value_t<Q>, row_value_t<K>> && std::same_as<row_value_t<Q>, row_value_t<V>>
             && std::same_as<row_value_t<Q>, row_value_t<O>> && std::same_as<row_value_t<Q>, row_value_t<DO>>
             && std::same_as<row_value_t<Q>, row_value_t<DQ>> && std::same_as<row_value_t<Q>, row_value_t<DK>>
             && std::same_as<row_value_t<Q>, row_value_t<DV>>
constexpr row_value_t<Q> attend_with_sink_backward(const Q& q, const K& keys, const V& values, const O& out, const DO& dout,
                                                   std::type_identity_t<row_value_t<Q>> lse,
                                                   std::type_identity_t<row_value_t<Q>> scale,
                                                   std::type_identity_t<row_value_t<Q>> sink, DQ&& dq, DK&& dkeys, DV&& dvalues)
{
    using T = row_value_t<Q>;
    const auto qs = as_row(q), ks = as_row(keys), vs = as_row(values), o = as_row(out), g = as_row(dout);
    const auto gq = as_mutable_row(dq), gk = as_mutable_row(dkeys), gv = as_mutable_row(dvalues);
    const std::size_t dk = qs.size(), dv = o.size(), n = dk == 0 ? 0 : ks.size() / dk;
    assert(dk > 0 && ks.size() == n * dk && vs.size() == n * dv && g.size() == dv && gq.size() == dk && gk.size() == ks.size()
           && gv.size() == vs.size());
    std::fill(gq.begin(), gq.end(), T(0));
    return detail::attention_sink::backward(
        n, [&](std::size_t j) { return scale * detail::attention::dot(qs, ks.subspan(j * dk, dk)); },
        [&](std::size_t j) { return vs.subspan(j * dv, dv); }, o, g, lse, sink,
        [&](std::size_t j, T p, T ds) {
            for (std::size_t i = 0; i < dv; ++i)
                gv[j * dv + i] += p * g[i];
            for (std::size_t i = 0; i < dk; ++i) {
                gq[i] += scale * ds * ks[j * dk + i];
                gk[j * dk + i] += scale * ds * qs[i];
            }
        });
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_attention_sink_hpp
