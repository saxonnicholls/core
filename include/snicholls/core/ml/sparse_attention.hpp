//
//  sparse_attention.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::attend_selected` and its backward pass, attention over the keys an index list selects; and DeepSeek Sparse Attention's lightning indexer: its scores, their backward pass, the top-k selection and the indexer's KL loss.
/// \ingroup core
///
/// **Overview.** Sparse attention lets each query read `k` of its `n` visible keys, so a long context costs `n · k`
/// instead of `n²`. DeepSeek Sparse Attention (DeepSeek-V3.2-Exp, 2025, section 2.1) chooses them with a lightning
/// indexer, a few cheap heads whose scores are
///
///     I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)       (eq. 1)
///
/// (`q_{t,j}` and `w_{t,j}` from the query token, `k_s` one shared key per token), takes the top `k` (2048 in the
/// report) and attends to those keys only (eq. 2), in MLA's MQA mode. The pieces, each a primitive:
///
/// - `indexer_scores(q, weights, keys, scores)`: eq. 1 for one query over `n` keys; `indexer_scores_backward` gives the
///   gradients in the indexer's query, weights and keys.
/// - `select_top_k(scores, indices)`: the `indices.size()` best keys (ties to the earlier key), written in position
///   order; returns how many (fewer when there are fewer keys).
/// - `attend_selected(q, keys, values, indices, out, scale, sink)`: attention over the selected rows of `keys` and
///   `values`, folded in the order given, with an optional sink (`ml/attention_sink.hpp`); with every key selected
///   in order it is `attend` to the bit. `attend_selected_backward` adds into the selected rows' gradients only.
/// - `indexer_kl_loss(target, scores, dscores)`: the indexer's training loss for one query, `KL(p ‖ softmax(I))`
///   (eqs. 3 and 4), and its gradient in the scores. The target `p` is the main attention's weights summed over the
///   heads and normalised to sum to 1, over every visible key in the dense warm-up and over the selected ones in the
///   sparse stage; the indexer's input is detached, so this loss trains the indexer alone and the language-model loss
///   the rest.
///
/// The indexer's projections (its queries from MLA's query latent, its key from the token through a layer norm, its
/// weights from the token), its RoPE, the Hadamard rotation and the FP8 storage are the model's, above these
/// primitives; as are its 64 heads of 128 (the released configuration's).
///
/// **Native Sparse Attention** (Yuan et al., 2025) has three branches, gated per query by a sigmoid: attention to
/// compressed blocks (an MLP over blocks of `l` keys at stride `d`), to the top `n` selected blocks of `l'` keys chosen
/// by the compressed branch's scores, and to a sliding window. Its selected branch is `attend_selected` over the
/// chosen blocks' positions, its window `attention_window`; the block compression, the block scoring and the gates
/// are deferred (TRAINING.md T12).
///
/// \snippet ml/sparse_attention_demo.cpp sparse_attention
///
/// **Relationships.** `ml/attention_sink.hpp` (the recurrence), `ml/attention.hpp`, `ml/latent_attention.hpp` (which
/// takes each token's selection), `transformer_config::sparse_top_k`. Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** `constexpr`, no allocation. `attend_selected` O(`k · (Dk + Dv)`); `indexer_scores` O(`n · H · D`);
/// `select_top_k` O(`n log k`) with a heap in the output, then a sort of the `k` chosen. Deterministic: ties go to the
/// earlier key.
///
/// **Errors.** Preconditions, asserted: every index below the number of keys; the rows whole; the gradients sized as
/// their inputs; `select_top_k` and `indexer_kl_loss` have as many scores as targets.
///
/// **Performance.** Not benchmarked on its own: `attend`'s loop over `k` keys, and the indexer's dot products.
///
/// **Tests and demos.** `tests/ml/sparse_attention_test.cpp`, `demos/ml/sparse_attention_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`,
/// `<type_traits>`, `ml/attention.hpp`, `ml/attention_sink.hpp`, `ml/row.hpp`.
///
/// **Known issues.** The report's indexer runs in FP8 with a Hadamard rotation; this reference runs in `T`.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_sparse_attention_hpp
#define snicholls_core_ml_sparse_attention_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/attention_sink.hpp>
#include <snicholls/core/ml/row.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief Attention for one query over the keys `indices` selects: `attend_with_sink` over rows `indices[0], ...` of
/// `keys` and `values`, in that order.
/// \param q The query, `Dk`.
/// \param keys `n` keys of `Dk`, end to end: every key the indices may name.
/// \param values `n` values of `Dv = out.size()`.
/// \param indices The selected rows, each below `n`; folded in the order given.
/// \param out The output, `Dv`.
/// \param scale The score scale.
/// \param sink A sink logit, or -infinity for none.
/// \return `logsumexp` of the selected scores and the sink.
template <row Q, row K, row V, mutable_row Out>
    requires std::same_as<row_value_t<Q>, row_value_t<K>> && std::same_as<row_value_t<Q>, row_value_t<V>>
             && std::same_as<row_value_t<Q>, row_value_t<Out>>
constexpr row_value_t<Q> attend_selected(const Q& q, const K& keys, const V& values, std::span<const std::uint32_t> indices,
                                         Out&& out, std::type_identity_t<row_value_t<Q>> scale,
                                         std::type_identity_t<row_value_t<Q>> sink = -std::numeric_limits<row_value_t<Q>>::infinity())
{
    const auto qs = as_row(q), ks = as_row(keys), vs = as_row(values);
    const auto y = as_mutable_row(out);
    const std::size_t dk = qs.size(), dv = y.size(), n = dk == 0 ? 0 : ks.size() / dk;
    assert(dk > 0 && ks.size() == n * dk && vs.size() == n * dv);
    assert(std::all_of(indices.begin(), indices.end(), [n](std::uint32_t i) { return i < n; }));
    (void)n;
    return detail::attention_sink::forward(
        indices.size(), [&](std::size_t j) { return scale * detail::attention::dot(qs, ks.subspan(indices[j] * dk, dk)); },
        [&](std::size_t j) { return vs.subspan(indices[j] * dv, dv); }, y, sink);
}

/// \brief The backward pass of `attend_selected`: writes `dq`, adds into the selected rows of `dkeys` and `dvalues`.
/// \param q The query. \param keys The keys. \param values The values. \param indices The selection.
/// \param out What the forward wrote. \param dout The gradient in `out`. \param lse What the forward returned.
/// \param scale The score scale. \param sink The sink logit, or -infinity. \param dq Written. \param dkeys Added into.
/// \param dvalues Added into.
/// \return The gradient in the sink logit.
template <row Q, row K, row V, row O, row DO, mutable_row DQ, mutable_row DK, mutable_row DV>
    requires std::same_as<row_value_t<Q>, row_value_t<K>> && std::same_as<row_value_t<Q>, row_value_t<V>>
             && std::same_as<row_value_t<Q>, row_value_t<O>> && std::same_as<row_value_t<Q>, row_value_t<DO>>
             && std::same_as<row_value_t<Q>, row_value_t<DQ>> && std::same_as<row_value_t<Q>, row_value_t<DK>>
             && std::same_as<row_value_t<Q>, row_value_t<DV>>
constexpr row_value_t<Q> attend_selected_backward(const Q& q, const K& keys, const V& values, std::span<const std::uint32_t> indices,
                                                  const O& out, const DO& dout, std::type_identity_t<row_value_t<Q>> lse,
                                                  std::type_identity_t<row_value_t<Q>> scale,
                                                  std::type_identity_t<row_value_t<Q>> sink, DQ&& dq, DK&& dkeys, DV&& dvalues)
{
    using T = row_value_t<Q>;
    const auto qs = as_row(q), ks = as_row(keys), vs = as_row(values), o = as_row(out), g = as_row(dout);
    const auto gq = as_mutable_row(dq), gk = as_mutable_row(dkeys), gv = as_mutable_row(dvalues);
    const std::size_t dk = qs.size(), dv = o.size(), n = dk == 0 ? 0 : ks.size() / dk;
    assert(dk > 0 && ks.size() == n * dk && vs.size() == n * dv && g.size() == dv && gq.size() == dk && gk.size() == ks.size()
           && gv.size() == vs.size());
    (void)n;
    std::fill(gq.begin(), gq.end(), T(0));
    return detail::attention_sink::backward(
        indices.size(), [&](std::size_t j) { return scale * detail::attention::dot(qs, ks.subspan(indices[j] * dk, dk)); },
        [&](std::size_t j) { return vs.subspan(indices[j] * dv, dv); }, o, g, lse, sink,
        [&](std::size_t j, T p, T ds) {
            const std::size_t r = indices[j];
            for (std::size_t i = 0; i < dv; ++i)
                gv[r * dv + i] += p * g[i];
            for (std::size_t i = 0; i < dk; ++i) {
                gq[i] += scale * ds * ks[r * dk + i];
                gk[r * dk + i] += scale * ds * qs[i];
            }
        });
}

/// \brief The lightning indexer's scores for one query: `I_s = Σ_j w_j · ReLU(q_j · k_s)` over `H` heads of `D`.
/// \param q The indexer's query, `H · D`, head by head.
/// \param weights The heads' weights `w_j`, `H`.
/// \param keys `n` indexer keys of `D`, end to end.
/// \param scores Receives `n` scores.
template <row Q, row W, row K, mutable_row S>
    requires std::same_as<row_value_t<Q>, row_value_t<W>> && std::same_as<row_value_t<Q>, row_value_t<K>>
             && std::same_as<row_value_t<Q>, row_value_t<S>>
constexpr void indexer_scores(const Q& q, const W& weights, const K& keys, S&& scores)
{
    using T = row_value_t<Q>;
    const auto qs = as_row(q), w = as_row(weights), ks = as_row(keys);
    const auto out = as_mutable_row(scores);
    const std::size_t h = w.size(), d = h == 0 ? 0 : qs.size() / h, n = out.size();
    assert(h > 0 && qs.size() == h * d && ks.size() == n * d);
    for (std::size_t s = 0; s < n; ++s) {
        T acc = T(0);
        for (std::size_t j = 0; j < h; ++j) {
            const T a = detail::attention::dot(qs.subspan(j * d, d), ks.subspan(s * d, d));
            acc += w[j] * (a > T(0) ? a : T(0));
        }
        out[s] = acc;
    }
}

/// \brief The backward pass of `indexer_scores`: adds into the gradients in the query, the weights and the keys.
///
/// With `a = q_j · k_s`, where `a > 0`: `dw_j += dI_s · a`, `dq_j += dI_s · w_j · k_s`, `dk_s += dI_s · w_j · q_j`.
/// \param q The query. \param weights The weights. \param keys The keys. \param dscores The gradients in the scores.
/// \param dq Added into, `q`'s size. \param dweights Added into, `H`. \param dkeys Added into, `keys`' size.
template <row Q, row W, row K, row DS, mutable_row DQ, mutable_row DW, mutable_row DK>
    requires std::same_as<row_value_t<Q>, row_value_t<W>> && std::same_as<row_value_t<Q>, row_value_t<K>>
             && std::same_as<row_value_t<Q>, row_value_t<DS>> && std::same_as<row_value_t<Q>, row_value_t<DQ>>
             && std::same_as<row_value_t<Q>, row_value_t<DW>> && std::same_as<row_value_t<Q>, row_value_t<DK>>
constexpr void indexer_scores_backward(const Q& q, const W& weights, const K& keys, const DS& dscores, DQ&& dq, DW&& dweights,
                                       DK&& dkeys)
{
    using T = row_value_t<Q>;
    const auto qs = as_row(q), w = as_row(weights), ks = as_row(keys), g = as_row(dscores);
    const auto gq = as_mutable_row(dq), gw = as_mutable_row(dweights), gk = as_mutable_row(dkeys);
    const std::size_t h = w.size(), d = h == 0 ? 0 : qs.size() / h, n = g.size();
    assert(h > 0 && qs.size() == h * d && ks.size() == n * d && gq.size() == qs.size() && gw.size() == h && gk.size() == ks.size());
    for (std::size_t s = 0; s < n; ++s)
        for (std::size_t j = 0; j < h; ++j) {
            const T a = detail::attention::dot(qs.subspan(j * d, d), ks.subspan(s * d, d));
            if (!(a > T(0)))
                continue;
            gw[j] += g[s] * a;
            const T c = g[s] * w[j];
            for (std::size_t i = 0; i < d; ++i) {
                gq[j * d + i] += c * ks[s * d + i];
                gk[s * d + i] += c * qs[j * d + i];
            }
        }
}

/// \brief The `indices.size()` highest scores' positions, in increasing position order; ties go to the earlier position.
/// \param scores `n` scores.
/// \param indices Receives the chosen positions: the first `min(k, n)` entries, `k = indices.size()`.
/// \return The number chosen, `min(k, n)`.
template <row S>
constexpr std::size_t select_top_k(const S& scores, std::span<std::uint32_t> indices)
{
    const auto s = as_row(scores);
    const std::size_t n = s.size(), k = std::min(indices.size(), n);
    assert(n <= std::numeric_limits<std::uint32_t>::max());
    // A min-heap of the best k under "a beats b": a higher score, or the same score at an earlier position
    const auto beats = [&](std::uint32_t a, std::uint32_t b) { return s[a] > s[b] || (s[a] == s[b] && a < b); };
    const auto heap = indices.first(k);
    for (std::size_t i = 0; i < n; ++i) {
        const auto pos = static_cast<std::uint32_t>(i);
        if (i < k) {
            heap[i] = pos;
            std::push_heap(heap.begin(), heap.begin() + std::ptrdiff_t(i + 1), beats);
        } else if (beats(pos, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), beats);
            heap.back() = pos;
            std::push_heap(heap.begin(), heap.end(), beats);
        }
    }
    std::sort(heap.begin(), heap.end());
    return k;
}

/// \brief The indexer's loss for one query, `KL(p ‖ softmax(I)) = Σ_s p_s (log p_s - log q_s)`, and its gradient in the
/// scores, `dI_s = (Σ p) q_s - p_s` (`q - p` for a target summing to 1).
/// \param target `p`, non-negative: the main attention's weights summed over heads and normalised.
/// \param scores The indexer's scores `I`, the same size.
/// \param dscores Receives the gradient, the same size.
/// \return The divergence, in nats; 0 with no scores.
template <row P, row S, mutable_row DS>
    requires std::same_as<row_value_t<P>, row_value_t<S>> && std::same_as<row_value_t<P>, row_value_t<DS>>
constexpr row_value_t<P> indexer_kl_loss(const P& target, const S& scores, DS&& dscores)
{
    using T = row_value_t<P>;
    const auto p = as_row(target), z = as_row(scores);
    const auto g = as_mutable_row(dscores);
    const std::size_t n = z.size();
    assert(p.size() == n && g.size() == n);
    if (n == 0)
        return T(0);
    T max = z[0];
    for (const T v : z)
        max = v > max ? v : max;
    T sum = T(0);
    for (const T v : z)
        sum += math::exp(v - max);
    const T lse = max + math::log(sum);
    T loss = T(0), mass = T(0);
    for (std::size_t s = 0; s < n; ++s) {
        mass += p[s];
        if (p[s] > T(0))
            loss += p[s] * (math::log(p[s]) - (z[s] - lse));
    }
    for (std::size_t s = 0; s < n; ++s)
        g[s] = mass * math::exp(z[s] - lse) - p[s];
    return loss;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_sparse_attention_hpp
