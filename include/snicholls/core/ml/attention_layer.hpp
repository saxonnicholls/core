//
//  attention_layer.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::attention_forward`, `attention_forward_sequence` and `attention_backward`: a transformer's attention for one token at one position, or over a sequence for training, and its backward pass: the Q, K and V projections, QK norm, RoPE, the KV cache, grouped-query attention, the soft cap and the output projection, as the configuration says.
/// \ingroup core
///
/// **Overview.** The generic multi-head attention layer. For a token `x` at position `pos`:
///
/// 1. `q = Wq x`, `k = Wk x`, `v = Wv x` (plus biases), from one prepared input;
/// 2. per head, an RMS norm of `q` and `k` before or after the rotation (`qk_norm`) and RoPE at `pos` (`rope`);
/// 3. `k` and `v` appended to the `kv_cache` at `pos`;
/// 4. each query head `h` attends to key-value head `h / group()` over the visible positions
///    (`attention_window(pos, window)`), each score `scale · q·k` capped by `attention_softcap · tanh(s / cap)` if set;
/// 5. `out = Wo y` (plus its bias) over the heads' outputs.
///
/// Prefill is this for each token of a run in order, so a prompt and a decode step are one code path. Attention
/// never forms the scores of a sequence: each head folds its keys in one pass (`attention_accumulator`), and a
/// sliding window's two runs are folded in position order, so a ring cache gives the flat cache's bits.
///
/// `attention_head` is one head on its own: the unit to run on threads or devices.
///
/// For training, `attention_forward_sequence` runs a whole sequence and saves `attention_activations` (each token's
/// output is `attention_forward`'s, to the bit), and `attention_backward` takes the gradient in the outputs back to
/// every weight, bias, QK-norm weight and input (`attention_gradients`): `Wo`'s backward, each head's
/// (`attention_head_backward`, the soft cap included; grouped-query heads add into the key-value head they share),
/// then QK norm and RoPE undone in the configuration's order, then the projections'. No `n x n` probabilities are
/// formed. Only dense weights train.
///
/// Both have a batched overload taking a `linalg` backend first: the projections are matrix products over the whole
/// sequence (`Q`, each key-value head's `K` and `V`, `Wo`; a BLAS or a GPU backend from another library, the same call), and QK
/// norm, RoPE and the heads are the per-token path's own code. Its backward needs `attention_gradients::y`, the heads'
/// output gradients. Both take a number of sequences (default one): a batch of sequences laid end to end, every
/// projection one product over all their rows, attention within each sequence, the activations each sequence's own
/// side by side (`attention_activations`).
///
/// **Masks and padding** (TRAINING.md T11). Over a sequence, a token sees the keys `attention_keys(C.mask, t, length,
/// C.window, C.mask_block)` allows: causal by default, bidirectional for a ViT or a JEPA encoder or predictor,
/// block-causal for a frame-causal predictor. `attention_activations::lengths` is the key-padding mask, one count of
/// valid tokens per sequence: the tokens from it on are seen by no query, and their own heads' outputs are zero (their
/// log-sum-exp -infinity) and pass no gradient. A rope source with a member `sequence(s)` (`gathered_rope`,
/// `ml/patch.hpp`) gives each sequence of a batch its own positions; any other source is shared by every sequence.
///
/// \snippet ml/attention_layer_demo.cpp attention_sequences
///
/// \snippet ml/attention_layer_demo.cpp attention_layer
///
/// \snippet ml/attention_layer_demo.cpp attention_backward
///
/// \snippet ml/attention_layer_demo.cpp attention_batched
///
/// **Relationships.** `ml/transformer_config.hpp`, `ml/linear.hpp`, `ml/kv_cache.hpp`, `ml/attention.hpp`,
/// `ml/rope.hpp` (`rope_table` is a rope source), `ml/norm.hpp`, `ml/activations.hpp` (`softcap`),
/// `ml/transformer_block.hpp` (the user). Group: \ref core.
///
/// **Implementations.** One, configured at compile time.
///
/// **Guarantees.** `constexpr`, no allocation (the projections are `std::array`s of the configuration's sizes),
/// O(`d_model · (q_size + 2 kv_size)` + `n_heads · visible · head_dim` + `q_size · d_model`) per token.
///
/// **Errors.** Preconditions, asserted: the matrices have the configuration's shapes (`q` is `q_size x d_model`,
/// `k` and `v` `kv_size x d_model`, `o` `d_model x q_size`), biases and QK-norm weights are empty or of their row's
/// size, the rope source covers `pos`, the cache's capacity covers the visible range. An invalid configuration is a
/// compile error.
///
/// **Performance.** Benchmarked through `ml/transformer_block.hpp`: its decode step, and its training passes, where
/// `attention_forward_sequence` and `attention_backward` run inside a layer within 5% and 3% of the same loops by hand.
///
/// **Tests and demos.** `tests/ml/attention_layer_test.cpp`, `demos/ml/attention_layer_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<limits>`, `<span>`,
/// `<type_traits>`, `linalg/backend.hpp`, and the `ml/` headers above.
///
/// **Known issues.** The score scale is `1 / sqrt(head_dim)`, rounded once to `T`.
///
/// **Status.** Experimental, new in 0.8.0; the masks, `lengths` and per-sequence rope sources new in 0.16.0.

#ifndef snicholls_core_ml_attention_layer_hpp
#define snicholls_core_ml_attention_layer_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The weights of an attention layer: views the caller fills.
/// \ingroup core
/// \tparam W A `weight_matrix`.
/// \tparam T The number type of the biases, norms and activations.
template <weight_matrix W, math::math_number T = float>
struct attention_weights
{
    W q{};                              ///< `q_size x d_model`.
    W k{};                              ///< `kv_size x d_model`.
    W v{};                              ///< `kv_size x d_model`.
    W o{};                              ///< `d_model x q_size`.
    std::span<const T> q_bias{};        ///< `q_size`, or empty.
    std::span<const T> k_bias{};        ///< `kv_size`, or empty.
    std::span<const T> v_bias{};        ///< `kv_size`, or empty.
    std::span<const T> o_bias{};        ///< `d_model`, or empty.
    std::span<const T> q_norm{};        ///< `head_dim` QK-norm weights, or empty for a parameter-free norm.
    std::span<const T> k_norm{};        ///< `head_dim` QK-norm weights, or empty.
};

/// \brief One query head over its visible keys: `out = softmax(cap(scale · q·k)) · v`, in one pass.
/// \tparam C The configuration (its `head_dim` and `attention_softcap`).
/// \param q The head's query, `head_dim`.
/// \param runs The visible keys and values, from a cache's `visible`, folded run after run.
/// \param out The head's output, `head_dim`.
/// \return `logsumexp` of the scores, for a backward pass; -infinity with no keys.
template <transformer_config C, math::math_number T, std::size_t N, row Q, mutable_row Out>
    requires std::same_as<row_value_t<Q>, T> && std::same_as<row_value_t<Out>, T>
constexpr T attention_head(const Q& q, const kv_runs<T, N>& runs, Out&& out)
{
    const auto qs = as_row(q);
    assert(qs.size() == C.head_dim);
    const T scale = T(1) / math::sqrt(T(C.head_dim));
    attention_accumulator<T, C.head_dim> acc{};
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t j = 0; j < runs.keys[r].size() / C.head_dim; ++j) {
            T s = scale * detail::attention::dot(qs, runs.keys[r].subspan(j * C.head_dim, C.head_dim));
            if constexpr (C.attention_softcap > 0)
                s = softcap(s, attention_softcap_v<C, T>);
            acc.add(s, runs.values[r].subspan(j * C.head_dim, C.head_dim));
        }
    acc.result(out);
    return acc.log_sum_exp();
}

namespace detail::attention_layer
{
    // RoPE then, or before, the QK norm of one head, as the configuration says
    template <transformer_config C, typename T, typename R>
    constexpr void position(std::span<T> head, std::span<const T> norm, std::size_t pos, const R& rope_src)
    {
        constexpr T eps = norm_eps_v<C, T>;
        if constexpr (C.qk_norm == qk_norm_kind::before_rope)
            rms_norm(head, norm, head, eps);
        if constexpr (C.rope != rope_kind::none)
            ml::rope(head, rope_src.cos_at(pos), rope_src.sin_at(pos), C.layout());
        if constexpr (C.qk_norm == qk_norm_kind::after_rope)
            rms_norm(head, norm, head, eps);
    }
} // namespace detail::attention_layer

/// \brief Attention for one token at position `pos`: projects, rotates, caches, attends and projects back.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The token's (normalised) input, `d_model`.
/// \param pos Its position.
/// \param cache The KV cache (`kv_cache_view`, `paged_kv_cache_view`); this token's keys and values are appended to
/// `layer` at `pos`.
/// \param layer The layer.
/// \param rope_src The cosines and sines by position (a `rope_table` of `head_dim`), or `no_rope`.
/// \param out The output, `d_model`; may not alias `x`.
template <transformer_config C, weight_matrix W, math::math_number T, row X, kv_store<T> S, typename R, mutable_row Out>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<Out>, T>
             && (C.rope == rope_kind::none || rope_source<R>)
constexpr void attention_forward(const attention_weights<W, T>& w, const X& x, std::size_t pos, const S& cache,
                                 std::size_t layer, const R& rope_src, Out&& out)
{
    static_assert(valid(C), "snicholls::ml::attention_forward: the transformer_config is not valid");
    static_assert(C.mask == attention_mask::causal,
                  "snicholls::ml::attention_forward: decoding one token at a time is causal; run a bidirectional or block-causal layer over the sequence");
    assert(w.q.rows == C.q_size() && w.k.rows == C.kv_size() && w.v.rows == C.kv_size() && w.o.rows == C.d_model
           && w.q.cols == C.d_model && w.k.cols == C.d_model && w.v.cols == C.d_model && w.o.cols == C.q_size());
    constexpr std::size_t d = C.head_dim;
    std::array<T, C.q_size()> q{};
    std::array<T, C.kv_size()> k{}, v{};
    const auto in = prepare_input<C.d_model>(w.q, x);
    linear(w.q, in, q, w.q_bias);
    linear(w.k, in, k, w.k_bias);
    linear(w.v, in, v, w.v_bias);
    for (std::size_t h = 0; h < C.n_heads; ++h)
        detail::attention_layer::position<C>(std::span<T>(q).subspan(h * d, d), w.q_norm, pos, rope_src);
    for (std::size_t h = 0; h < C.n_kv_heads; ++h)
        detail::attention_layer::position<C>(std::span<T>(k).subspan(h * d, d), w.k_norm, pos, rope_src);
    cache.append(layer, pos, k, v);
    std::array<T, C.q_size()> y{};
    const key_range visible = attention_window(pos, C.window);
    for (std::size_t h = 0; h < C.n_heads; ++h)
        attention_head<C>(std::span<const T>(q).subspan(h * d, d), cache.visible(layer, h / C.group(), visible),
                          std::span<T>(y).subspan(h * d, d));
    linear(w.o, prepare_input<C.q_size()>(w.o, y), out, w.o_bias);
}

/// \brief The backward pass of `attention_head` over one contiguous run of keys, soft cap included.
///
/// With `r_j = scale · q·k_j`, `s_j = cap(r_j)` and `p_j = e^(s_j - lse)`: `dv_j += p_j · dout`,
/// `ds_j = p_j · (dout·v_j - dout·out) · cap'(r_j)`, `dq = scale · sum ds_j · k_j`, `dk_j += scale · ds_j · q`.
/// Without a soft cap it is `attend_backward`, to the bit.
/// \tparam C The configuration (its `head_dim` and `attention_softcap`).
/// \param q The head's query, `head_dim`.
/// \param keys The visible keys, `head_dim` per position, end to end.
/// \param values The visible values, laid out as the keys.
/// \param out What `attention_head` wrote.
/// \param dout The gradient in `out`.
/// \param lse What `attention_head` returned.
/// \param dq The gradient in `q`: written.
/// \param dkeys Added into; `keys`' size.
/// \param dvalues Added into; `values`' size.
template <transformer_config C, math::math_number T, row Q, row K, row V, row O, row DO, mutable_row DQ, mutable_row DK,
          mutable_row DV>
    requires std::same_as<row_value_t<Q>, T> && std::same_as<row_value_t<K>, T> && std::same_as<row_value_t<V>, T>
             && std::same_as<row_value_t<O>, T> && std::same_as<row_value_t<DO>, T> && std::same_as<row_value_t<DQ>, T>
             && std::same_as<row_value_t<DK>, T> && std::same_as<row_value_t<DV>, T>
constexpr void attention_head_backward(const Q& q, const K& keys, const V& values, const O& out, const DO& dout,
                                       std::type_identity_t<T> lse, DQ&& dq, DK&& dkeys, DV&& dvalues)
{
    const T scale = T(1) / math::sqrt(T(C.head_dim));
    if constexpr (C.attention_softcap == 0) {
        attend_backward(q, keys, values, out, dout, lse, scale, dq, dkeys, dvalues);
    } else {
        constexpr std::size_t d = C.head_dim;
        const auto qs = as_row(q), ks = as_row(keys), vs = as_row(values), o = as_row(out), g = as_row(dout);
        const auto gq = as_mutable_row(dq), gk = as_mutable_row(dkeys), gv = as_mutable_row(dvalues);
        assert(qs.size() == d && o.size() == d && g.size() == d && gq.size() == d && ks.size() == vs.size()
               && ks.size() % d == 0 && gk.size() == ks.size() && gv.size() == vs.size());
        std::fill(gq.begin(), gq.end(), T(0));
        if (lse == -std::numeric_limits<T>::infinity())
            return;
        constexpr T cap = attention_softcap_v<C, T>;
        const T big_d = detail::attention::dot(g, o);
        for (std::size_t j = 0; j < ks.size() / d; ++j) {
            const auto k = ks.subspan(j * d, d);
            const auto v = vs.subspan(j * d, d);
            const T r = scale * detail::attention::dot(qs, k);
            const T p = math::exp(softcap(r, cap) - lse);
            const T ds = p * (detail::attention::dot(g, v) - big_d) * softcap_derivative(r, cap);
            for (std::size_t i = 0; i < d; ++i) {
                gv[j * d + i] += p * g[i];
                gq[i] += scale * ds * k[i];
                gk[j * d + i] += scale * ds * qs[i];
            }
        }
    }
}

/// \brief What `attention_forward_sequence` saves for `attention_backward`: views the caller owns.
///
/// For a batch of `sequences` sequences of `n` tokens (the batched passes' `sequences`), every field is the sequences'
/// own laid end to end, sequence-major: `q` is `sequences · n · q_size`, and `k` is `sequences` blocks of `n_kv_heads ·
/// n · head_dim`, each block one sequence's keys head by head. So a batch's activations are its sequences' activations
/// side by side, and a view of one sequence is a slice.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct attention_activations
{
    std::span<T> q{};       ///< `n · q_size`, token by token: the queries after QK norm and RoPE.
    std::span<T> k{};       ///< `n_kv_heads · n · head_dim`, head by head: the keys after QK norm and RoPE.
    std::span<T> v{};       ///< `n_kv_heads · n · head_dim`, head by head: the values.
    std::span<T> q_raw{};   ///< `n · q_size`: the projected queries before QK norm; with QK norm only, else empty.
    std::span<T> k_raw{};   ///< `n · kv_size`: the projected keys before QK norm; with QK norm only, else empty.
    std::span<T> y{};       ///< `n · q_size`, token by token: the heads' outputs, the input of `Wo`.
    std::span<T> lse{};     ///< `n · n_heads`, token by token: each head's log-sum-exp.
    std::span<const std::size_t> lengths{};   ///< The valid tokens of each sequence (key padding, TRAINING.md T11), one per sequence; or empty: all valid.
};

/// \brief The gradients of an attention layer: views the caller owns. `q`, `k`, `v` and `y` are laid out per sequence
/// as `attention_activations` is.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct attention_gradients
{
    std::span<T> q{};        ///< `n · q_size`: written, the gradients in the projected queries.
    std::span<T> k{};        ///< `n_kv_heads · n · head_dim`: written, in the projected keys, laid out as the activations.
    std::span<T> v{};        ///< `n_kv_heads · n · head_dim`: written, in the values.
    std::span<T> wq{};       ///< `q_size · d_model`: added into.
    std::span<T> wk{};       ///< `kv_size · d_model`: added into.
    std::span<T> wv{};       ///< `kv_size · d_model`: added into.
    std::span<T> wo{};       ///< `d_model · q_size`: added into.
    std::span<T> q_bias{};   ///< `q_size`: added into; or empty.
    std::span<T> k_bias{};   ///< `kv_size`: added into; or empty.
    std::span<T> v_bias{};   ///< `kv_size`: added into; or empty.
    std::span<T> o_bias{};   ///< `d_model`: added into; or empty.
    std::span<T> q_norm{};   ///< `head_dim`: added into; or empty.
    std::span<T> k_norm{};   ///< `head_dim`: added into; or empty.
    std::span<T> y{};        ///< `n · q_size`: the batched backward's working space (the heads' output gradients); else empty.
};

namespace detail::attention_layer
{
    // The backward of position(): the gradient in a head after it, to the gradient in its raw projection, in place
    template <transformer_config C, typename T, typename R>
    constexpr void position_backward(std::span<T> d, std::span<const T> raw, std::span<const T> norm,
                                     std::span<T> dnorm, std::size_t pos, const R& rope_src)
    {
        if constexpr (C.qk_norm == qk_norm_kind::after_rope) {
            constexpr T eps = norm_eps_v<C, T>;
            std::array<T, C.head_dim> x{}, scratch{};
            std::copy(raw.begin(), raw.end(), x.begin());   // the norm's input: the rotated projection
            if constexpr (C.rope != rope_kind::none)
                ml::rope(x, rope_src.cos_at(pos), rope_src.sin_at(pos), C.layout());
            const T rstd = rms_norm(x, norm, scratch, eps);
            rms_norm_backward(d, x, norm, rstd, d, dnorm);
        }
        if constexpr (C.rope != rope_kind::none)
            rope_inverse(d, rope_src.cos_at(pos), rope_src.sin_at(pos), C.layout());
        if constexpr (C.qk_norm == qk_norm_kind::before_rope) {
            constexpr T eps = norm_eps_v<C, T>;
            std::array<T, C.head_dim> x{}, scratch{};
            std::copy(raw.begin(), raw.end(), x.begin());
            const T rstd = rms_norm(x, norm, scratch, eps);
            rms_norm_backward(d, x, norm, rstd, d, dnorm);
        }
    }

    // The forward after the projections, for token t of n, shared by both paths: the raw copies, then QK norm and
    // RoPE of its heads in place (the query in act.q, the keys in act.k's [kv_head][t] rows)
    template <transformer_config C, typename W, typename T, typename R>
    constexpr void position_token(const attention_weights<W, T>& w, const attention_activations<T>& act, std::size_t n,
                                  std::size_t t, const R& rope_src)
    {
        constexpr std::size_t d = C.head_dim, qn = C.q_size(), kn = C.kv_size();
        const auto q = act.q.subspan(t * qn, qn);
        if constexpr (C.qk_norm != qk_norm_kind::none) {
            std::copy(q.begin(), q.end(), act.q_raw.begin() + std::ptrdiff_t(t * qn));
            for (std::size_t h = 0; h < C.n_kv_heads; ++h)
                std::copy_n(act.k.begin() + std::ptrdiff_t((h * n + t) * d), d, act.k_raw.begin() + std::ptrdiff_t(t * kn + h * d));
        }
        for (std::size_t h = 0; h < C.n_heads; ++h)
            position<C>(q.subspan(h * d, d), w.q_norm, t, rope_src);
        for (std::size_t h = 0; h < C.n_kv_heads; ++h)
            position<C>(act.k.subspan((h * n + t) * d, d), w.k_norm, t, rope_src);
    }

    // The keys token t of a sequence of n sees: the configuration's mask within the sequence's valid tokens
    template <transformer_config C, typename T>
    constexpr key_range keys(const attention_activations<T>& act, std::size_t n, std::size_t t)
    {
        return attention_keys(C.mask, t, act.lengths.empty() ? n : act.lengths[0], C.window, C.mask_block);
    }

    // The rope source of sequence s of a batch: the source itself, or its sequence(s) when it has one (a source
    // whose positions differ between sequences, such as gathered_rope)
    template <typename R>
    constexpr const R& rope_of(const R& r, std::size_t)
    {
        return r;
    }
    template <typename R>
        requires requires(const R& r, std::size_t s) { r.sequence(s); }
    constexpr auto rope_of(const R& r, std::size_t s)
    {
        return r.sequence(s);
    }

    // Token t's heads over its visible keys: its rows of act.y and act.lse
    template <transformer_config C, typename T>
    constexpr void attend_token(const attention_activations<T>& act, std::size_t n, std::size_t t)
    {
        constexpr std::size_t d = C.head_dim, qn = C.q_size();
        const key_range r = keys<C>(act, n, t);
        const auto yt = act.y.subspan(t * qn, qn);
        for (std::size_t h = 0; h < C.n_heads; ++h) {
            const std::size_t kvh = h / C.group();
            kv_runs<T, 1> run{};
            run.keys[0] = std::span<const T>(act.k).subspan((kvh * n + r.first) * d, r.size() * d);
            run.values[0] = std::span<const T>(act.v).subspan((kvh * n + r.first) * d, r.size() * d);
            act.lse[t * C.n_heads + h] = attention_head<C>(std::span<const T>(act.q).subspan(t * qn + h * d, d), run, yt.subspan(h * d, d));
        }
    }

    // The backward of token t's heads from the gradient in their outputs: its query gradients written, its keys'
    // and values' added into
    template <transformer_config C, typename T>
    constexpr void heads_backward_token(const attention_activations<T>& act, std::span<const T> dy,
                                        const attention_gradients<T>& grads, std::size_t n, std::size_t t)
    {
        constexpr std::size_t d = C.head_dim, qn = C.q_size();
        const auto yt = std::span<const T>(act.y).subspan(t * qn, qn);
        const key_range r = keys<C>(act, n, t);
        for (std::size_t h = 0; h < C.n_heads; ++h) {
            const std::size_t kvh = h / C.group(), at = (kvh * n + r.first) * d;
            attention_head_backward<C, T>(std::span<const T>(act.q).subspan(t * qn + h * d, d),
                                          std::span<const T>(act.k).subspan(at, r.size() * d),
                                          std::span<const T>(act.v).subspan(at, r.size() * d), yt.subspan(h * d, d),
                                          dy.subspan(h * d, d), act.lse[t * C.n_heads + h], grads.q.subspan(t * qn + h * d, d),
                                          grads.k.subspan(at, r.size() * d), grads.v.subspan(at, r.size() * d));
        }
    }

    // QK norm and RoPE undone for token t, in place: its row of grads.q and its [kv_head][t] rows of grads.k
    template <transformer_config C, typename W, typename T, typename R>
    constexpr void position_backward_token(const attention_weights<W, T>& w, const attention_activations<T>& act,
                                           const attention_gradients<T>& grads, std::size_t n, std::size_t t, const R& rope_src)
    {
        if constexpr (C.qk_norm != qk_norm_kind::none || C.rope != rope_kind::none) {
            constexpr std::size_t d = C.head_dim, qn = C.q_size(), kn = C.kv_size();
            constexpr bool raw = C.qk_norm != qk_norm_kind::none;
            for (std::size_t h = 0; h < C.n_heads; ++h)
                position_backward<C>(grads.q.subspan(t * qn + h * d, d),
                                     raw ? std::span<const T>(act.q_raw).subspan(t * qn + h * d, d) : std::span<const T>{},
                                     w.q_norm, grads.q_norm, t, rope_src);
            for (std::size_t h = 0; h < C.n_kv_heads; ++h)
                position_backward<C>(grads.k.subspan((h * n + t) * d, d),
                                     raw ? std::span<const T>(act.k_raw).subspan(t * kn + h * d, d) : std::span<const T>{},
                                     w.k_norm, grads.k_norm, t, rope_src);
        }
    }

    // Sequence s of a batch of sequences of n tokens: every per-sequence view is a slice, the batch's being the
    // sequences' own laid end to end
    template <typename T>
    constexpr std::span<T> part(std::span<T> x, std::size_t s, std::size_t size)
    {
        return x.empty() ? x : x.subspan(s * size, size);
    }
    template <transformer_config C, typename T>
    constexpr attention_activations<T> sequence(const attention_activations<T>& a, std::size_t n, std::size_t s)
    {
        constexpr std::size_t qn = C.q_size(), kn = C.kv_size();
        return {part(a.q, s, n * qn), part(a.k, s, n * kn), part(a.v, s, n * kn), part(a.q_raw, s, n * qn),
                part(a.k_raw, s, n * kn), part(a.y, s, n * qn), part(a.lse, s, n * C.n_heads), part(a.lengths, s, 1)};
    }
    template <transformer_config C, typename T>
    constexpr attention_gradients<T> sequence(const attention_gradients<T>& g, std::size_t n, std::size_t s)
    {
        constexpr std::size_t qn = C.q_size(), kn = C.kv_size();
        attention_gradients<T> r = g;
        r.q = part(g.q, s, n * qn);
        r.k = part(g.k, s, n * kn);
        r.v = part(g.v, s, n * kn);
        r.y = part(g.y, s, n * qn);
        return r;
    }

    template <typename T>
    constexpr void check_activations(const attention_activations<T>& act, std::size_t n, std::size_t qn, std::size_t kn,
                                     std::size_t heads, bool raw)
    {
        assert(act.q.size() == n * qn && act.k.size() == n * kn && act.v.size() == n * kn && act.y.size() == n * qn
               && act.lse.size() == n * heads && (!raw || (act.q_raw.size() == n * qn && act.k_raw.size() == n * kn)));
        (void)act, (void)n, (void)qn, (void)kn, (void)heads, (void)raw;
    }

    template <typename T>
    constexpr void check_lengths(const attention_activations<T>& act, std::size_t n, std::size_t sequences)
    {
        assert(act.lengths.empty() || act.lengths.size() == sequences);
        for ([[maybe_unused]] const std::size_t l : act.lengths)
            assert(l <= n);
        (void)act, (void)n, (void)sequences;
    }
} // namespace detail::attention_layer

/// \name The attention steps of the batched passes, on the backend (TRAINING.md T9)
/// The steps of `attention_forward_sequence` and `attention_backward` (batched) that are not matrix products, each over
/// a batch of `sequences` sequences of `n` tokens, laid out as `attention_activations` says. Attention stays within a
/// sequence: positions restart at 0 in each. The overloads here take any backend and run on the host, sequence by
/// sequence and token by token, with the per-token path's own code, so a batch gives each sequence's own call's bits; a
/// device library overloads them for its own backend type.
/// @{

/// \brief QK norm and RoPE of every token's heads, in place, after the raw copies: `act.q`, `act.k`, `act.q_raw`,
/// `act.k_raw`. \tparam C The configuration. \param w The weights (the norms'). \param act The activations.
/// \param n The tokens of each sequence. \param rope_src The rope source, or `no_rope`. \param sequences The sequences.
template <transformer_config C, typename B, typename W, math::math_number T, typename R>
constexpr void attention_position_rows(const B&, const attention_weights<W, T>& w, const attention_activations<T>& act, std::size_t n,
                                       const R& rope_src, std::size_t sequences = 1)
{
    for (std::size_t s = 0; s < sequences; ++s) {
        const auto a = detail::attention_layer::sequence<C>(act, n, s);
        const auto& r = detail::attention_layer::rope_of(rope_src, s);
        for (std::size_t t = 0; t < n; ++t)
            detail::attention_layer::position_token<C>(w, a, n, t, r);
    }
}

/// \brief Every token's heads over its visible keys, those of its own sequence: `act.y` and `act.lse`.
/// \tparam C The configuration. \param act The activations. \param n The tokens of each sequence.
/// \param sequences The sequences.
template <transformer_config C, typename B, math::math_number T>
constexpr void attention_attend_rows(const B&, const attention_activations<T>& act, std::size_t n, std::size_t sequences = 1)
{
    detail::attention_layer::check_lengths(act, n, sequences);
    for (std::size_t s = 0; s < sequences; ++s) {
        const auto a = detail::attention_layer::sequence<C>(act, n, s);
        for (std::size_t t = 0; t < n; ++t)
            detail::attention_layer::attend_token<C>(a, n, t);
    }
}

/// \brief The backward pass of `attention_attend_rows`: from the heads' output gradients `dy` (`sequences · n ·
/// q_size`), the queries' gradients written into `grads.q` and the keys' and values' added into `grads.k`, `grads.v`.
/// \tparam C The configuration. \param act The activations. \param dy The gradients in the heads' outputs.
/// \param grads The gradients. \param n The tokens of each sequence. \param sequences The sequences.
template <transformer_config C, typename B, math::math_number T>
constexpr void attention_attend_rows_backward(const B&, const attention_activations<T>& act, std::type_identity_t<std::span<const T>> dy,
                                              const attention_gradients<T>& grads, std::size_t n, std::size_t sequences = 1)
{
    for (std::size_t s = 0; s < sequences; ++s) {
        const auto a = detail::attention_layer::sequence<C>(act, n, s);
        const auto g = detail::attention_layer::sequence<C>(grads, n, s);
        for (std::size_t t = 0; t < n; ++t)
            detail::attention_layer::heads_backward_token<C>(a, dy.subspan((s * n + t) * C.q_size(), C.q_size()), g, n, t);
    }
}

/// \brief The backward pass of `attention_position_rows`, in place on `grads.q` and `grads.k`, the norms' gradients added
/// into. \tparam C The configuration. \param w The weights. \param act The activations. \param grads The gradients.
/// \param n The tokens of each sequence. \param rope_src The rope source. \param sequences The sequences.
template <transformer_config C, typename B, typename W, math::math_number T, typename R>
constexpr void attention_position_rows_backward(const B&, const attention_weights<W, T>& w, const attention_activations<T>& act,
                                                const attention_gradients<T>& grads, std::size_t n, const R& rope_src,
                                                std::size_t sequences = 1)
{
    for (std::size_t s = 0; s < sequences; ++s) {
        const auto a = detail::attention_layer::sequence<C>(act, n, s);
        const auto g = detail::attention_layer::sequence<C>(grads, n, s);
        const auto& r = detail::attention_layer::rope_of(rope_src, s);
        for (std::size_t t = 0; t < n; ++t)
            detail::attention_layer::position_backward_token<C>(w, a, g, n, t, r);
    }
}

/// @}

/// \brief Attention over a whole sequence, saving what `attention_backward` needs: each token's output is
/// `attention_forward`'s, to the bit.
/// \tparam C The configuration.
/// \param w The weights.
/// \param xs `n` tokens of `d_model`, end to end (normalised inputs).
/// \param rope_src The cosines and sines by position, covering `n - 1`, or `no_rope`.
/// \param act Receives the activations.
/// \param out Receives `n` outputs of `d_model`; may not alias `xs`.
template <transformer_config C, weight_matrix W, math::math_number T, row X, typename R, mutable_row Out>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<Out>, T>
             && (C.rope == rope_kind::none || rope_source<R>)
constexpr void attention_forward_sequence(const attention_weights<W, T>& w, const X& xs, const R& rope_src,
                                          const attention_activations<T>& act, Out&& out)
{
    static_assert(valid(C), "snicholls::ml::attention_forward_sequence: the transformer_config is not valid");
    constexpr std::size_t d = C.head_dim, qn = C.q_size(), kn = C.kv_size();
    const auto x = as_row(xs);
    const auto y = as_mutable_row(out);
    const std::size_t n = x.size() / C.d_model;
    assert(x.size() == n * C.d_model && y.size() == n * C.d_model);
    detail::attention_layer::check_activations(act, n, qn, kn, C.n_heads, C.qk_norm != qk_norm_kind::none);
    detail::attention_layer::check_lengths(act, n, 1);
    std::array<T, kn> k{}, v{};
    for (std::size_t t = 0; t < n; ++t) {
        const auto in = prepare_input<C.d_model>(w.q, x.subspan(t * C.d_model, C.d_model));
        linear(w.q, in, act.q.subspan(t * qn, qn), w.q_bias);
        linear(w.k, in, k, w.k_bias);
        linear(w.v, in, v, w.v_bias);
        for (std::size_t h = 0; h < C.n_kv_heads; ++h) {
            std::copy_n(k.begin() + std::ptrdiff_t(h * d), d, act.k.begin() + std::ptrdiff_t((h * n + t) * d));
            std::copy_n(v.begin() + std::ptrdiff_t(h * d), d, act.v.begin() + std::ptrdiff_t((h * n + t) * d));
        }
        detail::attention_layer::position_token<C>(w, act, n, t, rope_src);
    }
    // Every key is in place before any token attends: a bidirectional or block-causal token reads later keys
    for (std::size_t t = 0; t < n; ++t) {
        detail::attention_layer::attend_token<C>(act, n, t);
        linear(w.o, prepare_input<qn>(w.o, std::span<const T>(act.y).subspan(t * qn, qn)), y.subspan(t * C.d_model, C.d_model), w.o_bias);
    }
}

/// \brief Attention over a whole sequence with its projections as matrix products through a `linalg` backend,
/// saving what `attention_backward` needs.
///
/// The batched form of the overload above, for dense weights in the sequence's type: `Q = X Wqᵀ`, each key-value
/// head's `K` and `V` straight into the activations' layout, and `X_out = Y Woᵀ`, each one `gemm` (a BLAS or a
/// device does them); QK norm, RoPE and the heads are the same code as the per-token path. The projections are the
/// backend's (`linear_sequence`), so a result can differ from the per-token path's in the last bits.
///
/// **A batch of sequences.** With `sequences` > 1, `xs` holds that many sequences of `n` tokens end to end, and
/// attention stays within each: positions restart at 0 and no token sees another sequence's. Every projection is one
/// product over all `sequences · n` rows; the activations are each sequence's own laid end to end (see
/// `attention_activations`). The result is that of calling this once per sequence, to the bit.
/// \tparam C The configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights.
/// \param xs `sequences · n` tokens of `d_model`, end to end (normalised inputs).
/// \param rope_src The cosines and sines by position, covering `n - 1`, or `no_rope`.
/// \param act Receives the activations.
/// \param out Receives `sequences · n` outputs of `d_model`; may not alias `xs`.
/// \param sequences The sequences, at least one; `xs`'s rows divide evenly among them.
template <transformer_config C, typename B, math::math_number T, typename R>
    requires linalg::blas_backend<B, T> && (C.rope == rope_kind::none || rope_source<R>)
constexpr void attention_forward_sequence(const B& backend, const attention_weights<dense_weights<T>, T>& w,
                                          std::type_identity_t<std::span<const T>> xs, const R& rope_src,
                                          const attention_activations<T>& act, std::type_identity_t<std::span<T>> out,
                                          std::size_t sequences = 1)
{
    static_assert(valid(C), "snicholls::ml::attention_forward_sequence: the transformer_config is not valid");
    constexpr std::size_t qn = C.q_size(), kn = C.kv_size();
    const std::size_t rows = xs.size() / C.d_model, n = rows / sequences;
    assert(sequences > 0 && xs.size() == n * sequences * C.d_model && out.size() == xs.size());
    detail::attention_layer::check_activations(act, rows, qn, kn, C.n_heads, C.qk_norm != qk_norm_kind::none);
    linear_sequence(backend, w.q, xs, act.q, w.q_bias);
    linear_heads(backend, w.k, xs, act.k, w.k_bias, C.n_kv_heads, sequences);   // the keys and values head by head
    linear_heads(backend, w.v, xs, act.v, w.v_bias, C.n_kv_heads, sequences);
    attention_position_rows<C>(backend, w, act, n, rope_src, sequences);
    attention_attend_rows<C>(backend, act, n, sequences);
    linear_sequence(backend, w.o, std::span<const T>(act.y), out, w.o_bias);
}

/// \brief The backward pass of `attention_forward_sequence`, from what it saved.
///
/// Token by token, `Wo`'s backward gives the heads' output gradients. Each query head's backward
/// (`attention_head_backward`) writes its query gradient and adds into the key and value gradients of the key-value
/// head it shares, so grouped-query heads sum. QK norm and RoPE are undone in the configuration's order
/// (`rope_inverse`, `rms_norm_backward`, the norm weights' gradients summed over heads and positions), and
/// `linear_backward` takes the gradients through `Wq`, `Wk` and `Wv`. Only dense weights train.
/// \tparam C The configuration.
/// \param w The weights the forward pass used.
/// \param xs The tokens it read.
/// \param rope_src Its rope source.
/// \param act What it saved.
/// \param dout The gradient in its output, `n · d_model`.
/// \param dxs Added into, `n · d_model`; or empty to skip it.
/// \param grads Receives the gradients: `q`, `k`, `v` written, the rest added into.
template <transformer_config C, typename S, math::math_number T, row X, typename R, row DO, mutable_row DX>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<DO>, T> && std::same_as<row_value_t<DX>, T>
             && (C.rope == rope_kind::none || rope_source<R>)
constexpr void attention_backward(const attention_weights<dense_weights<S>, T>& w, const X& xs, const R& rope_src,
                                  const attention_activations<T>& act, const DO& dout, DX&& dxs,
                                  const attention_gradients<T>& grads)
{
    static_assert(valid(C), "snicholls::ml::attention_backward: the transformer_config is not valid");
    constexpr std::size_t d = C.head_dim, qn = C.q_size(), kn = C.kv_size();
    const auto x = as_row(xs);
    const auto g = as_row(dout);
    const auto gx = as_mutable_row(dxs);
    const std::size_t n = x.size() / C.d_model;
    assert(g.size() == x.size() && (gx.empty() || gx.size() == x.size()));
    assert(grads.q.size() == n * qn && grads.k.size() == n * kn && grads.v.size() == n * kn);
    std::fill(grads.k.begin(), grads.k.end(), T(0));
    std::fill(grads.v.begin(), grads.v.end(), T(0));
    for (std::size_t t = 0; t < n; ++t) {
        std::array<T, qn> dy{};
        linear_backward(w.o, std::span<const T>(act.y).subspan(t * qn, qn), g.subspan(t * C.d_model, C.d_model), dy, grads.wo, grads.o_bias);
        detail::attention_layer::heads_backward_token<C>(act, std::span<const T>(dy), grads, n, t);
    }
    const std::span<T> no_dx{};
    for (std::size_t t = 0; t < n; ++t) {
        detail::attention_layer::position_backward_token<C>(w, act, grads, n, t, rope_src);
        std::array<T, kn> dk{}, dv{};
        for (std::size_t h = 0; h < C.n_kv_heads; ++h)
            for (std::size_t i = 0; i < d; ++i) {
                dk[h * d + i] = grads.k[(h * n + t) * d + i];
                dv[h * d + i] = grads.v[(h * n + t) * d + i];
            }
        const auto xt = x.subspan(t * C.d_model, C.d_model);
        const auto dxt = gx.empty() ? no_dx : gx.subspan(t * C.d_model, C.d_model);
        linear_backward(w.q, xt, grads.q.subspan(t * qn, qn), dxt, grads.wq, grads.q_bias);
        linear_backward(w.k, xt, dk, dxt, grads.wk, grads.k_bias);
        linear_backward(w.v, xt, dv, dxt, grads.wv, grads.v_bias);
    }
}

/// \brief The backward pass of the batched `attention_forward_sequence`, its products through the backend.
///
/// `Wo`'s backward over the whole sequence (into `grads.y`, the heads' output gradients), the heads' backward and
/// the undoing of QK norm and RoPE token by token (the per-token path's code), then `Wq`'s, each key-value head's
/// `Wk` and `Wv`'s, each one pair of `gemm`s (`linear_sequence_backward`).
///
/// Over a batch of `sequences` (as the forward): the input's gradient, the biases', the QK norms' and `Wk`'s and `Wv`'s
/// (made sequence by sequence on the host) are those of calling this once per sequence into the same gradients, to the
/// bit; `Wq`'s and `Wo`'s are one product over every row, so their sums run in one pass where the per-sequence calls add
/// one sequence's sum at a time: equal within rounding.
/// \tparam C The configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param xs The tokens it read.
/// \param rope_src Its rope source.
/// \param act What it saved.
/// \param dout The gradient in its output, `n · d_model`.
/// \param dxs Added into, `n · d_model`; or empty to skip it.
/// \param grads Receives the gradients: `q`, `k`, `v` and the working space `y` (`n · q_size`) written, the rest
/// added into.
/// \param sequences The sequences the forward pass took.
template <transformer_config C, typename B, math::math_number T, typename R>
    requires linalg::blas_backend<B, T> && (C.rope == rope_kind::none || rope_source<R>)
constexpr void attention_backward(const B& backend, const attention_weights<dense_weights<T>, T>& w,
                                  std::type_identity_t<std::span<const T>> xs, const R& rope_src, const attention_activations<T>& act,
                                  std::type_identity_t<std::span<const T>> dout, std::type_identity_t<std::span<T>> dxs,
                                  const attention_gradients<T>& grads, std::size_t sequences = 1)
{
    static_assert(valid(C), "snicholls::ml::attention_backward: the transformer_config is not valid");
    [[maybe_unused]] constexpr std::size_t qn = C.q_size();   // each its own declaration: MSVC applies the attribute
    [[maybe_unused]] constexpr std::size_t kn = C.kv_size();  // to the first declarator only (C4189)
    const std::size_t rows = xs.size() / C.d_model, n = rows / sequences;
    assert(sequences > 0 && rows == n * sequences && dout.size() == xs.size() && (dxs.empty() || dxs.size() == xs.size()));
    assert(grads.q.size() == rows * qn && grads.k.size() == rows * kn && grads.v.size() == rows * kn && grads.y.size() == rows * qn);
    fill_rows(backend, grads.k, T(0));
    fill_rows(backend, grads.v, T(0));
    fill_rows(backend, grads.y, T(0));
    linear_sequence_backward(backend, w.o, std::span<const T>(act.y), dout, grads.y, grads.wo, grads.o_bias);
    attention_attend_rows_backward<C>(backend, act, std::span<const T>(grads.y), grads, n, sequences);
    attention_position_rows_backward<C>(backend, w, act, grads, n, rope_src, sequences);
    linear_sequence_backward(backend, w.q, xs, std::span<const T>(grads.q), dxs, grads.wq, grads.q_bias);
    linear_heads_backward(backend, w.k, xs, std::span<const T>(grads.k), dxs, grads.wk, grads.k_bias, C.n_kv_heads, sequences);
    linear_heads_backward(backend, w.v, xs, std::span<const T>(grads.v), dxs, grads.wv, grads.v_bias, C.n_kv_heads, sequences);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_attention_layer_hpp
