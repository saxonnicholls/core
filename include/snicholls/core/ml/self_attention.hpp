//
//  self_attention.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::self_attention_head` and its backward pass: one causal self-attention head over a sequence, from its own Q, K and V projections, with no configuration and no cache.
/// \ingroup core
///
/// **Overview.** The textbook head (Vaswani et al., 2017), on its own: for tokens `x_0 ... x_{n-1}` of `DModel`
/// channels and projections `Wq`, `Wk`, `Wv` of `DHead x DModel`,
///
/// 1. `q_t = Wq x_t`, `k_t = Wk x_t`, `v_t = Wv x_t` (plus biases), from one prepared input per token;
/// 2. optionally RoPE at position `t` on `q_t` and `k_t`;
/// 3. `out_t = softmax(q_t · K / sqrt(DHead)) · V` over the keys `attention_window(t, window)` allows: causal, and
///    within the last `window` positions if `window` is not 0.
///
/// It is the unit to study, test, train or put on a device: no `transformer_config`, no KV cache, no output
/// projection, no other heads. The projections land in `self_attention_activations`, buffers the caller owns,
/// because they are what a backward pass reads (with the log-sum-exp per position, as flash attention keeps it, never
/// the `n x n` probabilities). For a model's layer, with grouped-query heads, QK norm, a soft cap, a cache and decoding
/// one token at a time, use `attention_forward` (`ml/attention_layer.hpp`).
///
/// The weights are any `weight_matrix`: dense rows of float, double or bfloat16, or quantised blocks.
///
/// `self_attention_head_backward` is its backward pass, from what the forward saved: `attend_backward` per position
/// from the log-sum-exp, `rope_inverse`, then `linear_backward` through the projections. It writes the gradients in
/// the queries, keys and values (`self_attention_gradients`, the caller's buffers) and adds into those of the
/// weights, the biases and the tokens, so micro-batches and heads sharing an input accumulate. Only dense weights
/// train: a quantised head's backward is a compile error.
///
/// \snippet ml/self_attention_demo.cpp self_attention
///
/// \snippet ml/self_attention_demo.cpp self_attention_backward
///
/// **Relationships.** `ml/linear.hpp` (the projections), `ml/attention.hpp` (`attend`, `attention_window`),
/// `ml/rope.hpp` (`rope_table` is a rope source; `no_rope`), `ml/attention_layer.hpp` (the configured layer).
/// Group: \ref core.
///
/// **Implementations.** One. Each position's attention depends only on the projections, so positions may be split
/// across workers once the projections are written.
///
/// **Guarantees.** `constexpr`, no allocation, sizes of a head known at compile time. O(`n · 3 · DHead · DModel`) for
/// the projections and O(`n · visible · DHead`) for attention: `n² / 2 · DHead` causally, `n · window · DHead` with a
/// window. Each position's output is `attend`'s to the bit, so it equals the same head decoded one token at a time.
///
/// **Errors.** Preconditions, asserted: the input is a whole number of tokens; the matrices are `DHead x DModel`;
/// `q`, `k`, `v` and `out` have `n · DHead` elements, `lse` none or `n`; biases are empty or `DHead`; the rope source
/// covers position `n - 1`; the backward pass needs the log-sum-exps saved, and its gradient buffers sized as the
/// activations. An odd `DHead` with RoPE is a compile error.
///
/// **Performance.** Against the same projections and recurrence written by hand
/// (`benchmarks/ml/self_attention_benchmark.cpp`), one causal float head of 64 channels over tokens of 128, the bits
/// identical: within 6% at 128 tokens (3.4 ms against 3.2 ms) and 4% at 512 (23.1 ms against 22.2 ms) (Xeon W-3245,
/// Release, Apple clang 17, 2026-11-14, one thread). The backward pass, the same way: within 10% at 128 tokens
/// (1.94 ms against 1.76 ms) and level at 512 (20.8 ms against 20.9 ms). Both keep every sum in written order (`-ffp-contract=off`, no
/// reassociation), so neither is vectorised across a reduction.
///
/// **Tests and demos.** `tests/ml/self_attention_test.cpp`, `demos/ml/self_attention_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`, `snicholls/core/math/constexpr_math.hpp`,
/// `ml/attention.hpp`, `ml/linear.hpp`, `ml/rope.hpp`, `ml/row.hpp`.
///
/// **Known issues.** The projections are matrix-vector products, one token at a time, as `linear` does them; a
/// matrix-matrix backend for long sequences is to come. The score scale is `1 / sqrt(DHead)`, rounded once to `T`.
///
/// **Status.** Experimental, new in 0.8.1.

#ifndef snicholls_core_ml_self_attention_hpp
#define snicholls_core_ml_self_attention_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/row.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief The weights of one head: views the caller fills.
/// \ingroup core
/// \tparam W A `weight_matrix`, each `DHead x DModel`.
/// \tparam T The number type of the biases and activations.
template <weight_matrix W, math::math_number T = float>
struct self_attention_weights
{
    W q{};                         ///< `DHead x DModel`.
    W k{};                         ///< `DHead x DModel`.
    W v{};                         ///< `DHead x DModel`.
    std::span<const T> q_bias{};   ///< `DHead`, or empty.
    std::span<const T> k_bias{};   ///< `DHead`, or empty.
    std::span<const T> v_bias{};   ///< `DHead`, or empty.
};

/// \brief What one head writes besides its output, and a backward pass reads: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct self_attention_activations
{
    std::span<T> q{};     ///< `n · DHead`: the queries, after RoPE.
    std::span<T> k{};     ///< `n · DHead`: the keys, after RoPE.
    std::span<T> v{};     ///< `n · DHead`: the values.
    std::span<T> lse{};   ///< `n`: the log-sum-exp of each position's scores, or empty to skip it.
};

/// \brief One causal self-attention head over a sequence: projects every token, rotates, then attends each position.
/// \tparam DModel The channels of a token.
/// \tparam DHead The channels of the head.
/// \param w The weights.
/// \param xs `n` tokens of `DModel`, end to end.
/// \param act Receives the queries, keys, values and log-sum-exps.
/// \param out Receives `n` outputs of `DHead`, end to end; may not alias the inputs.
/// \param window The sliding window, counting a position's own key; 0 for fully causal.
/// \param rope_src The cosines and sines by position (a `rope_table` of `DHead`), or `no_rope`.
/// \param layout Which channels RoPE rotates together.
template <std::size_t DModel, std::size_t DHead, weight_matrix W, math::math_number T, row X, mutable_row Out,
          typename R = no_rope_t>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<Out>, T>
             && (std::same_as<R, no_rope_t> || (rope_source<R> && DHead % 2 == 0))
constexpr void self_attention_head(const self_attention_weights<W, T>& w, const X& xs,
                                   const self_attention_activations<T>& act, Out&& out, std::size_t window = 0,
                                   const R& rope_src = no_rope, rope_layout layout = rope_layout::half)
{
    static_assert(DModel > 0 && DHead > 0, "snicholls::ml::self_attention_head: a token and a head have channels");
    const auto x = as_row(xs);
    const auto y = as_mutable_row(out);
    const std::size_t n = x.size() / DModel;
    assert(x.size() == n * DModel && y.size() == n * DHead);
    assert(act.q.size() == n * DHead && act.k.size() == n * DHead && act.v.size() == n * DHead
           && (act.lse.empty() || act.lse.size() == n));
    assert(w.q.rows == DHead && w.k.rows == DHead && w.v.rows == DHead && w.q.cols == DModel && w.k.cols == DModel
           && w.v.cols == DModel);
    for (std::size_t t = 0; t < n; ++t) {
        const auto in = prepare_input<DModel>(w.q, x.subspan(t * DModel, DModel));
        const auto q = act.q.subspan(t * DHead, DHead), k = act.k.subspan(t * DHead, DHead);
        linear(w.q, in, q, w.q_bias);
        linear(w.k, in, k, w.k_bias);
        linear(w.v, in, act.v.subspan(t * DHead, DHead), w.v_bias);
        if constexpr (!std::same_as<R, no_rope_t>) {
            rope(q, rope_src.cos_at(t), rope_src.sin_at(t), layout);
            rope(k, rope_src.cos_at(t), rope_src.sin_at(t), layout);
        }
    }
    const T scale = T(1) / math::sqrt(T(DHead));
    for (std::size_t t = 0; t < n; ++t) {
        const key_range r = attention_window(t, window);
        const T l = attend(std::span<const T>(act.q.subspan(t * DHead, DHead)),
                           std::span<const T>(act.k.subspan(r.first * DHead, r.size() * DHead)),
                           std::span<const T>(act.v.subspan(r.first * DHead, r.size() * DHead)),
                           y.subspan(t * DHead, DHead), scale);
        if (!act.lse.empty())
            act.lse[t] = l;
    }
}

/// \brief The gradients of one head: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct self_attention_gradients
{
    std::span<T> q{};        ///< `n · DHead`: written, the gradients in the projected queries.
    std::span<T> k{};        ///< `n · DHead`: written, the gradients in the projected keys.
    std::span<T> v{};        ///< `n · DHead`: written, the gradients in the values.
    std::span<T> wq{};       ///< `DHead · DModel`: added into.
    std::span<T> wk{};       ///< `DHead · DModel`: added into.
    std::span<T> wv{};       ///< `DHead · DModel`: added into.
    std::span<T> q_bias{};   ///< `DHead`: added into; or empty.
    std::span<T> k_bias{};   ///< `DHead`: added into; or empty.
    std::span<T> v_bias{};   ///< `DHead`: added into; or empty.
};

/// \brief The backward pass of `self_attention_head`, from what its forward pass saved.
///
/// Per position, `attend_backward` from the saved log-sum-exp gives the gradients in the query, and adds into the
/// keys' and values' it saw; RoPE's are undone by `rope_inverse`; `linear_backward` takes them through the three
/// projections. No `n x n` probabilities are formed.
/// \tparam DModel The channels of a token.
/// \tparam DHead The channels of the head.
/// \param w The weights the forward pass used: dense storage (float, double, bfloat16).
/// \param xs The tokens it read.
/// \param act What it saved, the log-sum-exps included.
/// \param out What it wrote.
/// \param dout The gradient in `out`.
/// \param dxs Added into, `n · DModel`; or empty to skip it.
/// \param grads Receives the gradients: `q`, `k`, `v` written, the weights' and biases' added into.
/// \param window The forward pass's window.
/// \param rope_src The forward pass's rope source, or `no_rope`.
/// \param layout The forward pass's RoPE layout.
template <std::size_t DModel, std::size_t DHead, typename S, math::math_number T, row X, row O, row DO, mutable_row DX,
          typename R = no_rope_t>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<O>, T> && std::same_as<row_value_t<DO>, T>
             && std::same_as<row_value_t<DX>, T> && (std::same_as<R, no_rope_t> || (rope_source<R> && DHead % 2 == 0))
constexpr void self_attention_head_backward(const self_attention_weights<dense_weights<S>, T>& w, const X& xs,
                                            const self_attention_activations<T>& act, const O& out, const DO& dout,
                                            DX&& dxs, const self_attention_gradients<T>& grads, std::size_t window = 0,
                                            const R& rope_src = no_rope, rope_layout layout = rope_layout::half)
{
    const auto x = as_row(xs);
    const auto y = as_row(out);
    const auto g = as_row(dout);
    const auto gx = as_mutable_row(dxs);
    const std::size_t n = x.size() / DModel;
    assert(x.size() == n * DModel && y.size() == n * DHead && g.size() == n * DHead && (gx.empty() || gx.size() == x.size()));
    assert(act.q.size() == n * DHead && act.k.size() == n * DHead && act.v.size() == n * DHead && act.lse.size() == n);
    assert(grads.q.size() == n * DHead && grads.k.size() == n * DHead && grads.v.size() == n * DHead);
    std::fill(grads.k.begin(), grads.k.end(), T(0));
    std::fill(grads.v.begin(), grads.v.end(), T(0));
    const T scale = T(1) / math::sqrt(T(DHead));
    for (std::size_t t = 0; t < n; ++t) {
        const key_range r = attention_window(t, window);
        const auto keys = std::span<const T>(act.k.subspan(r.first * DHead, r.size() * DHead));
        const auto values = std::span<const T>(act.v.subspan(r.first * DHead, r.size() * DHead));
        attend_backward(std::span<const T>(act.q.subspan(t * DHead, DHead)), keys, values, y.subspan(t * DHead, DHead),
                        g.subspan(t * DHead, DHead), act.lse[t], scale, grads.q.subspan(t * DHead, DHead),
                        grads.k.subspan(r.first * DHead, r.size() * DHead),
                        grads.v.subspan(r.first * DHead, r.size() * DHead));
    }
    const std::span<T> no_dx{};
    for (std::size_t t = 0; t < n; ++t) {
        const auto dq = grads.q.subspan(t * DHead, DHead), dk = grads.k.subspan(t * DHead, DHead);
        if constexpr (!std::same_as<R, no_rope_t>) {
            rope_inverse(dq, rope_src.cos_at(t), rope_src.sin_at(t), layout);
            rope_inverse(dk, rope_src.cos_at(t), rope_src.sin_at(t), layout);
        }
        const auto xt = x.subspan(t * DModel, DModel);
        const auto dxt = gx.empty() ? no_dx : gx.subspan(t * DModel, DModel);
        linear_backward(w.q, xt, dq, dxt, grads.wq, grads.q_bias);
        linear_backward(w.k, xt, dk, dxt, grads.wk, grads.k_bias);
        linear_backward(w.v, xt, grads.v.subspan(t * DHead, DHead), dxt, grads.wv, grads.v_bias);
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_self_attention_hpp
