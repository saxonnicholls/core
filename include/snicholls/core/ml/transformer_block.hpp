//
//  transformer_block.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::block_forward`: one pre-norm transformer block, `x += attention(norm(x)); x += mlp(norm(x))`, for a token or a run of tokens; and for training, over a sequence with its backward pass.
/// \ingroup core
///
/// **Overview.** The layer every decoder-only transformer stacks. `block_forward` updates one token's residual
/// stream in place at its position, appending its keys and values to the cache; `block_prefill` does the same for
/// a run of consecutive tokens, which is the same thing token by token, so prefill and decode cannot disagree. The
/// normalisation (layer norm with weight and bias, or RMS norm with or without a weight), the attention and the MLP
/// are the configuration's. A whole model, the stack of blocks between an embedding and an LM head, belongs above
/// core (`INFERENCE.md` I9).
///
/// For training, `block_forward_sequence` runs a whole sequence in place and saves `block_activations` (each token's
/// result is `block_prefill`'s, to the bit); `block_backward` turns the gradient in the block's output into the
/// gradient in its input, in place, and adds every weight's gradient into `block_gradients`: the MLP's and second
/// norm's per token, the attention's over the sequence, the first norm's, each through its residual.
///
/// \snippet ml/transformer_block_demo.cpp block_backward
///
/// \snippet ml/transformer_block_demo.cpp transformer_block
///
/// **Relationships.** `ml/attention_layer.hpp`, `ml/mlp.hpp`, `ml/norm.hpp`, `ml/kv_cache.hpp`,
/// `ml/transformer_config.hpp`; the plans `INFERENCE.md` and `TRAINING.md`. Group: \ref core.
///
/// **Implementations.** One, configured at compile time.
///
/// **Guarantees.** `constexpr`, no allocation (the normalised row and each half's output are `std::array`s of
/// `d_model`), the cost of `attention_forward` plus `mlp_forward` per token.
///
/// **Errors.** Preconditions, asserted, as `attention_forward` and `mlp_forward`; the norm weights and biases are
/// empty or have `d_model`; a prefill's rows are `d_model` each.
///
/// **Performance.** One decode step of a GPT-2-small layer (768 wide, 12 heads, MLP of 3072) at position 256, against
/// the same layer written as loops over raw pointers with every operation in the same order, after a bit-for-bit
/// fairness check (`benchmarks/ml/transformer_block_benchmark.cpp`; Release, Apple clang 17, Xeon W-3245, macOS 26,
/// 2026-11-12, medians of five, on a shared machine with load average 45 to 120):
///
/// | Weights | Core | By hand |
/// |---|---|---|
/// | float | 8.45 ms | 8.31 ms |
/// | Q8_0 | 3.07 ms | 3.00 ms |
///
/// Level within the machine's noise. A decode step reads every weight once, so it is bound by memory: Q8_0's quarter
/// of the bytes makes it 2.8x faster than float.
///
/// **Tests and demos.** `tests/ml/transformer_block_test.cpp`, `demos/ml/transformer_block_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `ml/attention_layer.hpp`, `ml/mlp.hpp`,
/// `ml/norm.hpp`.
///
/// **Known issues.** Post-norm and sandwich-norm blocks (Gemma's extra norms) are not configurable.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_core_ml_transformer_block_hpp
#define snicholls_core_ml_transformer_block_hpp

#include <snicholls/core/ml/attention_layer.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/norm.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief The weights of a block: views the caller fills.
/// \ingroup core
/// \tparam W A `weight_matrix`.
/// \tparam T The number type of the norms, biases and activations.
template <weight_matrix W, math::math_number T = float>
struct block_weights
{
    std::span<const T> norm1_weight{};    ///< `d_model`, or empty: the norm before attention.
    std::span<const T> norm1_bias{};      ///< `d_model`, or empty; layer norm only.
    attention_weights<W, T> attention{};  ///< The attention.
    std::span<const T> norm2_weight{};    ///< `d_model`, or empty: the norm before the MLP.
    std::span<const T> norm2_bias{};      ///< `d_model`, or empty; layer norm only.
    mlp_weights<W, T> mlp{};              ///< The MLP.
};

/// \brief The configuration's normalisation of one row.
/// \tparam C The configuration.
/// \param x The row, `d_model`.
/// \param weight The scale, or empty.
/// \param bias The shift (layer norm), or empty.
/// \param out The normalised row; may be `x`.
template <transformer_config C, row X, mutable_row Out>
    requires std::same_as<row_value_t<X>, row_value_t<Out>>
constexpr void block_norm(const X& x, std::span<const row_value_t<X>> weight, std::span<const row_value_t<X>> bias, Out&& out)
{
    using T = row_value_t<X>;
    if constexpr (C.norm == norm_kind::layer)
        layer_norm(x, weight, bias, out, norm_eps_v<C, T>);
    else
        rms_norm(x, weight, out, norm_eps_v<C, T>);
}

/// \brief One block for one token: `x += attention(norm1(x)); x += mlp(norm2(x))`, appending to the cache at `pos`.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The token's residual stream, `d_model`, updated in place.
/// \param pos Its position.
/// \param cache The KV cache (`kv_cache_view`, `paged_kv_cache_view`).
/// \param layer This block's layer in the cache.
/// \param rope_src The cosines and sines by position, or `no_rope`.
template <transformer_config C, weight_matrix W, math::math_number T, mutable_row X, kv_store<T> S, typename R>
    requires std::same_as<row_value_t<X>, T>
constexpr void block_forward(const block_weights<W, T>& w, X&& x, std::size_t pos, const S& cache, std::size_t layer,
                             const R& rope_src)
{
    const auto s = as_mutable_row(x);
    assert(s.size() == C.d_model);
    std::array<T, C.d_model> h{}, a{};
    block_norm<C>(s, w.norm1_weight, w.norm1_bias, h);
    attention_forward<C>(w.attention, h, pos, cache, layer, rope_src, a);
    for (std::size_t i = 0; i < C.d_model; ++i)
        s[i] += a[i];
    block_norm<C>(s, w.norm2_weight, w.norm2_bias, h);
    mlp_forward<C>(w.mlp, h, a);
    for (std::size_t i = 0; i < C.d_model; ++i)
        s[i] += a[i];
}

/// \brief One block for a run of consecutive tokens, from position `first`: `block_forward` on each, in order.
/// \tparam C The configuration.
/// \param w The weights.
/// \param xs The tokens' residual streams, `d_model` each, end to end; updated in place.
/// \param first The position of the first token.
/// \param cache The KV cache.
/// \param layer This block's layer in the cache.
/// \param rope_src The cosines and sines by position, or `no_rope`.
template <transformer_config C, weight_matrix W, math::math_number T, mutable_row X, kv_store<T> S, typename R>
    requires std::same_as<row_value_t<X>, T>
constexpr void block_prefill(const block_weights<W, T>& w, X&& xs, std::size_t first, const S& cache, std::size_t layer,
                             const R& rope_src)
{
    const auto s = as_mutable_row(xs);
    assert(s.size() % C.d_model == 0);
    for (std::size_t t = 0; t < s.size() / C.d_model; ++t)
        block_forward<C>(w, s.subspan(t * C.d_model, C.d_model), first + t, cache, layer, rope_src);
}

/// \brief What `block_forward_sequence` saves for `block_backward`: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct block_activations
{
    std::span<T> x0{};                         ///< `n · d_model`: the block's input, the first norm's.
    std::span<T> h1{};                         ///< `n · d_model`: the first norm's output, the attention's input.
    std::span<T> x1{};                         ///< `n · d_model`: the residual after attention, the second norm's input.
    std::span<T> h2{};                         ///< `n · d_model`: the second norm's output, the MLP's input.
    attention_activations<T> attention{};      ///< The attention's, over the sequence.
    std::span<T> mlp_up{};                     ///< `n · d_ff`: each token's MLP up pre-activation.
    std::span<T> mlp_gate{};                   ///< `n · d_ff`: each token's gate pre-activation; SwiGLU only, else empty.
};

/// \brief The gradients of a block: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct block_gradients
{
    std::span<T> norm1_weight{};               ///< `d_model`: added into; or empty.
    std::span<T> norm1_bias{};                 ///< `d_model`: added into; or empty.
    std::span<T> norm2_weight{};               ///< `d_model`: added into; or empty.
    std::span<T> norm2_bias{};                 ///< `d_model`: added into; or empty.
    attention_gradients<T> attention{};        ///< The attention's.
    mlp_gradients<T> mlp{};                    ///< The MLP's.
    std::span<T> scratch{};                    ///< `n · d_model`: working space for the attention input's gradient.
};

namespace detail::block
{
    // The backward of block_norm for one row: dx written, the weight's and bias's gradients added into
    template <transformer_config C, typename T>
    constexpr void norm_backward(std::span<const T> dh, std::span<const T> x, std::span<const T> weight,
                                 std::span<const T> bias, std::span<T> dx, std::span<T> dweight, std::span<T> dbias)
    {
        std::array<T, C.d_model> scratch{};
        if constexpr (C.norm == norm_kind::layer) {
            const auto stats = layer_norm(x, weight, bias, scratch, norm_eps_v<C, T>);
            layer_norm_backward(dh, x, weight, stats, dx, dweight, dbias);
        } else {
            const T rstd = rms_norm(x, weight, scratch, norm_eps_v<C, T>);
            rms_norm_backward(dh, x, weight, rstd, dx, dweight);
        }
    }
} // namespace detail::block

/// \brief A block over a whole sequence, in place, saving what `block_backward` needs: each token's result is
/// `block_prefill`'s, to the bit.
/// \tparam C The configuration.
/// \param w The weights.
/// \param xs `n` residual rows of `d_model`, end to end: updated in place.
/// \param rope_src The cosines and sines by position, or `no_rope`.
/// \param act Receives the activations.
template <transformer_config C, weight_matrix W, math::math_number T, mutable_row X, typename R>
    requires std::same_as<row_value_t<X>, T> && (C.rope == rope_kind::none || rope_source<R>)
constexpr void block_forward_sequence(const block_weights<W, T>& w, X&& xs, const R& rope_src, const block_activations<T>& act)
{
    constexpr std::size_t d = C.d_model;
    const auto s = as_mutable_row(xs);
    const std::size_t n = s.size() / d;
    assert(s.size() == n * d && act.x0.size() == n * d && act.h1.size() == n * d && act.x1.size() == n * d
           && act.h2.size() == n * d && act.mlp_up.size() == n * C.d_ff
           && (C.mlp != mlp_kind::swiglu || act.mlp_gate.size() == n * C.d_ff));
    std::copy(s.begin(), s.end(), act.x0.begin());
    for (std::size_t t = 0; t < n; ++t)
        block_norm<C>(s.subspan(t * d, d), w.norm1_weight, w.norm1_bias, act.h1.subspan(t * d, d));
    attention_forward_sequence<C>(w.attention, std::span<const T>(act.h1), rope_src, act.attention, s);
    for (std::size_t i = 0; i < n * d; ++i)
        s[i] = act.x0[i] + s[i];
    std::copy(s.begin(), s.end(), act.x1.begin());
    std::array<T, d> a{};
    for (std::size_t t = 0; t < n; ++t) {
        block_norm<C>(s.subspan(t * d, d), w.norm2_weight, w.norm2_bias, act.h2.subspan(t * d, d));
        const mlp_activations<T> save{act.mlp_up.subspan(t * C.d_ff, C.d_ff),
                                      act.mlp_gate.empty() ? std::span<T>{} : act.mlp_gate.subspan(t * C.d_ff, C.d_ff)};
        mlp_forward<C>(w.mlp, std::span<const T>(act.h2).subspan(t * d, d), a, save);
        for (std::size_t i = 0; i < d; ++i)
            s[t * d + i] += a[i];
    }
}

/// \brief The backward pass of `block_forward_sequence`, in place: the gradient in the block's output becomes the
/// gradient in its input.
///
/// In reverse: each token's MLP backward and second norm's, plus the residual; the attention's over the sequence;
/// each token's first norm's, plus the residual. The weights' gradients are added into. Only dense weights train.
/// \tparam C The configuration.
/// \param w The weights the forward pass used.
/// \param rope_src Its rope source.
/// \param act What it saved.
/// \param dxs The gradient in its output, `n · d_model`: replaced by the gradient in its input.
/// \param grads Receives the gradients.
template <transformer_config C, typename S, math::math_number T, mutable_row DX, typename R>
    requires std::same_as<row_value_t<DX>, T> && (C.rope == rope_kind::none || rope_source<R>)
constexpr void block_backward(const block_weights<dense_weights<S>, T>& w, const R& rope_src,
                              const block_activations<T>& act, DX&& dxs, const block_gradients<T>& grads)
{
    constexpr std::size_t d = C.d_model;
    const auto g = as_mutable_row(dxs);
    const std::size_t n = g.size() / d;
    assert(g.size() == n * d && act.x0.size() == n * d && grads.scratch.size() == n * d);
    std::array<T, d> dh{}, dn{};
    for (std::size_t t = 0; t < n; ++t) {
        const auto gt = g.subspan(t * d, d);
        dh.fill(T(0));
        const mlp_activations<T> saved{act.mlp_up.subspan(t * C.d_ff, C.d_ff),
                                       act.mlp_gate.empty() ? std::span<T>{} : act.mlp_gate.subspan(t * C.d_ff, C.d_ff)};
        mlp_backward<C>(w.mlp, std::span<const T>(act.h2).subspan(t * d, d), saved, std::span<const T>(gt), dh, grads.mlp);
        detail::block::norm_backward<C, T>(dh, std::span<const T>(act.x1).subspan(t * d, d), w.norm2_weight, w.norm2_bias, dn,
                                           grads.norm2_weight, grads.norm2_bias);
        for (std::size_t i = 0; i < d; ++i)
            gt[i] += dn[i];
    }
    std::fill(grads.scratch.begin(), grads.scratch.end(), T(0));
    attention_backward<C>(w.attention, std::span<const T>(act.h1), rope_src, act.attention, std::span<const T>(g),
                          grads.scratch, grads.attention);
    for (std::size_t t = 0; t < n; ++t) {
        detail::block::norm_backward<C, T>(std::span<const T>(grads.scratch).subspan(t * d, d),
                                           std::span<const T>(act.x0).subspan(t * d, d), w.norm1_weight, w.norm1_bias, dn,
                                           grads.norm1_weight, grads.norm1_bias);
        for (std::size_t i = 0; i < d; ++i)
            g[t * d + i] += dn[i];
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_transformer_block_hpp
