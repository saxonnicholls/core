//
//  composed_block.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` transformer blocks composed from an attention part and an MLP part, either kind of each: `x += attention(norm(x)); x += mlp(norm(x))`, batched for training (returning the MLP's auxiliary losses) and token by token for decoding; DeepSeek-V3's block, multi-head latent attention with a mixture of experts, as `latent_moe_block_*` with its buffers' sizes and layouts.
/// \ingroup core
///
/// **Overview.** Every pre-norm block is the same eight steps: keep the input, the first norm, the attention, the
/// residual, keep it, the second norm, the MLP, the residual. Only the two parts vary: the attention is standard
/// (`ml/attention_layer.hpp`) or multi-head latent (`ml/latent_attention.hpp`), the MLP dense (`ml/mlp.hpp`) or a
/// mixture of experts (`ml/moe.hpp`). `composed_block_weights<Attention, Mlp, T>` holds one of each beside the norms,
/// and the overloads here of `block_forward_sequence`, `block_backward` and `block_forward` run the steps with the parts
/// chosen by their weights' types (an overload set, not a chain of type tests), so four kinds of block are one template
/// rather than four headers. A new kind of attention or MLP is one overload of each part's step, not a new block.
///
/// A linear-time mixer is a third kind of attention part (TRAINING.md T14): `mamba2_weights` or
/// `gated_deltanet_weights` (`ml/mixer_layer.hpp`), with `mixer_activations` and `mixer_gradients`. A hybrid stack is
/// composed blocks of both kinds, layer `i` attention exactly when `layer_is_attention(C, i)` (Qwen3-Next's three
/// gated-delta-rule layers then one attention layer; Jamba's one attention layer in eight); decoding goes through a
/// `hybrid_cache`, its KV cache holding the attention layers alone (`hybrid_attention_index`) and its `recurrent_cache`
/// the mixer layers' states (`hybrid_mixer_index`), or through a `recurrent_cache` for a stack of mixers alone.
///
/// - `block_forward_sequence<C>(backend, w, xs, rope_src, act, sequences) -> moe_losses<T>`: the batched training pass,
///   every step a backend call (TRAINING.md T9). It returns the mixture of experts' auxiliary losses (zero for a dense
///   MLP), the sequences being the sequence-wise balance loss's groups.
/// - `block_backward<C>(backend, w, rope_src, act, dxs, grads, sequences)`: its backward pass, the auxiliary losses'
///   gradients included.
/// - `block_forward<C>(w, x, pos, cache, layer, rope_src)`: one token, in place, through a `kv_cache_view` (standard
///   attention) or a `latent_kv_cache_view` (latent), the MLP through `mlp_forward` (dense, or the overload for
///   `moe_weights`).
///
/// DeepSeek-V3's block (DeepSeek-AI, 2024, sections 2.1 and 4.2) is latent attention with a mixture of experts:
/// `latent_moe_block_weights<T>`, `latent_moe_block_activations<T>` and `latent_moe_block_gradients<T>`, sized by
/// `latent_moe_block_parameter_count`, `latent_moe_block_activation_values`, `latent_moe_block_activation_indices` and
/// `latent_moe_block_gradient_work`, laid out by `latent_moe_block_weights_in`, `latent_moe_block_activations_in` and
/// `latent_moe_block_gradients_in` (the norms, then the attention in `latent_attention_weights_in`'s order, then the
/// mixture of experts in `moe_weights_in`'s). Its node-limited routing is the configuration's `expert_groups` and
/// `group_top_k` (`ml/moe.hpp`). A multi-token prediction module whose block is this one returns the block's losses
/// (`ml/multi_token_prediction.hpp`).
///
/// \snippet ml/composed_block_demo.cpp composed_block
///
/// \snippet ml/composed_block_demo.cpp composed_decode
///
/// **Relationships.** `ml/transformer_block.hpp` (`block_norm`, `block_norm_rows` and their backward passes; the dense
/// block), `ml/moe_block.hpp` and `ml/latent_block.hpp` (the standard-and-MoE and latent-and-dense blocks with their own
/// field names, kept for their users; the composed block with the same parts gives their bits, tested),
/// `ml/attention_layer.hpp`, `ml/latent_attention.hpp`, `ml/mlp.hpp`, `ml/moe.hpp`, `ml/multi_token_prediction.hpp`
/// (a user). Group: \ref core.
///
/// **Implementations.** One skeleton; the parts are standard or latent attention or a mixer, and a dense or
/// mixture-of-experts MLP.
///
/// **Guarantees.** `constexpr`, each step in a fixed order, so the bits do not depend on scheduling; no allocation
/// (decoding keeps two rows of `d_model` and the parts' own fixed-size scratch). The batched passes are the existing
/// blocks' steps with the parts' passes, to the bit: with standard attention and an MoE MLP it is `ml/moe_block.hpp`'s
/// block, with latent attention and a dense MLP `ml/latent_block.hpp`'s, and latent attention with one expert chosen
/// with weight 1 is the latent block's dense path (all tested). Decoding is the training pass within the latent
/// attention's stated rounding (its absorbed form) and the MoE's (per-token products against the batched ones).
/// Weight gradients are added into; `dxs` is replaced by the gradient in the block's input.
///
/// **Errors.** Preconditions, asserted, as the parts' and the norms' (empty or of `d_model`). Parts of no supported
/// kind, or a configuration the parts reject (a dense one under a mixture of experts, a standard one under latent
/// attention), are compile errors.
///
/// **Performance.** Not benchmarked as a whole: its parts are (`ml/latent_attention.hpp`, `ml/moe.hpp`,
/// `ml/transformer_block.hpp`), and the skeleton adds the norms and residuals the other blocks have.
///
/// **Tests and demos.** `tests/ml/composed_block_test.cpp`, `tests/ml/composed_block_hybrid_test.cpp` (mixer blocks
/// against central differences, hybrid stacks decoded against their batched passes, a hybrid stack trained),
/// `demos/ml/composed_block_demo.cpp`, `demos/ml/mixer_layer_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`,
/// `linalg/backend.hpp`, `ml/attention_layer.hpp`, `ml/latent_attention.hpp`, `ml/linear.hpp`, `ml/mixer_layer.hpp`, `ml/mlp.hpp`,
/// `ml/moe.hpp`, `ml/row.hpp`, `ml/transformer_block.hpp`, `ml/transformer_config.hpp`. It pulls in both attentions
/// and both MLPs.
///
/// **Known issues.** The decoding MoE runs one token's experts one after another (`mlp_forward`'s overload for
/// `moe_weights`): a batch of tokens decoding together is the batched layer's job, through the backend.
///
/// **Status.** Experimental, new in 0.16.0; mixers as the attention part and hybrid caches new in 0.17.0.

#ifndef snicholls_core_ml_composed_block_hpp
#define snicholls_core_ml_composed_block_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/ml/attention_layer.hpp>
#include <snicholls/core/ml/latent_attention.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/mixer_layer.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/moe.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The weights of a composed block: the norms' views and one attention part and one MLP part.
/// \ingroup core
/// \tparam Attention `attention_weights<W, T>` or `latent_attention_weights<W, T>`.
/// \tparam Mlp `mlp_weights<W, T>` or `moe_weights<T>`.
/// \tparam T The number type of the norms.
template <typename Attention, typename Mlp, math::math_number T>
struct composed_block_weights
{
    std::span<const T> norm1_weight{};   ///< `d_model`, or empty: the norm before attention.
    std::span<const T> norm1_bias{};     ///< `d_model`, or empty; layer norm only.
    Attention attention{};               ///< The attention.
    std::span<const T> norm2_weight{};   ///< `d_model`, or empty: the norm before the MLP.
    std::span<const T> norm2_bias{};     ///< `d_model`, or empty; layer norm only.
    Mlp mlp{};                           ///< The MLP.
};

/// \brief What the composed block's batched forward pass saves: views the caller owns, every field over all the rows.
/// \ingroup core
/// \tparam AttentionActivations `attention_activations<T>` or `latent_attention_activations<T>`.
/// \tparam MlpActivations `mlp_activations<T>` or `moe_activations<T>`.
/// \tparam T The number type.
template <typename AttentionActivations, typename MlpActivations, math::math_number T>
struct composed_block_activations
{
    std::span<T> x0{};                  ///< `rows · d_model`: the block's input.
    std::span<T> h1{};                  ///< `rows · d_model`: the first norm's output.
    std::span<T> x1{};                  ///< `rows · d_model`: the residual after attention.
    std::span<T> h2{};                  ///< `rows · d_model`: the second norm's output, the MLP's input.
    AttentionActivations attention{};   ///< The attention's.
    MlpActivations mlp{};               ///< The MLP's.
};

/// \brief The gradients of a composed block, and its working space: views the caller owns.
/// \ingroup core
/// \tparam AttentionGradients `attention_gradients<T>` or `latent_attention_gradients<T>`.
/// \tparam MlpGradients `mlp_gradients<T>` or `moe_gradients<T>`.
/// \tparam T The number type.
template <typename AttentionGradients, typename MlpGradients, math::math_number T>
struct composed_block_gradients
{
    std::span<T> norm1_weight{};      ///< `d_model`: added into; or empty.
    std::span<T> norm1_bias{};        ///< `d_model`: added into; or empty.
    std::span<T> norm2_weight{};      ///< `d_model`: added into; or empty.
    std::span<T> norm2_bias{};        ///< `d_model`: added into; or empty.
    AttentionGradients attention{};   ///< The attention's, with its working spaces.
    MlpGradients mlp{};               ///< The MLP's, with its working spaces.
    std::span<T> scratch{};           ///< `rows · d_model`: working space.
};

/// \brief DeepSeek-V3's block weights: multi-head latent attention and a mixture of experts.
/// \tparam T The number type.
template <math::math_number T>
using latent_moe_block_weights = composed_block_weights<latent_attention_weights<dense_weights<T>, T>, moe_weights<T>, T>;

/// \brief DeepSeek-V3's block activations. \tparam T The number type.
template <math::math_number T>
using latent_moe_block_activations = composed_block_activations<latent_attention_activations<T>, moe_activations<T>, T>;

/// \brief DeepSeek-V3's block gradients. \tparam T The number type.
template <math::math_number T>
using latent_moe_block_gradients = composed_block_gradients<latent_attention_gradients<T>, moe_gradients<T>, T>;

namespace detail::composed_block
{
    // The attention part's passes, chosen by the weights' type
    template <transformer_config C, typename B, typename T, typename R>
    constexpr void attention_pass(const B& b, const attention_weights<dense_weights<T>, T>& w, std::span<const T> h, const R& rope_src,
                                     const attention_activations<T>& act, std::span<T> out, std::size_t sequences)
    {
        ::snicholls::ml::attention_forward_sequence<C>(b, w, h, rope_src, act, out, sequences);
    }

    template <transformer_config C, typename B, typename T, typename R>
    constexpr void attention_pass(const B& b, const latent_attention_weights<dense_weights<T>, T>& w, std::span<const T> h,
                                     const R& rope_src, const latent_attention_activations<T>& act, std::span<T> out, std::size_t sequences)
    {
        ::snicholls::ml::latent_attention_forward_sequence<C>(b, w, h, rope_src, act, out, sequences);
    }

    template <transformer_config C, typename B, typename T, typename R>
    constexpr void attention_pass_backward(const B& b, const attention_weights<dense_weights<T>, T>& w, std::span<const T> h, const R& rope_src,
                                      const attention_activations<T>& act, std::span<const T> dout, std::span<T> dh,
                                      const attention_gradients<T>& grads, std::size_t sequences)
    {
        ::snicholls::ml::attention_backward<C>(b, w, h, rope_src, act, dout, dh, grads, sequences);
    }

    template <transformer_config C, typename B, typename T, typename R>
    constexpr void attention_pass_backward(const B& b, const latent_attention_weights<dense_weights<T>, T>& w, std::span<const T> h,
                                      const R& rope_src, const latent_attention_activations<T>& act, std::span<const T> dout,
                                      std::span<T> dh, const latent_attention_gradients<T>& grads, std::size_t sequences)
    {
        ::snicholls::ml::latent_attention_backward<C>(b, w, h, rope_src, act, dout, dh, grads, sequences);
    }

    // One token's attention: through a KV cache, or a latent cache in the absorbed form
    template <transformer_config C, typename W, typename T, typename Cache, typename R>
    constexpr void attention_token(const attention_weights<W, T>& w, std::span<const T> h, std::size_t pos, const Cache& cache,
                                   std::size_t layer, const R& rope_src, std::span<T> out)
    {
        ::snicholls::ml::attention_forward<C>(w, h, pos, cache, layer, rope_src, out);
    }

    template <transformer_config C, typename S, typename T, typename Cache, typename R>
    constexpr void attention_token(const latent_attention_weights<dense_weights<S>, T>& w, std::span<const T> h, std::size_t pos,
                                   const Cache& cache, std::size_t layer, const R& rope_src, std::span<T> out)
    {
        ::snicholls::ml::latent_attention_forward<C>(w, h, pos, cache, layer, rope_src, out);
    }

    // A linear-time mixer as the attention part (TRAINING.md T14): Mamba-2 or the gated delta rule, no positions
    template <typename W, typename T>
    concept mixer_part = std::same_as<W, mamba2_weights<T>> || std::same_as<W, gated_deltanet_weights<T>>;

    template <transformer_config C, typename B, typename T, typename W, typename R>
        requires mixer_part<W, T>
    constexpr void attention_pass(const B& b, const W& w, std::span<const T> h, const R&, const mixer_activations<T>& act, std::span<T> out,
                                  std::size_t sequences)
    {
        ::snicholls::ml::mixer_forward_sequence<C>(b, w, h, act, out, sequences);
    }

    template <transformer_config C, typename B, typename T, typename W, typename R>
        requires mixer_part<W, T>
    constexpr void attention_pass_backward(const B& b, const W& w, std::span<const T> h, const R&, const mixer_activations<T>& act,
                                           std::span<const T> dout, std::span<T> dh, const mixer_gradients<T>& grads, std::size_t sequences)
    {
        ::snicholls::ml::mixer_backward<C>(b, w, h, act, dout, dh, grads, sequences);
    }

    // One token of a mixer through its layer's state: a stack of mixers' cache, or a hybrid stack's
    template <transformer_config C, typename W, typename T, typename R>
        requires mixer_part<W, T>
    constexpr void attention_token(const W& w, std::span<const T> h, std::size_t, const recurrent_cache<T>& cache, std::size_t layer, const R&,
                                   std::span<T> out)
    {
        ::snicholls::ml::mixer_forward<C>(w, h, cache.layer(layer), out);
    }

    template <transformer_config C, typename W, typename T, typename KV, typename R>
        requires mixer_part<W, T>
    constexpr void attention_token(const W& w, std::span<const T> h, std::size_t, const hybrid_cache<KV, T>& cache, std::size_t layer, const R&,
                                   std::span<T> out)
    {
        ::snicholls::ml::mixer_forward<C>(w, h, cache.recurrent.layer(hybrid_mixer_index(C, layer)), out);
    }

    // Standard attention in a hybrid stack: its slot among the attention layers
    template <transformer_config C, typename W, typename T, typename KV, typename R>
    constexpr void attention_token(const attention_weights<W, T>& w, std::span<const T> h, std::size_t pos, const hybrid_cache<KV, T>& cache,
                                   std::size_t layer, const R& rope_src, std::span<T> out)
    {
        ::snicholls::ml::attention_forward<C>(w, h, pos, cache.kv, hybrid_attention_index(C, layer), rope_src, out);
    }

    // The MLP part's passes: a dense MLP has no auxiliary losses
    template <transformer_config C, typename B, typename T>
    constexpr moe_losses<T> mlp_pass(const B& b, const mlp_weights<dense_weights<T>, T>& w, std::span<const T> h, std::span<T> out,
                                        const mlp_activations<T>& act, std::size_t)
    {
        ::snicholls::ml::mlp_forward_sequence<C>(b, w, h, out, act);
        return {};
    }

    template <transformer_config C, typename B, typename T>
    constexpr moe_losses<T> mlp_pass(const B& b, const moe_weights<T>& w, std::span<const T> h, std::span<T> out,
                                        const moe_activations<T>& act, std::size_t sequences)
    {
        return ::snicholls::ml::moe_forward<C>(b, w, h, out, act, sequences);
    }

    template <transformer_config C, typename B, typename T>
    constexpr void mlp_pass_backward(const B& b, const mlp_weights<dense_weights<T>, T>& w, std::span<const T> h, const mlp_activations<T>& act,
                                std::span<const T> dout, std::span<T> dh, const mlp_gradients<T>& grads, std::size_t)
    {
        ::snicholls::ml::mlp_backward_sequence<C>(b, w, h, act, dout, dh, grads);
    }

    template <transformer_config C, typename B, typename T>
    constexpr void mlp_pass_backward(const B& b, const moe_weights<T>& w, std::span<const T> h, const moe_activations<T>& act,
                                std::span<const T> dout, std::span<T> dh, const moe_gradients<T>& grads, std::size_t sequences)
    {
        ::snicholls::ml::moe_backward<C>(b, w, h, act, dout, dh, grads, sequences);
    }
} // namespace detail::composed_block

/// \brief A composed block over a batch of sequences, in place, saving what `block_backward` needs: `copy_rows`, the
/// first norm, the attention part, the residual, the second norm, the MLP part, the residual.
/// \tparam C The configuration: latent for latent attention, with experts for a mixture of experts.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights, dense in the sequence's type.
/// \param xs `sequences · n` residual rows of `d_model`: updated in place.
/// \param rope_src The cosines and sines by position (`qk_rope_dim` wide for latent attention), or `no_rope`.
/// \param act Receives the activations.
/// \param sequences The sequences in `xs`, each of `n` tokens; attention stays within each, and they are the balance
/// loss's groups.
/// \return The MLP's auxiliary losses: a mixture of experts' `balance` and `z` as its configuration weights them, else 0.
template <transformer_config C, typename B, math::math_number T, typename Attention, typename Mlp, typename AttentionActivations,
          typename MlpActivations, typename R>
    requires linalg::blas_backend<B, T>
constexpr moe_losses<T> block_forward_sequence(const B& backend, const composed_block_weights<Attention, Mlp, T>& w,
                                               std::type_identity_t<std::span<T>> xs, const R& rope_src,
                                               const composed_block_activations<AttentionActivations, MlpActivations, T>& act,
                                               std::size_t sequences = 1)
{
    assert(xs.size() % C.d_model == 0 && act.x0.size() == xs.size() && act.h1.size() == xs.size() && act.x1.size() == xs.size()
           && act.h2.size() == xs.size());
    copy_rows(backend, std::span<const T>(xs), act.x0);
    block_norm_rows<C>(backend, std::span<const T>(xs), w.norm1_weight, w.norm1_bias, act.h1);
    detail::composed_block::attention_pass<C>(backend, w.attention, std::span<const T>(act.h1), rope_src, act.attention, xs, sequences);
    add_rows(backend, std::span<const T>(act.x0), xs);
    copy_rows(backend, std::span<const T>(xs), act.x1);
    block_norm_rows<C>(backend, std::span<const T>(xs), w.norm2_weight, w.norm2_bias, act.h2);
    const auto losses = detail::composed_block::mlp_pass<C>(backend, w.mlp, std::span<const T>(act.h2), xs, act.mlp, sequences);
    add_rows(backend, std::span<const T>(act.x1), xs);
    return losses;
}

/// \brief The backward pass of the batched composed block, in place: the MLP part's backward (a mixture of experts'
/// auxiliary losses' gradients included), the second norm's and the residual, the attention part's, the first norm's
/// and the residual.
/// \tparam C The configuration the forward pass took.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param rope_src Its rope source.
/// \param act What it saved.
/// \param dxs The gradient in its output: replaced by the gradient in its input.
/// \param grads The gradients and working spaces.
/// \param sequences The sequences the forward pass took.
template <transformer_config C, typename B, math::math_number T, typename Attention, typename Mlp, typename AttentionActivations,
          typename MlpActivations, typename AttentionGradients, typename MlpGradients, typename R>
    requires linalg::blas_backend<B, T>
constexpr void block_backward(const B& backend, const composed_block_weights<Attention, Mlp, T>& w, const R& rope_src,
                              const composed_block_activations<AttentionActivations, MlpActivations, T>& act,
                              std::type_identity_t<std::span<T>> dxs, const composed_block_gradients<AttentionGradients, MlpGradients, T>& grads,
                              std::size_t sequences = 1)
{
    assert(dxs.size() == act.x0.size() && grads.scratch.size() == dxs.size());
    fill_rows(backend, grads.scratch, T(0));
    detail::composed_block::mlp_pass_backward<C>(backend, w.mlp, std::span<const T>(act.h2), act.mlp, std::span<const T>(dxs), grads.scratch,
                                            grads.mlp, sequences);
    block_norm_rows_backward_add<C>(backend, std::span<const T>(grads.scratch), std::span<const T>(act.x1), w.norm2_weight, w.norm2_bias,
                                    dxs, grads.norm2_weight, grads.norm2_bias);
    fill_rows(backend, grads.scratch, T(0));
    detail::composed_block::attention_pass_backward<C>(backend, w.attention, std::span<const T>(act.h1), rope_src, act.attention,
                                                  std::span<const T>(dxs), grads.scratch, grads.attention, sequences);
    block_norm_rows_backward_add<C>(backend, std::span<const T>(grads.scratch), std::span<const T>(act.x0), w.norm1_weight, w.norm1_bias,
                                    dxs, grads.norm1_weight, grads.norm1_bias);
}

/// \brief One composed block for one token: `x += attention(norm1(x)); x += mlp(norm2(x))`, the token's keys and values
/// (or latent row) appended to the cache at `pos`.
/// \tparam C The configuration.
/// \param w The weights (dense for latent attention: the absorbed form's requirement).
/// \param x The token's residual stream, `d_model`, updated in place.
/// \param pos Its position.
/// \param cache A `kv_store` for standard attention, a `latent_kv_store` for latent attention.
/// \param layer This block's layer in the cache.
/// \param rope_src The cosines and sines by position, or `no_rope`.
template <transformer_config C, typename Attention, typename Mlp, math::math_number T, mutable_row X, typename Cache, typename R>
    requires std::same_as<row_value_t<X>, T>
constexpr void block_forward(const composed_block_weights<Attention, Mlp, T>& w, X&& x, std::size_t pos, const Cache& cache, std::size_t layer,
                             const R& rope_src)
{
    const auto s = as_mutable_row(x);
    assert(s.size() == C.d_model);
    std::array<T, C.d_model> h{}, a{};
    block_norm<C>(s, w.norm1_weight, w.norm1_bias, h);
    detail::composed_block::attention_token<C>(w.attention, std::span<const T>(h), pos, cache, layer, rope_src, std::span<T>(a));
    for (std::size_t i = 0; i < C.d_model; ++i)
        s[i] += a[i];
    block_norm<C>(s, w.norm2_weight, w.norm2_bias, h);
    mlp_forward<C>(w.mlp, h, a);
    for (std::size_t i = 0; i < C.d_model; ++i)
        s[i] += a[i];
}

/// \name DeepSeek-V3's block: sizes and layouts
/// The buffers of `latent_moe_block_*`: the parameters (the norms, the latent attention, the mixture of experts), the
/// activations a training pass saves (values and indices) and the gradients' working spaces.
/// @{

/// \brief The parameters of a latent-attention, mixture-of-experts block, in `latent_moe_block_weights_in`'s order: the
/// first norm's weight (and bias under layer norm), `latent_attention_parameter_count(c)`, the second norm's, then
/// `moe_parameter_count(c)`.
/// \param c A latent configuration with experts.
/// \return The number of values (the selection bias is state, not counted).
[[nodiscard]] constexpr std::size_t latent_moe_block_parameter_count(const transformer_config& c) noexcept
{
    const std::size_t norm = (c.norm == norm_kind::layer ? 2 : 1) * c.d_model;
    return 2 * norm + latent_attention_parameter_count(c) + moe_parameter_count(c);
}

/// \brief The numbers the block's batched forward pass saves for `rows` rows: the four residual-stream rows, the
/// attention's and the mixture of experts' values.
/// \param c A latent configuration with experts. \param rows The rows.
/// \return The number of values.
[[nodiscard]] constexpr std::size_t latent_moe_block_activation_values(const transformer_config& c, std::size_t rows) noexcept
{
    return 4 * rows * c.d_model + latent_attention_activation_values(c, rows) + moe_activation_values(c, rows);
}

/// \brief The indices the block's batched forward pass saves for `rows` rows: the mixture of experts'.
/// \param c A latent configuration with experts. \param rows The rows.
/// \return The number of `std::uint32_t`s.
[[nodiscard]] constexpr std::size_t latent_moe_block_activation_indices(const transformer_config& c, std::size_t rows) noexcept
{
    return moe_activation_indices(c, rows);
}

/// \brief The working space the block's backward pass needs for `rows` rows: its scratch, the attention's and the
/// mixture of experts'.
/// \param c A latent configuration with experts. \param rows The rows.
/// \return The number of values.
[[nodiscard]] constexpr std::size_t latent_moe_block_gradient_work(const transformer_config& c, std::size_t rows) noexcept
{
    return rows * c.d_model + latent_attention_gradient_work(c, rows) + moe_gradient_work(c, rows);
}

/// \brief The weights of a latent-attention, mixture-of-experts block laid out in one buffer
/// (`latent_moe_block_parameter_count`'s order).
/// \tparam C A latent configuration with experts.
/// \param params `latent_moe_block_parameter_count(C)` values.
/// \param selection_bias The mixture of experts' selection bias, `E`, or empty.
/// \return The views.
template <transformer_config C, math::math_number T>
    requires(C.latent() && C.moe())
constexpr latent_moe_block_weights<T> latent_moe_block_weights_in(std::span<const T> params, std::span<const T> selection_bias = {})
{
    assert(params.size() == latent_moe_block_parameter_count(C));
    constexpr std::size_t d = C.d_model, b = C.norm == norm_kind::layer ? d : 0;
    detail::cutter<const T> c{params};
    latent_moe_block_weights<T> w{};
    w.norm1_weight = c.take(d);
    w.norm1_bias = c.take(b);
    w.attention = latent_attention_weights_in<C>(c.take(latent_attention_parameter_count(C)));
    w.norm2_weight = c.take(d);
    w.norm2_bias = c.take(b);
    w.mlp = moe_weights_in<C>(c.take(moe_parameter_count(C)), selection_bias);
    return w;
}

/// \brief What the block's batched forward pass saves for `rows` rows, laid out in two buffers.
/// \tparam C A latent configuration with experts.
/// \param values `latent_moe_block_activation_values(C, rows)` values.
/// \param indices `latent_moe_block_activation_indices(C, rows)` indices.
/// \param rows The rows.
/// \param selected With `sparse_top_k`, the attention's `rows · sparse_top_k` key positions, or empty.
/// \return The views.
template <transformer_config C, math::math_number T>
    requires(C.latent() && C.moe())
constexpr latent_moe_block_activations<T> latent_moe_block_activations_in(std::span<T> values, std::span<std::uint32_t> indices,
                                                                          std::size_t rows, std::span<const std::uint32_t> selected = {})
{
    assert(values.size() == latent_moe_block_activation_values(C, rows) && indices.size() == latent_moe_block_activation_indices(C, rows));
    detail::cutter<T> v{values};
    latent_moe_block_activations<T> a{};
    a.x0 = v.take(rows * C.d_model);
    a.h1 = v.take(rows * C.d_model);
    a.x1 = v.take(rows * C.d_model);
    a.h2 = v.take(rows * C.d_model);
    a.attention = latent_attention_activations_in<C>(v.take(latent_attention_activation_values(C, rows)), rows, selected);
    a.mlp = moe_activations_in<C>(v.take(moe_activation_values(C, rows)), indices, rows);
    return a;
}

/// \brief The gradients of the block in `latent_moe_block_weights_in`'s layout, and its backward pass's working spaces.
/// \tparam C A latent configuration with experts.
/// \param grads `latent_moe_block_parameter_count(C)` values, added into.
/// \param work `latent_moe_block_gradient_work(C, rows)` values.
/// \param rows The rows.
/// \return The views.
template <transformer_config C, math::math_number T>
    requires(C.latent() && C.moe())
constexpr latent_moe_block_gradients<T> latent_moe_block_gradients_in(std::span<T> grads, std::span<T> work, std::size_t rows)
{
    assert(grads.size() == latent_moe_block_parameter_count(C) && work.size() == latent_moe_block_gradient_work(C, rows));
    constexpr std::size_t d = C.d_model, b = C.norm == norm_kind::layer ? d : 0;
    detail::cutter<T> c{grads}, w{work};
    latent_moe_block_gradients<T> g{};
    g.norm1_weight = c.take(d);
    g.norm1_bias = c.take(b);
    g.attention = latent_attention_gradients_in<C>(c.take(latent_attention_parameter_count(C)), w.take(latent_attention_gradient_work(C, rows)),
                                                   rows);
    g.norm2_weight = c.take(d);
    g.norm2_bias = c.take(b);
    g.mlp = moe_gradients_in<C>(c.take(moe_parameter_count(C)), w.take(moe_gradient_work(C, rows)), rows);
    g.scratch = w.take(rows * d);
    return g;
}

/// @}

} // namespace snicholls::ml

#endif // snicholls_core_ml_composed_block_hpp
