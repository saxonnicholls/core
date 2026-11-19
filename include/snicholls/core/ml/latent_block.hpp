//
//  latent_block.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` transformer blocks whose attention is multi-head latent attention: `x += mla(norm(x)); x += mlp(norm(x))`, for decoding with a latent cache and, for training, over a batch of sequences with its backward pass.
/// \ingroup core
///
/// **Overview.** `ml/transformer_block.hpp`'s pre-norm block with `ml/latent_attention.hpp` in place of the standard
/// attention, configured at compile time by the same `transformer_config` (its MLA fields, and the sinks, sparse
/// selection, window and YaRN scale they carry): the overloads of `block_forward`, `block_forward_sequence` and
/// `block_backward` taking `latent_block_weights` are chosen by the weights' type, so a model's code calls the same
/// names for either attention.
///
/// - `block_forward<C>(w, x, pos, cache, layer, rope_src)`: one token, in place, through the absorbed attention and a
///   `latent_kv_cache_view`.
/// - `block_forward_sequence<C>(backend, w, xs, rope_src, act, sequences)` and `block_backward<C>(backend, w, rope_src,
///   act, dxs, grads, sequences)`: the batched training passes, every step a backend call (TRAINING.md T9), as the
///   standard block's.
///
/// A DeepSeek-V3 block puts a mixture of experts where this block has its dense MLP (`ml/moe.hpp`'s layer); a block of
/// both is the composition of this file's attention half with that MLP and is not built here (TRAINING.md T12).
///
/// \snippet ml/latent_block_demo.cpp latent_block
///
/// **Relationships.** `ml/latent_attention.hpp`, `ml/transformer_block.hpp` (`block_norm`, `block_norm_rows` and their
/// backward passes), `ml/mlp.hpp`, `ml/multi_token_prediction.hpp` (a user). Group: \ref core.
///
/// **Implementations.** One, configured at compile time.
///
/// **Guarantees.** `constexpr`, no allocation in decoding, the cost of the attention plus the MLP. The batched passes
/// are the standard block's with the latent attention's passes in place of `attention_forward_sequence` and
/// `attention_backward`.
///
/// **Errors.** As `ml/latent_attention.hpp` and `ml/mlp.hpp`; the norm weights empty or of `d_model`.
///
/// **Performance.** Not benchmarked on its own: the latent attention (benchmarked) plus the standard block's MLP and
/// norms.
///
/// **Tests and demos.** `tests/ml/latent_block_test.cpp`, `demos/ml/latent_block_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`,
/// `ml/latent_attention.hpp`, `ml/mlp.hpp`, `ml/transformer_block.hpp`.
///
/// **Known issues.** None beyond the latent attention's.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_latent_block_hpp
#define snicholls_core_ml_latent_block_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/ml/latent_attention.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/transformer_block.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The weights of a latent-attention block: views the caller fills.
/// \ingroup core
/// \tparam W A `weight_matrix`. \tparam T The number type of the norms and activations.
template <weight_matrix W, math::math_number T = float>
struct latent_block_weights
{
    std::span<const T> norm1_weight{};           ///< `d_model`, or empty: the norm before attention.
    std::span<const T> norm1_bias{};             ///< `d_model`, or empty; layer norm only.
    latent_attention_weights<W, T> attention{};  ///< The latent attention.
    std::span<const T> norm2_weight{};           ///< `d_model`, or empty: the norm before the MLP.
    std::span<const T> norm2_bias{};             ///< `d_model`, or empty; layer norm only.
    mlp_weights<W, T> mlp{};                     ///< The MLP.
};

/// \brief What the batched `block_forward_sequence` saves: views the caller owns, every field over all the rows.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct latent_block_activations
{
    std::span<T> x0{};                              ///< `rows · d_model`: the block's input.
    std::span<T> h1{};                              ///< `rows · d_model`: the first norm's output.
    std::span<T> x1{};                              ///< `rows · d_model`: the residual after attention.
    std::span<T> h2{};                              ///< `rows · d_model`: the second norm's output.
    latent_attention_activations<T> attention{};    ///< The latent attention's.
    std::span<T> mlp_up{};                          ///< `rows · d_ff`.
    std::span<T> mlp_gate{};                        ///< `rows · d_ff`; SwiGLU only, else empty.
    std::span<T> mlp_hidden{};                      ///< `rows · d_ff`.
};

/// \brief The gradients of a latent-attention block: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct latent_block_gradients
{
    std::span<T> norm1_weight{};                 ///< Added into, or empty.
    std::span<T> norm1_bias{};                   ///< Added into, or empty.
    std::span<T> norm2_weight{};                 ///< Added into, or empty.
    std::span<T> norm2_bias{};                   ///< Added into, or empty.
    latent_attention_gradients<T> attention{};   ///< The latent attention's, with its working spaces.
    mlp_gradients<T> mlp{};                      ///< The MLP's, with `scratch`.
    std::span<T> scratch{};                      ///< `rows · d_model`: working space.
};

/// \brief One latent-attention block for one token: `x += mla(norm1(x)); x += mlp(norm2(x))`, the token's latent row
/// appended to the cache at `pos`.
/// \tparam C A latent configuration.
/// \param w The weights (dense: the absorbed attention's requirement).
/// \param x The token's residual stream, `d_model`, updated in place.
/// \param pos Its position.
/// \param cache The latent cache.
/// \param layer This block's layer in the cache.
/// \param rope_src A rope source of `qk_rope_dim`.
template <transformer_config C, typename S, math::math_number T, mutable_row X, latent_kv_store<T> Cache, typename R>
    requires std::same_as<row_value_t<X>, T> && rope_source<R>
constexpr void block_forward(const latent_block_weights<dense_weights<S>, T>& w, X&& x, std::size_t pos, const Cache& cache,
                             std::size_t layer, const R& rope_src)
{
    const auto s = as_mutable_row(x);
    assert(s.size() == C.d_model);
    std::array<T, C.d_model> h{}, a{};
    block_norm<C>(s, w.norm1_weight, w.norm1_bias, h);
    latent_attention_forward<C>(w.attention, h, pos, cache, layer, rope_src, a);
    for (std::size_t i = 0; i < C.d_model; ++i)
        s[i] += a[i];
    block_norm<C>(s, w.norm2_weight, w.norm2_bias, h);
    mlp_forward<C>(w.mlp, h, a);
    for (std::size_t i = 0; i < C.d_model; ++i)
        s[i] += a[i];
}

/// \brief A latent-attention block over a batch of sequences, in place, every product through the backend, saving what
/// the batched `block_backward` needs.
/// \tparam C A latent configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights, dense in the sequence's type.
/// \param xs `sequences · n` residual rows of `d_model`: updated in place.
/// \param rope_src A rope source of `qk_rope_dim` covering `n - 1`.
/// \param act Receives the activations.
/// \param sequences The sequences, each of `n` tokens; attention stays within each.
template <transformer_config C, typename B, math::math_number T, typename R>
    requires linalg::blas_backend<B, T> && rope_source<R>
constexpr void block_forward_sequence(const B& backend, const latent_block_weights<dense_weights<T>, T>& w,
                                      std::type_identity_t<std::span<T>> xs, const R& rope_src, const latent_block_activations<T>& act,
                                      std::size_t sequences = 1)
{
    assert(xs.size() % C.d_model == 0 && act.x0.size() == xs.size() && act.h1.size() == xs.size() && act.x1.size() == xs.size()
           && act.h2.size() == xs.size());
    copy_rows(backend, std::span<const T>(xs), act.x0);
    block_norm_rows<C>(backend, std::span<const T>(xs), w.norm1_weight, w.norm1_bias, act.h1);
    latent_attention_forward_sequence<C>(backend, w.attention, std::span<const T>(act.h1), rope_src, act.attention, xs, sequences);
    add_rows(backend, std::span<const T>(act.x0), xs);
    copy_rows(backend, std::span<const T>(xs), act.x1);
    block_norm_rows<C>(backend, std::span<const T>(xs), w.norm2_weight, w.norm2_bias, act.h2);
    mlp_forward_sequence<C>(backend, w.mlp, std::span<const T>(act.h2), xs, mlp_activations<T>{act.mlp_up, act.mlp_gate, act.mlp_hidden});
    add_rows(backend, std::span<const T>(act.x1), xs);
}

/// \brief The backward pass of the batched latent-attention block, in place: the gradient in its output becomes the
/// gradient in its input, every weight's gradient added into.
/// \tparam C A latent configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param rope_src Its rope source.
/// \param act What it saved.
/// \param dxs The gradient in its output: replaced by the gradient in its input.
/// \param grads The gradients and working spaces.
/// \param sequences The sequences the forward pass took.
template <transformer_config C, typename B, math::math_number T, typename R>
    requires linalg::blas_backend<B, T> && rope_source<R>
constexpr void block_backward(const B& backend, const latent_block_weights<dense_weights<T>, T>& w, const R& rope_src,
                              const latent_block_activations<T>& act, std::type_identity_t<std::span<T>> dxs,
                              const latent_block_gradients<T>& grads, std::size_t sequences = 1)
{
    assert(dxs.size() == act.x0.size() && grads.scratch.size() == dxs.size());
    fill_rows(backend, grads.scratch, T(0));
    mlp_backward_sequence<C>(backend, w.mlp, std::span<const T>(act.h2), mlp_activations<T>{act.mlp_up, act.mlp_gate, act.mlp_hidden},
                             std::span<const T>(dxs), grads.scratch, grads.mlp);
    block_norm_rows_backward_add<C>(backend, std::span<const T>(grads.scratch), std::span<const T>(act.x1), w.norm2_weight, w.norm2_bias,
                                    dxs, grads.norm2_weight, grads.norm2_bias);
    fill_rows(backend, grads.scratch, T(0));
    latent_attention_backward<C>(backend, w.attention, std::span<const T>(act.h1), rope_src, act.attention, std::span<const T>(dxs),
                                 grads.scratch, grads.attention, sequences);
    block_norm_rows_backward_add<C>(backend, std::span<const T>(grads.scratch), std::span<const T>(act.x0), w.norm1_weight, w.norm1_bias,
                                    dxs, grads.norm1_weight, grads.norm1_bias);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_latent_block_hpp
