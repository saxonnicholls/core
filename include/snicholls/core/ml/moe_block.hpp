//
//  moe_block.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::block_forward_sequence` and `block_backward` for a block whose MLP is a mixture of experts: `x += attention(norm(x)); x += moe(norm(x))`, batched through a backend.
/// \ingroup core
///
/// **Overview.** The pre-norm block of `ml/transformer_block.hpp` with its MLP replaced by `moe_forward`
/// (`ml/moe.hpp`), for a configuration with `experts > 0`: the same norms, the same attention over a batch of
/// sequences, the same residual adds, every step a call on the backend (TRAINING.md T9, T10). The weights, activations
/// and gradients are the dense block's with an MoE part (`moe_block_weights`, `moe_block_activations`,
/// `moe_block_gradients`); the forward pass returns the MoE's auxiliary losses, which the backward pass includes in its
/// gradients.
///
/// \snippet ml/moe_block_demo.cpp moe_block
///
/// **Relationships.** `ml/moe.hpp` (the MLP), `ml/transformer_block.hpp` (the dense block, whose norm steps these
/// reuse), `ml/attention_layer.hpp`, `ml/composed_block.hpp` (the same steps with either attention; DeepSeek-V3's block
/// of latent attention and a mixture of experts). Group: \ref core.
///
/// **Implementations.** One: the batched path, products through a `linalg` backend. Decoding one token is
/// `ml/composed_block.hpp`'s `block_forward` with these parts (`mlp_forward`'s overload for `moe_weights`).
///
/// **Guarantees.** As the batched dense block's: `constexpr`, no allocation, each step in a fixed order, so the bits do
/// not depend on scheduling. The forward pass's output is the dense block's steps with `moe_forward` in place of
/// `mlp_forward_sequence`, to the bit (tested by composing them by hand).
///
/// **Errors.** Preconditions, asserted, as the dense block's and `moe_forward`'s. A dense configuration is a compile
/// error here.
///
/// **Performance.** Not benchmarked as a whole: its parts are (`ml/transformer_block.hpp`, `ml/moe.hpp`).
///
/// **Tests and demos.** `tests/ml/moe_block_test.cpp`, `demos/ml/moe_block_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`,
/// `ml/attention_layer.hpp`, `ml/moe.hpp`, `ml/transformer_block.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_moe_block_hpp
#define snicholls_core_ml_moe_block_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/ml/attention_layer.hpp>
#include <snicholls/core/ml/moe.hpp>
#include <snicholls/core/ml/transformer_block.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The weights of a block with a mixture-of-experts MLP: views the caller fills.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct moe_block_weights
{
    std::span<const T> norm1_weight{};                  ///< `d_model`, or empty: the norm before attention.
    std::span<const T> norm1_bias{};                    ///< `d_model`, or empty; layer norm only.
    attention_weights<dense_weights<T>, T> attention{}; ///< The attention.
    std::span<const T> norm2_weight{};                  ///< `d_model`, or empty: the norm before the MLP.
    std::span<const T> norm2_bias{};                    ///< `d_model`, or empty; layer norm only.
    moe_weights<T> moe{};                               ///< The mixture of experts.
};

/// \brief What the MoE block's forward pass saves: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct moe_block_activations
{
    std::span<T> x0{};                      ///< `n · d_model`: the block's input.
    std::span<T> h1{};                      ///< `n · d_model`: the first norm's output.
    std::span<T> x1{};                      ///< `n · d_model`: the residual after attention.
    std::span<T> h2{};                      ///< `n · d_model`: the second norm's output, the MoE's input.
    attention_activations<T> attention{};   ///< The attention's.
    moe_activations<T> moe{};               ///< The mixture of experts' (`moe_activations_in` for `n` rows).
};

/// \brief The gradients of the MoE block: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct moe_block_gradients
{
    std::span<T> norm1_weight{};          ///< `d_model`: added into; or empty.
    std::span<T> norm1_bias{};            ///< `d_model`: added into; or empty.
    std::span<T> norm2_weight{};          ///< `d_model`: added into; or empty.
    std::span<T> norm2_bias{};            ///< `d_model`: added into; or empty.
    attention_gradients<T> attention{};   ///< The attention's.
    moe_gradients<T> moe{};               ///< The mixture of experts' (`moe_gradients_in` for `n` rows).
    std::span<T> scratch{};               ///< `n · d_model`: working space.
};

/// \brief A block with a mixture-of-experts MLP over a batch of sequences, in place, saving what `block_backward`
/// needs: `copy_rows`, the first norm, `attention_forward_sequence`, the residual, the second norm, `moe_forward`, the
/// residual, as the dense batched block with `moe_forward` for its MLP.
/// \tparam C A mixture-of-experts configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights.
/// \param xs `sequences · n` residual rows of `d_model`: updated in place.
/// \param rope_src The cosines and sines by position, or `no_rope`.
/// \param act Receives the activations.
/// \param sequences The sequences in `xs`, each of `n` tokens; attention stays within each, and they are the balance
/// loss's groups.
/// \return The MoE's auxiliary losses.
template <transformer_config C, typename B, math::math_number T, typename R>
    requires linalg::blas_backend<B, T> && (C.experts > 0) && (C.rope == rope_kind::none || rope_source<R>)
constexpr moe_losses<T> block_forward_sequence(const B& backend, const moe_block_weights<T>& w, std::type_identity_t<std::span<T>> xs,
                                               const R& rope_src, const moe_block_activations<T>& act, std::size_t sequences = 1)
{
    constexpr std::size_t d = C.d_model;
    [[maybe_unused]] const std::size_t n = xs.size() / d;
    assert(xs.size() == n * d && act.x0.size() == n * d && act.h1.size() == n * d && act.x1.size() == n * d && act.h2.size() == n * d);
    copy_rows(backend, std::span<const T>(xs), act.x0);
    block_norm_rows<C>(backend, std::span<const T>(xs), w.norm1_weight, w.norm1_bias, act.h1);
    attention_forward_sequence<C>(backend, w.attention, std::span<const T>(act.h1), rope_src, act.attention, xs, sequences);
    add_rows(backend, std::span<const T>(act.x0), xs);
    copy_rows(backend, std::span<const T>(xs), act.x1);
    block_norm_rows<C>(backend, std::span<const T>(xs), w.norm2_weight, w.norm2_bias, act.h2);
    const auto losses = moe_forward<C>(backend, w.moe, std::span<const T>(act.h2), xs, act.moe, sequences);
    add_rows(backend, std::span<const T>(act.x1), xs);
    return losses;
}

/// \brief The backward pass of the MoE block, in place: `moe_backward`, the second norm's backward and the residual,
/// the attention's backward, the first norm's backward and the residual. The MoE's auxiliary losses' gradients are
/// included.
/// \tparam C A mixture-of-experts configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param rope_src Its rope source.
/// \param act What it saved.
/// \param dxs The gradient in its output, `sequences · n · d_model`: replaced by the gradient in its input.
/// \param grads Receives the gradients; the working spaces cover all the rows.
/// \param sequences The sequences the forward pass took.
template <transformer_config C, typename B, math::math_number T, typename R>
    requires linalg::blas_backend<B, T> && (C.experts > 0) && (C.rope == rope_kind::none || rope_source<R>)
constexpr void block_backward(const B& backend, const moe_block_weights<T>& w, const R& rope_src, const moe_block_activations<T>& act,
                              std::type_identity_t<std::span<T>> dxs, const moe_block_gradients<T>& grads, std::size_t sequences = 1)
{
    constexpr std::size_t d = C.d_model;
    [[maybe_unused]] const std::size_t n = dxs.size() / d;
    assert(dxs.size() == n * d && act.x0.size() == n * d && grads.scratch.size() == n * d);
    fill_rows(backend, grads.scratch, T(0));
    moe_backward<C>(backend, w.moe, std::span<const T>(act.h2), act.moe, std::span<const T>(dxs), grads.scratch, grads.moe, sequences);
    block_norm_rows_backward_add<C>(backend, std::span<const T>(grads.scratch), std::span<const T>(act.x1), w.norm2_weight, w.norm2_bias,
                                    dxs, grads.norm2_weight, grads.norm2_bias);
    fill_rows(backend, grads.scratch, T(0));
    attention_backward<C>(backend, w.attention, std::span<const T>(act.h1), rope_src, act.attention, std::span<const T>(dxs), grads.scratch,
                          grads.attention, sequences);
    block_norm_rows_backward_add<C>(backend, std::span<const T>(grads.scratch), std::span<const T>(act.x0), w.norm1_weight, w.norm1_bias,
                                    dxs, grads.norm1_weight, grads.norm1_bias);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_moe_block_hpp
