//
//  multi_token_prediction.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` multi-token prediction (DeepSeek-V3): a chain of modules, each predicting one token further ahead from the previous depth's hidden states and the next token's embedding through one transformer block and the model's own output head; forward and backward passes over a batch of sequences, and the weighted losses.
/// \ingroup core
///
/// **Overview.** DeepSeek-V3 (DeepSeek-AI, 2024, section 2.2) trains each position to predict `D` tokens beyond the
/// next, sequentially, so the causal chain is kept at every depth. Module `k` (1 to `D`), at position `i`:
///
///     h'^k_i = M_k [RMSNorm(h^{k-1}_i); RMSNorm(Emb(t_{i+k}))]        (eq. 21; h^0 is the main model's output)
///     h^k = TRM_k(h'^k)                                              (eq. 22, one transformer block)
///     P^k_{i+k+1} = OutHead(h^k_i)                                   (eq. 23, the main model's head, shared)
///     L^k = -(1/T) Σ_{i=2+k}^{T+1} log P^k_i[t_i],   L_MTP = (λ/D) Σ_k L^k   (eqs. 24, 25)
///
/// `M_k` is `d x 2d`, its columns the previous depth's half then the embedding's; the embedding table and the head
/// are the main model's. Every depth here keeps the sequence's length `T` (as large-scale implementations do, so the
/// shapes never change): position `i` reads the embedding of token `i + k`, and the last `k` positions, which have no
/// such token, read a padding id (`mtp_input_ids`) and are left out of the loss, so they influence nothing that is
/// scored (each position sees only earlier ones). The loss of depth `k` divides by `T`, as eq. 24 does, not by the
/// `T - k` positions it scores.
///
/// - `mtp_combine_rows` and `mtp_combine_rows_backward`: the two RMS norms and the concatenation, on the backend
///   (TRAINING.md T9).
/// - `mtp_forward_sequence(backend, w, hidden_prev, embedded, rope_src, act, sequences)`: the combine, `M_k`, the block
///   (any block with batched passes: `block_weights`, `latent_block_weights`), and the norm before the shared head;
///   writes `act.output` (`h^k`, the next depth's `hidden_prev`) and `act.head_input`.
/// - `mtp_backward(...)`: from the gradient in the head's input (and in `h^k` from the depth after), the gradients in
///   `h^{k-1}` and in the embedded rows (added into; `embed_rows_backward` takes them to the shared table), and every
///   weight's.
/// - `mtp_loss_sequence(backend, head, head_input, targets, depth, ...)`: `lm_head_loss_sequence` over the scored
///   positions of each sequence with weight `λ/D` and denominator `sequences · T`, its gradients added into.
///
/// \snippet ml/multi_token_prediction_demo.cpp mtp
///
/// **At inference** the modules can be dropped: the main model does not read them (V3 section 2.2). Or they draft:
/// module 1 on the main model's last hidden state and the token just sampled proposes the token after it, which the
/// main model verifies in its next step (speculative decoding; V3 reports 85 to 90% acceptance of the second token,
/// section 5.4.3, recalled). Decoding a module one token at a time is its block's `block_forward` on `h'^k`, with a
/// cache of its own.
///
/// **Relationships.** `ml/lm_head.hpp` (the shared head), `ml/embedding.hpp` (`embed_rows`, `embed_rows_backward`),
/// `ml/transformer_block.hpp` and `ml/latent_block.hpp` (the blocks), `ml/norm.hpp`, `transformer_config::mtp_depth`
/// and `mtp_loss_weight`. Group: \ref core.
///
/// **Implementations.** One, generic over the block.
///
/// **Guarantees.** `constexpr`, no allocation. A module costs its block plus a `d x 2d` product per position. The
/// gradients are added into, so depths and micro-batches accumulate.
///
/// **Errors.** Preconditions, asserted: rows of `d_model`; `depth` below `T`; the projection `d x 2d`; working spaces of
/// their documented sizes.
///
/// **Performance.** Not benchmarked on its own: a block, one product and the head's loss, each benchmarked where it is
/// defined.
///
/// **Tests and demos.** `tests/ml/multi_token_prediction_test.cpp`, `demos/ml/multi_token_prediction_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`,
/// `linalg/backend.hpp`, `ml/linear.hpp`, `ml/lm_head.hpp`, `ml/norm.hpp`, `ml/transformer_block.hpp`,
/// `ml/transformer_config.hpp`.
///
/// **Known issues.** The block's overloads are found by argument-dependent lookup: include its header
/// (`ml/latent_block.hpp` for a latent-attention block). Released DeepSeek-V3 checkpoints store `M_k` with the embedding's columns first (`eh_proj` over
/// `[enorm(e); hnorm(h)]`, recalled from the open implementations): swap the halves when loading.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_multi_token_prediction_hpp
#define snicholls_core_ml_multi_token_prediction_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/lm_head.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The weights of one multi-token prediction module: views the caller fills.
/// \ingroup core
/// \tparam Block The block's weights: `block_weights<dense_weights<T>, T>` or `latent_block_weights<dense_weights<T>, T>`.
/// \tparam T The number type.
template <typename Block, math::math_number T = float>
struct mtp_weights
{
    std::span<const T> hidden_norm{};      ///< `d_model`, or empty: the RMS norm of `h^{k-1}`.
    std::span<const T> embedding_norm{};   ///< `d_model`, or empty: the RMS norm of the embedding.
    dense_weights<T> projection{};         ///< `M_k`, `d_model x 2 d_model`: the hidden state's columns, then the embedding's.
    Block block{};                         ///< `TRM_k`.
    std::span<const T> output_norm{};      ///< `d_model`, or empty: the norm before the shared head.
};

/// \brief What `mtp_forward_sequence` saves: views the caller owns.
/// \ingroup core
/// \tparam BlockActivations The block's activations type. \tparam T The number type.
template <typename BlockActivations, math::math_number T = float>
struct mtp_activations
{
    std::span<T> combined{};       ///< `rows · 2 d_model`: `[RMSNorm(h^{k-1}); RMSNorm(e)]`.
    BlockActivations block{};      ///< The block's.
    std::span<T> output{};         ///< `rows · d_model`: `h^k`, the next depth's input.
    std::span<T> head_input{};     ///< `rows · d_model`: the normalised `h^k`, the shared head's input.
};

/// \brief The gradients of one module, and its working spaces: views the caller owns.
/// \ingroup core
/// \tparam BlockGradients The block's gradients type. \tparam T The number type.
template <typename BlockGradients, math::math_number T = float>
struct mtp_gradients
{
    std::span<T> hidden_norm{};      ///< Added into, or empty.
    std::span<T> embedding_norm{};   ///< Added into, or empty.
    std::span<T> projection{};       ///< Added into, `d_model · 2 d_model`.
    BlockGradients block{};          ///< The block's.
    std::span<T> output_norm{};      ///< Added into, or empty.
    std::span<T> combined{};         ///< Working space, `rows · 2 d_model`.
    std::span<T> output{};           ///< Working space, `rows · d_model`.
};

/// \brief The ids module `depth` embeds: `out[s·T + i] = ids[s·T + i + depth]` while `i + depth < T`, else `pad`.
/// \param ids `sequences · T` input ids. \param depth The module, `k`, below `T`. \param sequences The sequences.
/// \param pad The id for the last `depth` positions of each sequence. \param out Receives `ids.size()` ids.
constexpr void mtp_input_ids(std::span<const std::uint32_t> ids, std::size_t depth, std::size_t sequences, std::uint32_t pad,
                             std::span<std::uint32_t> out)
{
    assert(sequences > 0 && ids.size() % sequences == 0 && out.size() == ids.size());
    const std::size_t n = ids.size() / sequences;
    assert(depth < n || n == 0);
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t i = 0; i < n; ++i)
            out[s * n + i] = i + depth < n ? ids[s * n + i + depth] : pad;
}

/// \name The steps of a module's batched passes, on the backend (TRAINING.md T9)
/// @{

/// \brief `combined_t = [RMSNorm(hidden_t) · w_h; RMSNorm(embedded_t) · w_e]` for every row.
/// \tparam C The configuration (`d_model`, `norm_eps`). \param hidden The previous depth's rows. \param embedded The
/// embedded rows. \param hidden_norm `w_h`, or empty. \param embedding_norm `w_e`, or empty. \param combined Receives
/// `rows · 2 d_model`.
template <transformer_config C, typename B, math::math_number T>
constexpr void mtp_combine_rows(const B&, std::type_identity_t<std::span<const T>> hidden, std::type_identity_t<std::span<const T>> embedded,
                                std::type_identity_t<std::span<const T>> hidden_norm,
                                std::type_identity_t<std::span<const T>> embedding_norm, std::span<T> combined)
{
    constexpr std::size_t d = C.d_model;
    constexpr T eps = norm_eps_v<C, T>;
    const std::size_t rows = hidden.size() / d;
    assert(hidden.size() == rows * d && embedded.size() == hidden.size() && combined.size() == 2 * hidden.size());
    for (std::size_t t = 0; t < rows; ++t) {
        rms_norm(hidden.subspan(t * d, d), hidden_norm, combined.subspan(2 * t * d, d), eps);
        rms_norm(embedded.subspan(t * d, d), embedding_norm, combined.subspan(2 * t * d + d, d), eps);
    }
}

/// \brief The backward pass of `mtp_combine_rows`: the halves of `dcombined` through their norms, added into
/// `dhidden` and `dembedded`, and the norms' weights' gradients added into.
/// \tparam C The configuration. \param dcombined The gradient in `combined`. \param hidden The forward's rows.
/// \param embedded Its embedded rows. \param hidden_norm `w_h`. \param embedding_norm `w_e`. \param dhidden Added into.
/// \param dembedded Added into. \param dhidden_norm Added into, or empty. \param dembedding_norm Added into, or empty.
template <transformer_config C, typename B, math::math_number T>
constexpr void mtp_combine_rows_backward(const B&, std::type_identity_t<std::span<const T>> dcombined,
                                         std::type_identity_t<std::span<const T>> hidden, std::type_identity_t<std::span<const T>> embedded,
                                         std::type_identity_t<std::span<const T>> hidden_norm,
                                         std::type_identity_t<std::span<const T>> embedding_norm, std::span<T> dhidden,
                                         std::type_identity_t<std::span<T>> dembedded, std::type_identity_t<std::span<T>> dhidden_norm,
                                         std::type_identity_t<std::span<T>> dembedding_norm)
{
    constexpr std::size_t d = C.d_model;
    constexpr T eps = norm_eps_v<C, T>;
    const std::size_t rows = hidden.size() / d;
    assert(dcombined.size() == 2 * hidden.size() && dhidden.size() == hidden.size() && dembedded.size() == embedded.size());
    std::array<T, d> scratch{}, dx{};
    for (std::size_t t = 0; t < rows; ++t) {
        const auto h = hidden.subspan(t * d, d), e = embedded.subspan(t * d, d);
        T rstd = rms_norm(h, hidden_norm, scratch, eps);
        rms_norm_backward(dcombined.subspan(2 * t * d, d), h, hidden_norm, rstd, dx, dhidden_norm);
        for (std::size_t i = 0; i < d; ++i)
            dhidden[t * d + i] += dx[i];
        rstd = rms_norm(e, embedding_norm, scratch, eps);
        rms_norm_backward(dcombined.subspan(2 * t * d + d, d), e, embedding_norm, rstd, dx, dembedding_norm);
        for (std::size_t i = 0; i < d; ++i)
            dembedded[t * d + i] += dx[i];
    }
}

/// @}

/// \brief One multi-token prediction module over a batch of sequences: combine, project, the block, the head's norm.
/// \tparam C The configuration of the module's block.
/// \param backend A `linalg::blas_backend`.
/// \param w The module's weights.
/// \param hidden_prev `h^{k-1}`: `sequences · T` rows of `d_model` (the main model's last block's output for `k = 1`).
/// \param embedded The embeddings of `mtp_input_ids(ids, k, ...)`, the same size.
/// \param rope_src The block's rope source.
/// \param act Receives the activations: `output` is `h^k`, `head_input` the shared head's input.
/// \param sequences The sequences.
template <transformer_config C, typename B, math::math_number T, typename Block, typename BlockActivations, typename R>
    requires linalg::blas_backend<B, T>
constexpr void mtp_forward_sequence(const B& backend, const mtp_weights<Block, T>& w, std::type_identity_t<std::span<const T>> hidden_prev,
                                    std::type_identity_t<std::span<const T>> embedded, const R& rope_src,
                                    const mtp_activations<BlockActivations, T>& act, std::size_t sequences = 1)
{
    assert(w.projection.rows == C.d_model && w.projection.cols == 2 * C.d_model && act.output.size() == hidden_prev.size()
           && act.head_input.size() == hidden_prev.size());
    mtp_combine_rows<C>(backend, hidden_prev, embedded, w.hidden_norm, w.embedding_norm, act.combined);
    linear_sequence(backend, w.projection, std::span<const T>(act.combined), act.output);
    block_forward_sequence<C>(backend, w.block, act.output, rope_src, act.block, sequences);
    block_norm_rows<C>(backend, std::span<const T>(act.output), w.output_norm, std::span<const T>{}, act.head_input);
}

/// \brief The backward pass of `mtp_forward_sequence`.
/// \tparam C The configuration of the module's block.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param hidden_prev Its `h^{k-1}`.
/// \param embedded Its embedded rows.
/// \param rope_src Its rope source.
/// \param act What it saved.
/// \param dhead_input The gradient in `act.head_input` (from `mtp_loss_sequence`).
/// \param doutput The gradient in `act.output` from the next depth's module, or empty for the last depth.
/// \param dhidden_prev Added into: the gradient in `h^{k-1}`.
/// \param dembedded Added into: the gradient in the embedded rows.
/// \param grads The gradients and working spaces.
/// \param sequences The sequences.
template <transformer_config C, typename B, math::math_number T, typename Block, typename BlockActivations, typename BlockGradients,
          typename R>
    requires linalg::blas_backend<B, T>
constexpr void mtp_backward(const B& backend, const mtp_weights<Block, T>& w, std::type_identity_t<std::span<const T>> hidden_prev,
                            std::type_identity_t<std::span<const T>> embedded, const R& rope_src,
                            const mtp_activations<BlockActivations, T>& act, std::type_identity_t<std::span<const T>> dhead_input,
                            std::type_identity_t<std::span<const T>> doutput, std::type_identity_t<std::span<T>> dhidden_prev,
                            std::type_identity_t<std::span<T>> dembedded, const mtp_gradients<BlockGradients, T>& grads,
                            std::size_t sequences = 1)
{
    assert(grads.output.size() == hidden_prev.size() && grads.combined.size() == 2 * hidden_prev.size()
           && (doutput.empty() || doutput.size() == hidden_prev.size()));
    if (doutput.empty())
        fill_rows(backend, grads.output, T(0));
    else
        copy_rows(backend, doutput, grads.output);
    block_norm_rows_backward_add<C>(backend, dhead_input, std::span<const T>(act.output), w.output_norm, std::span<const T>{},
                                    grads.output, grads.output_norm, std::span<T>{});
    block_backward<C>(backend, w.block, rope_src, act.block, grads.output, grads.block, sequences);
    fill_rows(backend, grads.combined, T(0));
    linear_sequence_backward(backend, w.projection, std::span<const T>(act.combined), std::span<const T>(grads.output), grads.combined,
                             grads.projection);
    mtp_combine_rows_backward<C>(backend, std::span<const T>(grads.combined), hidden_prev, embedded, w.hidden_norm, w.embedding_norm,
                                 dhidden_prev, dembedded, grads.hidden_norm, grads.embedding_norm);
}

/// \brief The loss of depth `k` through the shared head, with its gradients added into: `weight · Σ CE / (sequences · T)`
/// over the positions `i < T - k` of each sequence, which predict `targets[s·T + i + k]`.
/// \param backend A `linalg::blas_backend`.
/// \param head The main model's head, `V x d_model`.
/// \param head_input `sequences · T` rows: the module's `act.head_input`.
/// \param targets `sequences · T` next tokens, as the main model's loss reads them (`targets[i]` follows input `i`).
/// \param depth `k`, from 1, below `T`.
/// \param logits Working space, `(T - k) · V` at least.
/// \param losses Receives each scored position's cross-entropy, unweighted (`sequences · T`; 0 where unscored).
/// \param dhead_input Added into, `head_input`'s size.
/// \param dhead Added into, the head's size.
/// \param weight `λ / D`: `mtp_loss_weight_v<C, T> / C.mtp_depth`.
/// \param sequences The sequences.
/// \return The depth's weighted loss, `weight · L^k`, averaged over the batch.
template <math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr T mtp_loss_sequence(const B& backend, const dense_weights<T>& head, std::type_identity_t<std::span<const T>> head_input,
                              std::span<const std::uint32_t> targets, std::size_t depth, std::type_identity_t<std::span<T>> logits,
                              std::type_identity_t<std::span<T>> losses, std::type_identity_t<std::span<T>> dhead_input,
                              std::type_identity_t<std::span<T>> dhead, std::type_identity_t<T> weight, std::size_t sequences = 1)
{
    const std::size_t d = head.cols, rows = targets.size(), n = rows / sequences;
    assert(sequences > 0 && rows == n * sequences && depth > 0 && depth < n && head_input.size() == rows * d
           && dhead_input.size() == head_input.size() && losses.size() == rows && logits.size() >= (n - depth) * head.rows);
    fill_rows(backend, losses, T(0));
    T total = T(0);
    const std::size_t m = n - depth;
    for (std::size_t s = 0; s < sequences; ++s)
        total += lm_head_loss_sequence(backend, head, head_input.subspan(s * n * d, m * d), targets.subspan(s * n + depth, m),
                                       logits.first(m * head.rows), losses.subspan(s * n, m), dhead_input.subspan(s * n * d, m * d),
                                       dhead, 0, weight, rows);
    return total;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_multi_token_prediction_hpp
