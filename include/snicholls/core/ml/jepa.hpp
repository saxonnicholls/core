//
//  jepa.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` joint embedding predictive architectures: the predictor's configuration, sizes, input and output, the normalised targets, the prediction losses (L2, L1, smooth L1, cosine) with their gradients, and the target encoder as an exponential moving average with its momentum schedule.
/// \ingroup core
///
/// **Overview.** A joint embedding predictive architecture (JEPA; LeCun, 2022) learns by predicting, in
/// representation space, the parts of a sample it was not shown (TRAINING.md T11):
///
/// 1. a mask (`ml/masking.hpp`) splits each sample's tokens (`ml/patch.hpp`) into context and targets;
/// 2. the **context encoder**, a bidirectional transformer, reads the context tokens only (`gather_rows`,
///    `ml/gather.hpp`, with their position embeddings);
/// 3. the **target encoder** reads every token; its outputs at the targets, normalised by a layer norm without
///    parameters (`jepa_target_norm_rows`), are the targets, and no gradient flows into them;
/// 4. the **predictor**, a narrower transformer, reads the context encoder's outputs projected to its width, followed by
///    one learnable mask token per target plus that target's position embedding (`jepa_predictor_input_rows`), and its
///    outputs at the targets, projected back, are the predictions (`slice_rows`);
/// 5. the loss compares predictions and targets (`jepa_loss_rows`), and the optimiser steps the context encoder and the
///    predictor;
/// 6. the target encoder follows the context encoder as an exponential moving average, `target = m · target + (1 -
///    m) · context` (`ema_segments`), with `m` rising from 0.996 to 1 over training (`ema_momentum`).
///
/// I-JEPA (Assran et al., 2023, "Self-Supervised Learning from Images with a Joint-Embedding Predictive
/// Architecture") is this on image patches, with an L2 loss in the paper (its code uses `smooth_l1_loss`); V-JEPA
/// (Bardes et al., 2024, "Revisiting Feature Prediction for Learning Visual Representations from Video") on video
/// tubelets with an L1 loss (`mean(|z - h|^p) / p`, `p = 1`); V-JEPA 2 (Assran et al., 2025) adds 3-D RoPE
/// (`axial_rope_table`) and an action-conditioned predictor (V-JEPA 2-AC: action and state tokens laid out frame by
/// frame with `scatter_rows`, attended block-causally, `attention_mask::block_causal`). All three code bases update the
/// target encoder as `param_k.mul_(m).add_((1 - m) · param_q)` (read 2026-11-19), which `ema_rows` reproduces
/// operation for operation.
///
/// **The text variant.** LLM-JEPA (Huang, LeCun and Balestriero, 2025, "LLM-JEPA: Large Language Models Meet Joint
/// Embedding Predictive Architectures", read 2026-11-19) keeps a language model's next-token loss and adds
/// `λ · d(Pred(Enc(Text)), Enc(Code))` over two views of one thing (a description and its code): `Enc` is the last
/// token's hidden state of the last layer (`gather_rows` at each sequence's last position), `Pred` is the model itself
/// run on `k` appended `[PRED]` tokens (no separate network; `k = 0` is the identity), `d` is the cosine distance
/// (`jepa_loss::cosine`, `1 - cos`; the paper says "cosine similarity" and the `1 -` is assumed), and there is no EMA
/// and no stop-gradient stated, so both views take gradients (`d_target`). The two views run as separate sequences
/// of one batch (causal each, padded with `attention_activations::lengths`), which is what the paper's block-causal
/// packing computes. The total is `lm_head_loss + jepa_loss_rows(..., weight = λ)`, their gradients added.
///
/// **The predictor** is composition: `jepa_predictor_config` holds its width and depth (a `transformer_config` with a
/// bidirectional mask, its `d_model` the predictor's width), the encoder's width, and the sizes follow from it
/// (`jepa_predictor_parameters`). Its pass: `linear_sequence` (encoder width to predictor width),
/// `jepa_predictor_input_rows`, `depth` batched `block_forward_sequence`s, the final norm (`block_norm_rows`),
/// `linear_sequence` back to the encoder's width, and `slice_rows` at the targets; the backward the same in reverse.
///
/// \snippet ml/jepa_demo.cpp predictor_input
///
/// \snippet ml/jepa_demo.cpp loss
///
/// \snippet ml/jepa_demo.cpp ema
///
/// **Relationships.** `ml/gather.hpp`, `ml/masking.hpp`, `ml/patch.hpp`, `ml/collapse.hpp` (the collapse
/// diagnostics and regularisers), `ml/probe.hpp` (evaluation), `ml/transformer_block.hpp`, `ml/norm.hpp`,
/// `ml/optim.hpp`. Group: \ref core.
///
/// **Implementations.** One each, on the host for any backend; a device library overloads the `_rows` and
/// `_segments` functions for its own backend type (TRAINING.md T9).
///
/// **Guarantees.** `constexpr`, no allocation. Sums run in the stated order: a loss sums each row's terms in channel
/// order, then the rows in order, then divides once. `ema_rows` with `m = 0` copies `online` and with `m = 1` keeps
/// `target`, to the bit (finite values). `jepa_predictor_config` is trivially copyable and structural.
///
/// **Errors.** Preconditions, asserted: the sizes of the spans; `lengths`, when given, one per sequence and at most the
/// rows of one; a momentum in `[0, 1]`.
///
/// **Performance.** `jepa_loss_rows` and `ema_rows` against the loops by hand (`benchmarks/ml/jepa_benchmark.cpp`):
/// see `BENCHMARKS.md`.
///
/// **Tests and demos.** `tests/ml/jepa_test.cpp` (with a whole I-JEPA training run on random images, its loss falling
/// without collapse), `demos/ml/jepa_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<cstdint>`, `<numbers>`, `<span>`, `<type_traits>`,
/// `ml/norm.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** The losses are means over every element (PyTorch's `reduction='mean'`): the I-JEPA paper's L2,
/// a sum over channels averaged over tokens, is `width` times `jepa_loss::l2`. The cosine schedule of the momentum is
/// BYOL's (Grill et al., 2020), recalled; the JEPA code bases use the linear one.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_jepa_hpp
#define snicholls_core_ml_jepa_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief A JEPA predictor: its blocks' configuration (its own width, heads and depth) and the encoder's width.
/// \ingroup core
struct jepa_predictor_config
{
    std::size_t encoder_width = 0;   ///< The encoder's `d_model`: the predictor's input and output width.
    std::size_t depth = 0;           ///< The predictor's blocks.
    transformer_config block{};      ///< Each block; `block.d_model` is the predictor's width.
};

/// \brief Whether a predictor's configuration is consistent: a valid, non-causal block, a depth, an encoder width.
/// \param c The configuration.
/// \return Whether it is valid.
[[nodiscard]] constexpr bool valid(const jepa_predictor_config& c) noexcept
{
    return valid(c.block) && c.block.mask != attention_mask::causal && c.depth > 0 && c.encoder_width > 0;
}

/// \brief I-JEPA's predictor for ViT-H/14: 12 blocks 384 wide with the encoder's 16 heads (heads recalled).
inline constexpr jepa_predictor_config ijepa_vit_h14_predictor{
    1280, 12,
    {.d_model = 384, .n_heads = 16, .n_kv_heads = 16, .head_dim = 24, .d_ff = 1536, .norm = norm_kind::layer, .norm_eps = 1e-6,
     .mlp = mlp_kind::gelu, .bias = true, .mask = attention_mask::bidirectional}};

/// \brief The parameters of one dense block: its two norms (a weight, and a bias under layer norm), the four
/// projections, the MLP's matrices, and their biases when `c.bias`. QK-norm weights are not counted.
/// \param c The block.
/// \return The count.
[[nodiscard]] constexpr std::size_t block_parameter_count(const transformer_config& c) noexcept
{
    const std::size_t d = c.d_model, norms = 2 * (c.norm == norm_kind::layer ? 2 * d : d);
    const std::size_t attention = c.q_size() * d + 2 * c.kv_size() * d + d * c.q_size()
                                  + (c.bias ? c.q_size() + 2 * c.kv_size() + d : 0);
    const std::size_t mats = c.mlp == mlp_kind::swiglu ? 3 : 2;
    const std::size_t mlp = mats * c.d_ff * d + (c.bias ? (mats - 1) * c.d_ff + d : 0);
    return norms + attention + mlp;
}

/// \brief The parameters of a predictor: the input projection, the mask token, the blocks, the final norm and the
/// output projection (the projections with biases when `c.block.bias`).
/// \param c The predictor.
/// \return The count.
[[nodiscard]] constexpr std::size_t jepa_predictor_parameters(const jepa_predictor_config& c) noexcept
{
    const std::size_t dp = c.block.d_model, de = c.encoder_width, b = c.block.bias ? 1 : 0;
    return (dp * de + b * dp) + dp + c.depth * block_parameter_count(c.block) + (c.block.norm == norm_kind::layer ? 2 * dp : dp)
           + (de * dp + b * de);
}

/// \brief The predictor's input: each sequence's `context` rows plus their positions, then the mask token plus each
/// target's position.
///
/// `out(s, j) = context(s, j) + positions[context_index(s, j)]` for `j < kc`, and `out(s, kc + j) = mask_token +
/// positions[target_index(s, j)]` for `j < kt`, each one addition per element in that order (I-JEPA's predictor: `x +=
/// pos_embed[masks_x]`, `pred_tokens = mask_token + pos_embed[masks]`, `cat([x, pred_tokens])`). With no `positions`
/// (a predictor with RoPE: give it a `gathered_rope` over the same index lists) nothing is added.
/// \param context `sequences · kc` rows of `width`: the context encoder's outputs projected to the predictor's width.
/// \param mask_token `width` values.
/// \param positions `tokens · width` position embeddings, one row per token of the grid; or empty.
/// \param context_index `sequences · kc` token indices.
/// \param target_index `sequences · kt` token indices.
/// \param out Receives `sequences · (kc + kt)` rows.
/// \param width The predictor's width.
/// \param sequences The sequences.
template <typename B, math::math_number T>
constexpr void jepa_predictor_input_rows(const B&, std::type_identity_t<std::span<const T>> context, std::type_identity_t<std::span<const T>> mask_token,
                                         std::type_identity_t<std::span<const T>> positions, std::span<const std::uint32_t> context_index,
                                         std::span<const std::uint32_t> target_index, std::span<T> out, std::size_t width,
                                         std::size_t sequences = 1) noexcept(std::is_floating_point_v<T>)
{
    assert(width > 0 && sequences > 0 && mask_token.size() == width && context_index.size() % sequences == 0
           && target_index.size() % sequences == 0 && context.size() == context_index.size() * width
           && out.size() == (context_index.size() + target_index.size()) * width && positions.size() % width == 0);
    const std::size_t kc = context_index.size() / sequences, kt = target_index.size() / sequences, n = kc + kt;
    for (std::size_t s = 0; s < sequences; ++s) {
        for (std::size_t j = 0; j < kc; ++j)
            for (std::size_t c = 0; c < width; ++c) {
                const T x = context[(s * kc + j) * width + c];
                out[(s * n + j) * width + c] = positions.empty() ? x : x + positions[context_index[s * kc + j] * width + c];
            }
        for (std::size_t j = 0; j < kt; ++j)
            for (std::size_t c = 0; c < width; ++c)
                out[(s * n + kc + j) * width + c] =
                    positions.empty() ? mask_token[c] : mask_token[c] + positions[target_index[s * kt + j] * width + c];
    }
}

/// \brief The backward pass of `jepa_predictor_input_rows`: the context rows' gradients added into `dcontext`, and every
/// target row's gradient added into `dmask_token`, sequence by sequence, target by target, in order. The positions are
/// fixed and take none.
/// \param dout The gradients in its output, `sequences · (kc + kt)` rows.
/// \param kc The context rows of a sequence.
/// \param kt The target rows of a sequence.
/// \param dcontext Added into, `sequences · kc` rows; or empty.
/// \param dmask_token Added into, `width`.
/// \param width The predictor's width.
/// \param sequences The sequences.
template <typename B, math::math_number T>
constexpr void jepa_predictor_input_rows_backward(const B&, std::type_identity_t<std::span<const T>> dout, std::size_t kc, std::size_t kt,
                                                  std::span<T> dcontext, std::type_identity_t<std::span<T>> dmask_token, std::size_t width,
                                                  std::size_t sequences = 1) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = kc + kt;
    assert(dout.size() == sequences * n * width && (dcontext.empty() || dcontext.size() == sequences * kc * width)
           && dmask_token.size() == width);
    for (std::size_t s = 0; s < sequences; ++s) {
        if (!dcontext.empty())
            for (std::size_t i = 0; i < kc * width; ++i)
                dcontext[s * kc * width + i] += dout[s * n * width + i];
        for (std::size_t j = 0; j < kt; ++j)
            for (std::size_t c = 0; c < width; ++c)
                dmask_token[c] += dout[(s * n + kc + j) * width + c];
    }
}

/// \brief The targets: each row normalised by a layer norm without parameters, `(x - mean) / sqrt(var + eps)`
/// (I-JEPA's and V-JEPA's `F.layer_norm(h, (h.size(-1),))`). The targets take no gradient (stop-gradient), so there is
/// no backward pass.
/// \param x Rows of `width`: the target encoder's outputs.
/// \param out Receives the normalised rows; may be `x`.
/// \param width The values of a row.
/// \param eps Added to the variance: PyTorch's 1e-5.
template <typename B, math::math_number T>
constexpr void jepa_target_norm_rows(const B&, std::type_identity_t<std::span<const T>> x, std::span<T> out, std::size_t width,
                                     std::type_identity_t<T> eps = T(1e-5))
{
    assert(width > 0 && x.size() % width == 0 && out.size() == x.size());
    const std::span<const T> none{};
    for (std::size_t r = 0; r < x.size() / width; ++r)
        layer_norm(x.subspan(r * width, width), none, none, out.subspan(r * width, width), eps);
}

/// \brief The distance a JEPA's prediction is scored by.
enum class jepa_loss
{
    l2,          ///< `(p - t)²` per element: I-JEPA's paper (as a mean; see the file's Known issues).
    l1,          ///< `|p - t|` per element: V-JEPA.
    smooth_l1,   ///< `0.5 (p - t)²` below 1 in magnitude, `|p - t| - 0.5` above: I-JEPA's code (PyTorch's, beta 1).
    cosine,      ///< `1 - cos(p, t)` per row: LLM-JEPA.
};

namespace detail::jepa
{
    template <typename T>
    constexpr T sign(T x) noexcept
    {
        return x > T(0) ? T(1) : (x < T(0) ? T(-1) : T(0));
    }

    // One element's term and its derivative in the difference
    template <typename T>
    constexpr T term(jepa_loss k, T d) noexcept
    {
        const T a = d < T(0) ? -d : d;
        switch (k) {
        case jepa_loss::l1:
            return a;
        case jepa_loss::smooth_l1:
            return a < T(1) ? T(0.5) * d * d : a - T(0.5);
        default:
            return d * d;
        }
    }
    template <typename T>
    constexpr T slope(jepa_loss k, T d) noexcept
    {
        switch (k) {
        case jepa_loss::l1:
            return sign(d);
        case jepa_loss::smooth_l1:
            return (d < T(0) ? -d : d) < T(1) ? d : sign(d);
        default:
            return T(2) * d;
        }
    }
} // namespace detail::jepa

/// \brief The prediction loss over rows, with its gradients added into `dpred` (and `dtarget` if given).
///
/// Over the valid rows (all, or the first `lengths[s]` of each sequence's `rows / sequences`), `count` of them:
/// elementwise kinds give `weight · (Σ_rows Σ_c term(p - t)) / (count · width)`, cosine gives `weight · Σ_rows (1 -
/// cos(p, t)) / count` with `cos = p·t / (max(|p|, 1e-8) · max(|t|, 1e-8))` (PyTorch's). Each row's terms are summed in
/// channel order, then the rows in order, then divided once. The gradient of each element is added: `dpred += g ·
/// term'(p - t)` with `g = weight / (count · width)`, and `dtarget -=` the same; for cosine, `dpred += (weight / count)
/// · (c p / |p|² - t / (|p| |t|))` and symmetrically. Padding rows get none.
/// \param kind The distance.
/// \param pred The predictions, rows of `width`.
/// \param target The targets, the same shape.
/// \param width The values of a row.
/// \param dpred Added into; `pred`'s size, or empty.
/// \param dtarget Added into; `target`'s size, or empty for a stop-gradient target.
/// \param lengths The valid rows of each sequence, or empty for every row valid.
/// \param weight A factor on the loss and its gradients (LLM-JEPA's `λ`).
/// \return The loss; 0 with no valid rows.
template <typename B, math::math_number T>
constexpr T jepa_loss_rows(const B&, jepa_loss kind, std::type_identity_t<std::span<const T>> pred, std::type_identity_t<std::span<const T>> target,
                           std::size_t width, std::span<T> dpred, std::type_identity_t<std::span<T>> dtarget = {},
                           std::span<const std::size_t> lengths = {}, std::type_identity_t<T> weight = T(1))
{
    assert(width > 0 && pred.size() == target.size() && pred.size() % width == 0 && (dpred.empty() || dpred.size() == pred.size())
           && (dtarget.empty() || dtarget.size() == pred.size()));
    const std::size_t rows = pred.size() / width, sequences = lengths.empty() ? 1 : lengths.size(), per = rows / sequences;
    assert(rows == per * sequences);
    std::size_t count = 0;
    for (std::size_t s = 0; s < sequences; ++s) {
        assert(lengths.empty() || lengths[s] <= per);
        count += lengths.empty() ? per : lengths[s];
    }
    if (count == 0)
        return T(0);
    const bool cosine = kind == jepa_loss::cosine;
    const T g = cosine ? weight / T(count) : weight / T(count * width);
    T total = T(0);
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t r = s * per; r < s * per + (lengths.empty() ? per : lengths[s]); ++r) {
            const auto p = pred.subspan(r * width, width), t = target.subspan(r * width, width);
            if (cosine) {
                T pt = T(0), pp = T(0), tt = T(0);
                for (std::size_t c = 0; c < width; ++c) {
                    pt += p[c] * t[c];
                    pp += p[c] * p[c];
                    tt += t[c] * t[c];
                }
                const T np = math::sqrt(pp), nt = math::sqrt(tt), eps = T(1e-8);
                const T dp = np > eps ? np : eps, dt = nt > eps ? nt : eps, cs = pt / (dp * dt);
                total += T(1) - cs;
                for (std::size_t c = 0; c < width; ++c) {
                    // d(1 - cos)/dp = cos p / |p|² - t / (|p| |t|) when |p| > eps, else -t / (eps |t|)
                    if (!dpred.empty())
                        dpred[r * width + c] += g * ((np > eps ? cs * p[c] / (np * np) : T(0)) - t[c] / (dp * dt));
                    if (!dtarget.empty())
                        dtarget[r * width + c] += g * ((nt > eps ? cs * t[c] / (nt * nt) : T(0)) - p[c] / (dp * dt));
                }
            } else {
                T row = T(0);
                for (std::size_t c = 0; c < width; ++c) {
                    const T d = p[c] - t[c];
                    row += detail::jepa::term(kind, d);
                    const T dd = g * detail::jepa::slope(kind, d);
                    if (!dpred.empty())
                        dpred[r * width + c] += dd;
                    if (!dtarget.empty())
                        dtarget[r * width + c] -= dd;
                }
                total += row;
            }
        }
    return cosine ? weight * (total / T(count)) : weight * (total / T(count * width));
}

/// \brief One tensor of the target encoder as an exponential moving average of the context encoder's:
/// `target = target · m + (1 - m) · online`, elementwise.
///
/// In `T` with `m = T(momentum)` and `r = T(1 - momentum)` (the complement formed in `double` and rounded once, as
/// PyTorch's `(1. - m)` is a Python float), each element is `kept = target · m`, `added = r · online`, `target = kept +
/// added`, three statements (no FMA): `param_k.mul_(m).add_((1 - m) · param_q)`, operation for operation.
/// \param target The target encoder's tensor, updated.
/// \param online The context encoder's tensor, the same size.
/// \param momentum `m`, in `[0, 1]`.
template <typename B, math::math_number T>
constexpr void ema_rows(const B&, std::span<T> target, std::type_identity_t<std::span<const T>> online, double momentum) noexcept(std::is_floating_point_v<T>)
{
    assert(target.size() == online.size() && momentum >= 0 && momentum <= 1);
    const T m = static_cast<T>(momentum), r = static_cast<T>(1.0 - momentum);
    for (std::size_t i = 0; i < target.size(); ++i) {
        const T kept = target[i] * m;
        const T added = r * online[i];
        target[i] = kept + added;
    }
}

/// \brief A target tensor and the online tensor it follows.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct ema_segment
{
    std::span<T> target{};         ///< The target encoder's tensor.
    std::span<const T> online{};   ///< The context encoder's, the same size.
};

/// \brief `ema_rows` over every tensor of a model, in the order given: one call for a whole step, which a device
/// library runs as one kernel over the segments (as an optimiser over a model's tensors).
/// \param segments The tensors.
/// \param momentum `m`, in `[0, 1]`.
template <typename B, math::math_number T>
constexpr void ema_segments(const B& backend, std::span<const ema_segment<T>> segments, double momentum) noexcept(std::is_floating_point_v<T>)
{
    for (const ema_segment<T>& s : segments)
        ema_rows(backend, s.target, s.online, momentum);
}

/// \brief How the EMA's momentum rises over training.
enum class momentum_schedule
{
    linear,   ///< `m0 + step · (m1 - m0) / total`: I-JEPA's and V-JEPA's.
    cosine,   ///< `m1 - (m1 - m0) · (cos(π step / total) + 1) / 2`: BYOL's (recalled).
};

/// \brief The EMA's momentum at `step` of `total`: from `m0` at step 0 to `m1` at `total` and after.
/// \param step The step, from 0.
/// \param total The steps of the schedule (I-JEPA's iterations per epoch times epochs), at least 1.
/// \param m0 The first momentum: 0.996 in I-JEPA, 0.998 in V-JEPA.
/// \param m1 The last: 1.
/// \param schedule The shape.
/// \return The momentum.
[[nodiscard]] constexpr double ema_momentum(std::uint64_t step, std::uint64_t total, double m0 = 0.996, double m1 = 1.0,
                                            momentum_schedule schedule = momentum_schedule::linear) noexcept
{
    assert(total > 0);
    if (step >= total)
        return m1;
    const double f = double(step) / double(total);
    if (schedule == momentum_schedule::cosine)
        return m1 - (m1 - m0) * (math::cos(std::numbers::pi * f) + 1.0) / 2.0;
    return m0 + double(step) * (m1 - m0) / double(total);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_jepa_hpp
