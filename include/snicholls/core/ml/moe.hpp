//
//  moe.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` mixture of experts for training: routing with a selection bias, the expert permutation, gather and combine, the grouped products and the experts' MLPs, the balance losses and the bias rule, load statistics, and the layer `moe_forward` / `moe_backward` over a batch of rows.
/// \ingroup core
///
/// **Overview.** A mixture-of-experts (MoE) MLP replaces a transformer's one dense MLP by `E` smaller ones, the
/// routed experts, of which each token uses `K`, plus optionally shared experts every token uses. A router (a linear
/// map to `E` logits) chooses each token's experts and their weights; the token's output is the weighted sum of its
/// experts' outputs. The parameters grow with `E` while the work per token grows only with `K`. The configuration's
/// expert fields (`transformer_config::experts`, `top_k`, `shared_experts`, `expert_d_ff`, `scoring`, `route_scale`
/// and the balance settings) choose among the published designs:
///
/// - **Switch Transformer** (Fedus, Zoph and Shazeer, 2021) and **OLMoE** (Muennighoff et al., 2024):
///   `moe_scoring::softmax`, a softmax over every expert, the chosen probabilities as weights, and the auxiliary
///   balance loss `alpha · E · sum_e f_e P_e`.
/// - **Mixtral** (Jiang et al., 2024): `moe_scoring::softmax_top_k`, the `K` largest logits, a softmax over those.
///   This is `moe_gate` (`ml/mlp.hpp`) row by row, to the bit.
/// - **DeepSeekMoE** (Dai et al., 2024) and **DeepSeek-V3** (DeepSeek-AI, 2024): fine-grained experts (many, narrow),
///   shared experts, `moe_scoring::sigmoid` (a sigmoid per expert, the chosen ones normalised, times `route_scale`),
///   and auxiliary-loss-free balancing: a per-expert bias added to the scores for the *choice only*, never entering
///   the weights, moved after each step by `bias_e += gamma · sign(mean load - load_e)` (`moe_update_bias`), with an
///   optional small sequence-wise balance loss (`moe_balance_loss` with one sequence per group of rows).
/// - **ST-MoE** (Zoph et al., 2022): the router z-loss `beta · mean_t logsumexp(z_t)²` (`moe_z_loss`), which keeps
///   the logits small.
///
/// **The steps.** Every step is a free function taking the backend first, over rows laid end to end (TRAINING.md T9
/// and T10): core's overloads run on the host and define the result; a device library overloads each for its own
/// backend type and the layer runs where its data lives.
///
/// 1. `moe_route_rows`: the router logits `[rows x E]` to each row's `K` experts (`std::uint32_t`, slot order: best
///    first, ties to the lower expert) and weights; `moe_route_rows_backward` takes the weights' gradient back to the
///    logits.
/// 2. `moe_count_rows` (a histogram of the `rows · K` choices), `moe_offsets` (its exclusive prefix sum, `E + 1`) and
///    `moe_permutation`: the stable order of the `(row, slot)` pairs by `(expert, row, slot)` and its inverse. The
///    order is unique, so a parallel sort on a device must give the same permutation.
/// 3. `moe_gather_rows` (token rows into expert order, `[rows · K x d]`) and `moe_combine_rows` (back, weighted:
///    `y_r = w_r0 yp_(r,0) + w_r1 yp_(r,1) + ...` in slot order, so the combine is deterministic, no atomics), and their
///    backward passes.
/// 4. `grouped_linear`: expert `e` multiplies its segment of the permuted rows by its own matrix, with both backward
///    products in `grouped_linear_backward`; `moe_experts_forward` / `moe_experts_backward`, the experts' MLPs
///    (GELU, ReLU² or SwiGLU, the configuration's `mlp`) over the permuted rows, with the dense MLP's activation steps.
/// 5. Shared experts: one dense MLP of `shared_experts · expert_d_ff` (`moe_shared_config`), the dense passes of
///    `ml/mlp.hpp`, added to the routed output.
/// 6. Balancing: `moe_update_bias`, `moe_balance_loss`, `moe_z_loss`, and `moe_load`, per-expert load statistics that
///    merge across workers.
/// 7. The layer: `moe_forward` and `moe_backward` compose them; `moe_parameter_count`, `moe_activation_values`,
///    `moe_activation_indices` and `moe_gradient_work` size its buffers, and `moe_weights_in`, `moe_activations_in` and
///    `moe_gradients_in` lay them out. The block with an MoE MLP is `ml/moe_block.hpp`; expert parallelism across
///    ranks is `ml/expert_parallel.hpp`.
///
/// \snippet ml/moe_demo.cpp route
///
/// \snippet ml/moe_demo.cpp layer
///
/// \snippet ml/moe_demo.cpp balance
///
/// **The arithmetic, exactly** (so that a device can match it). Every product and sum is its own statement (no fused
/// multiply-add; GCC needs `-ffp-contract=off`, see `ml/quantize.hpp`), in `T`.
///
/// - Routing, per row `z` (logits) and selection bias `b` (or none): the affinities `s` are `z` itself
///   (`softmax_top_k`), `softmax(z)` (`softmax`, `ml/softmax.hpp`'s own) or `sigmoid(z_e)` (`sigmoid`); the key is
///   `s_e + b_e` (or `s_e`); the chosen experts are `top_k<K>` of the keys (`ml/sampling.hpp`: largest first, ties to
///   the lower index). Weights, slot `i` of expert `e_i`: `softmax_top_k`, `m` the largest chosen logit,
///   `u_i = exp(z_(e_i) - m)`, `S = u_0 + u_1 + ...` in slot order, `w_i = (u_i / S) · scale`; `softmax`,
///   `w_i = p_(e_i) · scale`; `sigmoid`, `S` the sum of the chosen `s` in slot order, `w_i = (s_(e_i) / S) · scale`.
/// - The combine: `y = w_0 · yp_0`, then `y = y + w_i · yp_i` for `i = 1 .. K-1`; its backward `dyp_i = w_i · dy` and
///   `dw_i = sum_j yp_i[j] · dy[j]` from 0 in `j` order; the gather's backward `dx = dx + dxp_i` for `i = 0 .. K-1`.
/// - The grouped products: each expert's segment is `linear_sequence` (one `gemm`), experts in order; on core's
///   reference backend a row's output depends only on that row, and an expert's weight gradient sums its segment's
///   rows in permuted order.
/// - The balance loss over `S` groups of `T = rows / S` rows (a group is a sequence: `S = 1` is Switch's batch-wise
///   loss, one per sequence DeepSeek-V3's sequence-wise one), with `D` the denominator (`S` unless given):
///   `f_e = c_e / (T · K)` from the group's counts, `P_e = (sum_t p_te) / T` summed in row order, the group's
///   `L = (alpha · E) · sum_e f_e P_e` in expert order, the loss `(sum of the groups' L in order) / D`; the gradient in
///   `p_te` is `((alpha · E) · f_e) / (T · D)`, taken through the softmax (or the normalised sigmoid) to the logits.
/// - The z-loss over `rows` rows with denominator `D` (`rows` unless given): `beta · (sum_t lse_t²) / D` in row order;
///   the gradient in `z_tj` is `((2 · beta) · lse_t / D) · softmax(z_t)_j`.
///
/// **Relationships.** `ml/mlp.hpp` (`moe_gate`, the per-token router; the dense MLP passes, reused), `ml/linear.hpp`
/// (`linear_sequence`, the products), `ml/softmax.hpp`, `ml/sampling.hpp` (`top_k`), `ml/transformer_config.hpp` (the
/// expert fields), `ml/moe_block.hpp`, `ml/expert_parallel.hpp`; the plan, TRAINING.md T10. Group: \ref core.
///
/// **Implementations.** One, for any backend, on the host. The grouped product loops `linear_sequence` over the
/// experts; a device library replaces it by one grouped kernel.
///
/// **Guarantees.** `constexpr`, no allocation (a row's affinities and keys are `std::array`s of `E`), deterministic: the
/// same inputs give the same bits whatever the backend's scheduling, because every reduction's order is stated above.
/// Routing is O(rows · E · K), the permutation O(rows · K + E), gather and combine O(rows · K · d), the experts' MLPs the
/// dense MLP's work for `rows · K` rows. Weight gradients are added into; the activations' gradients are written, except
/// the layer's input gradient, which is added into (TRAINING.md T3). `moe_load` is trivially copyable and merges.
///
/// **Errors.** Preconditions, asserted: every span has the size its documentation gives, expert ids are below `E`, the
/// rows divide into the groups. An invalid configuration, or a dense one where experts are needed, is a compile error.
///
/// **Performance.** The grouped product, the gather and the combine through `linalg::reference` against the same loops
/// by hand, after a bit-for-bit fairness check (`benchmarks/ml/moe_benchmark.cpp`); numbers in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/ml/moe_test.cpp`, `demos/ml/moe_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`,
/// `linalg/backend.hpp`, `math/constexpr_math.hpp`, `ml/activations.hpp`, `ml/linear.hpp`, `ml/mlp.hpp`,
/// `ml/sampling.hpp`, `ml/softmax.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** Expert capacity (dropping tokens past a fixed number per expert, as Switch does) is not
/// implemented: every chosen pair is computed, so the segments' sizes vary with the routing. The selection bias has no
/// gradient by design; it is state the caller keeps beside the weights.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_moe_hpp
#define snicholls_core_ml_moe_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/sampling.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The configuration of one routed expert's MLP: `c` with `d_ff = expert_d_ff` and no experts.
/// \param c A mixture-of-experts configuration.
/// \return The dense configuration each expert's MLP has.
[[nodiscard]] constexpr transformer_config moe_expert_config(transformer_config c) noexcept
{
    c.d_ff = c.expert_d_ff;
    c.experts = c.top_k = c.shared_experts = c.expert_d_ff = 0;
    return c;
}

/// \brief The configuration of the shared experts, as one dense MLP: `c` with `d_ff = shared_experts · expert_d_ff`.
///
/// `S` shared experts of width `f`, their outputs summed, are one MLP of width `S · f` (the hidden units side by side),
/// so the shared experts are the dense MLP passes of `ml/mlp.hpp` with this configuration.
/// \param c A mixture-of-experts configuration with `shared_experts > 0`.
/// \return The dense configuration of the shared MLP.
[[nodiscard]] constexpr transformer_config moe_shared_config(transformer_config c) noexcept
{
    c.d_ff = c.shared_experts * c.expert_d_ff;
    c.experts = c.top_k = c.shared_experts = c.expert_d_ff = 0;
    return c;
}

/// \brief The weight matrices of `experts` experts, one after another: `experts` matrices of `rows x cols`.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct grouped_weights
{
    std::span<const T> data{};   ///< `experts · rows · cols` elements, expert by expert, each row by row.
    std::size_t experts = 0;     ///< The matrices.
    std::size_t rows = 0;        ///< Each one's outputs.
    std::size_t cols = 0;        ///< Each one's inputs.

    /// \brief One expert's matrix.
    /// \param e The expert, below `experts`.
    /// \return Its `rows x cols` view.
    [[nodiscard]] constexpr dense_weights<T> expert(std::size_t e) const { return {data.subspan(e * rows * cols, rows * cols), rows, cols}; }

    /// \brief The matrices of a run of experts, as expert parallelism places them on one rank.
    /// \param first The first expert.
    /// \param count The experts.
    /// \return Their view.
    [[nodiscard]] constexpr grouped_weights slice(std::size_t first, std::size_t count) const
    {
        return {data.subspan(first * rows * cols, count * rows * cols), count, rows, cols};
    }
};

namespace detail::moe
{
    using ::snicholls::ml::softmax;   // the function, not namespace detail::softmax

    // The affinities of one row: z itself, softmax(z) or sigmoid(z), by the configuration's scoring
    template <transformer_config C, typename T>
    constexpr void affinities(std::span<const T> z, std::array<T, C.experts>& s)
    {
        switch (C.scoring) {
        case moe_scoring::softmax_top_k:
            for (std::size_t e = 0; e < C.experts; ++e)
                s[e] = z[e];
            return;
        case moe_scoring::softmax: softmax(z, s); return;
        case moe_scoring::sigmoid:
            for (std::size_t e = 0; e < C.experts; ++e)
                s[e] = sigmoid(z[e]);
            return;
        }
    }

    // The router probabilities of one row the balance loss reads: softmax(z), or sigmoid(z) normalised over every expert
    template <transformer_config C, typename T>
    constexpr void probabilities(std::span<const T> z, std::array<T, C.experts>& p)
    {
        if (C.scoring != moe_scoring::sigmoid) {
            softmax(z, p);
            return;
        }
        T sum = T(0);
        for (std::size_t e = 0; e < C.experts; ++e) {
            p[e] = sigmoid(z[e]);
            sum += p[e];
        }
        for (std::size_t e = 0; e < C.experts; ++e)
            p[e] = p[e] / sum;
    }

    // The gradient in the logits from the gradient g in probabilities(z), added into dz
    template <transformer_config C, typename T>
    constexpr void probabilities_backward(std::span<const T> z, const std::array<T, C.experts>& p, const std::array<T, C.experts>& g,
                                          std::span<T> dz)
    {
        T dot = T(0);
        for (std::size_t e = 0; e < C.experts; ++e)
            dot += p[e] * g[e];
        if (C.scoring != moe_scoring::sigmoid) {
            for (std::size_t e = 0; e < C.experts; ++e)
                dz[e] += p[e] * (g[e] - dot);
            return;
        }
        T sum = T(0);
        for (std::size_t e = 0; e < C.experts; ++e)
            sum += sigmoid(z[e]);
        for (std::size_t e = 0; e < C.experts; ++e)
            dz[e] += (g[e] - dot) / sum * sigmoid_derivative(z[e]);
    }

    // The chosen experts' softmax_top_k weights before the scale: u_i / S with u_i = exp(z_i - m)
    template <transformer_config C, typename T>
    constexpr std::array<T, C.top_k> top_softmax(std::span<const T> z, std::span<const std::uint32_t> ids)
    {
        std::array<T, C.top_k> q{};
        T m = z[ids[0]];
        for (std::size_t i = 1; i < C.top_k; ++i)
            m = z[ids[i]] > m ? z[ids[i]] : m;
        T sum = T(0);
        for (std::size_t i = 0; i < C.top_k; ++i) {
            q[i] = math::exp(z[ids[i]] - m);
            sum += q[i];
        }
        for (std::size_t i = 0; i < C.top_k; ++i)
            q[i] /= sum;
        return q;
    }
} // namespace detail::moe

/// \name 1. Routing
/// @{

/// \brief Routes every row: the `K` experts of the largest keys (the affinities plus the selection bias) and their
/// weights, as the configuration's `scoring` says (the file's "The arithmetic, exactly").
///
/// With `moe_scoring::softmax_top_k`, no bias and `route_scale` 1, each row is `moe_gate<K>` to the bit.
/// \tparam C A mixture-of-experts configuration.
/// \param logits The router logits, `rows · E`.
/// \param bias The selection bias, `E` (added to the affinities for the choice only), or empty.
/// \param experts Receives the chosen experts, `rows · K`, best first in each row; ties to the lower expert.
/// \param weights Receives their weights, `rows · K`.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr void moe_route_rows(const B&, std::type_identity_t<std::span<const T>> logits, std::type_identity_t<std::span<const T>> bias,
                              std::span<std::uint32_t> experts, std::span<T> weights)
{
    static_assert(valid(C), "snicholls::ml::moe_route_rows: the transformer_config is not valid");
    constexpr std::size_t E = C.experts, K = C.top_k;
    const std::size_t rows = logits.size() / E;
    assert(logits.size() == rows * E && experts.size() == rows * K && weights.size() == rows * K && (bias.empty() || bias.size() == E));
    constexpr T scale = route_scale_v<C, T>;
    std::array<T, E> s{}, key{};
    for (std::size_t t = 0; t < rows; ++t) {
        const auto z = logits.subspan(t * E, E);
        const auto ids = experts.subspan(t * K, K);
        const auto w = weights.subspan(t * K, K);
        detail::moe::affinities<C, T>(z, s);
        for (std::size_t e = 0; e < E; ++e)
            key[e] = bias.empty() ? s[e] : s[e] + bias[e];
        const auto best = top_k<K>(key);
        for (std::size_t i = 0; i < K; ++i)
            ids[i] = static_cast<std::uint32_t>(best.index[i]);
        switch (C.scoring) {
        case moe_scoring::softmax_top_k: {
            const auto q = detail::moe::top_softmax<C, T>(z, ids);
            for (std::size_t i = 0; i < K; ++i)
                w[i] = q[i] * scale;
            break;
        }
        case moe_scoring::softmax:
            for (std::size_t i = 0; i < K; ++i)
                w[i] = s[ids[i]] * scale;
            break;
        case moe_scoring::sigmoid: {
            T sum = T(0);
            for (std::size_t i = 0; i < K; ++i)
                sum += s[ids[i]];
            for (std::size_t i = 0; i < K; ++i)
                w[i] = s[ids[i]] / sum * scale;
            break;
        }
        }
    }
}

/// \brief The backward pass of `moe_route_rows`: the gradient in the logits from the gradient in the weights.
///
/// The choice itself has no gradient; the weights' does, recomputed from the logits. `softmax_top_k`: the chosen
/// experts' logits get `q_i (dq_i - sum_k q_k dq_k)`, `q` the softmax over the chosen, `dq = scale · dw`, the others 0
/// (`moe_gate_backward`'s, to the bit, at scale 1). `softmax`: `softmax_backward` of the whole row with
/// `dp_(e_i) = scale · dw_i`. `sigmoid`: the chosen get `(dq_i - sum_k q_k dq_k) / S · sigmoid'(z_(e_i))`, the others 0.
/// \tparam C A mixture-of-experts configuration.
/// \param logits The router logits the forward pass read, `rows · E`.
/// \param experts The experts it chose, `rows · K`.
/// \param dweights The gradient in its weights, `rows · K`.
/// \param dlogits Receives the gradient in the logits, `rows · E`: written.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr void moe_route_rows_backward(const B&, std::type_identity_t<std::span<const T>> logits, std::span<const std::uint32_t> experts,
                                       std::type_identity_t<std::span<const T>> dweights, std::span<T> dlogits)
{
    static_assert(valid(C), "snicholls::ml::moe_route_rows_backward: the transformer_config is not valid");
    constexpr std::size_t E = C.experts, K = C.top_k;
    const std::size_t rows = logits.size() / E;
    assert(logits.size() == rows * E && experts.size() == rows * K && dweights.size() == rows * K && dlogits.size() == rows * E);
    constexpr T scale = route_scale_v<C, T>;
    for (std::size_t t = 0; t < rows; ++t) {
        const auto z = logits.subspan(t * E, E);
        const auto ids = experts.subspan(t * K, K);
        const auto dw = dweights.subspan(t * K, K);
        const auto g = dlogits.subspan(t * E, E);
        switch (C.scoring) {
        case moe_scoring::softmax_top_k: {
            const auto q = detail::moe::top_softmax<C, T>(z, ids);
            T dot = T(0);
            for (std::size_t i = 0; i < K; ++i)
                dot += q[i] * (scale * dw[i]);
            for (T& v : g)
                v = T(0);
            for (std::size_t i = 0; i < K; ++i)
                g[ids[i]] = q[i] * (scale * dw[i] - dot);
            break;
        }
        case moe_scoring::softmax: {
            std::array<T, E> p{}, dp{};
            softmax(z, p);
            for (std::size_t i = 0; i < K; ++i)
                dp[ids[i]] = scale * dw[i];
            softmax_backward(p, dp, g);
            break;
        }
        case moe_scoring::sigmoid: {
            std::array<T, K> s{};
            T sum = T(0);
            for (std::size_t i = 0; i < K; ++i) {
                s[i] = sigmoid(z[ids[i]]);
                sum += s[i];
            }
            T dot = T(0);
            for (std::size_t i = 0; i < K; ++i)
                dot += s[i] / sum * (scale * dw[i]);
            for (T& v : g)
                v = T(0);
            for (std::size_t i = 0; i < K; ++i)
                g[ids[i]] = (scale * dw[i] - dot) / sum * sigmoid_derivative(z[ids[i]]);
            break;
        }
        }
    }
}

/// @}

/// \name 2. Counting and permutation
/// @{

/// \brief The rows routed to each expert: `counts[e]` is the number of the `rows · K` choices equal to `e`.
/// \tparam C A mixture-of-experts configuration.
/// \param experts The chosen experts, `rows · K`, each below `E`.
/// \param counts Receives the histogram, `E`.
template <transformer_config C, typename B>
    requires(C.experts > 0)
constexpr void moe_count_rows(const B&, std::span<const std::uint32_t> experts, std::span<std::uint32_t> counts)
{
    assert(counts.size() == C.experts && experts.size() % C.top_k == 0);
    for (auto& c : counts)
        c = 0;
    for (const std::uint32_t e : experts) {
        assert(e < C.experts);
        ++counts[e];
    }
}

/// \brief The exclusive prefix sum of the counts: expert `e`'s rows are `[offsets[e], offsets[e + 1])` of the permuted
/// rows, and `offsets[E]` is `rows · K`.
/// \tparam C A mixture-of-experts configuration.
/// \param counts The histogram, `E`.
/// \param offsets Receives `E + 1` offsets.
template <transformer_config C, typename B>
    requires(C.experts > 0)
constexpr void moe_offsets(const B&, std::span<const std::uint32_t> counts, std::span<std::uint32_t> offsets)
{
    assert(counts.size() == C.experts && offsets.size() == C.experts + 1);
    offsets[0] = 0;
    for (std::size_t e = 0; e < C.experts; ++e)
        offsets[e + 1] = offsets[e] + counts[e];
}

/// \brief The stable permutation of the `(row, slot)` pairs into expert order, and its inverse.
///
/// Pair `j = row · K + slot` goes to position `p = inverse[j]`, and `order[p] = j`; the positions of expert `e` are
/// `[offsets[e], offsets[e + 1])`, its pairs in increasing `j`, so the whole order is by `(expert, row, slot)`. That
/// order is unique: any correct sort gives these two arrays.
/// \tparam C A mixture-of-experts configuration.
/// \param experts The chosen experts, `rows · K`.
/// \param offsets Their offsets (`moe_offsets`), `E + 1`.
/// \param order Receives the pair at each position, `rows · K`.
/// \param inverse Receives the position of each pair, `rows · K`.
template <transformer_config C, typename B>
    requires(C.experts > 0)
constexpr void moe_permutation(const B&, std::span<const std::uint32_t> experts, std::span<const std::uint32_t> offsets,
                               std::span<std::uint32_t> order, std::span<std::uint32_t> inverse)
{
    assert(offsets.size() == C.experts + 1 && offsets[C.experts] == experts.size() && order.size() == experts.size()
           && inverse.size() == experts.size());
    std::array<std::uint32_t, C.experts> next{};
    for (std::size_t e = 0; e < C.experts; ++e)
        next[e] = offsets[e];
    for (std::size_t j = 0; j < experts.size(); ++j) {
        const std::uint32_t p = next[experts[j]]++;
        order[p] = static_cast<std::uint32_t>(j);
        inverse[j] = p;
    }
}

/// @}

/// \name 3. Gather and combine
/// @{

/// \brief The token rows in expert order: `xp[p] = xs[order[p] / K]`.
/// \tparam C A mixture-of-experts configuration.
/// \param xs `rows` rows of `d_model`.
/// \param order The permutation's order, `rows · K`.
/// \param xp Receives `rows · K` rows of `d_model`.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr void moe_gather_rows(const B&, std::type_identity_t<std::span<const T>> xs, std::span<const std::uint32_t> order, std::span<T> xp)
{
    constexpr std::size_t d = C.d_model, K = C.top_k;
    assert(xs.size() * K == order.size() * d && xp.size() == order.size() * d);
    for (std::size_t p = 0; p < order.size(); ++p) {
        const std::size_t r = order[p] / K;
        for (std::size_t i = 0; i < d; ++i)
            xp[p * d + i] = xs[r * d + i];
    }
}

/// \brief The backward pass of `moe_gather_rows`: each row's `K` contributions added into its gradient, in slot order,
/// `dxs_r = dxs_r + dxp[inverse[r · K + i]]` for `i = 0 .. K-1`.
/// \tparam C A mixture-of-experts configuration.
/// \param dxp The gradient in the permuted rows, `rows · K · d_model`.
/// \param inverse The permutation's inverse, `rows · K`.
/// \param dxs Added into, `rows · d_model`.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr void moe_gather_rows_backward(const B&, std::type_identity_t<std::span<const T>> dxp, std::span<const std::uint32_t> inverse,
                                        std::span<T> dxs)
{
    constexpr std::size_t d = C.d_model, K = C.top_k;
    const std::size_t rows = inverse.size() / K;
    assert(inverse.size() == rows * K && dxp.size() == rows * K * d && dxs.size() == rows * d);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t s = 0; s < K; ++s) {
            const std::size_t p = inverse[r * K + s];
            for (std::size_t i = 0; i < d; ++i)
                dxs[r * d + i] = dxs[r * d + i] + dxp[p * d + i];
        }
}

/// \brief The weighted combine of the experts' outputs back into token order, in slot order:
/// `ys_r = w_r0 · yp[inverse[r · K]]`, then `ys_r = ys_r + w_ri · yp[inverse[r · K + i]]`.
/// \tparam C A mixture-of-experts configuration.
/// \param yp The experts' outputs in permuted order, `rows · K · d_model`.
/// \param weights The routing weights, `rows · K`.
/// \param inverse The permutation's inverse, `rows · K`.
/// \param ys Receives `rows · d_model`: written.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr void moe_combine_rows(const B&, std::type_identity_t<std::span<const T>> yp, std::type_identity_t<std::span<const T>> weights,
                                std::span<const std::uint32_t> inverse, std::span<T> ys)
{
    constexpr std::size_t d = C.d_model, K = C.top_k;
    const std::size_t rows = inverse.size() / K;
    assert(inverse.size() == rows * K && weights.size() == rows * K && yp.size() == rows * K * d && ys.size() == rows * d);
    for (std::size_t r = 0; r < rows; ++r) {
        const auto y = ys.subspan(r * d, d);
        const std::size_t p0 = inverse[r * K];
        for (std::size_t i = 0; i < d; ++i)
            y[i] = weights[r * K] * yp[p0 * d + i];
        for (std::size_t s = 1; s < K; ++s) {
            const T w = weights[r * K + s];
            const std::size_t p = inverse[r * K + s];
            for (std::size_t i = 0; i < d; ++i) {
                const T v = w * yp[p * d + i];
                y[i] = y[i] + v;
            }
        }
    }
}

/// \brief The backward pass of `moe_combine_rows`: `dyp[inverse[r · K + i]] = w_ri · dys_r` and
/// `dweights[r · K + i] = sum_j yp[inverse[r · K + i]][j] · dys_r[j]`, from 0 in `j` order; both written.
/// \tparam C A mixture-of-experts configuration.
/// \param dys The gradient in the combined rows, `rows · d_model`.
/// \param yp The experts' outputs the forward pass combined, `rows · K · d_model`.
/// \param weights The routing weights, `rows · K`.
/// \param inverse The permutation's inverse, `rows · K`.
/// \param dyp Receives the gradient in `yp`, `rows · K · d_model`.
/// \param dweights Receives the gradient in the weights, `rows · K`.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr void moe_combine_rows_backward(const B&, std::type_identity_t<std::span<const T>> dys, std::type_identity_t<std::span<const T>> yp,
                                         std::type_identity_t<std::span<const T>> weights, std::span<const std::uint32_t> inverse,
                                         std::type_identity_t<std::span<T>> dyp, std::span<T> dweights)
{
    constexpr std::size_t d = C.d_model, K = C.top_k;
    const std::size_t rows = inverse.size() / K;
    assert(inverse.size() == rows * K && weights.size() == rows * K && yp.size() == rows * K * d && dys.size() == rows * d
           && dyp.size() == yp.size() && dweights.size() == rows * K);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t s = 0; s < K; ++s) {
            const T w = weights[r * K + s];
            const std::size_t p = inverse[r * K + s];
            T dot = T(0);
            for (std::size_t i = 0; i < d; ++i) {
                dyp[p * d + i] = w * dys[r * d + i];
                const T v = yp[p * d + i] * dys[r * d + i];
                dot = dot + v;
            }
            dweights[r * K + s] = dot;
        }
}

/// @}

/// \name 4. The experts' products
/// @{

/// \brief The grouped product: expert `e`'s segment of rows times its own matrix, `Y_e = X_e W_eᵀ + b_e`, for each
/// expert in order, each one `linear_sequence` (an empty segment is skipped).
/// \param backend A `linalg::blas_backend`.
/// \param w The experts' matrices, `experts` of `rows x cols`.
/// \param offsets The segments, `experts + 1`: expert `e`'s rows are `[offsets[e], offsets[e + 1])` (relative to
/// `offsets[0]`, which is usually 0).
/// \param xs The rows, `(offsets[experts] - offsets[0]) · cols`.
/// \param ys Receives `(offsets[experts] - offsets[0]) · rows`.
/// \param bias `experts · rows` biases, or empty.
template <typename B, math::math_number T>
    requires linalg::blas_backend<B, T>
constexpr void grouped_linear(const B& backend, const grouped_weights<T>& w, std::span<const std::uint32_t> offsets,
                              std::type_identity_t<std::span<const T>> xs, std::type_identity_t<std::span<T>> ys,
                              std::type_identity_t<std::span<const T>> bias = {})
{
    assert(offsets.size() == w.experts + 1 && w.data.size() == w.experts * w.rows * w.cols && (bias.empty() || bias.size() == w.experts * w.rows));
    [[maybe_unused]] const std::size_t n = offsets[w.experts] - offsets[0];
    assert(xs.size() == n * w.cols && ys.size() == n * w.rows);
    for (std::size_t e = 0; e < w.experts; ++e) {
        const std::size_t first = offsets[e] - offsets[0], count = offsets[e + 1] - offsets[e];
        if (count == 0)
            continue;
        linear_sequence(backend, w.expert(e), xs.subspan(first * w.cols, count * w.cols), ys.subspan(first * w.rows, count * w.rows),
                        bias.empty() ? bias : bias.subspan(e * w.rows, w.rows));
    }
}

/// \brief The backward pass of `grouped_linear`: per expert in order, `dX_e += dY_e W_e`, `dW_e += dY_eᵀ X_e`,
/// `db_e += sum dY_e` (`linear_sequence_backward`).
/// \param backend A `linalg::blas_backend`.
/// \param w The matrices the forward pass used.
/// \param offsets Its segments, `experts + 1`.
/// \param xs Its rows.
/// \param dys The gradients in its outputs.
/// \param dxs Added into, the size of `xs`; or empty to skip it.
/// \param dw Added into, `experts · rows · cols`.
/// \param db Added into, `experts · rows`; or empty.
template <typename B, math::math_number T>
    requires linalg::blas_backend<B, T>
constexpr void grouped_linear_backward(const B& backend, const grouped_weights<T>& w, std::span<const std::uint32_t> offsets,
                                       std::type_identity_t<std::span<const T>> xs, std::type_identity_t<std::span<const T>> dys,
                                       std::type_identity_t<std::span<T>> dxs, std::type_identity_t<std::span<T>> dw,
                                       std::type_identity_t<std::span<T>> db = {})
{
    assert(offsets.size() == w.experts + 1 && dw.size() == w.experts * w.rows * w.cols && (db.empty() || db.size() == w.experts * w.rows));
    [[maybe_unused]] const std::size_t n = offsets[w.experts] - offsets[0];
    assert(xs.size() == n * w.cols && dys.size() == n * w.rows && (dxs.empty() || dxs.size() == xs.size()));
    const std::size_t m = w.rows * w.cols;
    for (std::size_t e = 0; e < w.experts; ++e) {
        const std::size_t first = offsets[e] - offsets[0], count = offsets[e + 1] - offsets[e];
        if (count == 0)
            continue;
        linear_sequence_backward(backend, w.expert(e), xs.subspan(first * w.cols, count * w.cols), dys.subspan(first * w.rows, count * w.rows),
                                 dxs.empty() ? dxs : dxs.subspan(first * w.cols, count * w.cols), dw.subspan(e * m, m),
                                 db.empty() ? db : db.subspan(e * w.rows, w.rows));
    }
}

/// \brief The routed experts' MLPs over the permuted rows: `mlp_forward_sequence` with each product grouped, saving
/// what `moe_experts_backward` needs.
/// \tparam C A mixture-of-experts configuration (its `mlp` and `expert_d_ff`).
/// \param backend A `linalg::blas_backend`.
/// \param w The experts' weights: `up` and `gate` `E` of `expert_d_ff x d_model`, `down` `E` of `d_model x
/// expert_d_ff`, biases `E` times the dense sizes, or empty.
/// \param offsets The experts' segments, `E + 1`.
/// \param xp The permuted rows, `n · d_model`.
/// \param yp Receives the experts' outputs, `n · d_model`.
/// \param save Receives the pre-activations and the hidden rows, `n · expert_d_ff` each (the gate for SwiGLU only).
template <transformer_config C, typename B, math::math_number T>
    requires linalg::blas_backend<B, T> && (C.experts > 0)
constexpr void moe_experts_forward(const B& backend, const mlp_weights<grouped_weights<T>, T>& w, std::span<const std::uint32_t> offsets,
                                   std::type_identity_t<std::span<const T>> xp, std::type_identity_t<std::span<T>> yp,
                                   const mlp_activations<T>& save)
{
    static_assert(valid(C), "snicholls::ml::moe_experts_forward: the transformer_config is not valid");
    constexpr bool swiglu = C.mlp == mlp_kind::swiglu;
    [[maybe_unused]] const std::size_t n = xp.size() / C.d_model, m = n * C.expert_d_ff;
    assert(w.up.experts == C.experts && w.up.rows == C.expert_d_ff && w.up.cols == C.d_model && w.down.rows == C.d_model
           && w.down.cols == C.expert_d_ff && yp.size() == xp.size() && save.up.size() == m && save.hidden.size() == m
           && (!swiglu || save.gate.size() == m));
    grouped_linear(backend, w.up, offsets, xp, save.up, w.up_bias);
    if constexpr (swiglu)
        grouped_linear(backend, w.gate, offsets, xp, save.gate, w.gate_bias);
    mlp_activate_rows<C>(backend, save);
    grouped_linear(backend, w.down, offsets, std::span<const T>(save.hidden), yp, w.down_bias);
}

/// \brief The backward pass of `moe_experts_forward`: `mlp_backward_sequence` with each product grouped.
/// \tparam C A mixture-of-experts configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The experts' weights the forward pass used.
/// \param offsets Its segments.
/// \param xp Its permuted rows.
/// \param saved What it saved.
/// \param dyp The gradient in its outputs, `n · d_model`.
/// \param dxp Added into, `n · d_model`; or empty to skip it.
/// \param grads The experts' weight gradients (`E` times the dense sizes), added into; its `scratch` (`n · expert_d_ff`,
/// twice for SwiGLU) is written.
template <transformer_config C, typename B, math::math_number T>
    requires linalg::blas_backend<B, T> && (C.experts > 0)
constexpr void moe_experts_backward(const B& backend, const mlp_weights<grouped_weights<T>, T>& w, std::span<const std::uint32_t> offsets,
                                    std::type_identity_t<std::span<const T>> xp, const mlp_activations<T>& saved,
                                    std::type_identity_t<std::span<const T>> dyp, std::type_identity_t<std::span<T>> dxp,
                                    const mlp_gradients<T>& grads)
{
    static_assert(valid(C), "snicholls::ml::moe_experts_backward: the transformer_config is not valid");
    constexpr bool swiglu = C.mlp == mlp_kind::swiglu;
    const std::size_t n = xp.size() / C.d_model, m = n * C.expert_d_ff;
    assert(dyp.size() == xp.size() && saved.up.size() == m && saved.hidden.size() == m && (!swiglu || saved.gate.size() == m)
           && grads.scratch.size() >= (swiglu ? 2 : 1) * m);
    const auto dh = grads.scratch.first(m);
    const auto dgate = swiglu ? grads.scratch.subspan(m, m) : std::span<T>{};
    fill_rows(backend, dh, T(0));
    grouped_linear_backward(backend, w.down, offsets, std::span<const T>(saved.hidden), dyp, dh, grads.down, grads.down_bias);
    mlp_activate_rows_backward<C>(backend, saved, dh, dgate);
    if constexpr (swiglu)
        grouped_linear_backward(backend, w.gate, offsets, xp, std::span<const T>(dgate), dxp, grads.gate, grads.gate_bias);
    grouped_linear_backward(backend, w.up, offsets, xp, std::span<const T>(dh), dxp, grads.up, grads.up_bias);
}

/// \brief The experts' weights of a run of experts, as expert parallelism places them on one rank.
/// \param w Every expert's weights.
/// \param first The first expert.
/// \param count The experts.
/// \return The view of those experts' matrices and biases.
template <math::math_number T>
constexpr mlp_weights<grouped_weights<T>, T> moe_expert_range(const mlp_weights<grouped_weights<T>, T>& w, std::size_t first, std::size_t count)
{
    auto cut = [&](std::span<const T> b, std::size_t each) { return b.empty() ? b : b.subspan(first * each, count * each); };
    return {w.up.slice(first, count), w.gate.experts == 0 ? w.gate : w.gate.slice(first, count), w.down.slice(first, count),
            cut(w.up_bias, w.up.rows), cut(w.gate_bias, w.gate.rows), cut(w.down_bias, w.down.rows)};
}

/// \brief The experts' weight gradients of a run of experts: each tensor's slice, the scratch as it is.
/// \tparam C A mixture-of-experts configuration.
/// \param g Every expert's gradients.
/// \param first The first expert.
/// \param count The experts.
/// \return The view of those experts' gradients.
template <transformer_config C, math::math_number T>
    requires(C.experts > 0)
constexpr mlp_gradients<T> moe_expert_range(const mlp_gradients<T>& g, std::size_t first, std::size_t count)
{
    constexpr std::size_t d = C.d_model, f = C.expert_d_ff;
    auto cut = [&](std::span<T> b, std::size_t each) { return b.empty() ? b : b.subspan(first * each, count * each); };
    return {cut(g.up, f * d), cut(g.gate, f * d), cut(g.down, d * f), cut(g.up_bias, f), cut(g.gate_bias, f), cut(g.down_bias, d), g.scratch};
}

/// @}

/// \name 6. Balancing
/// @{

/// \brief DeepSeek-V3's auxiliary-loss-free balancing: `bias_e += gamma · sign(mean load - load_e)` with `gamma` the
/// configuration's `bias_update_rate`.
///
/// The sign is decided in integers, `sign(total - E · counts[e])`, so it is exact: an expert above the mean load is
/// made less likely to be chosen, one below more, one at it is left alone. The bias enters only the choice. Call it
/// once per step with the step's counts summed over every worker, so every worker keeps the same bias.
/// \tparam C A mixture-of-experts configuration.
/// \param counts The step's load per expert, `E`.
/// \param bias The selection bias, `E`: updated.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr void moe_update_bias(const B&, std::span<const std::uint32_t> counts, std::span<T> bias)
{
    assert(counts.size() == C.experts && bias.size() == C.experts);
    constexpr T gamma = bias_update_rate_v<C, T>;
    std::uint64_t total = 0;
    for (const auto c : counts)
        total += c;
    for (std::size_t e = 0; e < C.experts; ++e) {
        const std::uint64_t scaled = std::uint64_t(C.experts) * counts[e];
        if (scaled < total)
            bias[e] = bias[e] + gamma;
        else if (scaled > total)
            bias[e] = bias[e] - gamma;
    }
}

/// \brief The auxiliary balance loss over `groups` groups of rows, `alpha · E · sum_e f_e P_e` per group averaged, and
/// its gradient added into the logits' (the file's "The arithmetic, exactly").
///
/// `f_e` is the fraction of a group's `T · K` choices that went to `e` (no gradient), `P_e` the group's mean router
/// probability (a softmax over every expert, or for `sigmoid` the sigmoids normalised over every expert, as
/// DeepSeek-V3). One group is Switch's batch-wise loss; one group per sequence is DeepSeek-V3's sequence-wise loss. At
/// uniform routing the loss is `alpha`. `alpha` is the configuration's `balance_loss_weight`.
/// \tparam C A mixture-of-experts configuration.
/// \param logits The router logits, `rows · E`.
/// \param experts The chosen experts, `rows · K`.
/// \param dlogits The gradient in the logits, `rows · E`: added into; or empty for the loss alone.
/// \param groups The groups, dividing the rows evenly; at least 1.
/// \param denominator The loss is the groups' sum divided by this; 0 for `groups` (a worker holding part of a batch
/// passes the whole batch's number of groups, so the workers' losses and gradients add up to the batch's).
/// \return The loss.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr T moe_balance_loss(const B&, std::type_identity_t<std::span<const T>> logits, std::span<const std::uint32_t> experts,
                             std::span<T> dlogits, std::size_t groups = 1, std::size_t denominator = 0)
{
    constexpr std::size_t E = C.experts, K = C.top_k;
    const std::size_t rows = logits.size() / E;
    assert(groups > 0 && rows % groups == 0 && logits.size() == rows * E && experts.size() == rows * K
           && (dlogits.empty() || dlogits.size() == rows * E));
    const std::size_t n = rows / groups, den = denominator == 0 ? groups : denominator;
    if (n == 0)
        return T(0);
    const T ae = balance_loss_weight_v<C, T> * T(E);
    T loss = T(0);
    std::array<T, E> p{}, sum_p{}, f{}, g{};
    for (std::size_t s = 0; s < groups; ++s) {
        std::array<std::uint32_t, E> counts{};
        for (std::size_t j = s * n * K; j < (s + 1) * n * K; ++j)
            ++counts[experts[j]];
        sum_p.fill(T(0));
        for (std::size_t t = s * n; t < (s + 1) * n; ++t) {
            detail::moe::probabilities<C, T>(logits.subspan(t * E, E), p);
            for (std::size_t e = 0; e < E; ++e)
                sum_p[e] += p[e];
        }
        T dot = T(0);
        for (std::size_t e = 0; e < E; ++e) {
            f[e] = T(counts[e]) / T(n * K);
            dot += f[e] * (sum_p[e] / T(n));
        }
        loss += ae * dot;
        if (dlogits.empty())
            continue;
        for (std::size_t e = 0; e < E; ++e)
            g[e] = ae * f[e] / T(n * den);
        for (std::size_t t = s * n; t < (s + 1) * n; ++t) {
            const auto z = logits.subspan(t * E, E);
            detail::moe::probabilities<C, T>(z, p);
            detail::moe::probabilities_backward<C, T>(z, p, g, dlogits.subspan(t * E, E));
        }
    }
    return loss / T(den);
}

/// \brief The router z-loss of ST-MoE, `beta · (sum_t logsumexp(z_t)²) / D`, and its gradient added into the logits':
/// `((2 · beta) · lse_t / D) · softmax(z_t)`. `beta` is the configuration's `z_loss_weight`.
/// \tparam C A mixture-of-experts configuration.
/// \param logits The router logits, `rows · E`.
/// \param dlogits The gradient in the logits, `rows · E`: added into; or empty for the loss alone.
/// \param denominator `D`; 0 for `rows` (a worker holding part of a batch passes the batch's rows).
/// \return The loss.
template <transformer_config C, typename B, math::math_number T>
    requires(C.experts > 0)
constexpr T moe_z_loss(const B&, std::type_identity_t<std::span<const T>> logits, std::span<T> dlogits, std::size_t denominator = 0)
{
    constexpr std::size_t E = C.experts;
    const std::size_t rows = logits.size() / E;
    assert(logits.size() == rows * E && (dlogits.empty() || dlogits.size() == rows * E));
    const std::size_t den = denominator == 0 ? rows : denominator;
    if (rows == 0)
        return T(0);
    constexpr T beta = z_loss_weight_v<C, T>;
    T sum = T(0);
    std::array<T, E> p{};
    for (std::size_t t = 0; t < rows; ++t) {
        const auto z = logits.subspan(t * E, E);
        const T lse = logsumexp(z);
        sum += lse * lse;
        if (dlogits.empty())
            continue;
        softmax(z, p);
        const T c = T(2) * beta * lse / T(den);
        for (std::size_t e = 0; e < E; ++e)
            dlogits[t * E + e] += c * p[e];
    }
    return beta * sum / T(den);
}

/// \brief The load of each expert over any number of steps and workers: counts that add, and the statistics a
/// training run watches (the most loaded expert against the mean, the coefficient of variation).
/// \ingroup core
/// \tparam E The experts.
template <std::size_t E>
    requires(E > 0)
struct moe_load
{
    std::array<std::uint64_t, E> counts{};   ///< The choices of each expert seen.

    /// \brief Adds one step's (or one worker's) counts.
    /// \param step The counts, `E` (`moe_count_rows`).
    constexpr void add(std::span<const std::uint32_t> step) noexcept
    {
        for (std::size_t e = 0; e < E && e < step.size(); ++e)
            counts[e] += step[e];
    }

    /// \brief Merges another worker's load: the result is as if this one had seen both.
    /// \param other The other load.
    constexpr void merge(const moe_load& other) noexcept
    {
        for (std::size_t e = 0; e < E; ++e)
            counts[e] += other.counts[e];
    }

    /// \brief The choices seen. \return The sum of the counts.
    [[nodiscard]] constexpr std::uint64_t total() const noexcept
    {
        std::uint64_t t = 0;
        for (const auto c : counts)
            t += c;
        return t;
    }

    /// \brief The most loaded expert's count. \return The largest count.
    [[nodiscard]] constexpr std::uint64_t max() const noexcept
    {
        std::uint64_t m = 0;
        for (const auto c : counts)
            m = c > m ? c : m;
        return m;
    }

    /// \brief The least loaded expert's count. \return The smallest count.
    [[nodiscard]] constexpr std::uint64_t min() const noexcept
    {
        std::uint64_t m = counts[0];
        for (const auto c : counts)
            m = c < m ? c : m;
        return m;
    }

    /// \brief The mean load. \return `total / E`.
    [[nodiscard]] constexpr double mean() const noexcept { return double(total()) / double(E); }

    /// \brief The most loaded expert against the mean: 1 when balanced, `E` when every choice went to one expert.
    /// \return `max / mean`, or 0 before any choice.
    [[nodiscard]] constexpr double max_over_mean() const noexcept { return total() == 0 ? 0.0 : double(max()) / mean(); }

    /// \brief The coefficient of variation of the counts: their population standard deviation over their mean.
    /// \return 0 when balanced; 0 before any choice.
    [[nodiscard]] constexpr double coefficient_of_variation() const noexcept
    {
        if (total() == 0)
            return 0.0;
        const double m = mean();
        double s = 0;
        for (const auto c : counts) {
            const double d = double(c) - m;
            s += d * d;
        }
        return math::sqrt(s / double(E)) / m;
    }
};

/// @}

/// \name 7. The layer
/// @{

/// \brief The weights of a mixture-of-experts MLP: views the caller fills.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct moe_weights
{
    dense_weights<T> router{};                     ///< `E x d_model`, no bias.
    std::span<const T> selection_bias{};           ///< `E`, or empty: DeepSeek-V3's bias for the choice only (`moe_update_bias`).
    mlp_weights<grouped_weights<T>, T> experts{};  ///< The routed experts' MLPs, `E` of `expert_d_ff`.
    mlp_weights<dense_weights<T>, T> shared{};     ///< The shared experts as one MLP of `shared_experts · expert_d_ff`; empty when none.
};

/// \brief What `moe_forward` saves for `moe_backward`: views the caller owns (`moe_activations_in` lays them out).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct moe_activations
{
    std::span<T> logits{};                   ///< `rows · E`: the router logits.
    std::span<std::uint32_t> experts{};      ///< `rows · K`: the chosen experts.
    std::span<T> weights{};                  ///< `rows · K`: their weights.
    std::span<std::uint32_t> counts{};       ///< `E`: the load per expert.
    std::span<std::uint32_t> offsets{};      ///< `E + 1`: the segments.
    std::span<std::uint32_t> order{};        ///< `rows · K`: the permutation.
    std::span<std::uint32_t> inverse{};      ///< `rows · K`: its inverse.
    std::span<T> xp{};                       ///< `rows · K · d_model`: the permuted rows.
    mlp_activations<T> expert{};             ///< `rows · K · expert_d_ff` each: the experts' pre-activations and hidden rows.
    std::span<T> yp{};                       ///< `rows · K · d_model`: the experts' outputs.
    mlp_activations<T> shared{};             ///< `rows · shared_experts · expert_d_ff` each; empty when none.
    std::span<T> shared_y{};                 ///< `rows · d_model`: the shared experts' output; empty when none.
};

/// \brief The gradients of a mixture-of-experts MLP and its working space: views the caller owns (`moe_gradients_in`).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct moe_gradients
{
    std::span<T> router{};        ///< `E · d_model`: added into.
    mlp_gradients<T> experts{};   ///< The routed experts', `E` times the dense sizes, added into; `scratch` `rows · K · expert_d_ff` (twice for SwiGLU).
    mlp_gradients<T> shared{};    ///< The shared experts', added into; `scratch` `rows · shared_experts · expert_d_ff` (twice for SwiGLU); empty when none.
    std::span<T> dlogits{};       ///< `rows · E`: working space.
    std::span<T> dweights{};      ///< `rows · K`: working space.
    std::span<T> dyp{};           ///< `rows · K · d_model`: working space.
    std::span<T> dxp{};           ///< `rows · K · d_model`: working space.
};

/// \brief The auxiliary losses `moe_forward` adds to the training loss; 0 where the configuration's weight is 0.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct moe_losses
{
    T balance = T(0);   ///< `moe_balance_loss`, one group per sequence.
    T z = T(0);         ///< `moe_z_loss`.
};

namespace detail::moe
{
    // The sizes of one dense MLP's tensors of width f, in mlp_weights order: up, gate, down, up_bias, gate_bias, down_bias
    constexpr std::array<std::size_t, 6> mlp_sizes(const transformer_config& c, std::size_t f) noexcept
    {
        const bool g = c.mlp == mlp_kind::swiglu;
        const std::size_t d = c.d_model, b = c.bias ? 1 : 0;
        return {f * d, g ? f * d : 0, d * f, b * f, g ? b * f : 0, b * d};
    }
} // namespace detail::moe

/// \brief The parameters of a mixture-of-experts MLP, in `moe_weights_in`'s order: the router, every routed expert's
/// tensors, the shared MLP's (the selection bias is state, not a parameter, and not counted).
/// \param c A mixture-of-experts configuration.
/// \return The number of values.
[[nodiscard]] constexpr std::size_t moe_parameter_count(const transformer_config& c) noexcept
{
    std::size_t n = c.experts * c.d_model;
    for (const std::size_t s : detail::moe::mlp_sizes(c, c.expert_d_ff))
        n += c.experts * s;
    for (const std::size_t s : detail::moe::mlp_sizes(c, c.shared_experts * c.expert_d_ff))
        n += s;
    return n;
}

/// \brief The numbers `moe_forward` saves for `rows` rows (`moe_activations_in`'s `values`).
/// \param c A mixture-of-experts configuration.
/// \param rows The rows.
/// \return The number of values.
[[nodiscard]] constexpr std::size_t moe_activation_values(const transformer_config& c, std::size_t rows) noexcept
{
    const std::size_t n = rows * c.top_k, g = c.mlp == mlp_kind::swiglu ? 3 : 2, fs = c.shared_experts * c.expert_d_ff;
    return rows * c.experts + n + 2 * n * c.d_model + g * n * c.expert_d_ff + g * rows * fs + (fs > 0 ? rows * c.d_model : 0);
}

/// \brief The indices `moe_forward` saves for `rows` rows (`moe_activations_in`'s `indices`).
/// \param c A mixture-of-experts configuration.
/// \param rows The rows.
/// \return The number of `std::uint32_t`s.
[[nodiscard]] constexpr std::size_t moe_activation_indices(const transformer_config& c, std::size_t rows) noexcept
{
    return 3 * rows * c.top_k + 2 * c.experts + 1;
}

/// \brief The working space `moe_backward` needs for `rows` rows (`moe_gradients_in`'s `work`).
/// \param c A mixture-of-experts configuration.
/// \param rows The rows.
/// \return The number of values.
[[nodiscard]] constexpr std::size_t moe_gradient_work(const transformer_config& c, std::size_t rows) noexcept
{
    const std::size_t n = rows * c.top_k, g = c.mlp == mlp_kind::swiglu ? 2 : 1;
    return g * n * c.expert_d_ff + g * rows * c.shared_experts * c.expert_d_ff + rows * c.experts + n + 2 * n * c.d_model;
}

namespace detail::moe
{
    // Cuts consecutive pieces off a span
    template <typename T>
    struct cutter
    {
        std::span<T> rest;
        constexpr std::span<T> take(std::size_t n)
        {
            const auto s = rest.first(n);
            rest = rest.subspan(n);
            return s;
        }
    };
} // namespace detail::moe

/// \brief The weights of a mixture-of-experts MLP laid out in one buffer: the router (`E x d_model`), then for `up`,
/// `gate`, `down`, `up_bias`, `gate_bias`, `down_bias` in turn every expert's tensor, then the shared MLP's six.
/// \tparam C A mixture-of-experts configuration.
/// \param params `moe_parameter_count(C)` values.
/// \param selection_bias The selection bias, `E`, or empty.
/// \return The views.
template <transformer_config C, math::math_number T>
    requires(C.experts > 0)
constexpr moe_weights<T> moe_weights_in(std::span<const T> params, std::span<const T> selection_bias = {})
{
    assert(params.size() == moe_parameter_count(C));
    constexpr std::size_t d = C.d_model, f = C.expert_d_ff, fs = C.shared_experts * f, E = C.experts;
    constexpr auto es = detail::moe::mlp_sizes(C, f), ss = detail::moe::mlp_sizes(C, fs);
    detail::moe::cutter<const T> c{params};
    moe_weights<T> w{};
    w.router = {c.take(E * d), E, d};
    w.selection_bias = selection_bias;
    w.experts.up = {c.take(E * es[0]), E, f, d};
    w.experts.gate = {c.take(E * es[1]), es[1] == 0 ? 0 : E, es[1] == 0 ? 0 : f, es[1] == 0 ? 0 : d};
    w.experts.down = {c.take(E * es[2]), E, d, f};
    w.experts.up_bias = c.take(E * es[3]);
    w.experts.gate_bias = c.take(E * es[4]);
    w.experts.down_bias = c.take(E * es[5]);
    w.shared.up = {c.take(ss[0]), fs, d};
    w.shared.gate = {c.take(ss[1]), ss[1] == 0 ? 0 : fs, ss[1] == 0 ? 0 : d};
    w.shared.down = {c.take(ss[2]), d, fs};
    w.shared.up_bias = c.take(ss[3]);
    w.shared.gate_bias = c.take(ss[4]);
    w.shared.down_bias = c.take(ss[5]);
    return w;
}

/// \brief What `moe_forward` saves, laid out in two buffers.
/// \tparam C A mixture-of-experts configuration.
/// \param values `moe_activation_values(C, rows)` values.
/// \param indices `moe_activation_indices(C, rows)` indices.
/// \param rows The rows.
/// \return The views.
template <transformer_config C, math::math_number T>
    requires(C.experts > 0)
constexpr moe_activations<T> moe_activations_in(std::span<T> values, std::span<std::uint32_t> indices, std::size_t rows)
{
    assert(values.size() == moe_activation_values(C, rows) && indices.size() == moe_activation_indices(C, rows));
    constexpr bool swiglu = C.mlp == mlp_kind::swiglu;
    constexpr std::size_t d = C.d_model, f = C.expert_d_ff, fs = C.shared_experts * f, E = C.experts;
    const std::size_t n = rows * C.top_k;
    detail::moe::cutter<T> v{values};
    detail::moe::cutter<std::uint32_t> i{indices};
    moe_activations<T> a{};
    a.logits = v.take(rows * E);
    a.experts = i.take(n);
    a.weights = v.take(n);
    a.counts = i.take(E);
    a.offsets = i.take(E + 1);
    a.order = i.take(n);
    a.inverse = i.take(n);
    a.xp = v.take(n * d);
    a.expert = {v.take(n * f), v.take(swiglu ? n * f : 0), v.take(n * f)};
    a.yp = v.take(n * d);
    a.shared = {v.take(rows * fs), v.take(swiglu ? rows * fs : 0), v.take(rows * fs)};
    a.shared_y = v.take(fs > 0 ? rows * d : 0);
    return a;
}

/// \brief The gradients of a mixture-of-experts MLP in `moe_weights_in`'s layout, and `moe_backward`'s working space.
/// \tparam C A mixture-of-experts configuration.
/// \param grads `moe_parameter_count(C)` values, added into.
/// \param work `moe_gradient_work(C, rows)` values.
/// \param rows The rows.
/// \return The views.
template <transformer_config C, math::math_number T>
    requires(C.experts > 0)
constexpr moe_gradients<T> moe_gradients_in(std::span<T> grads, std::span<T> work, std::size_t rows)
{
    assert(grads.size() == moe_parameter_count(C) && work.size() == moe_gradient_work(C, rows));
    constexpr std::size_t d = C.d_model, f = C.expert_d_ff, fs = C.shared_experts * f, E = C.experts, g = C.mlp == mlp_kind::swiglu ? 2 : 1;
    constexpr auto es = detail::moe::mlp_sizes(C, f), ss = detail::moe::mlp_sizes(C, fs);
    const std::size_t n = rows * C.top_k;
    detail::moe::cutter<T> c{grads}, w{work};
    moe_gradients<T> out{};
    out.router = c.take(E * d);
    out.experts = {c.take(E * es[0]), c.take(E * es[1]), c.take(E * es[2]), c.take(E * es[3]), c.take(E * es[4]), c.take(E * es[5]), w.take(g * n * f)};
    out.shared = {c.take(ss[0]), c.take(ss[1]), c.take(ss[2]), c.take(ss[3]), c.take(ss[4]), c.take(ss[5]), w.take(g * rows * fs)};
    out.dlogits = w.take(rows * E);
    out.dweights = w.take(n);
    out.dyp = w.take(n * d);
    out.dxp = w.take(n * d);
    return out;
}

/// \brief A mixture-of-experts MLP over a batch of rows, saving what `moe_backward` needs.
///
/// In order: the router logits (`linear_sequence`), `moe_route_rows`, `moe_count_rows`, `moe_offsets`,
/// `moe_permutation`, `moe_gather_rows`, `moe_experts_forward`, `moe_combine_rows` into `ys`; then, with shared
/// experts, their MLP (`mlp_forward_sequence` with `moe_shared_config(C)`) into `shared_y` and `ys = shared_y + ys`
/// (`add_rows`); then the auxiliary losses the configuration weights.
/// \tparam C A mixture-of-experts configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights.
/// \param xs `rows` inputs of `d_model`.
/// \param ys Receives `rows` outputs of `d_model`; may not alias `xs`.
/// \param act Receives the activations (`moe_activations_in`).
/// \param sequences The sequences in the batch, dividing the rows: the balance loss's groups.
/// \return The auxiliary losses: `balance` if `balance_loss_weight > 0`, `z` if `z_loss_weight > 0`, else 0.
template <transformer_config C, typename B, math::math_number T>
    requires linalg::blas_backend<B, T> && (C.experts > 0)
constexpr moe_losses<T> moe_forward(const B& backend, const moe_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                                    std::type_identity_t<std::span<T>> ys, const moe_activations<T>& act, std::size_t sequences = 1)
{
    static_assert(valid(C), "snicholls::ml::moe_forward: the transformer_config is not valid");
    [[maybe_unused]] const std::size_t rows = xs.size() / C.d_model;
    assert(xs.size() == rows * C.d_model && ys.size() == xs.size() && act.logits.size() == rows * C.experts);
    linear_sequence(backend, w.router, xs, act.logits);
    moe_route_rows<C>(backend, std::span<const T>(act.logits), w.selection_bias, act.experts, act.weights);
    moe_count_rows<C>(backend, act.experts, act.counts);
    moe_offsets<C>(backend, act.counts, act.offsets);
    moe_permutation<C>(backend, act.experts, act.offsets, act.order, act.inverse);
    moe_gather_rows<C>(backend, xs, act.order, act.xp);
    moe_experts_forward<C>(backend, w.experts, act.offsets, std::span<const T>(act.xp), act.yp, act.expert);
    moe_combine_rows<C>(backend, std::span<const T>(act.yp), std::span<const T>(act.weights), act.inverse, ys);
    if constexpr (C.shared_experts > 0) {
        mlp_forward_sequence<moe_shared_config(C)>(backend, w.shared, xs, act.shared_y, act.shared);
        add_rows(backend, std::span<const T>(act.shared_y), ys);
    }
    moe_losses<T> losses{};
    if constexpr (C.balance_loss_weight > 0)
        losses.balance = moe_balance_loss<C>(backend, std::span<const T>(act.logits), act.experts, std::span<T>{}, sequences);
    if constexpr (C.z_loss_weight > 0)
        losses.z = moe_z_loss<C>(backend, std::span<const T>(act.logits), std::span<T>{});
    (void)sequences;
    return losses;
}

/// \brief The backward pass of `moe_forward`: the gradient in its output added into its input's, every weight's
/// gradient added into, the auxiliary losses' gradients included.
///
/// In order: `moe_combine_rows_backward` (`dyp`, `dweights`), `moe_experts_backward` into a zeroed `dxp`,
/// `moe_gather_rows_backward` (`dxs += ` the routed contributions, slot by slot), `moe_route_rows_backward` (`dlogits`),
/// the balance loss's and the z-loss's gradients added into `dlogits`, the router's `linear_sequence_backward`
/// (`dxs += dlogits W_router`), then the shared MLP's `mlp_backward_sequence` (`dxs +=` its input gradient).
/// \tparam C A mixture-of-experts configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param xs Its inputs.
/// \param act What it saved.
/// \param dys The gradient in its outputs, `rows · d_model`.
/// \param dxs Added into, `rows · d_model`.
/// \param grads Receives the gradients; its working spaces are written.
/// \param sequences The sequences the forward pass took.
template <transformer_config C, typename B, math::math_number T>
    requires linalg::blas_backend<B, T> && (C.experts > 0)
constexpr void moe_backward(const B& backend, const moe_weights<T>& w, std::type_identity_t<std::span<const T>> xs, const moe_activations<T>& act,
                            std::type_identity_t<std::span<const T>> dys, std::type_identity_t<std::span<T>> dxs, const moe_gradients<T>& grads,
                            std::size_t sequences = 1)
{
    static_assert(valid(C), "snicholls::ml::moe_backward: the transformer_config is not valid");
    [[maybe_unused]] const std::size_t rows = xs.size() / C.d_model;
    assert(dys.size() == xs.size() && dxs.size() == xs.size() && grads.dlogits.size() == rows * C.experts);
    moe_combine_rows_backward<C>(backend, dys, std::span<const T>(act.yp), std::span<const T>(act.weights), act.inverse, grads.dyp,
                                 grads.dweights);
    fill_rows(backend, grads.dxp, T(0));
    moe_experts_backward<C>(backend, w.experts, act.offsets, std::span<const T>(act.xp), act.expert, std::span<const T>(grads.dyp), grads.dxp,
                            grads.experts);
    moe_gather_rows_backward<C>(backend, std::span<const T>(grads.dxp), act.inverse, dxs);
    moe_route_rows_backward<C>(backend, std::span<const T>(act.logits), act.experts, std::span<const T>(grads.dweights), grads.dlogits);
    if constexpr (C.balance_loss_weight > 0)
        moe_balance_loss<C>(backend, std::span<const T>(act.logits), act.experts, grads.dlogits, sequences);
    if constexpr (C.z_loss_weight > 0)
        moe_z_loss<C>(backend, std::span<const T>(act.logits), grads.dlogits);
    linear_sequence_backward(backend, w.router, xs, std::span<const T>(grads.dlogits), dxs, grads.router);
    if constexpr (C.shared_experts > 0)
        mlp_backward_sequence<moe_shared_config(C)>(backend, w.shared, xs, act.shared, dys, dxs, grads.shared);
    (void)sequences;
}

/// @}

} // namespace snicholls::ml

#endif // snicholls_core_ml_moe_hpp
