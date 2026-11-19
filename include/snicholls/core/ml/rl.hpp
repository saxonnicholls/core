//
//  rl.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` reinforcement learning for language models: GRPO's group-normalised advantages, the clipped importance-ratio objective with the k3 KL penalty and an entropy bonus, the normalisations of GRPO, DAPO and Dr. GRPO, and PPO's GAE and value loss.
/// \ingroup core
///
/// **Overview.** Reinforcement learning from rewards samples responses from the policy, scores them, and raises the
/// probability of the better ones (TRAINING.md T11). A rollout is plain data, `rollout_view`: `prompts · group_size`
/// sequences padded to one `length`, laid out group by group (the `G` samples of one prompt are consecutive rows),
/// their targets, the response mask (1 on response tokens, 0 on the prompt and padding) and one reward a sequence.
/// The arithmetic, in the order of a step:
///
/// 1. **Advantages.** GRPO (Shao et al. 2024, DeepSeekMath, §4.1.2; DeepSeek-R1) uses no value model: each sample's
///    advantage is its reward normalised within its group, `Â_i = (r_i - mean(r)) / std(r)`, the same for every token
///    of the response (`group_advantages` with `advantage_scale::group_std`, then `expand_rows`). Dr. GRPO (Liu et al.
///    2025, "Understanding R1-Zero-Like Training", appendix A eq. 6) drops the division by the std, which weights
///    questions of low reward variance more (`advantage_scale::none`). PPO estimates per-token advantages with a value
///    model and GAE (`gae_rows`).
/// 2. **Ratios.** `r_t = π_θ(o_t) / π_old(o_t) = exp(logp_t - old_logp_t)`, from the log-probabilities of the policy
///    being trained and of the policy that sampled (`ml/logprobs.hpp`); 1 on the first step after sampling.
/// 3. **The objective of a token**, to be minimised (`policy_token_loss`): `-min(r_t Â_t, clip(r_t, 1 - ε_low,
///    1 + ε_high) Â_t) + β D_KL,t - c_H H_t`, the clipped surrogate of PPO (Schulman et al. 2017) as GRPO's eq. 3
///    writes it, DAPO's asymmetric clip ("clip-higher", `ε_low = 0.2`, `ε_high = 0.28`; Yu et al. 2025, eq. 10), the
///    KL to the reference by the k3 estimator `π_ref/π_θ - log(π_ref/π_θ) - 1` (GRPO's eq. 4; GRPO's `β = 0.04`,
///    DAPO's 0), and an entropy bonus `c_H H_t` (the entropy of the token's distribution, from `token_logprobs_rows`).
/// 4. **Aggregation** over a batch, the choice the variants disagree on (`token_aggregation`):
///    - `sequence_mean`: GRPO's `(1/G) Σ_i (1/|o_i|) Σ_t`, each response's mean, then the mean over responses. Every
///      response weighs the same, so a token of a long response weighs less: Dr. GRPO shows this favours short
///      correct and long wrong answers.
///    - `token_mean`: DAPO's `1 / Σ_i |o_i| · Σ_i Σ_t` (eq. 12), every token of the batch the same weight.
///    - `constant`: Dr. GRPO's `Σ_t ... / MAX_TOKENS` per response, then the mean over responses: a constant
///      (`max_tokens`, by default the padded length) in place of each length, so neither bias appears.
///
///    `policy_loss_rows` computes the aggregated loss and the gradient in every token's log-probability (and in its
///    entropy), ready for `token_logprobs_rows_backward`.
/// 5. **PPO's value loss** (`value_loss_rows`): `½ max((V - R)², (clip(V, V_old ± ε_v) - R)²)` over response tokens,
///    with the returns `R = Â + V_old` from GAE.
///
/// \snippet ml/rl_demo.cpp grpo
///
/// **Relationships.** `ml/logprobs.hpp` (log-probabilities, entropies and their backward), `ml/reward_model.hpp` (the
/// value head), `ml/preference.hpp` (the offline alternative). Sampling the rollouts is `ml/sampling.hpp`'s; the
/// rollout loop, the reward functions and the weights' synchronisation belong to the programs above core. Group: \ref
/// core.
///
/// **Implementations.** One each, on the host; the backend-first functions are the seam for a device (TRAINING.md T9).
///
/// **Guarantees.** `constexpr`, no allocation, `noexcept` for the standard floating-point types; linear in the
/// tokens. Every gradient is written, not added. `policy_loss_metrics` merges across workers.
///
/// **Errors.** Preconditions, asserted: sizes agree (`rows · length` per-token spans, `rows` per-sequence spans); the
/// rows divide into groups of `group_size`. A group whose rewards are all equal has advantage 0 in every sample (no
/// signal: DAPO's dynamic sampling drops such groups before the step; core leaves that to the caller). A response of
/// no tokens contributes nothing.
///
/// **Definitions.** Checked against the papers: GRPO's objective, the k3 estimator and the outcome advantage
/// (arXiv 2402.03300 eqs. 3-4, §4.1.2); DAPO's token-level loss, clip-higher and no KL (2503.14476 eqs. 8-12, §2.3,
/// §4.1); Dr. GRPO's removals and `MAX_TOKENS` (2503.20783 §3.2, appendix A). Recalled, not re-read: the std of
/// `group_std` is the sample std (`n - 1`) plus `ε = 1e-4`, as TRL's GRPO trainer computes it (the paper does not say);
/// PPO's clipped surrogate, GAE (Schulman et al. 2016) and the clipped value loss of OpenAI's PPO2 baseline;
/// `masked_whiten`'s sample variance and `ε = 1e-8`, as TRL's.
///
/// **Gradient at the clip.** Where the unclipped and clipped terms are equal (inside the range, or exactly on its
/// edge) the derivative is the unclipped one, `-r_t Â_t`; strictly outside, on the side the clip binds, it is 0. The
/// same convention for the value loss.
///
/// **Performance.** `policy_loss_rows` (GRPO's settings, with the metrics) against the same loop by hand, after a
/// bit-for-bit check (`benchmarks/ml/rl_benchmark.cpp`), 64 sequences of 1024 tokens: 1.9 ms against 2.1 ms, level
/// (Xeon W-3245, -O2, Apple clang 17, 2026-11-19, one thread, CPU time, load average about 100: a record, not a quiet-machine figure). Negligible next to the passes that produce its inputs.
///
/// **Tests and demos.** `tests/ml/rl_test.cpp`, `demos/ml/rl_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`.
///
/// **Known issues.** Groups are consecutive rows of one size; a rollout with groups of different sizes is split by
/// the caller. Sequence-level importance ratios (GSPO) are not here.
///
/// **Status.** Experimental, new in Unreleased.

#ifndef snicholls_core_ml_rl_hpp
#define snicholls_core_ml_rl_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief A batch of rollouts as plain data: `prompts · group_size` sequences of `length`, group by group.
/// \ingroup core
/// \tparam T The number type of the rewards.
template <math::math_number T>
struct rollout_view
{
    std::size_t prompts = 0;                       ///< The prompts.
    std::size_t group_size = 0;                    ///< `G`, the samples of each prompt, consecutive rows.
    std::size_t length = 0;                        ///< The padded length of every row.
    std::span<const std::uint32_t> targets{};      ///< `rows · length`: the token after each position.
    std::span<const std::uint8_t> response_mask{}; ///< `rows · length`: 1 where the target is a response token.
    std::span<const T> rewards{};                  ///< `rows`: each sequence's reward.

    /// \brief The sequences. \return `prompts · group_size`.
    [[nodiscard]] constexpr std::size_t rows() const noexcept { return prompts * group_size; }
    /// \brief The prompt a row samples. \param row The row. \return Its group.
    [[nodiscard]] constexpr std::size_t group_of(std::size_t row) const noexcept { return row / group_size; }
};

/// \brief How a group's centred rewards are scaled.
/// \ingroup core
enum class advantage_scale : std::uint8_t
{
    group_std,   ///< GRPO: divided by the group's sample std plus `epsilon`.
    none,        ///< Dr. GRPO: centred only.
};

/// \brief How the per-token losses of a batch are averaged.
/// \ingroup core
enum class token_aggregation : std::uint8_t
{
    sequence_mean,   ///< GRPO: each response's mean over its tokens, then the mean over responses.
    token_mean,      ///< DAPO: the mean over every response token of the batch.
    constant,        ///< Dr. GRPO: each response's sum divided by a constant, then the mean over responses.
};

/// \brief GRPO's group-normalised advantages: `(r_i - mean) / (std + ε)` within each group of consecutive rows, or `r_i - mean`.
/// \param rewards `rows` rewards, group by group.
/// \param group_size `G`: `rows` is a multiple of it.
/// \param advantages Receives `rows` advantages; may be `rewards`.
/// \param scale `group_std` (GRPO) or `none` (Dr. GRPO).
/// \param epsilon Added to the std: 1e-4 as TRL.
template <math::math_number T, typename B>
constexpr void group_advantages(const B&, std::span<const T> rewards, std::size_t group_size, std::type_identity_t<std::span<T>> advantages,
                                advantage_scale scale = advantage_scale::group_std, double epsilon = 1e-4) noexcept(std::is_floating_point_v<T>)
{
    assert(group_size > 0 && rewards.size() % group_size == 0 && advantages.size() == rewards.size());
    const T g = T(group_size);
    for (std::size_t b = 0; b < rewards.size(); b += group_size) {
        T mean = T(0);
        for (std::size_t i = b; i < b + group_size; ++i)
            mean += rewards[i];
        mean = mean / g;
        T div = T(1);
        if (scale == advantage_scale::group_std) {
            T ss = T(0);
            for (std::size_t i = b; i < b + group_size; ++i)
                ss += (rewards[i] - mean) * (rewards[i] - mean);
            const T sd = group_size > 1 ? math::sqrt(ss / T(group_size - 1)) : T(0);
            div = sd + T(epsilon);
        }
        for (std::size_t i = b; i < b + group_size; ++i)
            advantages[i] = (rewards[i] - mean) / div;
    }
}

/// \brief Each row's value repeated over its `length` positions: a sequence's advantage on every token.
/// \param per_row `rows` values. \param per_token Receives `rows · length`.
template <math::math_number T, typename B>
constexpr void expand_rows(const B&, std::span<const T> per_row, std::type_identity_t<std::span<T>> per_token) noexcept(std::is_floating_point_v<T>)
{
    assert(per_row.empty() ? per_token.empty() : per_token.size() % per_row.size() == 0);
    if (per_row.empty())
        return;
    const std::size_t len = per_token.size() / per_row.size();
    for (std::size_t r = 0; r < per_row.size(); ++r)
        for (std::size_t t = 0; t < len; ++t)
            per_token[r * len + t] = per_row[r];
}

/// \brief One token's clipped surrogate and KL terms, unweighted.
/// \ingroup core
template <math::math_number T>
struct policy_token
{
    T loss = T(0);       ///< `-min(r Â, clip(r) Â) + β k3`.
    T d_logp = T(0);     ///< Its derivative in the policy's log-probability.
    T ratio = T(1);      ///< `r = exp(logp - old_logp)`.
    T kl = T(0);         ///< The k3 estimate of the KL to the reference (0 without one).
    bool clipped = false;   ///< Whether the clipped term was strictly the smaller (its gradient 0).
};

/// \brief The k3 estimator of `KL(π_θ || π_ref)` at one token: `e^u - u - 1`, `u = ref_logp - logp`, computed as `expm1(u) - u`.
/// \param logp The policy's log-probability. \param ref_logp The reference's.
/// \return The estimate, at least 0.
template <math::math_number T>
constexpr T kl_k3(T logp, std::type_identity_t<T> ref_logp) noexcept(std::is_floating_point_v<T>)
{
    const T u = ref_logp - logp;
    return math::expm1(u) - u;
}

/// \brief The derivative of `kl_k3` in the policy's log-probability: `1 - e^u = -expm1(u)`.
/// \param logp The policy's log-probability. \param ref_logp The reference's. \return The derivative.
template <math::math_number T>
constexpr T kl_k3_derivative(T logp, std::type_identity_t<T> ref_logp) noexcept(std::is_floating_point_v<T>)
{
    return -math::expm1(ref_logp - logp);
}

/// \brief The settings of the policy loss: PPO's symmetric clip by default, no KL, no entropy bonus, GRPO's aggregation.
/// \ingroup core
struct policy_loss_config
{
    double clip_low = 0.2;    ///< `ε_low`: the ratio is clipped below at `1 - ε_low`.
    double clip_high = 0.2;   ///< `ε_high`: above at `1 + ε_high` (DAPO: 0.28).
    double kl_beta = 0.0;     ///< `β` of the k3 KL to the reference (GRPO: 0.04; DAPO: 0).
    double entropy_coef = 0.0;   ///< `c_H`, the weight of the entropy bonus.
    token_aggregation aggregation = token_aggregation::sequence_mean;   ///< How the tokens are averaged.
    std::size_t max_tokens = 0;   ///< The constant of `token_aggregation::constant`; 0 for the padded length.
};

/// \brief One token's objective: `-min(r Â, clip(r, 1 - ε_low, 1 + ε_high) Â) + β k3(logp, ref_logp)` and its derivative.
/// \param logp The policy's log-probability. \param old_logp The sampling policy's. \param ref_logp The reference's
/// (ignored when `c.kl_beta` is 0). \param advantage `Â`. \param c The settings.
/// \return The terms; the entropy bonus is applied by `policy_loss_rows`.
template <math::math_number T>
constexpr policy_token<T> policy_token_loss(T logp, std::type_identity_t<T> old_logp, std::type_identity_t<T> ref_logp,
                                            std::type_identity_t<T> advantage, const policy_loss_config& c) noexcept(std::is_floating_point_v<T>)
{
    policy_token<T> p;
    p.ratio = math::exp(logp - old_logp);
    const T lo = T(1) - T(c.clip_low), hi = T(1) + T(c.clip_high);
    const T clipped_ratio = p.ratio < lo ? lo : (p.ratio > hi ? hi : p.ratio);
    const T unclipped = p.ratio * advantage, clipped = clipped_ratio * advantage;
    p.clipped = clipped < unclipped;
    p.loss = p.clipped ? -clipped : -unclipped;
    p.d_logp = p.clipped ? T(0) : -unclipped;   // d(r Â)/d logp = r Â
    if (c.kl_beta != 0.0) {
        const T beta = T(c.kl_beta);
        p.kl = kl_k3(logp, ref_logp);
        p.loss += beta * p.kl;
        p.d_logp += beta * kl_k3_derivative(logp, ref_logp);
    }
    return p;
}

/// \brief The metrics of policy optimisation over response tokens, merged across workers.
/// \ingroup core
template <math::math_number T>
struct policy_loss_metrics
{
    std::uint64_t tokens = 0;    ///< Response tokens seen.
    std::uint64_t clipped = 0;   ///< Of which the clip bound.
    T kl = T(0);                 ///< The sum of the k3 estimates.
    T ratio = T(0);              ///< The sum of the ratios.
    T entropy = T(0);            ///< The sum of the entropies (when given).

    /// \brief Merges another worker's metrics. \param o The other.
    constexpr void merge(const policy_loss_metrics& o) noexcept(std::is_floating_point_v<T>)
    {
        tokens += o.tokens;
        clipped += o.clipped;
        kl += o.kl;
        ratio += o.ratio;
        entropy += o.entropy;
    }
    /// \brief The fraction of tokens whose ratio was clipped. \return The fraction; 0 before any.
    [[nodiscard]] constexpr double clip_fraction() const noexcept { return tokens == 0 ? 0.0 : static_cast<double>(clipped) / static_cast<double>(tokens); }
    /// \brief The mean k3 KL per token. \return The mean.
    [[nodiscard]] constexpr T mean_kl() const noexcept(std::is_floating_point_v<T>) { return tokens == 0 ? T(0) : kl / T(tokens); }
    /// \brief The mean ratio. \return The mean.
    [[nodiscard]] constexpr T mean_ratio() const noexcept(std::is_floating_point_v<T>) { return tokens == 0 ? T(0) : ratio / T(tokens); }
    /// \brief The mean entropy. \return The mean.
    [[nodiscard]] constexpr T mean_entropy() const noexcept(std::is_floating_point_v<T>) { return tokens == 0 ? T(0) : entropy / T(tokens); }
};

/// \brief The weight each response token of a row carries in the aggregated loss: `1/(rows·|o|)`, `1/Σ|o|` or `1/(rows·C)`.
/// \param mask `rows · length`. \param rows The rows. \param c The settings. \param row The row.
/// \param total The response tokens of the whole batch. \return The weight; 0 for an empty response.
constexpr double token_weight(std::span<const std::uint8_t> mask, std::size_t rows, const policy_loss_config& c, std::size_t row,
                              std::size_t total) noexcept
{
    const std::size_t len = mask.size() / rows;
    std::size_t n = 0;
    for (std::size_t t = row * len; t < (row + 1) * len; ++t)
        n += mask[t] != 0 ? 1 : 0;
    if (n == 0)
        return 0.0;
    switch (c.aggregation) {
    case token_aggregation::sequence_mean: return 1.0 / (static_cast<double>(rows) * static_cast<double>(n));
    case token_aggregation::token_mean: return 1.0 / static_cast<double>(total);
    case token_aggregation::constant: return 1.0 / (static_cast<double>(rows) * static_cast<double>(c.max_tokens == 0 ? len : c.max_tokens));
    }
    return 0.0;
}

/// \brief The policy-gradient loss of a batch (GRPO, DAPO, Dr. GRPO, PPO's policy term), and its gradient in every
/// token's log-probability and entropy, on the backend.
/// \param mask `rows · length` response mask.
/// \param logps The policy's token log-probabilities, `rows · length`.
/// \param old_logps The sampling policy's, `rows · length`.
/// \param ref_logps The reference's, `rows · length`, or empty when `c.kl_beta` is 0.
/// \param advantages Per token, `rows · length` (`expand_rows` for one per sequence).
/// \param entropies Per token, `rows · length`, or empty when `c.entropy_coef` is 0.
/// \param dlogps Written, `rows · length`: 0 off the response.
/// \param dentropies Written, `rows · length`, or empty (with `entropies`).
/// \param rows The sequences.
/// \param c The settings.
/// \param metrics Added into.
/// \return The aggregated loss.
template <math::math_number T, typename B>
constexpr T policy_loss_rows(const B&, std::span<const std::uint8_t> mask, std::span<const T> logps, std::type_identity_t<std::span<const T>> old_logps,
                             std::type_identity_t<std::span<const T>> ref_logps, std::type_identity_t<std::span<const T>> advantages,
                             std::type_identity_t<std::span<const T>> entropies, std::type_identity_t<std::span<T>> dlogps,
                             std::type_identity_t<std::span<T>> dentropies, std::size_t rows, const policy_loss_config& c,
                             policy_loss_metrics<T>& metrics) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = mask.size();
    assert(logps.size() == n && old_logps.size() == n && advantages.size() == n && dlogps.size() == n
           && (c.kl_beta == 0.0 || ref_logps.size() == n) && (entropies.empty() || entropies.size() == n)
           && dentropies.size() == entropies.size() && (rows == 0 ? n == 0 : n % rows == 0));
    if (rows == 0)
        return T(0);
    const std::size_t len = n / rows;
    std::size_t total = 0;
    for (const auto m : mask)
        total += m != 0 ? 1 : 0;
    const T coef = T(c.entropy_coef);
    T loss = T(0);
    for (std::size_t r = 0; r < rows; ++r) {
        const T w = T(token_weight(mask, rows, c, r, total));
        for (std::size_t t = r * len; t < (r + 1) * len; ++t) {
            if (mask[t] == 0) {
                dlogps[t] = T(0);
                if (!dentropies.empty())
                    dentropies[t] = T(0);
                continue;
            }
            const auto p = policy_token_loss(logps[t], old_logps[t], ref_logps.empty() ? T(0) : ref_logps[t], advantages[t], c);
            T l = p.loss;
            ++metrics.tokens;
            metrics.clipped += p.clipped ? 1 : 0;
            metrics.kl += p.kl;
            metrics.ratio += p.ratio;
            if (!entropies.empty()) {
                l -= coef * entropies[t];
                metrics.entropy += entropies[t];
                dentropies[t] = -coef * w;
            }
            loss += w * l;
            dlogps[t] = w * p.d_logp;
        }
    }
    return loss;
}

/// \brief Generalised advantage estimation over each row's response tokens (Schulman et al. 2016): walking back,
/// `δ_t = r_t + γ V_next - V_t`, `Â_t = δ_t + γλ Â_next`, with `V_next = 0` and `Â_next = 0` after the last response
/// token; the returns `R_t = Â_t + V_t`. Positions off the response are skipped (the next response token is "next")
/// and get 0.
/// \param mask `rows · length`. \param rewards Per token, `rows · length` (typically the sequence's reward on its
/// last token). \param values The value model's, per token. \param gamma The discount. \param lambda GAE's `λ`.
/// \param advantages Written, per token. \param returns Written, per token.
/// \param rows The sequences.
template <math::math_number T, typename B>
constexpr void gae_rows(const B&, std::span<const std::uint8_t> mask, std::span<const T> rewards, std::type_identity_t<std::span<const T>> values,
                        double gamma, double lambda, std::type_identity_t<std::span<T>> advantages, std::type_identity_t<std::span<T>> returns,
                        std::size_t rows) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = mask.size();
    assert(rewards.size() == n && values.size() == n && advantages.size() == n && returns.size() == n && (rows == 0 ? n == 0 : n % rows == 0));
    if (rows == 0)
        return;
    const std::size_t len = n / rows;
    const T g = T(gamma), gl = T(gamma) * T(lambda);
    for (std::size_t r = 0; r < rows; ++r) {
        T next_value = T(0), next_adv = T(0);
        for (std::size_t k = len; k-- > 0;) {
            const std::size_t t = r * len + k;
            if (mask[t] == 0) {
                advantages[t] = T(0);
                returns[t] = T(0);
                continue;
            }
            const T delta = rewards[t] + g * next_value - values[t];
            next_adv = delta + gl * next_adv;
            next_value = values[t];
            advantages[t] = next_adv;
            returns[t] = next_adv + values[t];
        }
    }
}

/// \brief Normalises the values on the mask to mean 0 and standard deviation 1 (sample variance, `ε = 1e-8`); 0 off it.
/// \param mask The mask. \param values The values. \param out Written; may be `values`.
template <math::math_number T, typename B>
constexpr void masked_whiten(const B&, std::span<const std::uint8_t> mask, std::span<const T> values, std::type_identity_t<std::span<T>> out) noexcept(std::is_floating_point_v<T>)
{
    assert(values.size() == mask.size() && out.size() == mask.size());
    std::size_t n = 0;
    T mean = T(0);
    for (std::size_t i = 0; i < mask.size(); ++i)
        if (mask[i] != 0) {
            mean += values[i];
            ++n;
        }
    if (n == 0) {
        for (T& v : out)
            v = T(0);
        return;
    }
    mean = mean / T(n);
    T ss = T(0);
    for (std::size_t i = 0; i < mask.size(); ++i)
        if (mask[i] != 0)
            ss += (values[i] - mean) * (values[i] - mean);
    const T inv = T(1) / math::sqrt((n > 1 ? ss / T(n - 1) : T(0)) + T(1e-8));
    for (std::size_t i = 0; i < mask.size(); ++i)
        out[i] = mask[i] != 0 ? (values[i] - mean) * inv : T(0);
}

/// \brief PPO's value loss over response tokens: the mean of `½ max((V - R)², (clip(V, V_old - ε, V_old + ε) - R)²)`,
/// or of `½ (V - R)²` when `clip` is 0, and its gradient.
/// \param mask `rows · length`. \param values The value head's, per token. \param old_values At sampling, per token.
/// \param returns The targets `R`, per token. \param clip `ε_v`, or 0 for no clip. \param dvalues Written, per token.
/// \return The mean over response tokens; 0 with none.
template <math::math_number T, typename B>
constexpr T value_loss_rows(const B&, std::span<const std::uint8_t> mask, std::span<const T> values, std::type_identity_t<std::span<const T>> old_values,
                            std::type_identity_t<std::span<const T>> returns, double clip, std::type_identity_t<std::span<T>> dvalues) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = mask.size();
    assert(values.size() == n && old_values.size() == n && returns.size() == n && dvalues.size() == n);
    std::size_t count = 0;
    for (const auto m : mask)
        count += m != 0 ? 1 : 0;
    const T w = count == 0 ? T(0) : T(1) / T(count);
    const T eps = T(clip);
    T loss = T(0);
    for (std::size_t t = 0; t < n; ++t) {
        if (mask[t] == 0) {
            dvalues[t] = T(0);
            continue;
        }
        const T e = values[t] - returns[t];
        T l = T(0.5) * e * e, d = e;
        if (clip > 0.0) {
            const T lo = old_values[t] - eps, hi = old_values[t] + eps;
            const T vc = values[t] < lo ? lo : (values[t] > hi ? hi : values[t]);
            const T ec = vc - returns[t];
            const T lc = T(0.5) * ec * ec;
            if (lc > l) {
                l = lc;
                d = T(0);   // the clipped value does not move with V where it binds
            }
        }
        loss += w * l;
        dvalues[t] = w * d;
    }
    return loss;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_rl_hpp
