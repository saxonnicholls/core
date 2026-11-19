//
//  preference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` preference losses over (chosen, rejected) pairs: DPO, IPO, SimPO and ORPO, each with its gradient in the policy's sequence log-probabilities, and their metrics as a mergeable accumulator.
/// \ingroup core
///
/// **Overview.** Preference optimisation trains a policy on pairs of responses to one prompt, one preferred
/// (`chosen`, `y_w`) and one not (`rejected`, `y_l`), without a reward model or sampling (TRAINING.md T11). Each loss
/// is a function of the policy's sequence log-probabilities of the two responses (`ml/logprobs.hpp`) and, for some,
/// a frozen reference model's; its gradient in those two numbers is taken back to the logits by
/// `sequence_logprobs_backward` and `token_logprobs_rows_backward`. The losses, each a parameter struct and an
/// overload of `preference_loss(params, chosen, rejected, ref_chosen, ref_rejected)`:
///
/// | Loss | Paper | Loss of one pair | Log-probabilities |
/// |---|---|---|---|
/// | `dpo{beta}` | Rafailov et al. 2023, eq. 7 | `-log σ(β[(π_w - ref_w) - (π_l - ref_l)])` | sums |
/// | `ipo{tau}` | Azar et al. 2023, eq. 17 | `(h - 1/(2τ))²`, `h = (π_w - ref_w) - (π_l - ref_l)` | sums (the caller's choice) |
/// | `simpo{beta, gamma}` | Meng et al. 2024, eq. 6 | `-log σ(β π̄_w - β π̄_l - γ)` | means; no reference |
/// | `orpo{lambda}` | Hong et al. 2024, eqs. 3-7 | `-π̄_w + λ · -log σ(log odds(π̄_w) - log odds(π̄_l))` | means; no reference |
///
/// Here `π_w` is `log π_θ(y_w | x)`, `ref_w` the reference's, and `π̄` a mean over the response's tokens
/// (`logprob_reduction::mean`). ORPO's `odds(p̄) = P / (1 - P)` with `P = e^π̄` (its eq. 3 and appendix A: the
/// geometric mean of the tokens' probabilities), so `log odds = π̄ - log(-expm1(π̄))`; its SFT term is the chosen
/// response's mean negative log-likelihood, `-π̄_w`. Each loss is computed through `softplus`, so it never takes
/// `log(0)`.
///
/// Every overload returns a `pair_loss`: the loss, its derivatives in the policy's two log-probabilities, and the
/// implicit rewards the training logs report (DPO: `β (π - ref)`, its section 5; IPO: `τ (π - ref)`, as TRL reports
/// it; SimPO: `β π̄`; ORPO: `λ π̄`). `preference_metrics` adds pairs and merges across workers: the mean loss, the
/// rewards, the margin and the accuracy (the fraction of pairs whose chosen reward is strictly the larger).
///
/// \snippet ml/preference_demo.cpp dpo
///
/// `preference_loss_rows(backend, params, ...)` is the batch on the backend (TRAINING.md T9): every pair's loss, the
/// gradients of the mean written per pair, the metrics added.
///
/// \snippet ml/preference_demo.cpp rows
///
/// **Relationships.** `ml/logprobs.hpp` (the sequence log-probabilities and their backward), `ml/activations.hpp`
/// (`softplus`, `sigmoid`), `ml/reward_model.hpp` (the Bradley-Terry loss, an overload of `preference_loss` in this
/// shape). Group: \ref core.
///
/// **Implementations.** One each. A loss of the caller's own joins by overloading `preference_loss` for its parameter
/// struct in its own namespace; `preference_loss_rows` finds it by argument-dependent lookup.
///
/// **Guarantees.** `constexpr`, no allocation, `noexcept` for the standard floating-point types; O(1) a pair. The
/// parameter structs, `pair_loss` and `preference_metrics` are trivially copyable; `merge` is commutative and
/// associative up to rounding of the sums.
///
/// **Errors.** Preconditions, asserted: the spans' sizes agree; ORPO's mean log-probabilities are below 0 (a response
/// of probability 1 has infinite odds). Identical chosen and rejected give DPO `log 2` with gradients `∓β/2`, IPO
/// `1/(4τ²)`.
///
/// **Definitions.** DPO, IPO, SimPO, ORPO and their gradients: checked against the papers (arXiv 2305.18290 eq. 7
/// and the gradient after it; 2310.12036 eq. 17; 2405.14734 eq. 6; 2403.07691 eqs. 3-10). The reported rewards of
/// IPO and ORPO follow TRL's logging, recalled, not from the papers. Default parameters: DPO's `β = 0.1` (the paper's
/// smallest), SimPO's `β = 2, γ = 1` (within its recommended `β ∈ [2, 2.5]`, `γ ∈ [0.5, 1.5]`), ORPO's `λ = 0.1` (its
/// Mistral runs); IPO's `τ = 0.1` is a common choice, recalled.
///
/// **Performance.** O(1) a pair; the batch is negligible next to the log-probabilities it consumes. Not benchmarked.
///
/// **Tests and demos.** `tests/ml/preference_test.cpp` (each loss against a direct computation, central differences
/// for every gradient, the edge cases, a tiny DPO run whose margin rises), `demos/ml/preference_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`,
/// `ml/activations.hpp`.
///
/// **Known issues.** DPO's label smoothing (conservative DPO), KTO and the other variants are not here; each would be
/// one more overload.
///
/// **Status.** Experimental, new in Unreleased.

#ifndef snicholls_core_ml_preference_hpp
#define snicholls_core_ml_preference_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/activations.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief One pair's loss, its derivatives in the policy's two log-probabilities, and its implicit rewards.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct pair_loss
{
    T loss = T(0);              ///< The pair's loss.
    T d_chosen = T(0);          ///< `∂loss / ∂ log π(y_w | x)` (or the chosen reward, for a reward model).
    T d_rejected = T(0);        ///< `∂loss / ∂ log π(y_l | x)`.
    T chosen_reward = T(0);     ///< The implicit reward of the chosen response.
    T rejected_reward = T(0);   ///< The implicit reward of the rejected response.
};

/// \brief Direct preference optimisation (Rafailov et al. 2023): `-log σ(β[(π_w - ref_w) - (π_l - ref_l)])`.
/// \ingroup core
struct dpo
{
    double beta = 0.1;   ///< The strength of the implicit KL to the reference: 0.1 to 0.5 in the paper.
};

/// \brief Identity preference optimisation (Azar et al. 2023, eq. 17): `(h - 1/(2τ))²`, `h` DPO's log-ratio difference.
/// \ingroup core
struct ipo
{
    double tau = 0.1;   ///< The regularisation `τ > 0`: the target margin is `1/(2τ)`.
};

/// \brief SimPO (Meng et al. 2024, eq. 6): `-log σ(β π̄_w - β π̄_l - γ)` on length-normalised log-probabilities, no reference.
/// \ingroup core
struct simpo
{
    double beta = 2.0;    ///< The reward scale: 2 to 2.5 recommended.
    double gamma = 1.0;   ///< The target reward margin: 0.5 to 1.5 recommended.
};

/// \brief ORPO (Hong et al. 2024): the chosen response's mean NLL plus `λ` times the odds-ratio loss, no reference.
/// \ingroup core
struct orpo
{
    double lambda = 0.1;   ///< The weight of the odds-ratio term: 0.1 to 0.25 in the paper's runs.
};

/// \brief The DPO loss of one pair and its gradient.
/// \param c The parameters.
/// \param chosen The policy's `log π(y_w | x)`, a sum over the response.
/// \param rejected The policy's `log π(y_l | x)`.
/// \param ref_chosen The reference's `log π_ref(y_w | x)`.
/// \param ref_rejected The reference's `log π_ref(y_l | x)`.
/// \return The loss, `∂/∂chosen = -β σ(-z)`, `∂/∂rejected = β σ(-z)` with `z = β[(π_w - ref_w) - (π_l - ref_l)]`, and the rewards `β (π - ref)`.
template <math::math_number T>
constexpr pair_loss<T> preference_loss(const dpo& c, T chosen, std::type_identity_t<T> rejected, std::type_identity_t<T> ref_chosen,
                                       std::type_identity_t<T> ref_rejected) noexcept(std::is_floating_point_v<T>)
{
    const T beta = T(c.beta);
    const T rw = beta * (chosen - ref_chosen), rl = beta * (rejected - ref_rejected);
    const T z = rw - rl;
    const T s = sigmoid(-z);
    return {softplus(-z), -beta * s, beta * s, rw, rl};
}

/// \brief The IPO loss of one pair and its gradient.
/// \param c The parameters.
/// \param chosen The policy's `log π(y_w | x)`.
/// \param rejected The policy's `log π(y_l | x)`.
/// \param ref_chosen The reference's.
/// \param ref_rejected The reference's.
/// \return `(h - 1/(2τ))²`, `∂/∂chosen = 2(h - 1/(2τ)) = -∂/∂rejected`, and the rewards `τ (π - ref)`.
template <math::math_number T>
constexpr pair_loss<T> preference_loss(const ipo& c, T chosen, std::type_identity_t<T> rejected, std::type_identity_t<T> ref_chosen,
                                       std::type_identity_t<T> ref_rejected) noexcept(std::is_floating_point_v<T>)
{
    const T tau = T(c.tau);
    const T lw = chosen - ref_chosen, ll = rejected - ref_rejected;
    const T e = (lw - ll) - T(1) / (T(2) * tau);
    return {e * e, T(2) * e, -T(2) * e, tau * lw, tau * ll};
}

/// \brief The SimPO loss of one pair and its gradient: reference-free, so the last two arguments are ignored.
/// \param c The parameters.
/// \param chosen The policy's mean token log-probability of `y_w`.
/// \param rejected The policy's mean token log-probability of `y_l`.
/// \return The loss, `∂/∂chosen = -β σ(-z)`, `∂/∂rejected = β σ(-z)` with `z = β(π̄_w - π̄_l) - γ`, and the rewards `β π̄`.
template <math::math_number T>
constexpr pair_loss<T> preference_loss(const simpo& c, T chosen, std::type_identity_t<T> rejected, std::type_identity_t<T> = T(0),
                                       std::type_identity_t<T> = T(0)) noexcept(std::is_floating_point_v<T>)
{
    const T beta = T(c.beta);
    const T rw = beta * chosen, rl = beta * rejected;
    const T z = (rw - rl) - T(c.gamma);
    const T s = sigmoid(-z);
    return {softplus(-z), -beta * s, beta * s, rw, rl};
}

/// \brief `log(p / (1 - p))` for `p = e^a`, `a < 0`: `a - log(-expm1(a))`, ORPO's log odds of a mean log-probability.
/// \param a A log-probability, below 0.
/// \return The log odds.
template <math::math_number T>
constexpr T log_odds_from_logprob(T a) noexcept(std::is_floating_point_v<T>)
{
    assert(a < T(0));
    return a - math::log(-math::expm1(a));
}

/// \brief The ORPO loss of one pair and its gradient: reference-free, so the last two arguments are ignored.
/// \param c The parameters.
/// \param chosen The policy's mean token log-probability of `y_w`, below 0.
/// \param rejected The policy's mean token log-probability of `y_l`, below 0.
/// \return `-π̄_w + λ softplus(-z)` with `z = log odds(π̄_w) - log odds(π̄_l)`; `∂/∂chosen = -1 - λ σ(-z) / (1 - e^π̄_w)`,
/// `∂/∂rejected = λ σ(-z) / (1 - e^π̄_l)` (the paper's eqs. 8-10); the rewards `λ π̄`.
template <math::math_number T>
constexpr pair_loss<T> preference_loss(const orpo& c, T chosen, std::type_identity_t<T> rejected, std::type_identity_t<T> = T(0),
                                       std::type_identity_t<T> = T(0)) noexcept(std::is_floating_point_v<T>)
{
    const T lambda = T(c.lambda);
    const T z = log_odds_from_logprob(chosen) - log_odds_from_logprob(rejected);
    const T s = sigmoid(-z);
    // d log odds(a) / da = 1 / (1 - e^a) = -1 / expm1(a)
    const T dw = T(-1) / math::expm1(chosen), dl = T(-1) / math::expm1(rejected);
    return {-chosen + lambda * softplus(-z), T(-1) - lambda * s * dw, lambda * s * dl, lambda * chosen, lambda * rejected};
}

/// \brief The metrics of preference training, added pair by pair and merged across workers.
/// \ingroup core
/// \tparam T The number type of the sums.
template <math::math_number T>
struct preference_metrics
{
    std::uint64_t pairs = 0;     ///< Pairs added.
    std::uint64_t correct = 0;   ///< Pairs whose chosen reward was strictly larger.
    T loss = T(0);               ///< The sum of the losses.
    T chosen = T(0);             ///< The sum of the chosen rewards.
    T rejected = T(0);           ///< The sum of the rejected rewards.

    /// \brief Adds one pair. \param p The pair's result.
    constexpr void add(const pair_loss<T>& p) noexcept(std::is_floating_point_v<T>)
    {
        ++pairs;
        correct += p.chosen_reward > p.rejected_reward ? 1 : 0;
        loss += p.loss;
        chosen += p.chosen_reward;
        rejected += p.rejected_reward;
    }

    /// \brief Merges another worker's metrics. \param o The other metrics.
    constexpr void merge(const preference_metrics& o) noexcept(std::is_floating_point_v<T>)
    {
        pairs += o.pairs;
        correct += o.correct;
        loss += o.loss;
        chosen += o.chosen;
        rejected += o.rejected;
    }

    /// \brief The mean loss; 0 before any pair. \return The mean.
    [[nodiscard]] constexpr T mean_loss() const noexcept(std::is_floating_point_v<T>) { return pairs == 0 ? T(0) : loss / T(pairs); }
    /// \brief The mean chosen reward. \return The mean.
    [[nodiscard]] constexpr T mean_chosen_reward() const noexcept(std::is_floating_point_v<T>) { return pairs == 0 ? T(0) : chosen / T(pairs); }
    /// \brief The mean rejected reward. \return The mean.
    [[nodiscard]] constexpr T mean_rejected_reward() const noexcept(std::is_floating_point_v<T>) { return pairs == 0 ? T(0) : rejected / T(pairs); }
    /// \brief The mean reward margin, chosen minus rejected. \return The mean.
    [[nodiscard]] constexpr T mean_margin() const noexcept(std::is_floating_point_v<T>) { return mean_chosen_reward() - mean_rejected_reward(); }
    /// \brief The fraction of pairs whose chosen reward was strictly larger; 0 before any pair. \return The accuracy.
    [[nodiscard]] constexpr double accuracy() const noexcept { return pairs == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(pairs); }
};

/// \brief A batch of pairs on the backend (TRAINING.md T9, T11): each pair's `preference_loss`, the gradients of the
/// mean loss written per pair (each pair's derivatives divided by the number of pairs), the metrics added.
/// \param params `dpo`, `ipo`, `simpo`, `orpo`, `bradley_terry`, or a parameter struct of the caller's with its own overload.
/// \param chosen The policy's chosen log-probabilities (or rewards), `n`.
/// \param rejected The rejected, `n`.
/// \param ref_chosen The reference's, `n`, or empty for a reference-free loss (zeros).
/// \param ref_rejected The reference's, `n`, or empty.
/// \param d_chosen Written, `n`: `∂ mean loss / ∂ chosen`.
/// \param d_rejected Written, `n`.
/// \param metrics Added into.
/// \return The mean loss over the pairs; 0 with none.
template <typename B, typename P, math::math_number T>
constexpr T preference_loss_rows(const B&, const P& params, std::span<const T> chosen, std::type_identity_t<std::span<const T>> rejected,
                                 std::type_identity_t<std::span<const T>> ref_chosen, std::type_identity_t<std::span<const T>> ref_rejected,
                                 std::type_identity_t<std::span<T>> d_chosen, std::type_identity_t<std::span<T>> d_rejected,
                                 preference_metrics<T>& metrics) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t n = chosen.size();
    assert(rejected.size() == n && d_chosen.size() == n && d_rejected.size() == n && (ref_chosen.empty() || ref_chosen.size() == n)
           && (ref_rejected.empty() || ref_rejected.size() == n));
    if (n == 0)
        return T(0);
    const T inv = T(1) / T(n);
    T sum = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        const pair_loss<T> p = preference_loss(params, chosen[i], rejected[i], ref_chosen.empty() ? T(0) : ref_chosen[i],
                                               ref_rejected.empty() ? T(0) : ref_rejected[i]);
        d_chosen[i] = p.d_chosen * inv;
        d_rejected[i] = p.d_rejected * inv;
        sum += p.loss;
        metrics.add(p);
    }
    return sum * inv;
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_preference_hpp
