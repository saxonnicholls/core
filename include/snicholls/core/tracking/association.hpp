//
//  association.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::pda_weights`, `pda_update` and `jpda_marginals`: probabilistic data association, which updates a track with every detection weighted by the chance it is the target's, rather than committing to one.
/// \ingroup core
///
/// **Overview.** In clutter, or where targets cross, committing each track to its nearest detection swaps
/// identities. Probabilistic data association keeps every hypothesis, weighted.
///
/// - `pda_weights(log_likelihoods, pd, pg, clutter_density, weights)`: one track's hypothesis weights, Stone Soup's
///   `PDAHypothesiser` normalised. Missed detection gets `1 - P_D P_G`; detection `i` gets `P_D N(νᵢ; 0, S) / λ`.
///   A detection outside the gate has log-likelihood `-∞` and weight 0.
/// - `pda_update(prediction, mp, innovations, weights)`: the moment-matched posterior (Bar-Shalom and Tse):
///   `x + K ν̄`, and `β₀ P + (1 - β₀) (P - K S Kᵀ) + K (Σ βᵢ νᵢ νᵢᵀ - ν̄ ν̄ᵀ) Kᵀ`, with `ν̄ = Σ βᵢ νᵢ`.
/// - `jpda_marginals<MaxTracks, MaxDetections>(likelihoods, tracks, detections, marginals)`: joint PDA over a
///   cluster. It enumerates every feasible joint event (each detection to at most one track), weights each by the
///   product of its tracks' entries, and returns each track's marginal probability of each detection (column 0:
///   missed), Stone Soup's `JPDA`. The enumeration is exact and grows exponentially, so it is for small clusters.
/// - `gate_volume(threshold, S)`: the volume of the gate `{ν : νᵀ S⁻¹ ν ≤ γ}`, for a clutter density from a count.
///
/// \snippet tracking/association_demo.cpp association
///
/// **Relationships.** `tracking/gaussian.hpp` (`log_likelihood`), `tracking/kalman.hpp`
/// (`measurement_prediction`, `kalman_update` gives each detection's `log_likelihood` and innovation),
/// `combinatorics/assignment.hpp` (the hard-decision alternative). Group: \ref core.
///
/// **Implementations.** Weights are formed in log space and normalised by `log_sum_exp`. `jpda_marginals` walks the
/// joint events depth first, iteratively, with a stack of partial products; a detection already used is skipped,
/// and so is a zero entry.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation, no recursion.
/// Weights sum to 1 and each track's marginals sum to 1.
///
/// **Errors.** `pda_weights` returns `false` when `weights` is not one longer than the likelihoods or no hypothesis
/// has weight. `pda_update` returns `linalg::error::not_positive_definite` for an `S` without a Cholesky factor.
/// `jpda_marginals` returns `false` for sizes beyond its maxima, spans too short, or no feasible event with weight.
///
/// **Performance.** Not benchmarked. `jpda_marginals` costs the number of feasible joint events, which for `t`
/// tracks all gating `d` detections is `Σₖ C(t, k) d! / (d - k)!`.
///
/// **Tests and demos.** `tests/tracking/association_test.cpp`, `demos/tracking/association_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<numbers>`, `<span>`, `linalg/fixed.hpp`,
/// `math/constexpr_math.hpp`, `result.hpp`, `stats/resample.hpp` (`log_sum_exp`), `tracking/gaussian.hpp`,
/// `tracking/kalman.hpp`.
///
/// **Known issues.** Loopy belief propagation and the efficient hypothesis management tree (Stone Soup's
/// `JPDAwithLBP` and `JPDAwithEHM`) are not here yet.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_association_hpp
#define snicholls_core_tracking_association_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/stats/resample.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/kalman.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>

namespace snicholls::tracking {

    /// \brief One track's PDA hypothesis weights: `weights[0]` missed detection, `weights[i + 1]` detection `i`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param log_likelihoods Each detection's `log N(νᵢ; 0, S)`; `-∞` for a detection outside the gate.
    /// \param pd The probability of detection.
    /// \param pg The gate probability.
    /// \param clutter_density The clutter's spatial density λ (per unit measurement volume), above zero.
    /// \param weights Receives the normalised weights; one longer than `log_likelihoods`.
    /// \return Whether the weights were formed.
    template <linalg::scalar T>
    constexpr bool pda_weights(std::span<const T> log_likelihoods, T pd, T pg, T clutter_density, std::span<T> weights) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        if (weights.size() != log_likelihoods.size() + 1 || !(clutter_density > T(0)))
            return false;
        const T miss = T(1) - pd * pg;
        weights[0] = miss > T(0) ? static_cast<T>(math::log(miss)) : -std::numeric_limits<T>::infinity();
        const T base = static_cast<T>(math::log(pd)) - static_cast<T>(math::log(clutter_density));
        for (std::size_t i = 0; i < log_likelihoods.size(); ++i)
            weights[i + 1] = log_likelihoods[i] + base;
        const T z = stats::normalise_log_weights(weights);
        if (!(z > -std::numeric_limits<T>::infinity()) || !(z < std::numeric_limits<T>::infinity()))
            return false;
        for (T& w : weights)
            w = static_cast<T>(math::exp(w));
        return true;
    }

    /// \brief The PDA posterior: the moment-matched mixture of the missed-detection and every detection's update.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam M The measurement dimension.
    /// \param prediction The predicted state.
    /// \param mp Its measurement prediction.
    /// \param innovations Each detection's innovation `zᵢ ⊖ ẑ` (wrapped as the sensor's kinds say).
    /// \param weights The weights from `pda_weights`: missed first, then one per innovation.
    /// \return The posterior, or `linalg::error::not_positive_definite` (also for mismatched lengths).
    template <linalg::scalar T, std::size_t N, std::size_t M>
    [[nodiscard]] constexpr result<gaussian<T, N>, linalg::error> pda_update(const gaussian<T, N>& prediction, const measurement_prediction<T, N, M>& mp,
                                                                            std::span<const linalg::fixed_vector<T, M>> innovations, std::span<const T> weights) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        if (weights.size() != innovations.size() + 1)
            return err(linalg::error::dimension_mismatch);
        const auto l = linalg::cholesky(mp.covariance);
        if (!l)
            return err(l.unwrap_err_unchecked());
        const auto k = linalg::transpose(linalg::cholesky_solve(*l, linalg::transpose(mp.cross_covariance)));
        linalg::fixed_vector<T, M> nu{};
        linalg::fixed_matrix<T, M, M> spread{};
        for (std::size_t i = 0; i < innovations.size(); ++i) {
            nu += innovations[i] * weights[i + 1];
            spread += linalg::outer(innovations[i], innovations[i]) * weights[i + 1];
        }
        spread -= linalg::outer(nu, nu);
        const T b0 = weights[0];
        const auto pc = prediction.covariance - linalg::sandwich(k, mp.covariance);
        return ok(gaussian<T, N>{prediction.mean + k * nu, linalg::symmetrise(prediction.covariance * b0 + pc * (T(1) - b0) + linalg::sandwich(k, spread))});
    }

    /// \brief The volume of the gate `{ν : νᵀ S⁻¹ ν ≤ γ}`: `π^(m/2) / Γ(m/2 + 1) · γ^(m/2) · √det S`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam M The measurement dimension.
    /// \param threshold The gate γ (a squared Mahalanobis distance).
    /// \param s The innovation covariance.
    /// \return The volume, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t M>
    [[nodiscard]] constexpr result<T, linalg::error> gate_volume(T threshold, const linalg::fixed_matrix<T, M, M>& s) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(s);
        if (!l)
            return err(l.unwrap_err_unchecked());
        const T half = static_cast<T>(M) / T(2);
        const T log_v = half * static_cast<T>(math::log(static_cast<T>(std::numbers::pi_v<long double>))) - static_cast<T>(math::lgamma(half + T(1))) +
                        half * static_cast<T>(math::log(threshold)) + linalg::cholesky_log_det(*l) / T(2);
        return ok(static_cast<T>(math::exp(log_v)));
    }

    /// \brief Joint PDA: each track's marginal association probabilities over a cluster, by exact enumeration.
    /// \ingroup core
    /// \tparam MaxTracks The most tracks.
    /// \tparam MaxDetections The most detections.
    /// \tparam T The number type.
    /// \param likelihoods Row-major, `tracks x (detections + 1)`: each track's unnormalised hypothesis weight for a
    /// missed detection (column 0) and for each detection (`pda_weights`' output, or before normalising); 0 forbids.
    /// \param tracks The tracks.
    /// \param detections The detections.
    /// \param marginals Receives, in the same shape, each track's probability of each hypothesis.
    /// \return Whether the marginals were formed.
    template <std::size_t MaxTracks, std::size_t MaxDetections, linalg::scalar T>
    constexpr bool jpda_marginals(std::span<const T> likelihoods, std::size_t tracks, std::size_t detections, std::span<T> marginals) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const std::size_t cols = detections + 1;
        if (tracks > MaxTracks || detections > MaxDetections || likelihoods.size() < tracks * cols || marginals.size() < tracks * cols)
            return false;
        for (std::size_t i = 0; i < tracks * cols; ++i)
            marginals[i] = T(0);
        if (tracks == 0)
            return true;
        std::array<std::size_t, MaxTracks + 1> choice{};   // the hypothesis chosen at each depth, or cols when exhausted
        std::array<T, MaxTracks + 1> partial{};             // the product of the chosen weights above each depth
        std::array<std::uint8_t, MaxDetections + 1> used{};
        T total(0);
        std::size_t depth = 0;
        partial[0] = T(1);
        choice[0] = 0;
        bool fresh = true;  // whether choice[depth] is still to be tried from its current value
        for (;;) {
            if (!fresh) {
                // Release the detection this depth used, then advance to the next choice
                if (choice[depth] > 0)
                    used[choice[depth]] = 0;
                ++choice[depth];
            }
            fresh = false;
            while (choice[depth] < cols) {
                const std::size_t c = choice[depth];
                if (likelihoods[depth * cols + c] > T(0) && (c == 0 || !used[c]))
                    break;
                ++choice[depth];
            }
            if (choice[depth] >= cols) {
                if (depth == 0)
                    break;
                --depth;
                continue;
            }
            const std::size_t c = choice[depth];
            if (c > 0)
                used[c] = 1;
            const T p = partial[depth] * likelihoods[depth * cols + c];
            if (depth + 1 == tracks) {
                total += p;
                for (std::size_t t = 0; t < tracks; ++t)
                    marginals[t * cols + choice[t]] += p;
                continue;  // not fresh: release and advance this depth
            }
            partial[depth + 1] = p;
            ++depth;
            choice[depth] = 0;
            fresh = true;
        }
        if (!(total > T(0)))
            return false;
        for (std::size_t i = 0; i < tracks * cols; ++i)
            marginals[i] /= total;
        return true;
    }

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_association_hpp */
