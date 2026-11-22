//
//  mixture.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::reduce_to_single`, `merge_pair`, `runnalls_cost` and `reduce_mixture`: Gaussian mixtures kept small enough to carry forward.
/// \ingroup core
///
/// **Overview.** Multi-hypothesis filters (PDA's components, GM-PHD, multiple models) multiply mixture components
/// every step. These keep the mixture bounded:
///
/// - `weighted_gaussian<T, N>`: a weight and a Gaussian.
/// - `reduce_to_single(components)`: the moment-matched single Gaussian (Stone Soup's `gm_reduce_single`).
/// - `merge_pair(a, b)`: two components as one, moments preserved, weight `min(w_a + w_b, 1)` as Stone Soup caps it.
/// - `runnalls_cost(a, b)`: the upper bound on the KL divergence a merge adds,
///   `[(w_a + w_b) log det P_ab - w_a log det P_a - w_b log det P_b] / 2`, for choosing what to merge.
/// - `reduce_mixture(components, options)`: in place, Stone Soup's `GaussianMixtureReducer`: prune components
///   below a weight (their weight shared among the rest), merge (the heaviest component absorbs every one within
///   a squared Mahalanobis distance of it, measured with that component's covariance, repeatedly), and truncate to
///   the heaviest `max_components`. Returns the new size.
///
/// \snippet tracking/mixture_demo.cpp mixture
///
/// **Relationships.** `tracking/gaussian.hpp`, `tracking/association.hpp` (PDA's mixture). Stone Soup's
/// `gm_reduce_single` and `GaussianMixtureReducer`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation, no recursion.
/// Merging preserves the mixture's weighted mean and covariance. Complexity O(n²) for `reduce_mixture`.
///
/// **Errors.** `reduce_to_single` of no components, or of zero total weight, returns a zero Gaussian.
/// `runnalls_cost` returns `linalg::error::not_positive_definite` for a covariance without a Cholesky factor. In
/// `reduce_mixture` a component whose covariance has none is never merged into another.
///
/// **Performance.** Not benchmarked.
///
/// **Tests and demos.** `tests/tracking/mixture_test.cpp`, `demos/tracking/mixture_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<limits>`, `<span>`, `linalg/fixed.hpp`, `result.hpp`, `tracking/gaussian.hpp`.
///
/// **Known issues.** Stone Soup's optional k-d-tree coarse gate is not applied; every pair is tested, which gives
/// the same result.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_mixture_hpp
#define snicholls_core_tracking_mixture_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <cstddef>
#include <limits>
#include <span>

namespace snicholls::tracking {

    /// \brief A mixture component: a weight and a Gaussian.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    template <linalg::scalar T, std::size_t N>
    struct weighted_gaussian {
        T weight{};               ///< The weight.
        gaussian<T, N> state{};   ///< The Gaussian.
    };

    /// \brief The single Gaussian with the mixture's mean and covariance.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param components The mixture (weights need not be normalised).
    /// \return `μ = Σ wᵢ μᵢ`, `P = Σ wᵢ (Pᵢ + (μᵢ - μ)(μᵢ - μ)ᵀ)` with normalised weights.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr gaussian<T, N> reduce_to_single(std::span<const weighted_gaussian<T, N>> components) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        T total(0);
        for (const auto& c : components)
            total += c.weight;
        gaussian<T, N> g{};
        if (!(total > T(0)) && !(total < T(0)))
            return g;
        for (const auto& c : components)
            g.mean += c.state.mean * (c.weight / total);
        for (const auto& c : components) {
            const auto d = c.state.mean - g.mean;
            g.covariance += (c.state.covariance + linalg::outer(d, d)) * (c.weight / total);
        }
        g.covariance = linalg::symmetrise(g.covariance);
        return g;
    }

    /// \brief Two components merged with their moments preserved; the weight is `min(w_a + w_b, 1)`.
    /// \tparam T The number type. \tparam N The dimension. \param a One component. \param b The other.
    /// \ingroup core
    /// \return The merged component.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr weighted_gaussian<T, N> merge_pair(const weighted_gaussian<T, N>& a, const weighted_gaussian<T, N>& b) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T w = a.weight + b.weight;
        const T wa = a.weight / w, wb = b.weight / w;
        const auto d = a.state.mean - b.state.mean;
        weighted_gaussian<T, N> m{};
        m.state.mean = a.state.mean * wa + b.state.mean * wb;
        m.state.covariance = linalg::symmetrise(a.state.covariance * wa + b.state.covariance * wb + linalg::outer(d, d) * (wa * wb));
        m.weight = w > T(1) ? T(1) : w;
        return m;
    }

    /// \brief Runnalls' merge cost: an upper bound on the KL divergence the merge of `a` and `b` adds.
    /// \tparam T The number type. \tparam N The dimension. \param a One component. \param b The other.
    /// \ingroup core
    /// \return `[(w_a + w_b) log det P_ab - w_a log det P_a - w_b log det P_b] / 2`, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<T, linalg::error> runnalls_cost(const weighted_gaussian<T, N>& a, const weighted_gaussian<T, N>& b) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        auto m = merge_pair(a, b);
        m.weight = a.weight + b.weight;
        const auto lm = linalg::cholesky(m.state.covariance);
        const auto la = linalg::cholesky(a.state.covariance);
        const auto lb = linalg::cholesky(b.state.covariance);
        if (!lm || !la || !lb)
            return err(linalg::error::not_positive_definite);
        return ok((m.weight * linalg::cholesky_log_det(*lm) - a.weight * linalg::cholesky_log_det(*la) - b.weight * linalg::cholesky_log_det(*lb)) / T(2));
    }

    /// \brief The settings of `reduce_mixture`, Stone Soup's `GaussianMixtureReducer` defaults.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct mixture_reduction {
        T prune_threshold = T(1e-9);   ///< Components lighter than this are pruned (0 to keep all).
        T merge_threshold = T(16);     ///< The squared Mahalanobis distance under which components merge (0 for no merging).
        std::size_t max_components = std::numeric_limits<std::size_t>::max();  ///< The most components kept.
    };

    /// \brief Prunes, merges and truncates a mixture in place, as Stone Soup's `GaussianMixtureReducer` does.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param components The mixture; on return its first entries are the reduced mixture, heaviest first.
    /// \param options The thresholds.
    /// \return The number of components kept.
    template <linalg::scalar T, std::size_t N>
    constexpr std::size_t reduce_mixture(std::span<weighted_gaussian<T, N>> components, const mixture_reduction<T>& options = {}) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        std::size_t n = components.size();
        if (n == 0)
            return 0;
        // Prune, sharing the pruned weight equally among the rest
        T pruned(0);
        std::size_t kept = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (components[i].weight < options.prune_threshold)
                pruned += components[i].weight;
            else
                components[kept++] = components[i];
        }
        n = kept;
        for (std::size_t i = 0; i < n; ++i)
            components[i].weight += pruned / static_cast<T>(n);
        // Heaviest first (stable insertion sort: equal weights keep their order, as Stone Soup's sort does)
        for (std::size_t i = 1; i < n; ++i)
            for (std::size_t j = i; j > 0 && components[j].weight > components[j - 1].weight; --j) {
                const auto t = components[j];
                components[j] = components[j - 1];
                components[j - 1] = t;
            }
        if (n > 1 && options.merge_threshold > T(0)) {
            // components[0, done) are merged results; [done, n) remain, heaviest first. Stone Soup pops the
            // heaviest and then tests the rest from the lightest up; the order matters because the merged
            // component moves as it absorbs.
            std::size_t done = 0;
            while (done < n) {
                auto best = components[done];
                std::size_t rest = done + 1;
                std::size_t write = n;  // compact survivors toward the end, preserving their order
                for (std::size_t i = n; i-- > rest;) {
                    const auto& c = components[i];
                    const auto d2 = squared_mahalanobis(best.state.mean - c.state.mean, c.state.covariance);
                    if (d2 && *d2 < options.merge_threshold)
                        best = merge_pair(best, c);
                    else
                        components[--write] = components[i];
                }
                components[done] = best;
                // The survivors sit in [write, n); move them down to follow the merged component
                std::size_t to = done + 1;
                for (std::size_t i = write; i < n; ++i)
                    components[to++] = components[i];
                n = to;
                ++done;
            }
            // Heaviest first again
            for (std::size_t i = 1; i < n; ++i)
                for (std::size_t j = i; j > 0 && components[j].weight > components[j - 1].weight; --j) {
                    const auto t = components[j];
                    components[j] = components[j - 1];
                    components[j - 1] = t;
                }
        }
        return n < options.max_components ? n : options.max_components;
    }

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_mixture_hpp */
