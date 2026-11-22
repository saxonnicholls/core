//
//  fusion.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::covariance_intersection`, `optimal_omega` and `fast_ci_weights`: fusing estimates whose cross-correlation is unknown, without becoming overconfident.
/// \ingroup core
///
/// **Overview.** Two sensors watching the same target are not independent. They share the target's motion, the
/// tracker's priors, perhaps the same splash, so fusing their tracks as if they were (adding information)
/// double-counts and is overconfident. Covariance intersection is consistent for *any* cross-correlation:
///
///     P⁻¹ = Σ ωᵢ Pᵢ⁻¹,   x = P Σ ωᵢ Pᵢ⁻¹ xᵢ,   Σ ωᵢ = 1, ωᵢ ≥ 0.
///
/// - `covariance_intersection(a, b, ω)` and `covariance_intersection(estimates, weights)` for n.
/// - `optimal_omega(a, b, lo, hi, criterion)`: the ω in `[lo, hi]` that minimises `tr P` (or `det P`), by
///   golden-section search, since both are unimodal in ω. The bounds encode trust: a camera still learning its pose
///   can be held to ω ≤ 0.3.
/// - `fast_ci_weights(estimates, weights, lo, hi)`: the fast approximation ωᵢ ∝ 1 / tr Pᵢ (Niehsen 2002) for n
///   sources, clamped to per-source bounds and renormalised.
///
/// \snippet tracking/fusion_demo.cpp fusion
///
/// **Relationships.** `tracking/gaussian.hpp`, `tracking/kalman.hpp` (the information form, for independent
/// sources). Stone Soup's `ChernoffUpdater` with ω, and `CovarianceIntersection`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. The fused
/// covariance is consistent (it bounds the true error covariance) for any correlation, given consistent inputs.
///
/// **Errors.** `linalg::error::not_positive_definite` for an input covariance (or the fused information) without a
/// Cholesky factor; `linalg::error::dimension_mismatch` for spans of different lengths or an empty set.
///
/// **Performance.** Not benchmarked: `optimal_omega` evaluates the fused trace about 60 times.
///
/// **Tests and demos.** `tests/tracking/fusion_test.cpp`, `demos/tracking/fusion_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `linalg/fixed.hpp`, `result.hpp`, `tracking/gaussian.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_fusion_hpp
#define snicholls_core_tracking_fusion_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::tracking {

    /// \brief What `optimal_omega` minimises.
    /// \ingroup core
    enum class ci_criterion : std::uint8_t {
        trace = 0,        ///< `tr P`, the mean squared error.
        determinant = 1,  ///< `det P`, the volume of the uncertainty ellipsoid.
    };

    /// \brief Covariance intersection of n estimates with given weights.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param estimates The estimates.
    /// \param weights One non-negative weight per estimate, summing to 1.
    /// \return The fused estimate, or a `linalg::error`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<gaussian<T, N>, linalg::error> covariance_intersection(std::span<const gaussian<T, N>> estimates, std::span<const T> weights) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        if (estimates.empty() || estimates.size() != weights.size())
            return err(linalg::error::dimension_mismatch);
        linalg::fixed_matrix<T, N, N> info{};
        linalg::fixed_vector<T, N> vec{};
        for (std::size_t i = 0; i < estimates.size(); ++i) {
            if (weights[i] == T(0))
                continue;
            const auto y = linalg::spd_inverse(estimates[i].covariance);
            if (!y)
                return err(y.unwrap_err_unchecked());
            info += *y * weights[i];
            vec += *y * estimates[i].mean * weights[i];
        }
        const auto l = linalg::cholesky(info);
        if (!l)
            return err(l.unwrap_err_unchecked());
        return ok(gaussian<T, N>{linalg::cholesky_solve(*l, vec), linalg::symmetrise(linalg::cholesky_solve(*l, linalg::fixed_matrix<T, N, N>::identity()))});
    }

    /// \brief Covariance intersection of two estimates: `P⁻¹ = ω P_a⁻¹ + (1 - ω) P_b⁻¹`.
    /// \tparam T The number type. \tparam N The dimension. \param a The first estimate.
    /// \ingroup core
    /// \param b The second. \param omega The first's weight, in `[0, 1]`. \return The fused estimate, or a `linalg::error`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<gaussian<T, N>, linalg::error> covariance_intersection(const gaussian<T, N>& a, const gaussian<T, N>& b, T omega) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const gaussian<T, N> e[2] = {a, b};
        const T w[2] = {omega, T(1) - omega};
        return covariance_intersection(std::span<const gaussian<T, N>>(e), std::span<const T>(w));
    }

    /// \brief The ω in `[lo, hi]` minimising the fused covariance's trace or determinant, by golden-section search.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param a The first estimate.
    /// \param b The second.
    /// \param lo The least weight the first may have.
    /// \param hi The most.
    /// \param criterion What to minimise.
    /// \return ω, or a `linalg::error`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<T, linalg::error> optimal_omega(const gaussian<T, N>& a, const gaussian<T, N>& b, T lo = T(0), T hi = T(1),
                                                                 ci_criterion criterion = ci_criterion::trace) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto ya = linalg::spd_inverse(a.covariance);
        const auto yb = linalg::spd_inverse(b.covariance);
        if (!ya || !yb)
            return err(linalg::error::not_positive_definite);
        // Cost of ω from the information matrices alone; a singular combination costs +∞
        auto cost = [&](T w) {
            const auto info = *ya * w + *yb * (T(1) - w);
            const auto l = linalg::cholesky(info);
            if (!l)
                return std::numeric_limits<T>::infinity();
            if (criterion == ci_criterion::determinant)
                return -linalg::cholesky_log_det(*l);
            return linalg::trace(linalg::cholesky_solve(*l, linalg::fixed_matrix<T, N, N>::identity()));
        };
        const T g = T(0.6180339887498948482);
        T x0 = lo, x3 = hi;
        T x1 = x3 - g * (x3 - x0), x2 = x0 + g * (x3 - x0);
        T f1 = cost(x1), f2 = cost(x2);
        for (int i = 0; i < 80; ++i) {
            if (f1 <= f2) {
                x3 = x2;
                x2 = x1;
                f2 = f1;
                x1 = x3 - g * (x3 - x0);
                f1 = cost(x1);
            } else {
                x0 = x1;
                x1 = x2;
                f1 = f2;
                x2 = x0 + g * (x3 - x0);
                f2 = cost(x2);
            }
        }
        T best = (x0 + x3) / T(2);
        T fb = cost(best);
        if (cost(lo) < fb) {
            best = lo;
            fb = cost(lo);
        }
        if (cost(hi) < fb)
            best = hi;
        return ok(best);
    }

    /// \brief The fast covariance-intersection weights for n estimates, `ωᵢ ∝ 1 / tr Pᵢ`, clamped to `[lo, hi]` and
    /// renormalised.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param estimates The estimates.
    /// \param weights Receives one weight per estimate.
    /// \param lo Each estimate's least weight (empty for 0).
    /// \param hi Each estimate's most weight (empty for 1).
    /// \return Whether the bounds admit weights summing to 1 (and the spans fit).
    template <linalg::scalar T, std::size_t N>
    constexpr bool fast_ci_weights(std::span<const gaussian<T, N>> estimates, std::span<T> weights, std::span<const T> lo = {}, std::span<const T> hi = {}) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const std::size_t n = estimates.size();
        if (n == 0 || weights.size() != n || (!lo.empty() && lo.size() != n) || (!hi.empty() && hi.size() != n))
            return false;
        T lo_sum(0), hi_sum(0);
        for (std::size_t i = 0; i < n; ++i) {
            lo_sum += lo.empty() ? T(0) : lo[i];
            hi_sum += hi.empty() ? T(1) : hi[i];
        }
        if (lo_sum > T(1) || hi_sum < T(1))
            return false;
        T s(0);
        for (std::size_t i = 0; i < n; ++i) {
            const T t = linalg::trace(estimates[i].covariance);
            weights[i] = t > T(0) ? T(1) / t : T(0);
            s += weights[i];
        }
        for (std::size_t i = 0; i < n; ++i)
            weights[i] = s > T(0) ? weights[i] / s : T(1) / static_cast<T>(n);
        // Clamp: ωᵢ = clamp(λ rᵢ, loᵢ, hiᵢ), rᵢ the proportional weights, with λ such that Σ ωᵢ = 1. The sum is
        // non-decreasing and piecewise linear in λ, with breakpoints at loᵢ / rᵢ and hiᵢ / rᵢ: take the largest
        // breakpoint at which it is at most 1 and the smallest at which it is at least 1, and interpolate (no
        // breakpoint lies between them). O(n²), exact up to rounding.
        auto bound = [&](std::size_t i, bool upper) { return upper ? (hi.empty() ? T(1) : hi[i]) : (lo.empty() ? T(0) : lo[i]); };
        auto clamped = [&](std::size_t i, T lambda) {
            const T w = lambda * weights[i], l = bound(i, false), h = bound(i, true);
            return w < l ? l : w > h ? h : w;
        };
        auto total = [&](T lambda) {
            T t(0);
            for (std::size_t i = 0; i < n; ++i)
                t += clamped(i, lambda);
            return t;
        };
        T below(0), f_below = total(T(0)), above(-1), f_above(0);
        for (std::size_t k = 0; k < 2 * n; ++k) {
            const std::size_t i = k / 2;
            if (!(weights[i] > T(0)))
                continue;
            const T lambda = bound(i, k % 2 == 1) / weights[i];
            const T f = total(lambda);
            if (f <= T(1) && lambda > below) {
                below = lambda;
                f_below = f;
            }
            if (f >= T(1) && (above < T(0) || lambda < above)) {
                above = lambda;
                f_above = f;
            }
        }
        const T lambda = above < T(0) || !(f_above > f_below) ? below : below + (above - below) * (T(1) - f_below) / (f_above - f_below);
        for (std::size_t i = 0; i < n; ++i)
            weights[i] = clamped(i, lambda);
        return true;
    }

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_fusion_hpp */
