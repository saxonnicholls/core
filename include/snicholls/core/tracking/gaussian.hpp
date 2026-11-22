//
//  gaussian.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::gaussian<T, N>` and the measures between states: Mahalanobis distance, log-likelihood, Hellinger distance, KL divergence, the χ² gate and the confidence ellipse.
/// \ingroup core
///
/// **Overview.** A Gaussian state is a mean and a covariance. Estimation keeps asking how far a measurement is from
/// a prediction, in units of the prediction's uncertainty, and how alike two estimates are. These functions answer
/// that on fixed-size values:
///
/// - `gaussian<T, N>`: `mean` and `covariance`, an aggregate, trivially copyable.
/// - `squared_mahalanobis(r, S)`: `rᵀ S⁻¹ r` by Cholesky; `whitened_squared_norm(r, L)` when `L = cholesky(S)` is in
///   hand; `mahalanobis` is its root.
/// - `log_likelihood(r, S)`: `log N(r; 0, S)`, the association score of PDA and of GNN's likelihood costs.
/// - `euclidean`, `euclidean_weighted`.
/// - `squared_hellinger(a, b)`, `hellinger(a, b)`, `kl_divergence(p, q)` between two Gaussians.
/// - `gate_threshold(probability, m)` = `chi_square_quantile(probability, m)`, and `in_gate(d², threshold)`.
/// - `confidence_ellipse(P, k)`: the semi-axes and orientation of the `k`-sigma ellipse of a 2 x 2 covariance (a
///   probability plume for a display).
///
/// Residuals with angle components are formed first with `geometry::residual<Kinds>`.
///
/// \snippet tracking/gaussian_demo.cpp gaussian
///
/// **Relationships.** `linalg/fixed.hpp` (Cholesky), `math/constexpr_math.hpp` (`chi_square_quantile`),
/// `geometry/angle.hpp` (residuals). Stone Soup's `measures` module (`Euclidean`, `EuclideanWeighted`,
/// `SquaredMahalanobis`, `Mahalanobis`, `SquaredGaussianHellinger`, `GaussianHellinger`, `KLDivergence`). Group:
/// \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. `gaussian<double, N>`
/// has the size of its `N + N²` doubles and is trivially copyable.
///
/// **Errors.** The functions that factorise a covariance return `linalg::error::not_positive_definite` when it is
/// not (numerically) positive definite.
///
/// **Performance.** Not benchmarked on its own: `tracking/kalman.hpp`'s benchmark covers the same Cholesky.
///
/// **Tests and demos.** `tests/tracking/gaussian_test.cpp`, `demos/tracking/gaussian_demo.cpp`.
///
/// **Dependencies.** `<numbers>`, `linalg/fixed.hpp`, `math/constexpr_math.hpp`, `result.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_gaussian_hpp
#define snicholls_core_tracking_gaussian_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <numbers>

namespace snicholls::tracking {

    /// \brief A Gaussian: mean and covariance.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    template <linalg::scalar T, std::size_t N>
    struct gaussian {
        using value_type = T;                       ///< The number type.
        static constexpr std::size_t dim = N;       ///< The dimension.
        linalg::fixed_vector<T, N> mean{};          ///< The mean.
        linalg::fixed_matrix<T, N, N> covariance{}; ///< The covariance, symmetric positive (semi-)definite.

        /// \brief Element-wise equality. \return Whether means and covariances are equal.
        friend constexpr bool operator==(const gaussian&, const gaussian&) = default;
    };

    /// \brief `|L⁻¹ r|²`, the squared Mahalanobis norm of `r` when `L` is the Cholesky factor of its covariance.
    /// \tparam T The number type. \tparam N The dimension. \param r The residual.
    /// \ingroup core
    /// \param l `cholesky(S)`. \return `rᵀ S⁻¹ r`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr T whitened_squared_norm(const linalg::fixed_vector<T, N>& r, const linalg::fixed_matrix<T, N, N>& l) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return linalg::squared_norm(linalg::lower_solve(l, r));
    }

    /// \brief The squared Mahalanobis distance `rᵀ S⁻¹ r`.
    /// \tparam T The number type. \tparam N The dimension. \param r The residual.
    /// \ingroup core
    /// \param s Its covariance. \return The distance², or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<T, linalg::error> squared_mahalanobis(const linalg::fixed_vector<T, N>& r, const linalg::fixed_matrix<T, N, N>& s) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(s);
        if (!l)
            return err(l.unwrap_err_unchecked());
        return ok(whitened_squared_norm(r, *l));
    }

    /// \brief The Mahalanobis distance `√(rᵀ S⁻¹ r)`.
    /// \tparam T The number type. \tparam N The dimension. \param r The residual.
    /// \ingroup core
    /// \param s Its covariance. \return The distance, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<T, linalg::error> mahalanobis(const linalg::fixed_vector<T, N>& r, const linalg::fixed_matrix<T, N, N>& s) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto d2 = squared_mahalanobis(r, s);
        if (!d2)
            return d2;
        return ok(static_cast<T>(math::sqrt(*d2)));
    }

    /// \brief `log N(r; 0, S) = -(rᵀ S⁻¹ r + log det S + m log 2π) / 2`.
    /// \tparam T The number type. \tparam M The dimension. \param r The residual (innovation).
    /// \ingroup core
    /// \param s Its covariance. \return The log-density, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t M>
    [[nodiscard]] constexpr result<T, linalg::error> log_likelihood(const linalg::fixed_vector<T, M>& r, const linalg::fixed_matrix<T, M, M>& s) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(s);
        if (!l)
            return err(l.unwrap_err_unchecked());
        const T log_2pi = static_cast<T>(math::log(T(2) * static_cast<T>(std::numbers::pi_v<long double>)));
        return ok(-(whitened_squared_norm(r, *l) + linalg::cholesky_log_det(*l) + static_cast<T>(M) * log_2pi) / T(2));
    }

    /// \brief The Euclidean distance. \tparam T The number type. \tparam N The dimension.
    /// \ingroup core
    /// \param a One point. \param b The other. \return `|a - b|`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr T euclidean(const linalg::fixed_vector<T, N>& a, const linalg::fixed_vector<T, N>& b) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return linalg::norm(a - b);
    }

    /// \brief The weighted Euclidean distance `√(Σ wᵢ (aᵢ - bᵢ)²)`. \tparam T The number type.
    /// \ingroup core
    /// \tparam N The dimension. \param a One point. \param b The other. \param w The weights. \return The distance.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr T euclidean_weighted(const linalg::fixed_vector<T, N>& a, const linalg::fixed_vector<T, N>& b, const linalg::fixed_vector<T, N>& w) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        T s(0);
        for (std::size_t i = 0; i < N; ++i)
            s += w[i] * (a[i] - b[i]) * (a[i] - b[i]);
        return static_cast<T>(math::sqrt(s));
    }

    /// \brief The squared Hellinger distance between two Gaussians, in `[0, 1]`:
    /// `1 - (det P₁ det P₂)^¼ / det(P̄)^½ · exp(-dᵀ P̄⁻¹ d / 8)`, `P̄ = (P₁ + P₂) / 2`, `d = μ₁ - μ₂`.
    /// \tparam T The number type. \tparam N The dimension. \param a One Gaussian. \param b The other.
    /// \ingroup core
    /// \return The distance², or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<T, linalg::error> squared_hellinger(const gaussian<T, N>& a, const gaussian<T, N>& b) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto la = linalg::cholesky(a.covariance);
        const auto lb = linalg::cholesky(b.covariance);
        const auto lm = linalg::cholesky((a.covariance + b.covariance) / T(2));
        if (!la || !lb || !lm)
            return err(linalg::error::not_positive_definite);
        const T log_coeff = (linalg::cholesky_log_det(*la) + linalg::cholesky_log_det(*lb)) / T(4) - linalg::cholesky_log_det(*lm) / T(2);
        const T e = whitened_squared_norm(a.mean - b.mean, *lm) / T(8);
        const T h2 = T(1) - static_cast<T>(math::exp(log_coeff - e));
        return ok(h2 < T(0) ? T(0) : h2);
    }

    /// \brief The Hellinger distance between two Gaussians, `√squared_hellinger`. \tparam T The number type.
    /// \ingroup core
    /// \tparam N The dimension. \param a One Gaussian. \param b The other.
    /// \return The distance, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<T, linalg::error> hellinger(const gaussian<T, N>& a, const gaussian<T, N>& b) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto h2 = squared_hellinger(a, b);
        if (!h2)
            return h2;
        return ok(static_cast<T>(math::sqrt(*h2)));
    }

    /// \brief The Kullback-Leibler divergence `KL(p ‖ q) = (tr(Q⁻¹ P) + dᵀ Q⁻¹ d - n + log(det Q / det P)) / 2`, `d = μ_q - μ_p`.
    /// \tparam T The number type. \tparam N The dimension. \param p The first Gaussian.
    /// \ingroup core
    /// \param q The second. \return The divergence, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<T, linalg::error> kl_divergence(const gaussian<T, N>& p, const gaussian<T, N>& q) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto lp = linalg::cholesky(p.covariance);
        const auto lq = linalg::cholesky(q.covariance);
        if (!lp || !lq)
            return err(linalg::error::not_positive_definite);
        const T tr = linalg::trace(linalg::cholesky_solve(*lq, p.covariance));
        const T md = whitened_squared_norm(q.mean - p.mean, *lq);
        return ok((tr + md - static_cast<T>(N) + linalg::cholesky_log_det(*lq) - linalg::cholesky_log_det(*lp)) / T(2));
    }

    /// \brief The gate on a squared Mahalanobis distance of an `m`-dimensional residual that a correct measurement
    /// passes with `probability`: `chi_square_quantile(probability, m)`.
    /// \tparam T The number type. \param probability The gate probability, such as 0.99.
    /// \ingroup core
    /// \param m The measurement dimension. \return The threshold.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T gate_threshold(T probability, std::size_t m) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return static_cast<T>(math::chi_square_quantile(probability, static_cast<T>(m)));
    }

    /// \brief Whether a squared Mahalanobis distance is inside a gate. \tparam T The number type.
    /// \ingroup core
    /// \param distance2 The distance². \param threshold The gate. \return `distance2 <= threshold`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr bool in_gate(T distance2, T threshold) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return distance2 <= threshold;
    }

    /// \brief An ellipse: semi-axes and the major axis's angle from x.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct ellipse {
        T semi_major{};  ///< The longer semi-axis.
        T semi_minor{};  ///< The shorter semi-axis.
        T angle{};       ///< The major axis's angle from the x axis, radians, in `(-π/2, π/2]`.
    };

    /// \brief The `k`-sigma ellipse of a 2 x 2 covariance: semi-axes `k √λ` for its eigenvalues `λ`.
    /// \tparam T The number type. \param p The covariance (symmetric; the upper element is read).
    /// \ingroup core
    /// \param k The scale: 1 for one sigma, `√gate_threshold(0.95, 2)` for 95% containment.
    /// \return The ellipse; a negative eigenvalue (indefinite input) counts as zero.
    template <linalg::scalar T>
    [[nodiscard]] constexpr ellipse<T> confidence_ellipse(const linalg::fixed_matrix<T, 2, 2>& p, T k = T(1)) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T a = p(0, 0), b = p(0, 1), d = p(1, 1);
        const T mean = (a + d) / T(2);
        const T h = static_cast<T>(math::sqrt((a - d) * (a - d) / T(4) + b * b));
        const T l1 = mean + h, l2 = mean - h;
        // The major axis is at θ = atan2(2b, a - d) / 2, in (-π/2, π/2]
        const T ang = (b == T(0) && a == d) ? T(0) : static_cast<T>(math::atan2(T(2) * b, a - d)) / T(2);
        return {k * static_cast<T>(math::sqrt(l1 > T(0) ? l1 : T(0))), k * static_cast<T>(math::sqrt(l2 > T(0) ? l2 : T(0))), ang};
    }

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_gaussian_hpp */
