//
//  statistical.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::metric::mahalanobis` and the Gaussian measures `hellinger`, `squared_hellinger` and `kl_divergence` as plug-in metrics.
/// \ingroup core
///
/// **Overview.** Distances that account for uncertainty, packaged as metrics so that any search structure or set
/// metric can use them:
///
/// - `mahalanobis<T, N>`: `√((a - b)ᵀ S⁻¹ (a - b))` for a fixed covariance `S`, made by
///   `make_mahalanobis(S)`, which factorises `S` once. It holds `L = cholesky(S)` and is the Euclidean distance after
///   whitening by `L⁻¹`: a metric, with the squared distance as its reduced form. Gating detections against one
///   track's innovation covariance is a radius search with it.
/// - `hellinger`, `squared_hellinger`, `kl_divergence`: between two `tracking::gaussian`s, the functions of
///   `tracking/gaussian.hpp` as metric values. `hellinger` is a metric; the other two are not (`kl_divergence` is not
///   even symmetric). A covariance without a Cholesky factor gives NaN, which never compares below a bound.
///
/// \snippet metric/statistical_demo.cpp statistical
///
/// **Relationships.** `metric/concepts.hpp`, `tracking/gaussian.hpp` (the functions), `linalg/fixed.hpp`
/// (`cholesky`, `lower_solve`), `result.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. `mahalanobis` is
/// trivially copyable (it holds an `N x N` factor).
///
/// **Errors.** `make_mahalanobis` returns `linalg::error::not_positive_definite` for a covariance without a Cholesky
/// factor. The Gaussian measures return NaN in that case.
///
/// **Performance.** Not benchmarked: `mahalanobis` is a triangular solve per pair, O(N²).
///
/// **Tests and demos.** `tests/metric/statistical_test.cpp`, `demos/metric/statistical_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<limits>`, `linalg/fixed.hpp`, `math/constexpr_math.hpp`, `metric/concepts.hpp`, `result.hpp`,
/// `tracking/gaussian.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.20.0.

#ifndef snicholls_core_metric_statistical_hpp
#define snicholls_core_metric_statistical_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <cstddef>
#include <limits>

namespace snicholls::metric {

    /// \brief The Mahalanobis distance for a fixed covariance `S`, through its Cholesky factor `L`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    template <linalg::scalar T, std::size_t N>
    struct mahalanobis {
        static constexpr bool is_metric = true;  ///< It is a Euclidean distance after whitening.
        linalg::fixed_matrix<T, N, N> factor{};  ///< `L` with `L Lᵀ = S`.

        /// \brief The reduced form, `(a - b)ᵀ S⁻¹ (a - b)`. \tparam A A `point` of dimension N. \tparam B Another.
        /// \param a One point. \param b The other. \return The squared distance.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr T reduced(const A& a, const B& b) const noexcept(linalg::detail::nothrow_scalar<T> && noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            linalg::fixed_vector<T, N> d{};
            for (std::size_t i = 0; i < N; ++i)
                d[i] = static_cast<T>(coordinate(a, i)) - static_cast<T>(coordinate(b, i));
            return tracking::whitened_squared_norm(d, factor);
        }
        /// \brief The distance of a reduced value. \tparam U The number type. \param r The squared distance. \return `√r`.
        template <typename U>
        [[nodiscard]] constexpr U expand(U r) const noexcept
        {
            return static_cast<U>(math::sqrt(r));
        }
        /// \brief The reduced value of a distance. \tparam U The number type. \param d The distance. \return `d²`.
        template <typename U>
        [[nodiscard]] constexpr U contract(U d) const noexcept
        {
            return d * d;
        }
        /// \brief The distance. \tparam A A `point` of dimension N. \tparam B Another. \param a One point.
        /// \param b The other. \return `√((a - b)ᵀ S⁻¹ (a - b))`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr T operator()(const A& a, const B& b) const noexcept(linalg::detail::nothrow_scalar<T> && noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return expand(reduced(a, b));
        }
    };

    /// \brief The Mahalanobis metric of a covariance, factorised once.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param s The covariance.
    /// \return The metric, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<mahalanobis<T, N>, linalg::error> make_mahalanobis(const linalg::fixed_matrix<T, N, N>& s) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(s);
        if (!l)
            return err(l.unwrap_err_unchecked());
        return ok(mahalanobis<T, N>{*l});
    }

    /// \brief The Hellinger distance between two Gaussians, in `[0, 1]`: a metric.
    /// \ingroup core
    struct hellinger {
        static constexpr bool is_metric = true;  ///< It is a metric on distributions.
        /// \brief The distance. \tparam T The number type. \tparam N The dimension. \param a One Gaussian.
        /// \param b The other. \return The distance, or NaN for a covariance without a Cholesky factor.
        template <linalg::scalar T, std::size_t N>
        [[nodiscard]] constexpr T operator()(const tracking::gaussian<T, N>& a, const tracking::gaussian<T, N>& b) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto h = tracking::hellinger(a, b);
            return h ? *h : std::numeric_limits<T>::quiet_NaN();
        }
    };

    /// \brief The squared Hellinger distance between two Gaussians: not a metric.
    /// \ingroup core
    struct squared_hellinger {
        static constexpr bool is_metric = false;  ///< The triangle inequality fails.
        /// \brief The value. \tparam T The number type. \tparam N The dimension. \param a One Gaussian.
        /// \param b The other. \return The squared distance, or NaN.
        template <linalg::scalar T, std::size_t N>
        [[nodiscard]] constexpr T operator()(const tracking::gaussian<T, N>& a, const tracking::gaussian<T, N>& b) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto h = tracking::squared_hellinger(a, b);
            return h ? *h : std::numeric_limits<T>::quiet_NaN();
        }
    };

    /// \brief The Kullback-Leibler divergence `KL(a ‖ b)` between two Gaussians: not symmetric, not a metric.
    /// \ingroup core
    struct kl_divergence {
        static constexpr bool is_metric = false;  ///< It is not symmetric.
        /// \brief The divergence. \tparam T The number type. \tparam N The dimension. \param a The first Gaussian.
        /// \param b The second. \return `KL(a ‖ b)`, or NaN.
        template <linalg::scalar T, std::size_t N>
        [[nodiscard]] constexpr T operator()(const tracking::gaussian<T, N>& a, const tracking::gaussian<T, N>& b) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto k = tracking::kl_divergence(a, b);
            return k ? *k : std::numeric_limits<T>::quiet_NaN();
        }
    };

} // namespace snicholls::metric

#endif /* snicholls_core_metric_statistical_hpp */
