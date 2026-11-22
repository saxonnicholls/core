//
//  unscented.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::make_sigma_points`, `make_cubature_points`, `point_transform` and the unscented and cubature Kalman steps: propagating a Gaussian through a nonlinear function by deterministic samples.
/// \ingroup core
///
/// **Overview.** Linearising a strongly nonlinear sensor (a bearing at short range, a camera) loses accuracy. The
/// unscented transform instead pushes `2n + 1` sigma points through the function and refits a Gaussian to them;
/// the cubature rule does the same with `2n` equally weighted points. Neither needs a Jacobian.
///
/// - `make_sigma_points(g, params)`: the scaled sigma points of Julier and Uhlmann, `x ± √(n + λ) Lᵢ`, with
///   `λ = α² (n + κ) - n`, mean weights `λ / (n + λ)` and `1 / (2(n + λ))`, and covariance weight
///   `λ / (n + λ) + 1 - α² + β` at the centre. `unscented_parameters` defaults to Stone Soup's: `α = 0.5`, `β = 2`,
///   `κ = 3 - n`.
/// - `make_cubature_points(g, α)`: `x ± √(α n) Lᵢ` (Arasaratnam and Haykin), equal weights.
/// - `point_transform<Kinds>(points, f, noise)`: the mean (circular in angle components), covariance (plus the
///   additive noise) and cross-covariance of `f` over the points.
/// - `unscented_predict(prior, model, dt)`, `unscented_predict_measurement(prediction, sensor)`, and the cubature
///   forms; the update is `tracking::update<S::kinds>(prediction, z, mp)` from `tracking/kalman.hpp`.
///
/// \snippet tracking/unscented_demo.cpp unscented
///
/// **Relationships.** `tracking/gaussian.hpp`, `tracking/kalman.hpp` (`measurement_prediction`, `update`),
/// `tracking/transition.hpp`, `tracking/measurement.hpp`, `geometry/angle.hpp` (`weighted_mean`, `residual`). Stone
/// Soup's `gauss2sigma`, `unscented_transform`, `gauss2cubature`, `cubature_transform`, `UnscentedKalmanPredictor`,
/// `UnscentedKalmanUpdater`, `CubatureKalmanPredictor`, `CubatureKalmanUpdater`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types and a `noexcept` function, no
/// allocation: points live in `std::array`s of compile-time size. Exact for linear functions, where it reproduces
/// the Kalman filter.
///
/// **Errors.** `linalg::error::not_positive_definite` when the input covariance has no Cholesky factor.
///
/// **Performance.** Not benchmarked: `2n + 1` evaluations of the function and one Cholesky.
///
/// **Tests and demos.** `tests/tracking/unscented_test.cpp`, `demos/tracking/unscented_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<optional>`, `linalg/fixed.hpp`, `geometry/angle.hpp`,
/// `tracking/gaussian.hpp`, `tracking/kalman.hpp`, `result.hpp`.
///
/// **Known issues.** With the default `κ = 3 - n` and `n > 3` the centre mean weight is negative, so the predicted
/// covariance can lose definiteness for a strongly nonlinear function; the remedy is the
/// caller's: raise `α`, or use cubature points, whose weights are all positive.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_unscented_hpp
#define snicholls_core_tracking_unscented_hpp

#include <snicholls/core/geometry/angle.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/kalman.hpp>

#include <array>
#include <cstddef>
#include <optional>

namespace snicholls::tracking {

    /// \brief The unscented transform's scaling: Stone Soup's defaults.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct unscented_parameters {
        T alpha = T(0.5);          ///< The spread of the points about the mean.
        T beta = T(2);             ///< Prior knowledge of the distribution: 2 is optimal for a Gaussian.
        std::optional<T> kappa{};  ///< The secondary spread; empty for `3 - n`.
    };

    /// \brief A set of weighted points that represent a Gaussian.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \tparam P The number of points.
    template <linalg::scalar T, std::size_t N, std::size_t P>
    struct point_set {
        std::array<linalg::fixed_vector<T, N>, P> points{};  ///< The points; the first is the mean for sigma points.
        std::array<T, P> mean_weights{};                     ///< The weights of the mean.
        std::array<T, P> covariance_weights{};               ///< The weights of the covariance.
    };

    /// \brief The `2N + 1` scaled sigma points of a Gaussian.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param g The Gaussian.
    /// \param params The scaling.
    /// \return The points and weights, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<point_set<T, N, 2 * N + 1>, linalg::error> make_sigma_points(const gaussian<T, N>& g, const unscented_parameters<T>& params = {}) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(g.covariance);
        if (!l)
            return err(l.unwrap_err_unchecked());
        const T n = static_cast<T>(N);
        const T kappa = params.kappa ? *params.kappa : T(3) - n;
        const T a2 = params.alpha * params.alpha;
        const T lambda = a2 * (n + kappa) - n;
        const T c = n + lambda;
        const T root = static_cast<T>(math::sqrt(c));
        point_set<T, N, 2 * N + 1> s{};
        s.points[0] = g.mean;
        for (std::size_t i = 0; i < N; ++i) {
            linalg::fixed_vector<T, N> li{};
            for (std::size_t r = 0; r < N; ++r)
                li[r] = (*l)(r, i) * root;
            s.points[1 + i] = g.mean + li;
            s.points[1 + N + i] = g.mean - li;
        }
        for (std::size_t p = 0; p < 2 * N + 1; ++p)
            s.mean_weights[p] = s.covariance_weights[p] = T(0.5) / c;
        s.mean_weights[0] = lambda / c;
        s.covariance_weights[0] = lambda / c + (T(1) - a2 + params.beta);
        return ok(s);
    }

    /// \brief The `2N` cubature points of a Gaussian, `x ± √(α N) Lᵢ`, each of weight `1 / 2N` (`1 / (2 N α)` for the covariance).
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param g The Gaussian.
    /// \param alpha The spread (Stone Soup's `alpha`), 1 for the third-degree spherical-radial rule.
    /// \return The points and weights, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<point_set<T, N, 2 * N>, linalg::error> make_cubature_points(const gaussian<T, N>& g, T alpha = T(1)) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(g.covariance);
        if (!l)
            return err(l.unwrap_err_unchecked());
        const T root = static_cast<T>(math::sqrt(alpha * static_cast<T>(N)));
        point_set<T, N, 2 * N> s{};
        for (std::size_t i = 0; i < N; ++i) {
            linalg::fixed_vector<T, N> li{};
            for (std::size_t r = 0; r < N; ++r)
                li[r] = (*l)(r, i) * root;
            s.points[i] = g.mean + li;
            s.points[N + i] = g.mean - li;
        }
        for (std::size_t p = 0; p < 2 * N; ++p) {
            s.mean_weights[p] = T(1) / static_cast<T>(2 * N);
            s.covariance_weights[p] = T(1) / (static_cast<T>(2 * N) * alpha);
        }
        return ok(s);
    }

    /// \brief A Gaussian pushed through a function by points: mean, covariance and cross-covariance.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The input dimension.
    /// \tparam M The output dimension.
    template <linalg::scalar T, std::size_t N, std::size_t M>
    struct transformed {
        linalg::fixed_vector<T, M> mean{};                 ///< The output mean.
        linalg::fixed_matrix<T, M, M> covariance{};        ///< The output covariance, noise included.
        linalg::fixed_matrix<T, N, M> cross_covariance{};  ///< The input-output cross-covariance.
    };

    /// \brief The weighted mean, covariance and cross-covariance of `f` over a point set; angle components (as
    /// `Kinds` says) are averaged on the circle and differenced the short way round.
    /// \ingroup core
    /// \tparam Kinds The output's `angle_kinds<M>`.
    /// \tparam M The output dimension.
    /// \tparam F The function, `fixed_vector<T, N>` to `fixed_vector<T, M>`.
    /// \tparam T The number type.
    /// \tparam N The input dimension.
    /// \tparam P The number of points.
    /// \param s The points.
    /// \param f The function.
    /// \param noise Additive noise covariance on the output.
    /// \return The transformed moments.
    template <auto Kinds, std::size_t M, typename F, linalg::scalar T, std::size_t N, std::size_t P>
    [[nodiscard]] constexpr transformed<T, N, M> point_transform(const point_set<T, N, P>& s, F&& f, const linalg::fixed_matrix<T, M, M>& noise)
    {
        std::array<linalg::fixed_vector<T, M>, P> y{};
        for (std::size_t p = 0; p < P; ++p)
            y[p] = f(s.points[p]);
        transformed<T, N, M> t{};
        t.mean = geometry::weighted_mean<Kinds>(y, s.mean_weights);
        linalg::fixed_vector<T, N> xm{};
        for (std::size_t p = 0; p < P; ++p)
            xm += s.points[p] * s.mean_weights[p];
        for (std::size_t p = 0; p < P; ++p) {
            const auto dy = geometry::residual<Kinds>(y[p], t.mean);
            t.covariance += linalg::outer(dy, dy) * s.covariance_weights[p];
            t.cross_covariance += linalg::outer(s.points[p] - xm, dy) * s.covariance_weights[p];
        }
        t.covariance = linalg::symmetrise(t.covariance + noise);
        return t;
    }

    /// \brief The unscented Kalman prediction: the prior's sigma points through the motion model, plus `Q(dt)`.
    /// \ingroup core
    /// \tparam M A `transition_model`.
    /// \param prior The state now.
    /// \param model The motion model.
    /// \param dt The time step, seconds.
    /// \param params The scaling.
    /// \return The prediction, or `linalg::error::not_positive_definite`.
    template <transition_model M>
    [[nodiscard]] constexpr result<gaussian<typename M::value_type, M::dim>, linalg::error> unscented_predict(const gaussian<typename M::value_type, M::dim>& prior, const M& model, typename M::value_type dt,
                                                                                                            const unscented_parameters<typename M::value_type>& params = {})
    {
        const auto s = make_sigma_points(prior, params);
        if (!s)
            return err(s.unwrap_err_unchecked());
        const auto t = point_transform<geometry::all_linear<M::dim>, M::dim>(*s, [&](const auto& x) { return model.apply(x, dt); }, model.covariance(dt));
        return ok(gaussian<typename M::value_type, M::dim>{t.mean, t.covariance});
    }

    /// \brief The unscented measurement prediction: the predicted state's sigma points through the sensor, plus `R`.
    /// \ingroup core
    /// \tparam S A `measurement_model`.
    /// \param prediction The predicted state.
    /// \param sensor The sensor model.
    /// \param params The scaling.
    /// \return The measurement prediction for `update<S::kinds>`, or `linalg::error::not_positive_definite`.
    template <measurement_model S>
    [[nodiscard]] constexpr result<measurement_prediction<typename S::value_type, S::state_dim, S::measurement_dim>, linalg::error>
    unscented_predict_measurement(const gaussian<typename S::value_type, S::state_dim>& prediction, const S& sensor, const unscented_parameters<typename S::value_type>& params = {})
    {
        const auto s = make_sigma_points(prediction, params);
        if (!s)
            return err(s.unwrap_err_unchecked());
        const auto t = point_transform<S::kinds, S::measurement_dim>(*s, [&](const auto& x) { return sensor.measure(x); }, sensor.covariance());
        return ok(measurement_prediction<typename S::value_type, S::state_dim, S::measurement_dim>{t.mean, t.covariance, t.cross_covariance});
    }

    /// \brief The cubature Kalman prediction.
    /// \ingroup core
    /// \tparam M A `transition_model`.
    /// \param prior The state now.
    /// \param model The motion model.
    /// \param dt The time step, seconds.
    /// \param alpha The cubature spread.
    /// \return The prediction, or `linalg::error::not_positive_definite`.
    template <transition_model M>
    [[nodiscard]] constexpr result<gaussian<typename M::value_type, M::dim>, linalg::error> cubature_predict(const gaussian<typename M::value_type, M::dim>& prior, const M& model, typename M::value_type dt,
                                                                                                           typename M::value_type alpha = 1)
    {
        const auto s = make_cubature_points(prior, alpha);
        if (!s)
            return err(s.unwrap_err_unchecked());
        const auto t = point_transform<geometry::all_linear<M::dim>, M::dim>(*s, [&](const auto& x) { return model.apply(x, dt); }, model.covariance(dt));
        return ok(gaussian<typename M::value_type, M::dim>{t.mean, t.covariance});
    }

    /// \brief The cubature measurement prediction.
    /// \ingroup core
    /// \tparam S A `measurement_model`.
    /// \param prediction The predicted state.
    /// \param sensor The sensor model.
    /// \param alpha The cubature spread.
    /// \return The measurement prediction for `update<S::kinds>`, or `linalg::error::not_positive_definite`.
    template <measurement_model S>
    [[nodiscard]] constexpr result<measurement_prediction<typename S::value_type, S::state_dim, S::measurement_dim>, linalg::error>
    cubature_predict_measurement(const gaussian<typename S::value_type, S::state_dim>& prediction, const S& sensor, typename S::value_type alpha = 1)
    {
        const auto s = make_cubature_points(prediction, alpha);
        if (!s)
            return err(s.unwrap_err_unchecked());
        const auto t = point_transform<S::kinds, S::measurement_dim>(*s, [&](const auto& x) { return sensor.measure(x); }, sensor.covariance());
        return ok(measurement_prediction<typename S::value_type, S::state_dim, S::measurement_dim>{t.mean, t.covariance, t.cross_covariance});
    }

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_unscented_hpp */
