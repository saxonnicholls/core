//
//  kalman.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::predict`, `update`, `predict_measurement` and friends: the Kalman and extended Kalman filter steps, the information form, the Rauch-Tung-Striebel smoothing step, the alpha-beta filter and Fisher information.
/// \ingroup core
///
/// **Overview.** The filter's two steps, as functions of fixed-size values:
///
/// - `predict(prior, model, dt)`: `x ← f(x)`, `P ← F P Fᵀ + Q`. A `linear_transition` uses `F = matrix(dt)`; any
///   other `transition_model` is linearised at the prior mean (the extended Kalman filter). `predict(prior, F, Q)`
///   takes the matrices already formed, for many targets stepped over the same `dt`. `predict(prior, model,
///   dt, B, u)` adds a known control input `B u` (a current that carries a swimmer, an engine command).
/// - `predict_measurement(prediction, sensor)`: the predicted measurement `ẑ`, its covariance `S = H P Hᵀ + R`,
///   and the cross-covariance `P Hᵀ`, linear or linearised in the same way.
/// - `update(prediction, z, sensor)`: the posterior, with the innovation `ν = z ⊖ ẑ` (wrapped as the sensor's
///   `kinds` say), `S`, the gain `K`, the squared Mahalanobis distance `νᵀ S⁻¹ ν` and the log-likelihood
///   `log N(ν; 0, S)`, everything a gate and an associator need, from one Cholesky of `S`. `covariance_form::joseph`
///   uses `(I - K H) P (I - K H)ᵀ + K R Kᵀ`, which stays positive definite under rounding; the default is
///   `P - K S Kᵀ` (Stone Soup's default).
/// - `update(prediction, z, measurement_prediction, kinds)`: the same from a measurement prediction made any other
///   way (the unscented transform, cubature).
/// - Information form: `to_information`, `from_information`, `information_update` (`Y ← Y + Hᵀ R⁻¹ H`,
///   `y ← y + Hᵀ R⁻¹ z`: sensors add), `fisher_information(H, R)` and `gdop(J)`, the geometric dilution of
///   precision `√tr(J⁻¹)` that a placement optimiser minimises.
/// - `rts_smooth(filtered, predicted, smoothed_next, F)`: one backward step of the Rauch-Tung-Striebel smoother,
///   also the retrodiction step for an out-of-sequence measurement.
/// - `numerical_jacobian(f, x)`: central differences, for a model without an analytic Jacobian.
/// - `alpha_beta_update<Pos, Vel>(x, z, α, β, dt)`: the fixed-gain filter (Stone Soup's `AlphaBetaUpdater`).
///
/// \snippet tracking/kalman_demo.cpp kalman
///
/// **Relationships.** `tracking/gaussian.hpp` (`gaussian`), `tracking/transition.hpp`, `tracking/measurement.hpp`,
/// `geometry/angle.hpp` (residuals), `linalg/fixed.hpp`. Stone Soup's `KalmanPredictor`, `ExtendedKalmanPredictor`,
/// `KalmanUpdater`, `ExtendedKalmanUpdater`, `InformationKalmanPredictor` / `Updater`, `KalmanSmoother`,
/// `AlphaBetaUpdater`. Group: \ref core.
///
/// **Implementations.** The gain is `K = P Hᵀ S⁻¹` by one Cholesky factorisation of `S`, reused for the distance and
/// the log-likelihood. Posterior covariances are made exactly symmetric.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation, no virtual call.
/// A predict on a 4-state constant-velocity target is a few hundred multiply-adds.
///
/// **Errors.** `update`, `to_information`, `from_information`, `information_update`, `fisher_information`, `gdop`
/// and `rts_smooth` return `linalg::error::not_positive_definite` when a matrix they must factorise is not.
///
/// **Performance.** A 2-D constant-velocity predict and position update, the whole `kalman_update` record written,
/// against the same arithmetic in the same order written by hand over raw arrays, after a bit-for-bit check, 4096
/// tracks (`benchmarks/tracking/kalman_benchmark.cpp`; Release, Apple clang 17, Xeon W-3245, 2026-11-23): 7.9 million
/// tracks a second (by hand 11.0). The predict is at parity; the update is slower by the copies of its 36-double
/// record through `result` and of the intermediate measurement prediction. See Known issues.
///
/// **Tests and demos.** `tests/tracking/kalman_test.cpp`, `demos/tracking/kalman_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `linalg/fixed.hpp`, `geometry/angle.hpp`,
/// `tracking/gaussian.hpp`, `tracking/measurement.hpp`, `tracking/transition.hpp`, `result.hpp`.
///
/// **Known issues.** Stone Soup's extended filters differentiate numerically; ours use each model's analytic
/// Jacobian, so results agree to the numerical derivative's accuracy (about 1e-7 relative), not to rounding. The
/// update runs at 0.72 of the hand-written loop's speed (Performance): closing the gap needs a form that writes
/// into the caller's storage, planned with the batched device forms.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_kalman_hpp
#define snicholls_core_tracking_kalman_hpp

#include <snicholls/core/geometry/angle.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/measurement.hpp>
#include <snicholls/core/tracking/transition.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace snicholls::tracking {

    /// \brief How `update` forms the posterior covariance.
    /// \ingroup core
    enum class covariance_form : std::uint8_t {
        standard = 0,  ///< `P - K S Kᵀ`, then symmetrised.
        joseph = 1,    ///< `(I - K H) P (I - K H)ᵀ + K R Kᵀ`: positive definite under rounding, a little dearer.
    };

    /// \brief The prediction of a measurement: its mean, covariance and cross-covariance with the state.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam M The measurement dimension.
    template <linalg::scalar T, std::size_t N, std::size_t M>
    struct measurement_prediction {
        linalg::fixed_vector<T, M> mean{};                ///< `ẑ`.
        linalg::fixed_matrix<T, M, M> covariance{};       ///< `S`, the innovation covariance, noise included.
        linalg::fixed_matrix<T, N, M> cross_covariance{}; ///< `P Hᵀ`, the state-measurement cross-covariance.
    };

    /// \brief What one update produced.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam M The measurement dimension.
    template <linalg::scalar T, std::size_t N, std::size_t M>
    struct kalman_update {
        gaussian<T, N> posterior{};                     ///< The updated state.
        linalg::fixed_vector<T, M> innovation{};        ///< `ν = z ⊖ ẑ`, angles wrapped.
        linalg::fixed_matrix<T, M, M> innovation_covariance{};  ///< `S`.
        linalg::fixed_matrix<T, N, M> gain{};           ///< `K`.
        T distance2{};                                  ///< `νᵀ S⁻¹ ν`, for a χ² gate.
        T log_likelihood{};                             ///< `log N(ν; 0, S)`.
    };

    /// \brief The predicted state of a linear model: `F x`, `F P Fᵀ + Q`.
    /// \ingroup core
    /// \tparam M A `linear_transition`.
    /// \param prior The state now.
    /// \param model The motion model.
    /// \param dt The time step, seconds.
    /// \return The prediction.
    template <linear_transition M>
    [[nodiscard]] constexpr gaussian<typename M::value_type, M::dim> predict(const gaussian<typename M::value_type, M::dim>& prior, const M& model, typename M::value_type dt) noexcept(linalg::detail::nothrow_scalar<typename M::value_type>)
    {
        const auto f = model.matrix(dt);
        return {f * prior.mean, linalg::sandwich(f, prior.covariance) + model.covariance(dt)};
    }

    /// \brief The predicted state from a transition matrix and process noise already in hand: `F x`, `F P Fᵀ + Q`.
    /// Many targets stepped over the same `dt` share one `F` and one `Q`; form them once with `model.matrix(dt)` and
    /// `model.covariance(dt)`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \param prior The state now.
    /// \param f The transition matrix `F(dt)`.
    /// \param q The process noise `Q(dt)`.
    /// \return The prediction.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr gaussian<T, N> predict(const gaussian<T, N>& prior, const linalg::fixed_matrix<T, N, N>& f, const linalg::fixed_matrix<T, N, N>& q) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return {f * prior.mean, linalg::sandwich(f, prior.covariance) + q};
    }

    /// \brief The predicted state of a nonlinear model, linearised at the prior mean (the extended Kalman filter):
    /// `f(x)`, `F P Fᵀ + Q` with `F = ∂f/∂x`.
    /// \ingroup core
    /// \tparam M A `transition_model`.
    /// \param prior The state now.
    /// \param model The motion model.
    /// \param dt The time step, seconds.
    /// \return The prediction.
    template <transition_model M>
    [[nodiscard]] constexpr gaussian<typename M::value_type, M::dim> predict(const gaussian<typename M::value_type, M::dim>& prior, const M& model, typename M::value_type dt) noexcept(linalg::detail::nothrow_scalar<typename M::value_type>)
    {
        const auto f = model.jacobian(prior.mean, dt);
        return {model.apply(prior.mean, dt), linalg::sandwich(f, prior.covariance) + model.covariance(dt)};
    }

    /// \brief The predicted state with a known control input: `f(x) + B u`, `F P Fᵀ + Q + B Q_u Bᵀ`.
    /// \ingroup core
    /// \tparam M A `transition_model`.
    /// \tparam C The control dimension.
    /// \param prior The state now.
    /// \param model The motion model.
    /// \param dt The time step, seconds.
    /// \param b The control matrix (for a velocity input over `dt`, `dt` in the position rows).
    /// \param u The control input.
    /// \param control_covariance `Q_u`, the input's uncertainty; zero if it is exact.
    /// \return The prediction.
    template <transition_model M, std::size_t C>
    [[nodiscard]] constexpr gaussian<typename M::value_type, M::dim> predict(const gaussian<typename M::value_type, M::dim>& prior, const M& model, typename M::value_type dt,
                                                                           const linalg::fixed_matrix<typename M::value_type, M::dim, C>& b, const linalg::fixed_vector<typename M::value_type, C>& u,
                                                                           const linalg::fixed_matrix<typename M::value_type, C, C>& control_covariance = {}) noexcept(linalg::detail::nothrow_scalar<typename M::value_type>)
    {
        auto g = predict(prior, model, dt);
        g.mean += b * u;
        g.covariance += linalg::sandwich(b, control_covariance);
        return g;
    }

    /// \brief The predicted measurement of a linear sensor: `H x`, `H P Hᵀ + R`, `P Hᵀ`.
    /// \ingroup core
    /// \tparam S A `linear_measurement_model`.
    /// \param prediction The predicted state.
    /// \param sensor The sensor model.
    /// \return The measurement prediction.
    template <linear_measurement_model S>
    [[nodiscard]] constexpr measurement_prediction<typename S::value_type, S::state_dim, S::measurement_dim> predict_measurement(const gaussian<typename S::value_type, S::state_dim>& prediction,
                                                                                                                             const S& sensor) noexcept(linalg::detail::nothrow_scalar<typename S::value_type>)
    {
        const auto h = sensor.matrix();
        return {sensor.measure(prediction.mean), linalg::sandwich(h, prediction.covariance) + sensor.covariance(), prediction.covariance * linalg::transpose(h)};
    }

    /// \brief The predicted measurement of a nonlinear sensor, linearised at the predicted mean (the extended
    /// Kalman filter): `h(x)`, `H P Hᵀ + R`, `P Hᵀ` with `H = ∂h/∂x`.
    /// \ingroup core
    /// \tparam S A `measurement_model`.
    /// \param prediction The predicted state.
    /// \param sensor The sensor model.
    /// \return The measurement prediction.
    template <measurement_model S>
    [[nodiscard]] constexpr measurement_prediction<typename S::value_type, S::state_dim, S::measurement_dim> predict_measurement(const gaussian<typename S::value_type, S::state_dim>& prediction,
                                                                                                                             const S& sensor) noexcept(linalg::detail::nothrow_scalar<typename S::value_type>)
    {
        const auto h = sensor.jacobian(prediction.mean);
        return {sensor.measure(prediction.mean), linalg::sandwich(h, prediction.covariance) + sensor.covariance(), prediction.covariance * linalg::transpose(h)};
    }

    namespace detail::kalman {
        template <auto Kinds, typename T, std::size_t N, std::size_t M>
        constexpr result<kalman_update<T, N, M>, linalg::error> core(const gaussian<T, N>& prediction, const linalg::fixed_vector<T, M>& z,
                                                                      const measurement_prediction<T, N, M>& mp) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto l = linalg::cholesky(mp.covariance);
            // The update is built inside the result, the only object returned, so it is never copied
            result<kalman_update<T, N, M>, linalg::error> out{std::in_place};
            if (!l) {
                out = err(l.unwrap_err_unchecked());
                return out;
            }
            kalman_update<T, N, M>& u = *out;
            u.innovation = geometry::residual<Kinds>(z, mp.mean);
            u.innovation_covariance = mp.covariance;
            u.gain = linalg::transpose(linalg::cholesky_solve(*l, linalg::transpose(mp.cross_covariance)));
            u.distance2 = whitened_squared_norm(u.innovation, *l);
            const T log_2pi = static_cast<T>(math::log(T(2) * geometry::pi<T>));
            u.log_likelihood = -(u.distance2 + linalg::cholesky_log_det(*l) + static_cast<T>(M) * log_2pi) / T(2);
            u.posterior.mean = prediction.mean + u.gain * u.innovation;
            u.posterior.covariance = linalg::symmetrise(prediction.covariance - linalg::sandwich(u.gain, mp.covariance));
            return out;
        }
    } // namespace detail::kalman

    /// \brief The Kalman update from a measurement prediction made any way (linearised, unscented, cubature).
    /// \ingroup core
    /// \tparam Kinds The measurement's `angle_kinds`, for the innovation.
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam M The measurement dimension.
    /// \param prediction The predicted state.
    /// \param z The measurement.
    /// \param mp The measurement prediction.
    /// \return The update, or `linalg::error::not_positive_definite` if `S` is not.
    template <auto Kinds, linalg::scalar T, std::size_t N, std::size_t M>
    [[nodiscard]] constexpr result<kalman_update<T, N, M>, linalg::error> update(const gaussian<T, N>& prediction, const linalg::fixed_vector<T, M>& z,
                                                                                const measurement_prediction<T, N, M>& mp) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return detail::kalman::core<Kinds>(prediction, z, mp);
    }

    /// \brief The Kalman (or, for a nonlinear sensor, extended Kalman) update.
    /// \ingroup core
    /// \tparam S A `measurement_model`.
    /// \param prediction The predicted state.
    /// \param z The measurement.
    /// \param sensor The sensor model.
    /// \param form How to form the posterior covariance.
    /// \return The update, or `linalg::error::not_positive_definite` if `S` is not.
    template <measurement_model S>
    [[nodiscard]] constexpr result<kalman_update<typename S::value_type, S::state_dim, S::measurement_dim>, linalg::error> update(const gaussian<typename S::value_type, S::state_dim>& prediction,
                                                                                                                              const linalg::fixed_vector<typename S::value_type, S::measurement_dim>& z, const S& sensor,
                                                                                                                              covariance_form form = covariance_form::standard) noexcept(linalg::detail::nothrow_scalar<typename S::value_type>)
    {
        using T = typename S::value_type;
        constexpr std::size_t N = S::state_dim;
        auto u = detail::kalman::core<S::kinds>(prediction, z, predict_measurement(prediction, sensor));
        if (u && form == covariance_form::joseph) {
            const auto h = sensor.jacobian(prediction.mean);
            const auto ikh = linalg::fixed_matrix<T, N, N>::identity() - u->gain * h;
            u->posterior.covariance = linalg::symmetrise(linalg::sandwich(ikh, prediction.covariance) + linalg::sandwich(u->gain, sensor.covariance()));
        }
        return u;
    }

    /// \brief A Gaussian in information form: `Y = P⁻¹`, `y = P⁻¹ x`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    template <linalg::scalar T, std::size_t N>
    struct information_state {
        linalg::fixed_vector<T, N> vector{};      ///< `y = P⁻¹ x`.
        linalg::fixed_matrix<T, N, N> matrix{};   ///< `Y = P⁻¹`, the information (precision) matrix.
    };

    /// \brief A Gaussian in information form. \tparam T The number type. \tparam N The dimension.
    /// \ingroup core
    /// \param g The Gaussian. \return `(P⁻¹ x, P⁻¹)`, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<information_state<T, N>, linalg::error> to_information(const gaussian<T, N>& g) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto y = linalg::spd_inverse(g.covariance);
        if (!y)
            return err(y.unwrap_err_unchecked());
        return ok(information_state<T, N>{*y * g.mean, *y});
    }

    /// \brief A Gaussian from information form. \tparam T The number type. \tparam N The dimension.
    /// \ingroup core
    /// \param s The information state. \return `(Y⁻¹ y, Y⁻¹)`, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<gaussian<T, N>, linalg::error> from_information(const information_state<T, N>& s) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto p = linalg::spd_inverse(s.matrix);
        if (!p)
            return err(p.unwrap_err_unchecked());
        return ok(gaussian<T, N>{*p * s.vector, *p});
    }

    /// \brief The Fisher information of one linear(ised) measurement, `Hᵀ R⁻¹ H`. Summed over sensors, its inverse
    /// is the best covariance the geometry allows.
    /// \tparam T The number type. \tparam M The measurement dimension. \tparam N The state dimension.
    /// \ingroup core
    /// \param h The (Jacobian) matrix. \param r The noise covariance. \return The information, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t M, std::size_t N>
    [[nodiscard]] constexpr result<linalg::fixed_matrix<T, N, N>, linalg::error> fisher_information(const linalg::fixed_matrix<T, M, N>& h, const linalg::fixed_matrix<T, M, M>& r) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(r);
        if (!l)
            return err(l.unwrap_err_unchecked());
        const auto w = linalg::lower_solve(*l, h);  // L⁻¹ H, so Hᵀ R⁻¹ H = wᵀ w
        return ok(linalg::symmetrise(linalg::transpose(w) * w));
    }

    /// \brief The information update: `Y ← Y + Hᵀ R⁻¹ H`, `y ← y + Hᵀ R⁻¹ z`. Independent sensors' updates add.
    /// \tparam T The number type. \tparam N The state dimension. \tparam M The measurement dimension.
    /// \ingroup core
    /// \param s The information state. \param z The measurement. \param h The measurement matrix. \param r The noise covariance.
    /// \return The updated information state, or `linalg::error::not_positive_definite`.
    template <linalg::scalar T, std::size_t N, std::size_t M>
    [[nodiscard]] constexpr result<information_state<T, N>, linalg::error> information_update(const information_state<T, N>& s, const linalg::fixed_vector<T, M>& z,
                                                                                            const linalg::fixed_matrix<T, M, N>& h, const linalg::fixed_matrix<T, M, M>& r) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(r);
        if (!l)
            return err(l.unwrap_err_unchecked());
        const auto w = linalg::lower_solve(*l, h);
        const auto wz = linalg::lower_solve(*l, z);
        return ok(information_state<T, N>{s.vector + linalg::transpose(w) * wz, linalg::symmetrise(s.matrix + linalg::transpose(w) * w)});
    }

    /// \brief The geometric dilution of precision of an information matrix: `√tr(J⁻¹)`.
    /// \tparam T The number type. \tparam N The dimension. \param j The (summed) Fisher information.
    /// \ingroup core
    /// \return GDOP, or `linalg::error::not_positive_definite` when the geometry does not determine the state.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<T, linalg::error> gdop(const linalg::fixed_matrix<T, N, N>& j) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto c = linalg::spd_inverse(j);
        if (!c)
            return err(c.unwrap_err_unchecked());
        return ok(static_cast<T>(math::sqrt(linalg::trace(*c))));
    }

    /// \brief One backward step of the Rauch-Tung-Striebel smoother: with `C = P_f Fᵀ P_p⁻¹`,
    /// `x_s = x_f + C (x_s' - x_p)`, `P_s = P_f + C (P_s' - P_p) Cᵀ`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The dimension.
    /// \param filtered The filtered state at step k.
    /// \param predicted The prediction for step k + 1 made from it.
    /// \param smoothed_next The smoothed state at step k + 1.
    /// \param f The transition matrix (or Jacobian) from k to k + 1.
    /// \return The smoothed state at step k, or `linalg::error::not_positive_definite` if `P_p` is not.
    template <linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr result<gaussian<T, N>, linalg::error> rts_smooth(const gaussian<T, N>& filtered, const gaussian<T, N>& predicted, const gaussian<T, N>& smoothed_next,
                                                                            const linalg::fixed_matrix<T, N, N>& f) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto l = linalg::cholesky(predicted.covariance);
        if (!l)
            return err(l.unwrap_err_unchecked());
        const auto c = linalg::transpose(linalg::cholesky_solve(*l, f * filtered.covariance));  // P_f Fᵀ P_p⁻¹
        return ok(gaussian<T, N>{filtered.mean + c * (smoothed_next.mean - predicted.mean),
                                 linalg::symmetrise(filtered.covariance + c * (smoothed_next.covariance - predicted.covariance) * linalg::transpose(c))});
    }

    /// \brief The Jacobian of `f` at `x` by central differences, step `∛ε · max(|xᵢ|, 1)` per component.
    /// \ingroup core
    /// \tparam M The output dimension.
    /// \tparam F The function, `fixed_vector<T, N>` to `fixed_vector<T, M>`.
    /// \tparam T The number type.
    /// \tparam N The input dimension.
    /// \param f The function.
    /// \param x The point.
    /// \return `∂f/∂x`, accurate to about `ε^(2/3)`.
    template <std::size_t M, typename F, linalg::scalar T, std::size_t N>
    [[nodiscard]] constexpr linalg::fixed_matrix<T, M, N> numerical_jacobian(F&& f, const linalg::fixed_vector<T, N>& x)
    {
        const T step = static_cast<T>(math::pow(std::numeric_limits<T>::epsilon(), T(1) / T(3)));
        linalg::fixed_matrix<T, M, N> j{};
        for (std::size_t k = 0; k < N; ++k) {
            const T ax = x[k] < T(0) ? -x[k] : x[k];
            const T h = step * (ax > T(1) ? ax : T(1));
            auto xp = x, xm = x;
            xp[k] += h;
            xm[k] -= h;
            const linalg::fixed_vector<T, M> d = (f(xp) - f(xm)) / ((xp[k] - xm[k]));
            for (std::size_t i = 0; i < M; ++i)
                j(i, k) = d[i];
        }
        return j;
    }

    /// \brief The alpha-beta update: `p ← p + α (z - p)`, `v ← v + (β / dt) (z - p)` on the mapped position and
    /// velocity components (Stone Soup's `AlphaBetaUpdater`).
    /// \ingroup core
    /// \tparam Pos The state's position components, in measurement order.
    /// \tparam Vel The matching velocity components.
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam D The measurement dimension.
    /// \param x The predicted state.
    /// \param z The position measurement.
    /// \param alpha The position gain.
    /// \param beta The velocity gain.
    /// \param dt The time since the last update, seconds.
    /// \return The updated state.
    template <std::array Pos, std::array Vel, linalg::scalar T, std::size_t N, std::size_t D>
    [[nodiscard]] constexpr linalg::fixed_vector<T, N> alpha_beta_update(const linalg::fixed_vector<T, N>& x, const linalg::fixed_vector<T, D>& z, T alpha, T beta, T dt) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        static_assert(Pos.size() == D && Vel.size() == D, "snicholls::tracking::alpha_beta_update: one position and one velocity index per measured component");
        auto y = x;
        for (std::size_t i = 0; i < D; ++i) {
            const auto p = static_cast<std::size_t>(Pos[i]), v = static_cast<std::size_t>(Vel[i]);
            const T r = z[i] - x[p];
            y[p] = x[p] + alpha * r;
            y[v] = x[v] + beta / dt * r;
        }
        return y;
    }

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_kalman_hpp */
