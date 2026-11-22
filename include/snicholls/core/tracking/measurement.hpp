//
//  measurement.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::linear_gaussian`, `bearing_range`, `elevation_bearing_range` and the other sensor models: what a sensor measures of a state, its Jacobian, its noise, and which components are angles.
/// \ingroup core
///
/// **Overview.** A measurement model maps a state to what a sensor reports. Every model here satisfies
/// `measurement_model`:
///
/// - `state_dim`, `measurement_dim`, `value_type`, and `kinds`, an `angle_kinds<measurement_dim>` saying which
///   components are bearings or elevations (so residuals wrap);
/// - `measure(x)`: the noise-free measurement `h(x)`;
/// - `jacobian(x)`: `∂h/∂x`, analytic;
/// - `covariance()`: the measurement noise `R`.
///
/// Which state components are the position (and velocity) is a template argument, a `std::array` of indices
/// (Stone Soup's `mapping`), so the gather is resolved at compile time. A sensor's pose is a translation and a
/// rotation (`geometry::sensor_rotation`, identity by default) applied as `R (x - t)`.
///
/// | Model | Measures | Kinds | Reversible |
/// |---|---|---|---|
/// | `linear_gaussian<T, N, Map>` | the mapped components | linear | (`matrix()` is `H`) |
/// | `bearing<T, N, Map>` (2-D) | `φ` | bearing | no |
/// | `bearing_range<T, N, Map>` (2-D) | `(φ, ρ)` | bearing, linear | yes |
/// | `elevation_bearing<T, N, Map>` (3-D) | `(θ, φ)` | elevation, bearing | no |
/// | `elevation_bearing_range<T, N, Map>` (3-D) | `(θ, φ, ρ)` | elevation, bearing, linear | yes |
/// | `bearing_range_rate<T, N, Map, VelMap>` (2-D or 3-D) | `(φ, ρ, ρ̇)` | bearing, linear, linear | no |
///
/// The functions and component order are Stone Soup's (`LinearGaussian`, `Cartesian2DToBearing`,
/// `CartesianToBearingRange`, `CartesianToElevationBearing`, `CartesianToElevationBearingRange`,
/// `CartesianToBearingRangeRate`, `CartesianToBearingRangeRate2D`). Where Stone Soup differentiates numerically,
/// these differentiate exactly.
///
/// \snippet tracking/measurement_demo.cpp measurement
///
/// **Relationships.** `linalg/fixed.hpp`, `geometry/angle.hpp` (`angle_kinds`), `geometry/coordinates.hpp`;
/// `tracking/kalman.hpp` and `tracking/unscented.hpp` update with them. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. Every model is an
/// aggregate, trivially copyable. At the sensor's own position (zero range) the angles are 0 and the Jacobian's
/// angle rows are zero, not NaN.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked on their own: a gather (linear) or a rotation, a square root and an `atan2`.
///
/// **Tests and demos.** `tests/tracking/measurement_test.cpp`, `demos/tracking/measurement_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `linalg/fixed.hpp`, `geometry/angle.hpp`,
/// `geometry/coordinates.hpp`, `math/constexpr_math.hpp`.
///
/// **Known issues.** The 2-D models use the upper-left 2 x 2 block of the rotation, as Stone Soup does (a roll or
/// pitch is ignored in the plane).
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_measurement_hpp
#define snicholls_core_tracking_measurement_hpp

#include <snicholls/core/geometry/angle.hpp>
#include <snicholls/core/geometry/coordinates.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

#include <array>
#include <concepts>
#include <cstddef>

namespace snicholls::tracking {

    /// \brief A sensor model: `state_dim`, `measurement_dim`, `kinds`, `measure(x)`, `jacobian(x)`, `covariance()`.
    /// \ingroup core
    /// \tparam M The model.
    template <typename M>
    concept measurement_model = requires(const M& m, const linalg::fixed_vector<typename M::value_type, M::state_dim>& x) {
        { M::kinds } -> std::convertible_to<geometry::angle_kinds<M::measurement_dim>>;
        { m.measure(x) } -> std::same_as<linalg::fixed_vector<typename M::value_type, M::measurement_dim>>;
        { m.jacobian(x) } -> std::same_as<linalg::fixed_matrix<typename M::value_type, M::measurement_dim, M::state_dim>>;
        { m.covariance() } -> std::same_as<linalg::fixed_matrix<typename M::value_type, M::measurement_dim, M::measurement_dim>>;
    };

    /// \brief A linear sensor model: a `measurement_model` with `matrix()`, `H`, and `measure(x) = H x`.
    /// \ingroup core
    /// \tparam M The model.
    template <typename M>
    concept linear_measurement_model = measurement_model<M> && requires(const M& m) {
        { m.matrix() } -> std::same_as<linalg::fixed_matrix<typename M::value_type, M::measurement_dim, M::state_dim>>;
    };

    /// \brief A model whose measurement determines a state's mapped components: `inverse(z)`.
    /// \ingroup core
    /// \tparam M The model.
    template <typename M>
    concept reversible_measurement_model = measurement_model<M> && requires(const M& m, const linalg::fixed_vector<typename M::value_type, M::measurement_dim>& z) {
        { m.inverse(z) } -> std::same_as<linalg::fixed_vector<typename M::value_type, M::state_dim>>;
    };

    namespace detail::measurement {
        template <std::size_t N, auto Map>
        constexpr bool valid_map() noexcept
        {
            for (std::size_t i = 0; i < Map.size(); ++i)
                if (!(static_cast<long long>(Map[i]) >= 0) || static_cast<std::size_t>(Map[i]) >= N)
                    return false;
            return true;
        }

        template <typename T, std::size_t D, std::size_t N, auto Map>
        constexpr linalg::fixed_vector<T, D> gather(const linalg::fixed_vector<T, N>& x) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            linalg::fixed_vector<T, D> p{};
            for (std::size_t i = 0; i < D; ++i)
                p[i] = x[static_cast<std::size_t>(Map[i])];
            return p;
        }

        template <typename T, std::size_t D>
        constexpr linalg::fixed_matrix<T, D, D> upper_left(const linalg::fixed_matrix<T, 3, 3>& r) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return linalg::block<0, 0, D, D>(r);
        }

        // Writes the M x D derivative with respect to the mapped components into the M x N Jacobian
        template <std::size_t N, auto Map, typename T, std::size_t M, std::size_t D>
        constexpr linalg::fixed_matrix<T, M, N> scatter(const linalg::fixed_matrix<T, M, D>& d) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            linalg::fixed_matrix<T, M, N> j{};
            for (std::size_t r = 0; r < M; ++r)
                for (std::size_t k = 0; k < D; ++k)
                    j(r, static_cast<std::size_t>(Map[k])) += d(r, k);
            return j;
        }

        // d(bearing)/dp and d(range)/dp for p in the plane (rows), zero at the origin
        template <typename T>
        struct polar_rows {
            T phi, rho;
            linalg::fixed_matrix<T, 1, 2> dphi, drho;
        };
        template <typename T>
        constexpr polar_rows<T> polar(const linalg::fixed_vector<T, 2>& p) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const T r2 = p[0] * p[0] + p[1] * p[1];
            if (r2 == T(0))
                return {T(0), T(0), {}, {}};
            const T r = static_cast<T>(math::sqrt(r2));
            return {static_cast<T>(math::atan2(p[1], p[0])), r, {{-p[1] / r2, p[0] / r2}}, {{p[0] / r, p[1] / r}}};
        }

        // (elevation, bearing, range) and their derivatives for p in space
        template <typename T>
        struct sphere_rows {
            T theta, phi, rho;
            linalg::fixed_matrix<T, 1, 3> dtheta, dphi, drho;
        };
        template <typename T>
        constexpr sphere_rows<T> sphere(const linalg::fixed_vector<T, 3>& p) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const T h2 = p[0] * p[0] + p[1] * p[1];
            const T r2 = h2 + p[2] * p[2];
            if (r2 == T(0))
                return {T(0), T(0), T(0), {}, {}, {}};
            const T r = static_cast<T>(math::sqrt(r2)), h = static_cast<T>(math::sqrt(h2));
            sphere_rows<T> s{static_cast<T>(math::asin(p[2] / r)), h2 == T(0) ? T(0) : static_cast<T>(math::atan2(p[1], p[0])), r, {}, {}, {{p[0] / r, p[1] / r, p[2] / r}}};
            if (h2 > T(0)) {
                s.dtheta = {{-p[0] * p[2] / (r2 * h), -p[1] * p[2] / (r2 * h), h / r2}};
                s.dphi = {{-p[1] / h2, p[0] / h2, T(0)}};
            }
            return s;
        }
    } // namespace detail::measurement

    /// \brief The mapped state components plus Gaussian noise: `z = H x + v` (Stone Soup's `LinearGaussian`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam Map The measured state components, in measurement order.
    template <linalg::scalar T, std::size_t N, std::array Map>
    struct linear_gaussian {
        static_assert(detail::measurement::valid_map<N, Map>(), "snicholls::tracking::linear_gaussian: a mapped index is outside the state");
        using value_type = T;                                         ///< The number type.
        static constexpr std::size_t state_dim = N;                   ///< The state dimension.
        static constexpr std::size_t measurement_dim = Map.size();    ///< The measurement dimension.
        static constexpr geometry::angle_kinds<measurement_dim> kinds{};  ///< All linear.
        linalg::fixed_matrix<T, measurement_dim, measurement_dim> noise{};  ///< `R`.

        /// \brief `H`, built at compile time. \return The 0/1 selection matrix.
        [[nodiscard]] static constexpr linalg::fixed_matrix<T, measurement_dim, N> matrix() noexcept(linalg::detail::nothrow_scalar<T>)
        {
            linalg::fixed_matrix<T, measurement_dim, N> h{};
            for (std::size_t i = 0; i < measurement_dim; ++i)
                h(i, static_cast<std::size_t>(Map[i])) = T(1);
            return h;
        }
        /// \brief `H x`, a gather. \param x The state. \return The measurement.
        [[nodiscard]] constexpr linalg::fixed_vector<T, measurement_dim> measure(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return detail::measurement::gather<T, measurement_dim, N, Map>(x);
        }
        /// \brief `H`. \param x The state (unused). \return `H`.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, measurement_dim, N> jacobian(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            static_cast<void>(x);
            return matrix();
        }
        /// \brief `R`. \return The noise covariance.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, measurement_dim, measurement_dim> covariance() const noexcept { return noise; }
        /// \brief A state whose mapped components are `z` and the rest zero. \param z The measurement. \return The state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, N> inverse(const linalg::fixed_vector<T, measurement_dim>& z) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            linalg::fixed_vector<T, N> x{};
            for (std::size_t i = 0; i < measurement_dim; ++i)
                x[static_cast<std::size_t>(Map[i])] = z[i];
            return x;
        }
    };

    /// \brief The bearing of a target in the plane (Stone Soup's `Cartesian2DToBearing`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam Map The state's (x, y) components.
    template <linalg::scalar T, std::size_t N, std::array Map>
    struct bearing {
        static_assert(Map.size() == 2 && detail::measurement::valid_map<N, Map>(), "snicholls::tracking::bearing: map (x, y) inside the state");
        using value_type = T;                          ///< The number type.
        static constexpr std::size_t state_dim = N;    ///< The state dimension.
        static constexpr std::size_t measurement_dim = 1;  ///< The measurement dimension.
        static constexpr geometry::angle_kinds<1> kinds{geometry::angle_kind::bearing};  ///< A bearing.
        linalg::fixed_matrix<T, 1, 1> noise{};          ///< `R`, the bearing variance.
        linalg::fixed_vector<T, 2> translation{};       ///< The sensor's position.
        linalg::fixed_matrix<T, 3, 3> rotation = linalg::fixed_matrix<T, 3, 3>::identity();  ///< World to sensor.

        /// \brief `(φ)`. \param x The state. \return The bearing in the sensor's frame.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 1> measure(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto p = detail::measurement::upper_left<T, 2>(rotation) * (detail::measurement::gather<T, 2, N, Map>(x) - translation);
            return {{detail::measurement::polar(p).phi}};
        }
        /// \brief `∂φ/∂x`. \param x The state. \return The Jacobian.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 1, N> jacobian(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto a = detail::measurement::upper_left<T, 2>(rotation);
            const auto pr = detail::measurement::polar(a * (detail::measurement::gather<T, 2, N, Map>(x) - translation));
            return detail::measurement::scatter<N, Map>(pr.dphi * a);
        }
        /// \brief `R`. \return The noise covariance.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 1, 1> covariance() const noexcept { return noise; }
    };

    /// \brief The bearing and range of a target in the plane, `(φ, ρ)` (Stone Soup's `CartesianToBearingRange`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam Map The state's (x, y) components.
    template <linalg::scalar T, std::size_t N, std::array Map>
    struct bearing_range {
        static_assert(Map.size() == 2 && detail::measurement::valid_map<N, Map>(), "snicholls::tracking::bearing_range: map (x, y) inside the state");
        using value_type = T;                          ///< The number type.
        static constexpr std::size_t state_dim = N;    ///< The state dimension.
        static constexpr std::size_t measurement_dim = 2;  ///< The measurement dimension.
        static constexpr geometry::angle_kinds<2> kinds{geometry::angle_kind::bearing, geometry::angle_kind::linear};  ///< Bearing, range.
        linalg::fixed_matrix<T, 2, 2> noise{};          ///< `R`.
        linalg::fixed_vector<T, 2> translation{};       ///< The sensor's position.
        linalg::fixed_matrix<T, 3, 3> rotation = linalg::fixed_matrix<T, 3, 3>::identity();  ///< World to sensor.

        /// \brief `(φ, ρ)`. \param x The state. \return The measurement.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 2> measure(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto pr = detail::measurement::polar(detail::measurement::upper_left<T, 2>(rotation) * (detail::measurement::gather<T, 2, N, Map>(x) - translation));
            return {{pr.phi, pr.rho}};
        }
        /// \brief `∂(φ, ρ)/∂x`. \param x The state. \return The Jacobian.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 2, N> jacobian(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto a = detail::measurement::upper_left<T, 2>(rotation);
            const auto pr = detail::measurement::polar(a * (detail::measurement::gather<T, 2, N, Map>(x) - translation));
            return detail::measurement::scatter<N, Map>(linalg::stack(pr.dphi, pr.drho) * a);
        }
        /// \brief `R`. \return The noise covariance.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 2, 2> covariance() const noexcept { return noise; }
        /// \brief The state at `(φ, ρ)`, other components zero. \param z The measurement. \return The state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, N> inverse(const linalg::fixed_vector<T, 2>& z) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto p = linalg::transpose(detail::measurement::upper_left<T, 2>(rotation)) * geometry::polar_to_cart(z[1], z[0]) + translation;
            linalg::fixed_vector<T, N> x{};
            x[static_cast<std::size_t>(Map[0])] = p[0];
            x[static_cast<std::size_t>(Map[1])] = p[1];
            return x;
        }
    };

    /// \brief The elevation and bearing of a target in space, `(θ, φ)` (Stone Soup's `CartesianToElevationBearing`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam Map The state's (x, y, z) components.
    template <linalg::scalar T, std::size_t N, std::array Map>
    struct elevation_bearing {
        static_assert(Map.size() == 3 && detail::measurement::valid_map<N, Map>(), "snicholls::tracking::elevation_bearing: map (x, y, z) inside the state");
        using value_type = T;                          ///< The number type.
        static constexpr std::size_t state_dim = N;    ///< The state dimension.
        static constexpr std::size_t measurement_dim = 2;  ///< The measurement dimension.
        static constexpr geometry::angle_kinds<2> kinds{geometry::angle_kind::elevation, geometry::angle_kind::bearing};  ///< Elevation, bearing.
        linalg::fixed_matrix<T, 2, 2> noise{};          ///< `R`.
        linalg::fixed_vector<T, 3> translation{};       ///< The sensor's position.
        linalg::fixed_matrix<T, 3, 3> rotation = linalg::fixed_matrix<T, 3, 3>::identity();  ///< World to sensor.

        /// \brief `(θ, φ)`. \param x The state. \return The measurement.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 2> measure(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto s = detail::measurement::sphere(rotation * (detail::measurement::gather<T, 3, N, Map>(x) - translation));
            return {{s.theta, s.phi}};
        }
        /// \brief `∂(θ, φ)/∂x`. \param x The state. \return The Jacobian.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 2, N> jacobian(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto s = detail::measurement::sphere(rotation * (detail::measurement::gather<T, 3, N, Map>(x) - translation));
            return detail::measurement::scatter<N, Map>(linalg::stack(s.dtheta, s.dphi) * rotation);
        }
        /// \brief `R`. \return The noise covariance.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 2, 2> covariance() const noexcept { return noise; }
    };

    /// \brief The elevation, bearing and range of a target in space, `(θ, φ, ρ)` (Stone Soup's
    /// `CartesianToElevationBearingRange`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam Map The state's (x, y, z) components.
    template <linalg::scalar T, std::size_t N, std::array Map>
    struct elevation_bearing_range {
        static_assert(Map.size() == 3 && detail::measurement::valid_map<N, Map>(), "snicholls::tracking::elevation_bearing_range: map (x, y, z) inside the state");
        using value_type = T;                          ///< The number type.
        static constexpr std::size_t state_dim = N;    ///< The state dimension.
        static constexpr std::size_t measurement_dim = 3;  ///< The measurement dimension.
        static constexpr geometry::angle_kinds<3> kinds{geometry::angle_kind::elevation, geometry::angle_kind::bearing, geometry::angle_kind::linear};  ///< Elevation, bearing, range.
        linalg::fixed_matrix<T, 3, 3> noise{};          ///< `R`.
        linalg::fixed_vector<T, 3> translation{};       ///< The sensor's position.
        linalg::fixed_matrix<T, 3, 3> rotation = linalg::fixed_matrix<T, 3, 3>::identity();  ///< World to sensor.

        /// \brief `(θ, φ, ρ)`. \param x The state. \return The measurement.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 3> measure(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto s = detail::measurement::sphere(rotation * (detail::measurement::gather<T, 3, N, Map>(x) - translation));
            return {{s.theta, s.phi, s.rho}};
        }
        /// \brief `∂(θ, φ, ρ)/∂x`. \param x The state. \return The Jacobian.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, N> jacobian(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto s = detail::measurement::sphere(rotation * (detail::measurement::gather<T, 3, N, Map>(x) - translation));
            return detail::measurement::scatter<N, Map>(linalg::stack(linalg::stack(s.dtheta, s.dphi), s.drho) * rotation);
        }
        /// \brief `R`. \return The noise covariance.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> covariance() const noexcept { return noise; }
        /// \brief The state at `(θ, φ, ρ)`, other components zero. \param z The measurement. \return The state.
        [[nodiscard]] constexpr linalg::fixed_vector<T, N> inverse(const linalg::fixed_vector<T, 3>& z) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto p = linalg::transpose(rotation) * geometry::sphere_to_cart(z[2], z[1], z[0]) + translation;
            linalg::fixed_vector<T, N> x{};
            for (std::size_t i = 0; i < 3; ++i)
                x[static_cast<std::size_t>(Map[i])] = p[i];
            return x;
        }
    };

    /// \brief Bearing, range and range rate, `(φ, ρ, ρ̇)`, in the plane or in space (Stone Soup's
    /// `CartesianToBearingRangeRate2D` and `CartesianToBearingRangeRate`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam N The state dimension.
    /// \tparam Map The state's position components, 2 or 3 of them.
    /// \tparam VelMap The state's velocity components, as many.
    template <linalg::scalar T, std::size_t N, std::array Map, std::array VelMap>
    struct bearing_range_rate {
        static constexpr std::size_t d = Map.size();  ///< The space's dimension, 2 or 3.
        static_assert((d == 2 || d == 3) && VelMap.size() == d && detail::measurement::valid_map<N, Map>() && detail::measurement::valid_map<N, VelMap>(),
                      "snicholls::tracking::bearing_range_rate: map 2 or 3 position and as many velocity components inside the state");
        using value_type = T;                          ///< The number type.
        static constexpr std::size_t state_dim = N;    ///< The state dimension.
        static constexpr std::size_t measurement_dim = 3;  ///< The measurement dimension.
        static constexpr geometry::angle_kinds<3> kinds{geometry::angle_kind::bearing, geometry::angle_kind::linear, geometry::angle_kind::linear};  ///< Bearing, range, range rate.
        linalg::fixed_matrix<T, 3, 3> noise{};          ///< `R`.
        linalg::fixed_vector<T, d> translation{};       ///< The sensor's position.
        linalg::fixed_matrix<T, 3, 3> rotation = linalg::fixed_matrix<T, 3, 3>::identity();  ///< World to sensor.
        linalg::fixed_vector<T, d> velocity{};          ///< The sensor's velocity.

    private:
        struct parts {
            linalg::fixed_vector<T, d> rel, vel;
            T phi, rho, rate;
            linalg::fixed_matrix<T, 1, d> dphi;
        };
        constexpr parts compute(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto a = detail::measurement::upper_left<T, d>(rotation);
            const auto rel = detail::measurement::gather<T, d, N, Map>(x) - translation;
            const auto vel = detail::measurement::gather<T, d, N, VelMap>(x) - velocity;
            const auto p = a * rel;
            const linalg::fixed_vector<T, 2> pxy{{p[0], p[1]}};
            const auto pr = detail::measurement::polar(pxy);
            const T rho = linalg::norm(rel);
            linalg::fixed_matrix<T, 1, d> dphi_p{};
            dphi_p(0, 0) = pr.dphi(0, 0);
            dphi_p(0, 1) = pr.dphi(0, 1);
            return {rel, vel, pr.phi, rho, rho > T(0) ? linalg::dot(rel, vel) / rho : T(0), dphi_p * a};
        }

    public:
        /// \brief `(φ, ρ, ρ̇)`. \param x The state. \return The measurement.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 3> measure(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto c = compute(x);
            return {{c.phi, c.rho, c.rate}};
        }
        /// \brief `∂(φ, ρ, ρ̇)/∂x`. \param x The state. \return The Jacobian.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, N> jacobian(const linalg::fixed_vector<T, N>& x) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const auto c = compute(x);
            linalg::fixed_matrix<T, 3, N> j{};
            if (c.rho == T(0))
                return j;
            for (std::size_t k = 0; k < d; ++k) {
                const auto pk = static_cast<std::size_t>(Map[k]), vk = static_cast<std::size_t>(VelMap[k]);
                j(0, pk) += c.dphi(0, k);
                j(1, pk) += c.rel[k] / c.rho;
                j(2, pk) += c.vel[k] / c.rho - c.rate * c.rel[k] / (c.rho * c.rho);
                j(2, vk) += c.rel[k] / c.rho;
            }
            return j;
        }
        /// \brief `R`. \return The noise covariance.
        [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> covariance() const noexcept { return noise; }
    };

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_measurement_hpp */



