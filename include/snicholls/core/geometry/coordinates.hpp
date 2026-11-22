//
//  coordinates.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::geometry::cart_to_polar`, `cart_to_sphere`, `rotation_z` and friends: Cartesian, polar and spherical coordinates, and rotations, in Stone Soup's conventions.
/// \ingroup core
///
/// **Overview.** Sensors measure range and angles; targets move in Cartesian space. These convert between the two,
/// each returning a `fixed_vector`:
///
/// - `cart_to_polar(x, y)` → `(range, bearing)`, bearing `atan2(y, x)`; `polar_to_cart(range, bearing)` → `(x, y)`.
/// - `cart_to_sphere(x, y, z)` → `(range, bearing, elevation)`, elevation `asin(z / range)` from the x-y plane;
///   `sphere_to_cart(range, bearing, elevation)`; `cart_to_angles(x, y, z)` → `(bearing, elevation)`.
/// - `cart_to_az_el_range(x, y, z)` → `(azimuth, elevation, range)`, the radar-face convention in which the boresight
///   is `z`: azimuth `asin(x / r)`, elevation `asin(y / r)`; `az_el_range_to_cart`.
/// - `rotation_x`, `rotation_y`, `rotation_z(θ)`: the 3 x 3 rotations about each axis; `rotation_2d(θ)`;
///   `sensor_rotation(roll, pitch, yaw)` = `R_z(-yaw) R_y(pitch) R_x(-roll)`, the map from the world into a
///   sensor's frame (Stone Soup's `build_rotation_matrix`).
///
/// At the origin, range is 0 and the angles are 0 (not NaN).
///
/// \snippet geometry/coordinates_demo.cpp coordinates
///
/// **Relationships.** `linalg/fixed.hpp`, `math/constexpr_math.hpp`; `geometry/angle.hpp` for wrapping;
/// `tracking/measurement.hpp` builds its sensor models on these. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. The rotations are
/// orthogonal with determinant 1 to rounding.
///
/// **Errors.** None. A point exactly at the origin has zero angles; `az_el_range_to_cart` takes the square root of
/// `max(0, 1 - sin² az - sin² el)`, so an impossible pair of angles maps onto the face's edge rather than to NaN.
///
/// **Performance.** Not benchmarked: a square root and one or two trigonometric functions each.
///
/// **Tests and demos.** `tests/geometry/coordinates_test.cpp`, `demos/geometry/coordinates_demo.cpp`.
///
/// **Dependencies.** `linalg/fixed.hpp`, `math/constexpr_math.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_geometry_coordinates_hpp
#define snicholls_core_geometry_coordinates_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

namespace snicholls::geometry {

    /// \brief `(range, bearing)` of a point in the plane. \tparam T The number type.
    /// \ingroup core
    /// \param x The x coordinate. \param y The y coordinate. \return `(√(x² + y²), atan2(y, x))`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 2> cart_to_polar(T x, T y) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T r = static_cast<T>(math::sqrt(x * x + y * y));
        return {{r, r == T(0) ? T(0) : static_cast<T>(math::atan2(y, x))}};
    }

    /// \brief `(x, y)` of a range and bearing. \tparam T The number type.
    /// \ingroup core
    /// \param range The range. \param bearing The bearing from the x axis, radians. \return The point.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 2> polar_to_cart(T range, T bearing) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return {{range * static_cast<T>(math::cos(bearing)), range * static_cast<T>(math::sin(bearing))}};
    }

    /// \brief `(range, bearing, elevation)` of a point. \tparam T The number type.
    /// \ingroup core
    /// \param x The x coordinate. \param y The y coordinate. \param z The z coordinate.
    /// \return `(r, atan2(y, x), asin(z / r))`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 3> cart_to_sphere(T x, T y, T z) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T r = static_cast<T>(math::sqrt(x * x + y * y + z * z));
        if (r == T(0))
            return {{T(0), T(0), T(0)}};
        const T h = static_cast<T>(math::sqrt(x * x + y * y));
        return {{r, h == T(0) ? T(0) : static_cast<T>(math::atan2(y, x)), static_cast<T>(math::asin(z / r))}};
    }

    /// \brief `(x, y, z)` of a range, bearing and elevation. \tparam T The number type.
    /// \ingroup core
    /// \param range The range. \param bearing The bearing, radians. \param elevation The elevation, radians. \return The point.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 3> sphere_to_cart(T range, T bearing, T elevation) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T ce = static_cast<T>(math::cos(elevation));
        return {{range * static_cast<T>(math::cos(bearing)) * ce, range * static_cast<T>(math::sin(bearing)) * ce, range * static_cast<T>(math::sin(elevation))}};
    }

    /// \brief `(bearing, elevation)` of a point. \tparam T The number type.
    /// \ingroup core
    /// \param x The x coordinate. \param y The y coordinate. \param z The z coordinate. \return The two angles.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 2> cart_to_angles(T x, T y, T z) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto s = cart_to_sphere(x, y, z);
        return {{s[1], s[2]}};
    }

    /// \brief `(azimuth, elevation, range)` in the radar-face convention (boresight `z`). \ingroup core
    /// \tparam T The number type. \param x The x coordinate. \param y The y coordinate. \param z The z coordinate.
    /// \return `(asin(x / r), asin(y / r), r)`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 3> cart_to_az_el_range(T x, T y, T z) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T r = static_cast<T>(math::sqrt(x * x + y * y + z * z));
        if (r == T(0))
            return {{T(0), T(0), T(0)}};
        return {{static_cast<T>(math::asin(x / r)), static_cast<T>(math::asin(y / r)), r}};
    }

    /// \brief `(x, y, z)` of an azimuth, elevation and range in the radar-face convention. \ingroup core
    /// \tparam T The number type. \param azimuth The azimuth, radians. \param elevation The elevation, radians.
    /// \param range The range. \return The point, `z ≥ 0`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 3> az_el_range_to_cart(T azimuth, T elevation, T range) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T sa = static_cast<T>(math::sin(azimuth)), se = static_cast<T>(math::sin(elevation));
        const T q = T(1) - sa * sa - se * se;
        return {{range * sa, range * se, range * static_cast<T>(math::sqrt(q > T(0) ? q : T(0)))}};
    }

    /// \brief The rotation by `theta` about the x axis. \tparam T The number type.
    /// \ingroup core
    /// \param theta The angle, radians. \return `[[1, 0, 0], [0, c, -s], [0, s, c]]`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> rotation_x(T theta) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T c = static_cast<T>(math::cos(theta)), s = static_cast<T>(math::sin(theta));
        return {{T(1), T(0), T(0), T(0), c, -s, T(0), s, c}};
    }

    /// \brief The rotation by `theta` about the y axis. \tparam T The number type.
    /// \ingroup core
    /// \param theta The angle, radians. \return `[[c, 0, s], [0, 1, 0], [-s, 0, c]]`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> rotation_y(T theta) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T c = static_cast<T>(math::cos(theta)), s = static_cast<T>(math::sin(theta));
        return {{c, T(0), s, T(0), T(1), T(0), -s, T(0), c}};
    }

    /// \brief The rotation by `theta` about the z axis. \tparam T The number type.
    /// \ingroup core
    /// \param theta The angle, radians. \return `[[c, -s, 0], [s, c, 0], [0, 0, 1]]`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> rotation_z(T theta) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T c = static_cast<T>(math::cos(theta)), s = static_cast<T>(math::sin(theta));
        return {{c, -s, T(0), s, c, T(0), T(0), T(0), T(1)}};
    }

    /// \brief The rotation by `theta` in the plane. \tparam T The number type.
    /// \ingroup core
    /// \param theta The angle, radians, anticlockwise. \return `[[c, -s], [s, c]]`.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_matrix<T, 2, 2> rotation_2d(T theta) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T c = static_cast<T>(math::cos(theta)), s = static_cast<T>(math::sin(theta));
        return {{c, -s, s, c}};
    }

    /// \brief The rotation from the world into a sensor's frame, given the sensor's roll, pitch and yaw:
    /// `R_z(-yaw) R_y(pitch) R_x(-roll)` (Stone Soup's `build_rotation_matrix`).
    /// \ingroup core
    /// \tparam T The number type.
    /// \param roll The rotation about x, radians.
    /// \param pitch The rotation about y (elevation), radians.
    /// \param yaw The rotation about z (azimuth), radians.
    /// \return The rotation.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> sensor_rotation(T roll, T pitch, T yaw) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return rotation_z(-yaw) * rotation_y(pitch) * rotation_x(-roll);
    }

} // namespace snicholls::geometry

#endif /* snicholls_core_geometry_coordinates_hpp */
