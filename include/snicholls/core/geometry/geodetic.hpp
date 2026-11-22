//
//  geodetic.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::geometry::lla_to_ecef`, `ecef_to_lla` and `local_frame`: WGS 84 geodetic, Earth-centred and local East-North-Up coordinates.
/// \ingroup core
///
/// **Overview.** A GNSS fix is latitude, longitude and height above the ellipsoid; trackers work in metres on a local
/// tangent plane. These convert between the three frames on an ellipsoid (WGS 84 by default):
///
/// - `lla_to_ecef(lla)` and `ecef_to_lla(ecef)`: geodetic `(latitude, longitude, height)`, radians and metres, to
///   and from Earth-centred, Earth-fixed `(x, y, z)` metres.
/// - `enu_rotation(lat, lon)`: the rotation from ECEF into East-North-Up at a point.
/// - `local_frame<T>`: an ENU frame anchored at an origin, with its rotation and the origin's ECEF position
///   computed once at construction (`constexpr`), then `to_enu(lla)`, `to_lla(enu)`, `ecef_to_enu`, `enu_to_ecef`,
///   and `to_ned` / `from_ned` for North-East-Down.
///
/// \snippet geometry/geodetic_demo.cpp geodetic
///
/// **Relationships.** `linalg/fixed.hpp`, `math/constexpr_math.hpp`, `geometry/angle.hpp` (`radians`). The world
/// frame of multi-sensor fusion, anchored at a session origin. Group: \ref core.
///
/// **Implementations.** `lla_to_ecef` is the closed form. `ecef_to_lla` iterates the latitude
/// `φ ← atan2(z + e² N(φ) sin φ, p)` from `atan2(z, p (1 - e²))` until it stops changing, at most 64 times. The error
/// shrinks by about `e² a / r` each step at distance `r` from the centre: three or four steps near the surface, more
/// deep inside the Earth, so the result is exact to rounding for any point from about 100 km from the centre to
/// geostationary orbit (within `e² a`, 43 km, of the centre a point has more than one geodetic latitude). The
/// height is `p cos φ + z sin φ - a √(1 - e² sin² φ)`, which stays exact at the poles, where `p / cos φ` would not.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation, no recursion. Round
/// trips agree to well under a millimetre in `double`.
///
/// **Errors.** None. At the Earth's centre the latitude is 0 and the height is `-b`.
///
/// **Performance.** Not benchmarked: a handful of trigonometric calls each.
///
/// **Tests and demos.** `tests/geometry/geodetic_test.cpp`, `demos/geometry/geodetic_demo.cpp`.
///
/// **Dependencies.** `linalg/fixed.hpp`, `math/constexpr_math.hpp`.
///
/// **Known issues.** Heights are above the ellipsoid, not mean sea level: a geoid model is outside core.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_geometry_geodetic_hpp
#define snicholls_core_geometry_geodetic_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

namespace snicholls::geometry {

    /// \brief An ellipsoid of revolution: semi-major axis `a` and flattening `f`.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct ellipsoid {
        T a;  ///< The semi-major (equatorial) axis, metres.
        T f;  ///< The flattening, `(a - b) / a`.

        /// \brief The first eccentricity squared, `f (2 - f)`. \return It.
        [[nodiscard]] constexpr T e2() const noexcept(linalg::detail::nothrow_scalar<T>) { return f * (T(2) - f); }
        /// \brief The semi-minor (polar) axis, `a (1 - f)`. \return It.
        [[nodiscard]] constexpr T b() const noexcept(linalg::detail::nothrow_scalar<T>) { return a * (T(1) - f); }
    };

    /// \brief The WGS 84 ellipsoid, `a = 6378137 m`, `1 / f = 298.257223563`. \tparam T The number type.
    /// \ingroup core
    template <linalg::scalar T>
    inline constexpr ellipsoid<T> wgs84{T(6378137.0), T(1) / T(298.257223563)};

    /// \brief ECEF `(x, y, z)` of a geodetic point.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param lla `(latitude, longitude, height)`: radians, radians, metres above the ellipsoid.
    /// \param e The ellipsoid.
    /// \return `(x, y, z)`, metres.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 3> lla_to_ecef(const linalg::fixed_vector<T, 3>& lla, const ellipsoid<T>& e = wgs84<T>) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T sl = static_cast<T>(math::sin(lla[0])), cl = static_cast<T>(math::cos(lla[0]));
        const T so = static_cast<T>(math::sin(lla[1])), co = static_cast<T>(math::cos(lla[1]));
        const T n = e.a / static_cast<T>(math::sqrt(T(1) - e.e2() * sl * sl));
        return {{(n + lla[2]) * cl * co, (n + lla[2]) * cl * so, (n * (T(1) - e.e2()) + lla[2]) * sl}};
    }

    /// \brief The geodetic point of an ECEF position.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param ecef `(x, y, z)`, metres.
    /// \param e The ellipsoid.
    /// \return `(latitude, longitude, height)`: radians, radians, metres.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_vector<T, 3> ecef_to_lla(const linalg::fixed_vector<T, 3>& ecef, const ellipsoid<T>& e = wgs84<T>) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T x = ecef[0], y = ecef[1], z = ecef[2];
        const T e2 = e.e2();
        const T p = static_cast<T>(math::sqrt(x * x + y * y));
        if (p == T(0) && z == T(0))
            return {{T(0), T(0), -e.b()}};
        T lat = static_cast<T>(math::atan2(z, p * (T(1) - e2)));
        for (int i = 0; i < 64; ++i) {
            const T s = static_cast<T>(math::sin(lat));
            const T n = e.a / static_cast<T>(math::sqrt(T(1) - e2 * s * s));
            const T next = static_cast<T>(math::atan2(z + e2 * n * s, p));
            if (next == lat)
                break;
            lat = next;
        }
        const T s = static_cast<T>(math::sin(lat)), c = static_cast<T>(math::cos(lat));
        const T h = p * c + z * s - e.a * static_cast<T>(math::sqrt(T(1) - e2 * s * s));
        const T lon = p == T(0) ? T(0) : static_cast<T>(math::atan2(y, x));
        return {{lat, lon, h}};
    }

    /// \brief The rotation from ECEF into East-North-Up at a latitude and longitude.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param lat The latitude, radians.
    /// \param lon The longitude, radians.
    /// \return The rows are east, north and up, in ECEF.
    template <linalg::scalar T>
    [[nodiscard]] constexpr linalg::fixed_matrix<T, 3, 3> enu_rotation(T lat, T lon) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const T sl = static_cast<T>(math::sin(lat)), cl = static_cast<T>(math::cos(lat));
        const T so = static_cast<T>(math::sin(lon)), co = static_cast<T>(math::cos(lon));
        return {{-so, co, T(0), -sl * co, -sl * so, cl, cl * co, cl * so, sl}};
    }

    /// \brief A local East-North-Up frame anchored at a geodetic origin; its rotation and the origin's ECEF position
    /// are computed once.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    class local_frame {
    public:
        /// \brief The frame at `origin`. \param origin `(latitude, longitude, height)`, radians and metres.
        /// \param e The ellipsoid.
        constexpr explicit local_frame(const linalg::fixed_vector<T, 3>& origin, const ellipsoid<T>& e = wgs84<T>) noexcept(linalg::detail::nothrow_scalar<T>)
            : origin_(origin), ellipsoid_(e), origin_ecef_(lla_to_ecef(origin, e)), rotation_(enu_rotation(origin[0], origin[1]))
        {
        }

        /// \brief The origin. \return `(latitude, longitude, height)`.
        [[nodiscard]] constexpr const linalg::fixed_vector<T, 3>& origin() const noexcept { return origin_; }
        /// \brief The rotation from ECEF into this frame. \return It.
        [[nodiscard]] constexpr const linalg::fixed_matrix<T, 3, 3>& rotation() const noexcept { return rotation_; }

        /// \brief ENU of an ECEF position. \param ecef The position, metres. \return `(east, north, up)`, metres.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 3> ecef_to_enu(const linalg::fixed_vector<T, 3>& ecef) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return rotation_ * (ecef - origin_ecef_);
        }
        /// \brief ECEF of an ENU position. \param enu `(east, north, up)`, metres. \return The ECEF position.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 3> enu_to_ecef(const linalg::fixed_vector<T, 3>& enu) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return linalg::transpose(rotation_) * enu + origin_ecef_;
        }
        /// \brief ENU of a geodetic point. \param lla `(latitude, longitude, height)`. \return `(east, north, up)`, metres.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 3> to_enu(const linalg::fixed_vector<T, 3>& lla) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return ecef_to_enu(lla_to_ecef(lla, ellipsoid_));
        }
        /// \brief The geodetic point of an ENU position. \param enu `(east, north, up)`, metres.
        /// \return `(latitude, longitude, height)`.
        [[nodiscard]] constexpr linalg::fixed_vector<T, 3> to_lla(const linalg::fixed_vector<T, 3>& enu) const noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return ecef_to_lla(enu_to_ecef(enu), ellipsoid_);
        }
        /// \brief North-East-Down of an ENU vector. \param enu `(east, north, up)`. \return `(north, east, down)`.
        [[nodiscard]] static constexpr linalg::fixed_vector<T, 3> to_ned(const linalg::fixed_vector<T, 3>& enu) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return {{enu[1], enu[0], -enu[2]}};
        }
        /// \brief East-North-Up of a NED vector. \param ned `(north, east, down)`. \return `(east, north, up)`.
        [[nodiscard]] static constexpr linalg::fixed_vector<T, 3> from_ned(const linalg::fixed_vector<T, 3>& ned) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            return {{ned[1], ned[0], -ned[2]}};
        }

    private:
        linalg::fixed_vector<T, 3> origin_;
        ellipsoid<T> ellipsoid_;
        linalg::fixed_vector<T, 3> origin_ecef_;
        linalg::fixed_matrix<T, 3, 3> rotation_;
    };

} // namespace snicholls::geometry

#endif /* snicholls_core_geometry_geodetic_hpp */
