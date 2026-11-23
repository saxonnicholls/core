//
//  geodesic.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::metric::haversine`: the great-circle distance between two `(latitude, longitude)` points on a sphere.
/// \ingroup core
///
/// **Overview.** Positions on the Earth are compared along its surface. `haversine{r}` takes points whose first two
/// coordinates are latitude and longitude in radians (a `fixed_vector` from `geometry::ecef_to_lla`, an
/// `std::array<double, 2>`, a struct of the caller's through `point_traits`) and returns
/// `2 r asin √(sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2))`, in the units of `r`. The default radius is the Earth's mean
/// radius, 6371008.8 m (IUGG). A third coordinate (height) is ignored.
///
/// It is a metric, so trees that prune by the triangle inequality can use it. It is exact on a sphere and within
/// about 0.5% on the WGS 84 ellipsoid; for survey work, convert with `geometry::local_frame` and use `euclidean`.
///
/// \snippet metric/geodesic_demo.cpp geodesic
///
/// **Relationships.** `metric/concepts.hpp`, `geometry/geodetic.hpp` (`ecef_to_lla`, `local_frame`),
/// `geometry/angle.hpp` (`radians`), `math/constexpr_math.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` when the coordinate reads are, no allocation. The haversine form is
/// accurate for small distances, where the spherical law of cosines loses digits; the argument of `asin` is clamped
/// to `[0, 1]`, so antipodal points give `π r`, never NaN.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked: five trigonometric calls and a square root.
///
/// **Tests and demos.** `tests/metric/geodesic_test.cpp`, `demos/metric/geodesic_demo.cpp`.
///
/// **Dependencies.** `math/constexpr_math.hpp`, `metric/concepts.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.20.0.

#ifndef snicholls_core_metric_geodesic_hpp
#define snicholls_core_metric_geodesic_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/metric/concepts.hpp>

namespace snicholls::metric {

    /// \brief The great-circle distance on a sphere of radius `radius`, between `(latitude, longitude)` points in radians.
    /// \ingroup core
    struct haversine {
        static constexpr bool is_metric = true;  ///< It is a metric on the sphere.
        double radius = 6371008.8;               ///< The sphere's radius; the Earth's mean radius in metres by default.

        /// \brief The distance. \tparam A A `point` of at least two coordinates. \tparam B Another. \param a One position.
        /// \param b The other. \return The distance along the sphere, in the units of `radius`.
        template <point A, point B>
        [[nodiscard]] constexpr double operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            const double p1 = static_cast<double>(coordinate(a, 0)), p2 = static_cast<double>(coordinate(b, 0));
            const double dl = static_cast<double>(coordinate(b, 1)) - static_cast<double>(coordinate(a, 1));
            const double sp = math::sin((p2 - p1) / 2), sl = math::sin(dl / 2);
            double h = sp * sp + math::cos(p1) * math::cos(p2) * sl * sl;
            h = h < 0 ? 0 : h > 1 ? 1 : h;
            return 2 * radius * math::asin(math::sqrt(h));
        }
    };

} // namespace snicholls::metric

#endif /* snicholls_core_metric_geodesic_hpp */
