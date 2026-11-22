//
//  geodetic_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/geometry/geodetic.hpp>: the WGS 84 constants, lla_to_ecef at the equator, the poles and a
//  surveyed point, ecef_to_lla against Heikkinen's independent closed form, round trips from the Earth's centre to
//  geostationary height, the poles and the centre, a spherical ellipsoid, the ENU rotation, local_frame's conversions
//  and NED, and constant evaluation.
//

#include <snicholls/core/geometry/geodetic.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
using v3 = la::fixed_vector<double, 3>;
using m33 = la::fixed_matrix<double, 3, 3>;
constexpr double pi = std::numbers::pi;
constexpr double deg = pi / 180;

static_assert(std::is_aggregate_v<geo::ellipsoid<double>> && std::is_trivially_copyable_v<geo::ellipsoid<double>> && sizeof(geo::ellipsoid<double>) == 2 * sizeof(double));
static_assert(std::is_trivially_copyable_v<geo::local_frame<double>>);
static_assert(noexcept(geo::lla_to_ecef(std::declval<const v3&>())) && noexcept(geo::ecef_to_lla(std::declval<const v3&>())) && noexcept(geo::enu_rotation(1.0, 2.0)));
static_assert(noexcept(geo::local_frame<double>(std::declval<const v3&>())) && noexcept(geo::local_frame<double>::to_ned(std::declval<const v3&>())));
static_assert(noexcept(std::declval<const geo::local_frame<double>&>().to_enu(std::declval<const v3&>())) && noexcept(geo::wgs84<double>.e2()));

// WGS 84: a = 6378137, 1/f = 298.257223563 (NIMA TR8350.2); b and e² from them
static_assert(geo::wgs84<double>.a == 6378137.0 && geo::wgs84<double>.f == 1 / 298.257223563);

// Constant evaluation
static_assert(geo::lla_to_ecef(v3{{0, 0, 0}}) == v3{{6378137.0, 0, 0}});
static_assert(geo::lla_to_ecef(v3{{0, 0, 100}}) == v3{{6378237.0, 0, 0}});
static_assert(geo::ecef_to_lla(v3{{0, 0, 0}}) == v3{{0, 0, -geo::wgs84<double>.b()}});
static_assert(geo::ecef_to_lla(v3{{6378137.0, 0, 0}}) == v3{{0, 0, 0}});
static_assert(geo::enu_rotation(0.0, 0.0) == m33{{0, 1, 0, 0, 0, 1, 1, 0, 0}});
constexpr geo::local_frame<double> frame0{v3{{0, 0, 0}}};
static_assert(frame0.to_enu(v3{{0, 0, 10}}) == v3{{0, 0, 10}});
static_assert(geo::local_frame<double>::to_ned(v3{{1, 2, 3}}) == v3{{2, 1, -3}});

void expect_lla_near(const v3& a, const v3& b, double angle_tol, double height_tol)
{
    EXPECT_NEAR(a[0], b[0], angle_tol);
    EXPECT_NEAR(a[1], b[1], angle_tol);
    EXPECT_NEAR(a[2], b[2], height_tol);
}

TEST(GeometryGeodeticEllipsoid, DerivedConstantsOfWgs84)
{
    // Python: f * (2 - f) and a * (1 - f) with f = 1 / 298.257223563
    EXPECT_DOUBLE_EQ(geo::wgs84<double>.e2(), 0.0066943799901413165);
    EXPECT_DOUBLE_EQ(geo::wgs84<double>.b(), 6356752.314245179);
    constexpr geo::ellipsoid<double> sphere{1000.0, 0.0};
    static_assert(sphere.e2() == 0 && sphere.b() == 1000.0);
    EXPECT_FLOAT_EQ(geo::wgs84<float>.b(), 6356752.314245179f);
}

TEST(GeometryGeodeticLlaToEcef, EquatorAndPoles)
{
    const double a = geo::wgs84<double>.a, b = geo::wgs84<double>.b();
    const auto e90 = geo::lla_to_ecef(v3{{0, pi / 2, 0}});
    EXPECT_NEAR(e90[0], 0, 1e-9);
    EXPECT_DOUBLE_EQ(e90[1], a);
    EXPECT_EQ(e90[2], 0);
    const auto north = geo::lla_to_ecef(v3{{pi / 2, 0, 0}});
    EXPECT_NEAR(north[0], 0, 1e-9);
    EXPECT_NEAR(north[2], b, 1e-8);
    const auto south = geo::lla_to_ecef(v3{{-pi / 2, 1.0, 50}});
    EXPECT_NEAR(south[2], -(b + 50), 1e-8);
    EXPECT_NEAR(std::hypot(south[0], south[1]), 0, 1e-9);
}

TEST(GeometryGeodeticLlaToEcef, MatchesTheClosedFormAtGreenwich)
{
    // 51.4778 N, 0.0014 W, 45 m: Python with the textbook closed form
    const auto e = geo::lla_to_ecef(v3{{51.4778 * deg, -0.0014 * deg, 45.0}});
    EXPECT_NEAR(e[0], 3980609.2372733257, 1e-6);
    EXPECT_NEAR(e[1], -97.26463241536489, 1e-6);
    EXPECT_NEAR(e[2], 4966859.728504293, 1e-6);
}

TEST(GeometryGeodeticEcefToLla, AgreesWithHeikkinensClosedForm)
{
    // Heikkinen (1982), an independent exact solution, evaluated in Python
    expect_lla_near(geo::ecef_to_lla(v3{{3980609.2372733257, -97.26463241536489, 4966859.728504293}}), v3{{0.898457101683137, -2.4434609527920612e-05, 45.0}}, 1e-14,
                    1e-6);
    expect_lla_near(geo::ecef_to_lla(v3{{-2694044.4111565403, -4266368.805493665, 3888310.6034}}), v3{{0.6598107666789524, -2.1340280844260433, 0.0006884422056727774}},
                    1e-14, 1e-6);
}

TEST(GeometryGeodeticEcefToLla, RoundTripsFromTheCentreToGeostationary)
{
    for (const double h : {-6.2e6, -6.0e6, -3.0e6, -1000.0, 0.0, 8848.0, 4.0e5, 3.5786e7})
        for (double lat = -90; lat <= 90; lat += 15)
            for (double lon = -180; lon < 180; lon += 45) {
                const v3 lla{{lat * deg, lon * deg, h}};
                const auto back = geo::ecef_to_lla(geo::lla_to_ecef(lla));
                const bool pole = std::abs(lat) == 90;
                EXPECT_NEAR(back[0], lla[0], 1e-13) << lat << " " << lon << " " << h;
                if (!pole) {
                    EXPECT_NEAR(std::remainder(back[1] - lla[1], 2 * pi), 0, 1e-13) << lat << " " << lon << " " << h;
                }
                EXPECT_NEAR(back[2], h, 1e-6) << lat << " " << lon << " " << h;   // well under a millimetre
                // And the forward map of the result gives the same position
                const auto e = geo::lla_to_ecef(lla), e2 = geo::lla_to_ecef(back);
                EXPECT_NEAR(la::norm(e - e2), 0, 1e-6);
            }
}

TEST(GeometryGeodeticEcefToLla, PolesAndCentre)
{
    const double b = geo::wgs84<double>.b();
    const auto n = geo::ecef_to_lla(v3{{0, 0, b + 100}});
    EXPECT_DOUBLE_EQ(n[0], pi / 2);
    EXPECT_EQ(n[1], 0);
    EXPECT_NEAR(n[2], 100, 1e-8);
    const auto s = geo::ecef_to_lla(v3{{0, 0, -b}});
    EXPECT_DOUBLE_EQ(s[0], -pi / 2);
    EXPECT_NEAR(s[2], 0, 1e-8);
    EXPECT_EQ(geo::ecef_to_lla(v3{{0, 0, 0}}), (v3{{0, 0, -b}}));
}

TEST(GeometryGeodeticEllipsoid, ASphereHasGeocentricLatitudes)
{
    constexpr geo::ellipsoid<double> sphere{1000.0, 0.0};
    const auto e = geo::lla_to_ecef(v3{{pi / 6, pi / 3, 10}}, sphere);
    EXPECT_NEAR(e[0], 1010 * std::cos(pi / 6) * std::cos(pi / 3), 1e-12);
    EXPECT_NEAR(e[1], 1010 * std::cos(pi / 6) * std::sin(pi / 3), 1e-12);
    EXPECT_NEAR(e[2], 1010 * std::sin(pi / 6), 1e-12);
    const auto l = geo::ecef_to_lla(v3{{3, 4, 12}}, sphere);   // |x| = 13
    EXPECT_NEAR(l[0], std::atan2(12.0, 5.0), 1e-15);
    EXPECT_NEAR(l[1], std::atan2(4.0, 3.0), 1e-15);
    EXPECT_NEAR(l[2], 13 - 1000, 1e-12);
}

TEST(GeometryGeodeticEnuRotation, RowsAreEastNorthUp)
{
    const double lat = 0.6, lon = -1.1;
    const auto r = geo::enu_rotation(lat, lon);
    EXPECT_NEAR(la::norm(r * la::transpose(r) - m33::identity()), 0, 1e-15);
    EXPECT_NEAR(la::determinant(r), 1, 1e-15);
    // Up is the ellipsoid normal; north is the derivative by latitude; east by longitude
    const v3 up{{std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)}};
    const auto rup = r * up;
    EXPECT_NEAR(rup[0], 0, 1e-15);
    EXPECT_NEAR(rup[1], 0, 1e-15);
    EXPECT_NEAR(rup[2], 1, 1e-15);
    const double h = 1e-6;
    const auto dn = (geo::lla_to_ecef(v3{{lat + h, lon, 0}}) - geo::lla_to_ecef(v3{{lat - h, lon, 0}}));
    const auto de = (geo::lla_to_ecef(v3{{lat, lon + h, 0}}) - geo::lla_to_ecef(v3{{lat, lon - h, 0}}));
    const auto rn = r * (dn / la::norm(dn)), re = r * (de / la::norm(de));
    EXPECT_NEAR(rn[1], 1, 1e-9);
    EXPECT_NEAR(re[0], 1, 1e-9);
}

TEST(GeometryGeodeticLocalFrame, OriginRotationAndHeights)
{
    const v3 origin{{-33.8568 * deg, 151.2153 * deg, 20}};
    const geo::local_frame<double> f(origin);
    EXPECT_EQ(f.origin(), origin);
    EXPECT_EQ(f.rotation(), geo::enu_rotation(origin[0], origin[1]));
    EXPECT_NEAR(la::norm(f.to_enu(origin)), 0, 1e-8);
    const auto above = f.to_enu(v3{{origin[0], origin[1], 120}});
    EXPECT_NEAR(above[0], 0, 1e-8);
    EXPECT_NEAR(above[1], 0, 1e-8);
    EXPECT_NEAR(above[2], 100, 1e-8);
    // A point a little north is north (and slightly below, by the Earth's curvature)
    const auto north = f.to_enu(v3{{origin[0] + 0.001 * deg, origin[1], 20}});
    EXPECT_NEAR(north[0], 0, 1e-6);
    EXPECT_NEAR(north[1], 110.9, 0.5);
    EXPECT_LT(north[2], 0);
    const auto east = f.to_enu(v3{{origin[0], origin[1] + 0.001 * deg, 20}});
    EXPECT_GT(east[0], 90);
    EXPECT_NEAR(east[1], 0, 1e-3);
}

TEST(GeometryGeodeticLocalFrame, ConversionsRoundTrip)
{
    const geo::local_frame<double> f(v3{{51.5 * deg, -0.12 * deg, 30}});
    for (const v3 enu : {v3{{0, 0, 0}}, v3{{1000, -2000, 50}}, v3{{-15000, 30000, -100}}, v3{{0.25, 0.5, 2}}}) {
        EXPECT_NEAR(la::norm(f.ecef_to_enu(f.enu_to_ecef(enu)) - enu), 0, 1e-8);
        EXPECT_NEAR(la::norm(f.to_enu(f.to_lla(enu)) - enu), 0, 1e-7);
        EXPECT_EQ(geo::local_frame<double>::from_ned(geo::local_frame<double>::to_ned(enu)), enu);
    }
    const v3 lla{{51.51 * deg, -0.1 * deg, 75}};
    expect_lla_near(f.to_lla(f.to_enu(lla)), lla, 1e-13, 1e-7);
    const auto ecef = geo::lla_to_ecef(lla);
    EXPECT_NEAR(la::norm(f.ecef_to_enu(ecef) - f.to_enu(lla)), 0, 1e-9);
}

TEST(GeometryGeodeticLocalFrame, NedIsNorthEastDown)
{
    EXPECT_EQ(geo::local_frame<double>::to_ned(v3{{1, 2, 3}}), (v3{{2, 1, -3}}));
    EXPECT_EQ(geo::local_frame<double>::from_ned(v3{{2, 1, -3}}), (v3{{1, 2, 3}}));
}

TEST(GeometryGeodeticLocalFrame, ACustomEllipsoidIsUsedThroughout)
{
    constexpr geo::ellipsoid<double> sphere{1000.0, 0.0};
    const geo::local_frame<double> f(v3{{0, 0, 0}}, sphere);
    const auto p = f.to_enu(v3{{0, pi / 2, 0}});   // a quarter turn east on a 1 km sphere
    EXPECT_NEAR(p[0], 1000, 1e-9);
    EXPECT_NEAR(p[2], -1000, 1e-9);
    expect_lla_near(f.to_lla(p), v3{{0, pi / 2, 0}}, 1e-15, 1e-9);
}

TEST(GeometryGeodeticFloat, RoundTripInFloat)
{
    using f3 = la::fixed_vector<float, 3>;
    const f3 lla{{0.6f, 2.0f, 100.0f}};
    const auto back = geo::ecef_to_lla(geo::lla_to_ecef(lla));
    EXPECT_NEAR(back[0], lla[0], 1e-6f);
    EXPECT_NEAR(back[1], lla[1], 1e-6f);
    EXPECT_NEAR(back[2], lla[2], 2.0f);   // float has 24 bits for six million metres
}

} // namespace
