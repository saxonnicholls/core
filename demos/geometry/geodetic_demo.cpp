//
//  geodetic_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/geometry/geodetic.hpp>: a session origin on the WGS 84 ellipsoid, GNSS fixes turned into
//  metres East-North-Up and North-East-Down and back, ECEF both ways, the poles and the Earth's centre, and an
//  ellipsoid of the caller's own. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/geometry/geodetic.hpp>

#include <cmath>
#include <cstdio>
#include <numbers>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
using v3 = la::fixed_vector<double, 3>;
constexpr double pi = std::numbers::pi;
constexpr double deg = pi / 180;

bool check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core geometry geodetic demo\n");
    bool ok = true;

    //! [geodetic]
    // The session's origin, a beach at 33.89 S, 151.27 E, 2 m above the ellipsoid; built once
    const geo::local_frame<double> frame(v3{{-33.8915 * deg, 151.2767 * deg, 2.0}});
    // A swimmer's GNSS fix, a little to the north-east
    const v3 fix{{-33.8906 * deg, 151.2781 * deg, 2.0}};
    const auto enu = frame.to_enu(fix);                        // metres east, north, up
    const auto ned = geo::local_frame<double>::to_ned(enu);   // north, east, down for the flight controller
    // And back: a point 50 m out to sea (east), on the tangent plane
    const auto buoy = frame.to_lla(v3{{50.0, 0.0, 0.0}});
    //! [geodetic]
    std::printf("  fix: %.2f m east, %.2f m north, %.3f m up\n", enu[0], enu[1], enu[2]);
    std::printf("  buoy: %.6f deg, %.6f deg, %.4f m\n", buoy[0] / deg, buoy[1] / deg, buoy[2]);
    // 0.0009 deg of latitude is about 99.8 m there, 0.0014 deg of longitude about 129.5 m
    ok &= check(std::abs(enu[1] - 99.8) < 0.5 && std::abs(enu[0] - 129.5) < 0.5 && enu[2] < 0 && enu[2] > -0.01, "ENU of a nearby fix (the curvature drops it a little)");
    ok &= check(ned[0] == enu[1] && ned[1] == enu[0] && ned[2] == -enu[2] && geo::local_frame<double>::from_ned(ned) == enu, "to_ned and from_ned");
    ok &= check(la::norm(frame.to_enu(buoy) - v3{{50.0, 0.0, 0.0}}) < 1e-8, "to_lla inverts to_enu");
    ok &= check(frame.origin()[2] == 2.0 && frame.rotation() == geo::enu_rotation(frame.origin()[0], frame.origin()[1]), "origin and rotation");
    const auto ecef = geo::lla_to_ecef(fix);
    ok &= check(la::norm(frame.ecef_to_enu(ecef) - enu) < 1e-9 && la::norm(frame.enu_to_ecef(enu) - ecef) < 1e-8, "ecef_to_enu and enu_to_ecef");

    // ECEF both ways, from below the surface to geostationary height
    double worst = 0;
    for (const double h : {-1.0e6, 0.0, 1.0e4, 3.5786e7})
        for (double lat = -80; lat <= 80; lat += 20) {
            const v3 lla{{lat * deg, 0.7, h}};
            const auto back = geo::ecef_to_lla(geo::lla_to_ecef(lla));
            const double e = std::abs(back[2] - h) + 6.4e6 * (std::abs(back[0] - lla[0]) + std::abs(back[1] - lla[1]));
            worst = e > worst ? e : worst;
        }
    std::printf("  worst LLA -> ECEF -> LLA error: %.2e m\n", worst);
    ok &= check(worst < 1e-6, "round trips agree to well under a millimetre");
    const auto eq = geo::lla_to_ecef(v3{{0.0, 0.0, 0.0}});
    ok &= check(eq[0] == geo::wgs84<double>.a && eq[1] == 0 && eq[2] == 0, "the equator at Greenwich is (a, 0, 0)");
    const auto np = geo::ecef_to_lla(v3{{0.0, 0.0, geo::wgs84<double>.b() + 10}});
    ok &= check(std::abs(np[0] - pi / 2) < 1e-15 && np[1] == 0 && std::abs(np[2] - 10) < 1e-8, "the north pole, 10 m up");
    const auto centre = geo::ecef_to_lla(v3{{0.0, 0.0, 0.0}});
    ok &= check(centre[0] == 0 && centre[2] == -geo::wgs84<double>.b(), "the centre: latitude 0, height -b");
    ok &= check(std::abs(geo::wgs84<double>.e2() - 0.0066943799901413165) < 1e-18 && std::abs(geo::wgs84<double>.b() - 6356752.314245179) < 1e-6, "WGS 84's e^2 and b");

    // An ellipsoid of the caller's own: a sphere of 1 km, on which latitudes are geocentric
    constexpr geo::ellipsoid<double> ball{1000.0, 0.0};
    const auto l = geo::ecef_to_lla(v3{{3.0, 4.0, 12.0}}, ball);
    ok &= check(std::abs(l[0] - std::atan2(12.0, 5.0)) < 1e-15 && std::abs(l[2] + 987) < 1e-12, "ecef_to_lla on a sphere");
    const geo::local_frame<double> on_ball(v3{{0.0, 0.0, 0.0}}, ball);
    ok &= check(la::norm(on_ball.to_enu(v3{{0.0, pi / 2, 0.0}}) - v3{{1000.0, 0.0, -1000.0}}) < 1e-9, "a quarter turn east on the sphere");

    std::printf(ok ? "geometry geodetic demo: ok\n" : "geometry geodetic demo: FAILED\n");
    return ok ? 0 : 1;
}
