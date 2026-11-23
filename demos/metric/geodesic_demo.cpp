//
//  geodesic_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/metric/geodesic.hpp>: great-circle distances between cities with the Earth's mean radius,
//  the nearest city to a position by brute force, a position with a height (ignored), antipodal points, and the same
//  metric on a unit sphere and on Mars. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/metric/geodesic.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numbers>

namespace {

namespace la = snicholls::linalg;
namespace me = snicholls::metric;

constexpr double pi = std::numbers::pi;

constexpr std::array<double, 2> degrees(double lat, double lon) { return {lat * pi / 180, lon * pi / 180}; }

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core geodesic demo\n");
    bool ok = true;

    //! [geodesic]
    // (latitude, longitude) in radians; the Earth's mean radius by default, so the answer is in metres
    const std::array<std::array<double, 2>, 4> cities{degrees(51.5074, -0.1278), degrees(48.8566, 2.3522), degrees(40.7128, -74.0060),
                                                      degrees(-33.8688, 151.2093)};
    const char* names[] = {"London", "Paris", "New York", "Sydney"};
    const me::haversine earth{};
    const double london_paris = earth(cities[0], cities[1]);

    // The nearest city to a ship at sea, given with its height (the third coordinate is ignored)
    const la::fixed_vector<double, 3> ship{{45.0 * pi / 180, -30.0 * pi / 180, 25.0}};
    std::size_t nearest = 0;
    for (std::size_t i = 1; i < cities.size(); ++i)
        if (earth(ship, cities[i]) < earth(ship, cities[nearest]))
            nearest = i;

    // Any sphere: radius 1 gives the central angle, Mars its own radius
    const double angle = me::haversine{1.0}(cities[0], cities[3]);
    const double on_mars = me::haversine{3389500.0}(cities[0], cities[3]);
    //! [geodesic]

    std::printf("  London-Paris %.1f km; nearest city to the ship: %s, %.1f km\n", london_paris / 1000, names[nearest], earth(ship, cities[nearest]) / 1000);
    ok &= check(std::abs(london_paris - 343556.535) < 0.01, "London-Paris is 343.557 km");
    ok &= check(std::abs(earth(cities[2], cities[3]) - 15988777.592) < 0.01, "New York-Sydney is 15988.778 km");
    ok &= check(std::abs(earth(cities[1], cities[0]) - london_paris) < 1e-6, "symmetric");
    ok &= check(earth(cities[0], cities[0]) == 0, "a city to itself is 0");
    ok &= check(nearest == 0, "the ship at 45 N 30 W is nearest London");
    ok &= check(std::abs(angle * earth.radius - earth(cities[0], cities[3])) < 1e-6, "radius 1 gives the central angle");
    ok &= check(std::abs(on_mars - 3389500.0 * angle) < 1e-6, "on Mars, the same angle at Mars's radius");
    const double antipode = earth(degrees(0, 0), degrees(0, 180));
    ok &= check(!std::isnan(antipode) && std::abs(antipode - pi * earth.radius) < 1e-6, "antipodal points are pi r, never NaN");

    std::printf(ok ? "geodesic demo: ok\n" : "geodesic demo: FAILED\n");
    return ok ? 0 : 1;
}
