//
//  measurement_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/tracking/measurement.hpp>: a position sensor, a radar on a yawed platform measuring
//  bearing and range and turning a report back into a position, a passive bearing sensor, the 3-D models (elevation
//  and bearing, with range, with range rate), each model's Jacobian and noise, and the sensor's own position and the
//  pole, where nothing is NaN. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/tracking/measurement.hpp>

#include <snicholls/core/geometry/coordinates.hpp>
#include <snicholls/core/linalg/fixed.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numbers>

namespace {

namespace tr = snicholls::tracking;
namespace la = snicholls::linalg;
namespace ge = snicholls::geometry;

bool check(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

constexpr double deg = std::numbers::pi / 180;

// The state of a target in the plane is (x, vx, y, vy); in space (x, vx, y, vy, z, vz)
constexpr std::array<std::size_t, 2> xy{0, 2};
constexpr std::array<std::size_t, 2> vxy{1, 3};
constexpr std::array<std::size_t, 3> xyz{0, 2, 4};
constexpr std::array<std::size_t, 3> vxyz{1, 3, 5};

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

} // namespace

int main()
{
    std::printf("snicholls-core measurement demo\n");
    bool ok = true;

    //! [measurement]
    // A radar at (1000, 500) on a platform heading 30 degrees left of east: bearing and range in its own frame
    tr::bearing_range<double, 4, xy> radar{};
    radar.translation = {{1000, 500}};
    radar.rotation = ge::sensor_rotation(0.0, 0.0, 30 * deg);
    radar.noise = {{0.5 * deg * 0.5 * deg, 0, 0, 25}};   // 0.5 degree and 5 m

    const la::fixed_vector<double, 4> target{{2000, 10, 1500, -3}};   // 45 degrees and 1414 m from the radar
    const auto z = radar.measure(target);          // (15 degrees, 1414.2 m)
    const auto h = radar.jacobian(target);         // exact, 2 x 4: zero in the velocity columns
    const auto where = radar.inverse(z);           // back to (2000, 0, 1500, 0)
    std::printf("  radar: bearing %.2f deg, range %.1f m -> position (%.1f, %.1f)\n", z[0] / deg, z[1], where[0], where[2]);
    static_assert(tr::reversible_measurement_model<decltype(radar)>);
    //! [measurement]

    ok &= check(near(z[0], 15 * deg, 1e-14) && near(z[1], 1000 * std::sqrt(2.0), 1e-9), "bearing is the world bearing less the yaw; range unchanged");
    ok &= check(near(where[0], 2000, 1e-9) && near(where[2], 1500, 1e-9) && where[1] == 0 && where[3] == 0, "inverse(measure(x)) is x's position");
    ok &= check(near(h(1, 0), std::sqrt(0.5), 1e-15) && h(0, 1) == 0 && h(1, 3) == 0 && near(h(0, 0) * 1000 + h(0, 2) * 1000, 0, 1e-15), "range rows are the unit line of sight");
    ok &= check(radar.covariance()(1, 1) == 25, "covariance() is R");

    // A position sensor: H gathers (x, y), its inverse fills them
    const tr::linear_gaussian<double, 4, xy> gps{{{4, 0, 0, 4}}};
    const auto hz = gps.measure(target);
    ok &= check(hz[0] == 2000 && hz[1] == 1500 && gps.matrix()(1, 2) == 1 && gps.jacobian(target) == gps.matrix() && gps.covariance()(0, 0) == 4, "linear: H x is the mapped components");
    ok &= check(gps.inverse(hz)[2] == 1500 && gps.inverse(hz)[1] == 0, "linear inverse");

    // A passive bearing sensor: no range, so no inverse
    const tr::bearing<double, 4, xy> passive{{{1e-4}}, {{1000, 500}}, radar.rotation};
    ok &= check(passive.measure(target)[0] == z[0] && passive.jacobian(target)(0, 0) == h(0, 0) && passive.covariance()(0, 0) == 1e-4, "bearing alone: the radar's first component");
    ok &= check(!tr::reversible_measurement_model<decltype(passive)>, "a bearing sensor is not reversible");

    // In space: relative position (3, 4, 12), climbing at 1 m/s
    const la::fixed_vector<double, 6> aircraft{{4, 0, 5, 0, 13, 1}};
    const la::fixed_vector<double, 3> site{{1, 1, 1}};
    const tr::elevation_bearing<double, 6, xyz> eo{{{1e-6, 0, 0, 1e-6}}, site};
    const tr::elevation_bearing_range<double, 6, xyz> radar3{{{1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1}}, site, ge::sensor_rotation(2 * deg, -5 * deg, 60 * deg)};
    const tr::bearing_range_rate<double, 6, xyz, vxyz> doppler{{{1e-6, 0, 0, 0, 1, 0, 0, 0, 0.01}}, site};
    const auto ze = eo.measure(aircraft);
    const auto zr = radar3.measure(aircraft);
    const auto zd = doppler.measure(aircraft);
    std::printf("  3-D: elevation %.2f deg, bearing %.2f deg, range %.1f m, range rate %.4f m/s\n", ze[0] / deg, ze[1] / deg, zr[2], zd[2]);
    ok &= check(near(ze[0], std::asin(12.0 / 13), 1e-15) && near(ze[1], std::atan2(4.0, 3.0), 1e-15) && eo.covariance()(0, 0) == 1e-6, "elevation and bearing");
    ok &= check(near(zr[2], 13, 1e-13) && radar3.covariance()(2, 2) == 1, "range is unchanged by the sensor's attitude");
    const auto back3 = radar3.inverse(zr);
    ok &= check(near(back3[0], 4, 1e-12) && near(back3[2], 5, 1e-12) && near(back3[4], 13, 1e-12), "the 3-D radar's inverse recovers the position");
    ok &= check(near(zd[2], 12.0 / 13, 1e-15) && near(zd[1], 13, 1e-14) && near(zd[0], ze[1], 1e-15) && doppler.covariance()(2, 2) == 0.01, "range rate is the velocity along the line of sight");
    const auto je = eo.jacobian(aircraft);
    const auto jr = radar3.jacobian(aircraft);
    const auto jd = doppler.jacobian(aircraft);
    ok &= check(near(je(1, 0), -4.0 / 25, 1e-15) && near(jd(2, 5), 12.0 / 13, 1e-15) && near(jr(2, 4), 12.0 / 13, 1e-15), "3-D Jacobians: bearing, range and rate rows");

    // In the plane with a moving sensor: a sensor keeping pace with the target sees no range rate
    tr::bearing_range_rate<double, 4, xy, vxy> doppler2{};
    doppler2.velocity = {{10, -3}};
    ok &= check(doppler2.measure(target)[2] == 0 && doppler2.jacobian(target)(2, 1) > 0 && doppler2.covariance()(0, 0) == 0, "a sensor moving with the target: zero rate");

    // At the sensor itself the angles are 0 and their rows zero; straight up, elevation 90 degrees
    const la::fixed_vector<double, 4> on_top{{1000, 0, 500, 0}};
    const la::fixed_vector<double, 6> overhead{{1, 0, 1, 0, 9, 0}};
    const auto j0 = radar.jacobian(on_top);
    ok &= check(radar.measure(on_top)[0] == 0 && j0(0, 0) == 0 && j0(1, 2) == 0 && passive.jacobian(on_top)(0, 2) == 0, "zero range: zero angle, zero rows, no NaN");
    ok &= check(near(radar3.measure(overhead)[2], 8, 1e-14) && near(eo.measure(overhead)[0], std::numbers::pi / 2, 1e-15) && eo.measure(overhead)[1] == 0 &&
                    eo.jacobian(overhead)(1, 0) == 0,
                "the pole: elevation pi/2, bearing 0");
    ok &= check(doppler.jacobian({{1, 0, 1, 0, 1, 0}})(1, 0) == 0 && doppler.measure({{1, 0, 1, 0, 1, 0}})[2] == 0, "range rate at zero range is zero");

    std::printf(ok ? "measurement demo: ok\n" : "measurement demo: FAILED\n");
    return ok ? 0 : 1;
}
