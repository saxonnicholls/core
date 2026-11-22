//
//  coordinates_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/geometry/coordinates.hpp>: a target seen by a rotated sensor, as range, bearing and
//  elevation and in the radar-face convention, converted back to Cartesian; polar coordinates, the origin, the edge
//  of the face, and the rotations. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/geometry/coordinates.hpp>

#include <cmath>
#include <cstdio>
#include <numbers>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
constexpr double pi = std::numbers::pi;

bool check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

template <std::size_t R, std::size_t C>
bool near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol = 1e-12)
{
    return la::norm(a - b) <= tol;
}

} // namespace

int main()
{
    std::printf("snicholls-core geometry coordinates demo\n");
    bool ok = true;

    //! [coordinates]
    // A sensor yawed 30° and pitched 5°; a target 1 km north-east and 100 m up, in the world frame
    const auto to_sensor = geo::sensor_rotation(0.0, 5 * pi / 180, 30 * pi / 180);
    const auto target = la::make_vector(707.0, 707.0, 100.0);
    const auto local = to_sensor * target;
    // What the sensor measures: (range, bearing, elevation)
    const auto rbe = geo::cart_to_sphere(local[0], local[1], local[2]);
    // ... and back, through the transpose of the rotation
    const auto back = la::transpose(to_sensor) * geo::sphere_to_cart(rbe[0], rbe[1], rbe[2]);
    //! [coordinates]
    std::printf("  range %.2f m, bearing %+.3f deg, elevation %+.3f deg\n", rbe[0], rbe[1] * 180 / pi, rbe[2] * 180 / pi);
    ok &= check(std::abs(rbe[0] - la::norm(target)) < 1e-9, "a rotation keeps the range");
    ok &= check(near(back, target, 1e-9), "sphere_to_cart inverts cart_to_sphere");
    ok &= check(near(to_sensor, geo::rotation_z(-30 * pi / 180) * geo::rotation_y(5 * pi / 180) * geo::rotation_x(-0.0)), "sensor_rotation = R_z(-yaw) R_y(pitch) R_x(-roll)");
    const auto angles = geo::cart_to_angles(local[0], local[1], local[2]);
    ok &= check(angles[0] == rbe[1] && angles[1] == rbe[2], "cart_to_angles is the bearing and elevation");

    // Polar
    const auto p = geo::cart_to_polar(-3.0, 4.0);
    ok &= check(p[0] == 5 && std::abs(p[1] - std::atan2(4.0, -3.0)) < 1e-15, "cart_to_polar(-3, 4) = (5, atan2(4, -3))");
    ok &= check(near(geo::polar_to_cart(p[0], p[1]), la::make_vector(-3.0, 4.0)), "polar_to_cart inverts it");

    // The origin and the poles have zero angles, not NaN
    ok &= check(geo::cart_to_polar(0.0, 0.0) == la::make_vector(0.0, 0.0) && geo::cart_to_sphere(0.0, 0.0, 0.0) == la::make_vector(0.0, 0.0, 0.0), "the origin");
    const auto pole = geo::cart_to_sphere(0.0, 0.0, 2.0);
    ok &= check(pole[1] == 0 && std::abs(pole[2] - pi / 2) < 1e-15, "straight up: bearing 0, elevation 90 deg");

    // The radar face, boresight z
    const auto aer = geo::cart_to_az_el_range(3.0, 0.0, 4.0);
    ok &= check(std::abs(aer[0] - std::asin(0.6)) < 1e-15 && aer[1] == 0 && aer[2] == 5, "cart_to_az_el_range(3, 0, 4)");
    ok &= check(near(geo::az_el_range_to_cart(aer[0], aer[1], aer[2]), la::make_vector(3.0, 0.0, 4.0)), "az_el_range_to_cart inverts it");
    ok &= check(geo::cart_to_az_el_range(0.0, 0.0, 0.0) == la::make_vector(0.0, 0.0, 0.0), "the face's origin");
    const auto edge = geo::az_el_range_to_cart(1.2, 1.2, 10.0);   // sin^2 + sin^2 > 1: no such direction
    ok &= check(edge[2] == 0 && !std::isnan(edge[0]), "impossible angles land on the face's edge");

    // The rotations
    ok &= check(near(geo::rotation_x(pi / 2) * la::make_vector(0.0, 1.0, 0.0), la::make_vector(0.0, 0.0, 1.0)), "rotation_x takes y to z");
    ok &= check(near(geo::rotation_y(pi / 2) * la::make_vector(0.0, 0.0, 1.0), la::make_vector(1.0, 0.0, 0.0)), "rotation_y takes z to x");
    ok &= check(near(geo::rotation_z(pi / 2) * la::make_vector(1.0, 0.0, 0.0), la::make_vector(0.0, 1.0, 0.0)), "rotation_z takes x to y");
    ok &= check(near(geo::rotation_2d(pi / 2) * la::make_vector(1.0, 0.0), la::make_vector(0.0, 1.0)), "rotation_2d is anticlockwise");
    const auto r = geo::sensor_rotation(0.3, -0.2, 1.1);
    ok &= check(near(r * la::transpose(r), la::fixed_matrix<double, 3, 3>::identity(), 1e-15) && std::abs(la::determinant(r) - 1) < 1e-15, "orthogonal, determinant 1");

    std::printf(ok ? "geometry coordinates demo: ok\n" : "geometry coordinates demo: FAILED\n");
    return ok ? 0 : 1;
}
