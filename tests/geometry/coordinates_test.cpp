//
//  coordinates_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/geometry/coordinates.hpp>: polar, spherical and radar-face coordinates against Stone
//  Soup's formulas (cart2pol, cart2sphere, cart2az_el_rg), their inverses and round trips, the zero angles at the
//  origin and on the poles, the edge of the radar face, the rotations' orthogonality and known values, and
//  sensor_rotation against Stone Soup's build_rotation_matrix (computed with numpy).
//

#include <snicholls/core/geometry/coordinates.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
using m33 = la::fixed_matrix<double, 3, 3>;
constexpr double pi = std::numbers::pi;

static_assert(noexcept(geo::cart_to_polar(1.0, 2.0)) && noexcept(geo::cart_to_sphere(1.0, 2.0, 3.0)) && noexcept(geo::cart_to_az_el_range(1.0f, 2.0f, 3.0f)));
static_assert(noexcept(geo::sensor_rotation(1.0, 2.0, 3.0)) && noexcept(geo::rotation_2d(1.0L)) && noexcept(geo::az_el_range_to_cart(1.0, 2.0, 3.0)));

// Constant evaluation
static_assert(geo::cart_to_polar(3.0, 4.0)[0] == 5.0);
static_assert(geo::cart_to_polar(0.0, 0.0) == la::make_vector(0.0, 0.0));
static_assert(geo::cart_to_sphere(0.0, 0.0, 0.0) == la::make_vector(0.0, 0.0, 0.0));
static_assert(geo::cart_to_sphere(1.0, 2.0, 2.0)[0] == 3.0);
static_assert(geo::cart_to_az_el_range(0.0, 0.0, 5.0) == la::make_vector(0.0, 0.0, 5.0));
static_assert(geo::rotation_x(0.0) == m33::identity() && geo::rotation_y(0.0) == m33::identity() && geo::rotation_z(0.0) == m33::identity());
static_assert(geo::sensor_rotation(0.0, 0.0, 0.0) == m33::identity());
static_assert(geo::polar_to_cart(2.0, 0.0) == la::make_vector(2.0, 0.0));
constexpr auto rz = geo::rotation_z(pi / 2);
static_assert(rz(0, 1) == -1.0 && rz(1, 0) == 1.0 && rz(0, 0) < 1e-15 && rz(0, 0) > -1e-15);

template <std::size_t R, std::size_t C>
void expect_near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol)
{
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol) << "(" << i << ", " << j << ")";
}

void expect_rotation(const m33& r)
{
    expect_near(r * la::transpose(r), m33::identity(), 1e-15);
    EXPECT_NEAR(la::determinant(r), 1.0, 1e-15);
}

TEST(GeometryCoordinatesPolar, IsHypotAndAtan2)
{
    const auto p = geo::cart_to_polar(3.0, 4.0);
    EXPECT_EQ(p[0], 5.0);
    EXPECT_DOUBLE_EQ(p[1], std::atan2(4.0, 3.0));
    EXPECT_DOUBLE_EQ(geo::cart_to_polar(-1.0, 0.0)[1], pi);
    EXPECT_DOUBLE_EQ(geo::cart_to_polar(0.0, -2.0)[1], -pi / 2);
    EXPECT_DOUBLE_EQ(geo::cart_to_polar(-1.0, -1.0)[1], -0.75 * pi);
    EXPECT_EQ(geo::cart_to_polar(0.0, 0.0), la::make_vector(0.0, 0.0));
}

TEST(GeometryCoordinatesPolar, RoundTrips)
{
    for (double x = -5; x <= 5; x += 1.25)
        for (double y = -5; y <= 5; y += 1.25) {
            const auto p = geo::cart_to_polar(x, y);
            const auto c = geo::polar_to_cart(p[0], p[1]);
            EXPECT_NEAR(c[0], x, 1e-14);
            EXPECT_NEAR(c[1], y, 1e-14);
        }
    const auto c = geo::polar_to_cart(2.0, pi / 2);
    EXPECT_NEAR(c[0], 0.0, 1e-15);
    EXPECT_DOUBLE_EQ(c[1], 2.0);
    EXPECT_EQ(geo::polar_to_cart(0.0, 1.0), la::make_vector(0.0, 0.0));
}

TEST(GeometryCoordinatesSphere, IsStoneSoupCart2Sphere)
{
    const auto s = geo::cart_to_sphere(1.0, 2.0, 2.0);
    EXPECT_EQ(s[0], 3.0);
    EXPECT_DOUBLE_EQ(s[1], std::atan2(2.0, 1.0));
    EXPECT_DOUBLE_EQ(s[2], std::asin(2.0 / 3.0));
    const auto below = geo::cart_to_sphere(-3.0, 0.0, -4.0);
    EXPECT_EQ(below[0], 5.0);
    EXPECT_DOUBLE_EQ(below[1], pi);
    EXPECT_DOUBLE_EQ(below[2], std::asin(-0.8));
}

TEST(GeometryCoordinatesSphere, OriginAndPolesHaveZeroBearing)
{
    EXPECT_EQ(geo::cart_to_sphere(0.0, 0.0, 0.0), la::make_vector(0.0, 0.0, 0.0));
    const auto up = geo::cart_to_sphere(0.0, 0.0, 5.0);
    EXPECT_EQ(up[0], 5.0);
    EXPECT_EQ(up[1], 0.0);
    EXPECT_DOUBLE_EQ(up[2], pi / 2);
    const auto down = geo::cart_to_sphere(0.0, 0.0, -2.0);
    EXPECT_EQ(down[1], 0.0);
    EXPECT_DOUBLE_EQ(down[2], -pi / 2);
}

TEST(GeometryCoordinatesSphere, RoundTripsAndAnglesAgree)
{
    for (double x = -3; x <= 3; x += 1.5)
        for (double y = -3; y <= 3; y += 1.5)
            for (double z = -3; z <= 3; z += 1.5) {
                const auto s = geo::cart_to_sphere(x, y, z);
                const auto c = geo::sphere_to_cart(s[0], s[1], s[2]);
                EXPECT_NEAR(c[0], x, 1e-14);
                EXPECT_NEAR(c[1], y, 1e-14);
                EXPECT_NEAR(c[2], z, 1e-14);
                const auto a = geo::cart_to_angles(x, y, z);
                EXPECT_EQ(a[0], s[1]);
                EXPECT_EQ(a[1], s[2]);
            }
    EXPECT_EQ(geo::cart_to_angles(0.0, 0.0, 0.0), la::make_vector(0.0, 0.0));
    const auto c = geo::sphere_to_cart(2.0, 0.0, pi / 2);
    EXPECT_NEAR(c[0], 0.0, 1e-15);
    EXPECT_DOUBLE_EQ(c[2], 2.0);
}

TEST(GeometryCoordinatesAzElRange, BoresightIsZ)
{
    EXPECT_EQ(geo::cart_to_az_el_range(0.0, 0.0, 5.0), la::make_vector(0.0, 0.0, 5.0));
    const auto a = geo::cart_to_az_el_range(3.0, 0.0, 4.0);
    EXPECT_DOUBLE_EQ(a[0], std::asin(0.6));
    EXPECT_EQ(a[1], 0.0);
    EXPECT_EQ(a[2], 5.0);
    const auto b = geo::cart_to_az_el_range(0.0, -3.0, 4.0);
    EXPECT_EQ(b[0], 0.0);
    EXPECT_DOUBLE_EQ(b[1], std::asin(-0.6));
    EXPECT_EQ(geo::cart_to_az_el_range(0.0, 0.0, 0.0), la::make_vector(0.0, 0.0, 0.0));
}

TEST(GeometryCoordinatesAzElRange, RoundTripsInFrontOfTheFace)
{
    for (double x = -2; x <= 2; x += 1)
        for (double y = -2; y <= 2; y += 1)
            for (double z = 0.5; z <= 3; z += 1.25) {
                const auto a = geo::cart_to_az_el_range(x, y, z);
                const auto c = geo::az_el_range_to_cart(a[0], a[1], a[2]);
                EXPECT_NEAR(c[0], x, 1e-14);
                EXPECT_NEAR(c[1], y, 1e-14);
                EXPECT_NEAR(c[2], z, 1e-14);
            }
}

TEST(GeometryCoordinatesAzElRange, ImpossibleAnglesMapToTheFaceEdge)
{
    const auto c = geo::az_el_range_to_cart(1.2, 1.2, 10.0);   // sin^2 1.2 * 2 = 1.74 > 1
    EXPECT_EQ(c[2], 0.0);
    EXPECT_FALSE(std::isnan(c[0]) || std::isnan(c[1]) || std::isnan(c[2]));
    EXPECT_DOUBLE_EQ(c[0], 10 * std::sin(1.2));
    const auto edge = geo::az_el_range_to_cart(pi / 2, 0.0, 3.0);
    EXPECT_DOUBLE_EQ(edge[0], 3.0);
    EXPECT_NEAR(edge[2], 0.0, 1e-7);
}

TEST(GeometryCoordinatesRotation, AxisRotationsHaveTheDocumentedSigns)
{
    const double t = 0.7, c = std::cos(t), s = std::sin(t);
    expect_near(geo::rotation_x(t), m33{{1, 0, 0, 0, c, -s, 0, s, c}}, 1e-16);
    expect_near(geo::rotation_y(t), m33{{c, 0, s, 0, 1, 0, -s, 0, c}}, 1e-16);
    expect_near(geo::rotation_z(t), m33{{c, -s, 0, s, c, 0, 0, 0, 1}}, 1e-16);
    expect_near(geo::rotation_2d(t), la::fixed_matrix<double, 2, 2>{{c, -s, s, c}}, 1e-16);
    // Anticlockwise: x goes to y about z, y to z about x, z to x about y
    expect_near(geo::rotation_z(pi / 2) * la::make_vector(1.0, 0.0, 0.0), la::make_vector(0.0, 1.0, 0.0), 1e-15);
    expect_near(geo::rotation_x(pi / 2) * la::make_vector(0.0, 1.0, 0.0), la::make_vector(0.0, 0.0, 1.0), 1e-15);
    expect_near(geo::rotation_y(pi / 2) * la::make_vector(0.0, 0.0, 1.0), la::make_vector(1.0, 0.0, 0.0), 1e-15);
    expect_near(geo::rotation_2d(pi / 2) * la::make_vector(1.0, 0.0), la::make_vector(0.0, 1.0), 1e-15);
}

TEST(GeometryCoordinatesRotation, AreOrthogonalWithDeterminantOne)
{
    for (double t = -4; t <= 4; t += 0.37) {
        expect_rotation(geo::rotation_x(t));
        expect_rotation(geo::rotation_y(t));
        expect_rotation(geo::rotation_z(t));
        expect_rotation(geo::sensor_rotation(t, 0.5 * t, -0.3 * t));
        const auto r2 = geo::rotation_2d(t);
        expect_near(r2 * la::transpose(r2), la::fixed_matrix<double, 2, 2>::identity(), 1e-15);
    }
}

TEST(GeometryCoordinatesRotation, SensorRotationIsStoneSoupBuildRotationMatrix)
{
    // numpy: rotz(-0.3) @ roty(0.2) @ rotx(-0.1), Stone Soup's build_rotation_matrix([[0.1], [0.2], [0.3]])
    const m33 ref{{0.9362933635841992, 0.2750958473182437, 0.21835066314633444, -0.28962947762551555, 0.9564250858492325, 0.03695701352462508,
                   -0.19866933079506122, -0.09784339500725571, 0.975170327201816}};
    expect_near(geo::sensor_rotation(0.1, 0.2, 0.3), ref, 1e-15);
    expect_near(geo::sensor_rotation(0.1, 0.2, 0.3), geo::rotation_z(-0.3) * geo::rotation_y(0.2) * geo::rotation_x(-0.1), 1e-16);
    // A sensor yawed 90 degrees sees the world's y axis ahead, along its own x
    expect_near(geo::sensor_rotation(0.0, 0.0, pi / 2) * la::make_vector(0.0, 1.0, 0.0), la::make_vector(1.0, 0.0, 0.0), 1e-15);
}

TEST(GeometryCoordinatesFloat, WorksInFloat)
{
    const auto p = geo::cart_to_polar(3.0f, 4.0f);
    EXPECT_FLOAT_EQ(p[0], 5.0f);
    const auto s = geo::cart_to_sphere(1.0f, 2.0f, 2.0f);
    EXPECT_FLOAT_EQ(s[0], 3.0f);
    EXPECT_NEAR(la::determinant(geo::sensor_rotation(0.1f, 0.2f, 0.3f)), 1.0f, 1e-6f);
}

} // namespace
