//
//  angle_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/geometry/angle.hpp>: degrees and radians, wrap_pi, wrap_two_pi and wrap_elevation against
//  Stone Soup's mod_bearing and mod_elevation (Python's floor modulo), their ranges at the rounding boundaries and for
//  NaN and infinity, difference, residual and wrapped by angle_kind, the circular mean with equal, negative and zero
//  weights, the weighted mean of vectors, and constant evaluation of each.
//

#include <snicholls/core/geometry/angle.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <utility>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
using geo::angle_kind;
constexpr double pi = std::numbers::pi;

static_assert(sizeof(angle_kind) == 1 && static_cast<int>(angle_kind::linear) == 0 && static_cast<int>(angle_kind::bearing) == 1
              && static_cast<int>(angle_kind::elevation) == 2);
static_assert(geo::all_linear<3> == geo::angle_kinds<3>{angle_kind::linear, angle_kind::linear, angle_kind::linear});
static_assert(geo::pi<double> == std::numbers::pi && geo::pi<float> == std::numbers::pi_v<float>);

// noexcept for the standard floating-point types
static_assert(noexcept(geo::wrap_pi(1.0)) && noexcept(geo::wrap_pi(1.0f)) && noexcept(geo::wrap_pi(1.0L)));
static_assert(noexcept(geo::wrap_elevation(1.0)) && noexcept(geo::wrap_two_pi(1.0)) && noexcept(geo::difference(angle_kind::bearing, 1.0, 2.0)));
static_assert(noexcept(geo::circular_mean<double>({}, {})) && noexcept(geo::radians(1.0)) && noexcept(geo::degrees(1.0)));

// Constant evaluation
static_assert(geo::wrap_pi(0.0) == 0.0 && geo::wrap_pi(pi) == -pi && geo::wrap_pi(-pi) == -pi);
static_assert(geo::wrap_pi(3 * pi) == -pi && geo::wrap_pi(2 * pi + 0.5) > 0.4999999 && geo::wrap_pi(2 * pi + 0.5) < 0.5000001);
static_assert(geo::wrap_two_pi(-0.5) > 2 * pi - 0.5 - 1e-12 && geo::wrap_two_pi(-0.5) < 2 * pi - 0.5 + 1e-12);
static_assert(geo::wrap_elevation(pi / 2 + 0.25) > pi / 2 - 0.25 - 1e-12 && geo::wrap_elevation(pi / 2 + 0.25) < pi / 2 - 0.25 + 1e-12);
static_assert(geo::radians(180.0) == pi && geo::degrees(pi) == 180.0);
static_assert(geo::wrap(angle_kind::linear, 10.0) == 10.0);
constexpr std::array<double, 2> opposite{0.3, 0.3};
constexpr std::array<double, 2> cancel{1.0, -1.0};
static_assert(geo::circular_mean<double>(std::span<const double>(opposite), std::span<const double>(cancel)) == 0.0);
static_assert(geo::circular_mean<double>(std::span<const double>(opposite)) > 0.2999999 && geo::circular_mean<double>(std::span<const double>(opposite)) < 0.3000001);

// Stone Soup's functions, as Python computes them: (x + pi) % (2 pi) - pi, and mod_elevation
double mod_bearing(double x)
{
    const double m = std::fmod(x + pi, 2 * pi);
    return (m < 0 ? m + 2 * pi : m) - pi;
}

TEST(GeometryAngleUnits, DegreesAndRadians)
{
    EXPECT_DOUBLE_EQ(geo::radians(90.0), pi / 2);
    EXPECT_DOUBLE_EQ(geo::degrees(pi / 4), 45.0);
    EXPECT_EQ(geo::radians(0.0), 0.0);
    EXPECT_NEAR(geo::degrees(geo::radians(-123.456)), -123.456, 1e-12);
    EXPECT_FLOAT_EQ(geo::radians(180.0f), std::numbers::pi_v<float>);
}

TEST(GeometryAngleWrapPi, MatchesStoneSoupModBearing)
{
    // Values from Python: (x + math.pi) % (2 * math.pi) - math.pi
    EXPECT_NEAR(geo::wrap_pi(4.0), -2.2831853071795862, 1e-15);
    EXPECT_NEAR(geo::wrap_pi(-4.0), 2.2831853071795862, 1e-15);
    EXPECT_NEAR(geo::wrap_pi(10.0), -2.5663706143591725, 1e-15);
    EXPECT_NEAR(geo::wrap_pi(-10.0), 2.5663706143591725, 1e-15);
    EXPECT_NEAR(geo::wrap_pi(100.0), -0.5309649148733762, 1e-13);
    EXPECT_NEAR(geo::wrap_pi(1e6), -0.3575641670518799, 1e-9);
    for (double x = -50; x < 50; x += 0.0137)
        EXPECT_NEAR(geo::wrap_pi(x), mod_bearing(x), 1e-12) << x;
}

TEST(GeometryAngleWrapPi, IsInTheHalfOpenRangeAtTheBoundaries)
{
    EXPECT_EQ(geo::wrap_pi(pi), -pi);
    EXPECT_EQ(geo::wrap_pi(-pi), -pi);
    EXPECT_EQ(geo::wrap_pi(0.0), 0.0);
    for (const double centre : {-5 * pi, -3 * pi, -pi, pi, 3 * pi, 5 * pi, 0.0, 2 * pi, -2 * pi}) {
        double x = centre;
        for (int k = 0; k < 20; ++k)
            x = std::nextafter(x, -1e300);
        for (int k = 0; k < 40; ++k, x = std::nextafter(x, 1e300)) {
            const double w = geo::wrap_pi(x);
            EXPECT_GE(w, -pi) << x;
            EXPECT_LT(w, pi) << x;
        }
    }
    EXPECT_LT(geo::wrap_pi(std::nextafter(pi, 0.0)), pi);
}

TEST(GeometryAngleWrapPi, NaNAndInfinityGiveNaN)
{
    EXPECT_TRUE(std::isnan(geo::wrap_pi(std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(geo::wrap_pi(std::numeric_limits<double>::infinity())));
    EXPECT_TRUE(std::isnan(geo::wrap_pi(-std::numeric_limits<double>::infinity())));
}

TEST(GeometryAngleWrapPi, FloatAndLongDouble)
{
    EXPECT_NEAR(geo::wrap_pi(4.0f), -2.2831853f, 1e-6f);
    EXPECT_NEAR(static_cast<double>(geo::wrap_pi(4.0L)), -2.2831853071795862, 1e-15);
    EXPECT_EQ(geo::wrap_pi(std::numbers::pi_v<float>), -std::numbers::pi_v<float>);
}

TEST(GeometryAngleWrapTwoPi, IsInZeroToTwoPi)
{
    EXPECT_EQ(geo::wrap_two_pi(0.0), 0.0);
    EXPECT_NEAR(geo::wrap_two_pi(-pi / 2), 1.5 * pi, 1e-15);
    EXPECT_NEAR(geo::wrap_two_pi(7.0), 7.0 - 2 * pi, 1e-15);
    EXPECT_NEAR(geo::wrap_two_pi(pi), pi, 1e-15);
    for (double x = -30; x < 30; x += 0.071) {
        const double w = geo::wrap_two_pi(x);
        EXPECT_GE(w, 0.0) << x;
        EXPECT_LT(w, 2 * pi) << x;
        EXPECT_NEAR(std::cos(w), std::cos(x), 1e-12);
        EXPECT_NEAR(std::sin(w), std::sin(x), 1e-12);
    }
}

TEST(GeometryAngleWrapElevation, MatchesStoneSoupModElevationInEveryQuadrant)
{
    EXPECT_NEAR(geo::wrap_elevation(0.3), 0.3, 1e-15);                      // quadrant 0: unchanged
    EXPECT_NEAR(geo::wrap_elevation(pi / 2 + 0.2), pi / 2 - 0.2, 1e-15);    // quadrant 1: back down
    EXPECT_NEAR(geo::wrap_elevation(pi + 0.3), -0.3, 1e-15);                // quadrant 2: the other side, below
    EXPECT_NEAR(geo::wrap_elevation(1.5 * pi + 0.2), -pi / 2 + 0.2, 1e-15); // quadrant 3: x - 2 pi
    EXPECT_NEAR(geo::wrap_elevation(-0.3), -0.3, 1e-15);
    EXPECT_NEAR(geo::wrap_elevation(pi / 2), pi / 2, 1e-15);
    EXPECT_NEAR(geo::wrap_elevation(-pi / 2), -pi / 2, 1e-15);
    // Values from Stone Soup's mod_elevation in Python
    EXPECT_NEAR(geo::wrap_elevation(4.0), -0.8584073464102069, 1e-15);
    EXPECT_NEAR(geo::wrap_elevation(-4.0), 0.8584073464102069, 1e-15);
    EXPECT_NEAR(geo::wrap_elevation(10.0), -0.5752220392306207, 1e-14);
    EXPECT_NEAR(geo::wrap_elevation(-10.0), 0.5752220392306207, 1e-14);
    EXPECT_NEAR(geo::wrap_elevation(100.0), -0.5309649148733797, 1e-13);
    for (double x = -20; x < 20; x += 0.0173) {
        const double w = geo::wrap_elevation(x);
        EXPECT_GE(w, -pi / 2) << x;
        EXPECT_LE(w, pi / 2) << x;
        EXPECT_NEAR(std::sin(w), std::sin(x), 1e-12) << x;   // the same height above the plane
    }
    // Just below zero, the reduction lands at (or rounds to) the top of the circle
    EXPECT_NEAR(geo::wrap_elevation(-1e-17), 0.0, 1e-15);
    EXPECT_NEAR(geo::wrap_elevation(-5e-16), 0.0, 1e-15);
}

TEST(GeometryAngleWrap, DispatchesOnTheKind)
{
    EXPECT_EQ(geo::wrap(angle_kind::linear, 7.0), 7.0);
    EXPECT_EQ(geo::wrap(angle_kind::bearing, 7.0), geo::wrap_pi(7.0));
    EXPECT_EQ(geo::wrap(angle_kind::elevation, 2.0), geo::wrap_elevation(2.0));
}

TEST(GeometryAngleDifference, TakesTheShorterWayRoundForABearing)
{
    const double a = geo::radians(179.0), b = geo::radians(-179.0);
    EXPECT_NEAR(geo::difference(angle_kind::bearing, a, b), geo::radians(-2.0), 1e-14);
    EXPECT_NEAR(geo::difference(angle_kind::bearing, b, a), geo::radians(2.0), 1e-14);
    EXPECT_NEAR(geo::difference(angle_kind::linear, a, b), geo::radians(358.0), 1e-14);
    EXPECT_EQ(geo::difference(angle_kind::bearing, 1.0, 1.0), 0.0);
    EXPECT_NEAR(geo::difference(angle_kind::elevation, 1.4, -0.4), pi - 1.8, 1e-14);
}

TEST(GeometryAngleResidual, WrapsOnlyTheAngleComponents)
{
    constexpr geo::angle_kinds<3> kinds{angle_kind::bearing, angle_kind::elevation, angle_kind::linear};
    const auto a = la::make_vector(geo::radians(179.0), 1.4, 1000.0);
    const auto b = la::make_vector(geo::radians(-179.0), -0.4, 10.0);
    const auto r = geo::residual<kinds>(a, b);
    EXPECT_NEAR(r[0], geo::radians(-2.0), 1e-14);
    EXPECT_NEAR(r[1], pi - 1.8, 1e-14);
    EXPECT_EQ(r[2], 990.0);
    EXPECT_EQ(geo::residual<geo::all_linear<3>>(a, b), a - b);
}

TEST(GeometryAngleWrapped, WrapsEachComponentAsItsKindSays)
{
    constexpr geo::angle_kinds<3> kinds{angle_kind::linear, angle_kind::bearing, angle_kind::elevation};
    const auto w = geo::wrapped<kinds>(la::make_vector(10.0, 10.0, 10.0));
    EXPECT_EQ(w[0], 10.0);
    EXPECT_NEAR(w[1], -2.5663706143591725, 1e-15);
    EXPECT_NEAR(w[2], -0.5752220392306207, 1e-14);
    constexpr auto cw = geo::wrapped<kinds>(la::make_vector(1.0, 3 * pi, 0.25));
    static_assert(cw[0] == 1.0 && cw[1] == -pi && cw[2] == 0.25);
}

TEST(GeometryAngleCircularMean, AveragesAcrossTheCut)
{
    const std::array<double, 2> across{geo::radians(179.0), geo::radians(-179.0)};
    EXPECT_NEAR(std::abs(geo::circular_mean<double>(across)), pi, 1e-14);
    const std::array<double, 3> a{geo::radians(170.0), geo::radians(-170.0), geo::radians(175.0)};
    const std::array<double, 3> w{0.5, 0.3, 0.2};
    EXPECT_NEAR(geo::circular_mean<double>(a, w), 3.0887985108789158, 1e-14);   // atan2(sum w sin, sum w cos), Python
    const std::array<double, 3> neg{2.0, -0.5, -0.5};                           // an unscented centre weight is negative
    EXPECT_NEAR(geo::circular_mean<double>(a, neg), 2.7620579215285295, 1e-14);
    const std::array<double, 1> one{-1.0};
    EXPECT_NEAR(geo::circular_mean<double>(one), -1.0, 1e-15);
}

TEST(GeometryAngleCircularMean, EmptyAndZeroResultantGiveZero)
{
    EXPECT_EQ(geo::circular_mean<double>({}), 0.0);
    EXPECT_EQ(geo::circular_mean<double>({}, {}), 0.0);
    const std::array<double, 2> a{0.7, 0.7}, w{1.0, -1.0};
    EXPECT_EQ(geo::circular_mean<double>(a, w), 0.0);
    const std::array<double, 2> zw{0.0, 0.0};
    EXPECT_EQ(geo::circular_mean<double>(a, zw), 0.0);
}

TEST(GeometryAngleWeightedMean, IsArithmeticInLinearAndCircularInAngles)
{
    constexpr geo::angle_kinds<3> kinds{angle_kind::linear, angle_kind::bearing, angle_kind::elevation};
    const std::array<la::fixed_vector<double, 3>, 3> points{la::make_vector(1.0, geo::radians(170.0), 0.1), la::make_vector(2.0, geo::radians(-170.0), 0.2),
                                                           la::make_vector(4.0, geo::radians(175.0), 0.3)};
    const std::array<double, 3> w{0.5, 0.3, 0.2};
    const auto m = geo::weighted_mean<kinds>(points, w);
    EXPECT_NEAR(m[0], 0.5 * 1 + 0.3 * 2 + 0.2 * 4, 1e-15);
    EXPECT_NEAR(m[1], 3.0887985108789158, 1e-14);
    const double s = 0.5 * std::sin(0.1) + 0.3 * std::sin(0.2) + 0.2 * std::sin(0.3);
    const double c = 0.5 * std::cos(0.1) + 0.3 * std::cos(0.2) + 0.2 * std::cos(0.3);
    EXPECT_NEAR(m[2], std::atan2(s, c), 1e-15);
    // All linear: the plain weighted mean
    const auto lin = geo::weighted_mean<geo::all_linear<3>>(points, w);
    EXPECT_NEAR(lin[1], 0.5 * points[0][1] + 0.3 * points[1][1] + 0.2 * points[2][1], 1e-15);
}

TEST(GeometryAngleWeightedMean, RunsInAConstantExpression)
{
    constexpr geo::angle_kinds<2> kinds{angle_kind::linear, angle_kind::bearing};
    constexpr std::array<la::fixed_vector<double, 2>, 2> points{la::make_vector(2.0, 0.5), la::make_vector(4.0, 0.5)};
    constexpr std::array<double, 2> w{0.5, 0.5};
    constexpr auto m = geo::weighted_mean<kinds>(points, w);
    static_assert(m[0] == 3.0 && m[1] > 0.4999999 && m[1] < 0.5000001);
    EXPECT_NEAR(m[1], 0.5, 1e-15);
}

} // namespace
