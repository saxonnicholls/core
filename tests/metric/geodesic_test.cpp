//
//  geodesic_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/metric/geodesic.hpp>: haversine against city-pair distances computed independently, the
//  same point, antipodal points (π r, never NaN), short distances against the flat-Earth approximation, a custom
//  radius, a third coordinate ignored, mixed point types, the metric axioms on random points, the concepts it
//  satisfies, and evaluation in a constant expression.
//

#include <snicholls/core/metric/geodesic.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace me = snicholls::metric;
namespace st = snicholls::stats;

using ll = std::array<double, 2>;   // (latitude, longitude), radians

constexpr double pi = std::numbers::pi;
constexpr double earth = 6371008.8;

constexpr ll degrees(double lat, double lon) { return {lat * pi / 180, lon * pi / 180}; }

static_assert(me::true_metric<me::haversine>);
static_assert(me::distance_for<me::haversine, ll>);
static_assert(me::distance_for<me::haversine, la::fixed_vector<double, 3>, ll>);
static_assert(!me::reducible<me::haversine, ll>);
static_assert(me::haversine{}.radius == earth);
static_assert(noexcept(me::haversine{}(ll{}, ll{})));
static_assert(noexcept(me::haversine{}(la::fixed_vector<double, 3>{}, ll{})));

constexpr bool compile_time()
{
    const me::haversine h{};
    const ll london = degrees(51.5074, -0.1278), paris = degrees(48.8566, 2.3522);
    const double d = h(london, paris);
    const double quarter = me::haversine{1.0}(ll{0, 0}, ll{0, pi / 2});
    return h(london, london) == 0 && d > 343556.0 && d < 343557.0 && quarter > pi / 2 - 1e-12 && quarter < pi / 2 + 1e-12;
}
static_assert(compile_time());

// Reference distances from Python 3 / numpy, independently of core: each position as a unit vector
// (cos φ cos λ, cos φ sin λ, sin φ), the angle as atan2(|u × v|, u · v), times 6371008.8 m.
TEST(MetricHaversine, MatchesIndependentlyComputedCityPairs)
{
    const ll london = degrees(51.5074, -0.1278), paris = degrees(48.8566, 2.3522), new_york = degrees(40.7128, -74.0060),
             sydney = degrees(-33.8688, 151.2093), tokyo = degrees(35.6762, 139.6503);
    const me::haversine h{};
    EXPECT_NEAR(h(london, paris), 343556.53488088306, 1e-3);
    EXPECT_NEAR(h(london, new_york), 5570229.873656523, 1e-3);
    EXPECT_NEAR(h(sydney, tokyo), 7825829.425997214, 1e-3);
    EXPECT_NEAR(h(new_york, sydney), 15988777.591649339, 1e-3);
}

TEST(MetricHaversine, TheSamePointIsZero)
{
    const me::haversine h{};
    for (const ll p : {ll{0, 0}, degrees(51.5, -0.1), degrees(90, 0), degrees(-90, 123), degrees(10, 180), degrees(-10, -180)})
        EXPECT_EQ(h(p, p), 0.0);
}

TEST(MetricHaversine, AntipodalPointsArePiRNeverNaN)
{
    const me::haversine h{};
    EXPECT_NEAR(h(ll{0, 0}, ll{0, pi}), pi * earth, 1e-6);
    EXPECT_NEAR(h(ll{pi / 2, 0}, ll{-pi / 2, 0}), pi * earth, 1e-6);
    st::wyrand g(5);
    for (int i = 0; i < 2000; ++i) {
        const double lat = (st::uniform_real(g) - 0.5) * pi, lon = (2 * st::uniform_real(g) - 1) * pi;
        const double d = h(ll{lat, lon}, ll{-lat, lon + pi});
        ASSERT_FALSE(std::isnan(d)) << i;
        EXPECT_LE(d, pi * earth) << i;
        EXPECT_NEAR(d, pi * earth, 1.0) << i;   // the haversine form near h = 1 keeps about half the digits
    }
}

TEST(MetricHaversine, ShortDistancesMatchTheFlatEarthApproximation)
{
    const me::haversine h{};
    st::wyrand g(7);
    for (int i = 0; i < 500; ++i) {
        const double lat = (st::uniform_real(g) - 0.5) * 2.8, lon = (2 * st::uniform_real(g) - 1) * pi;   // |φ| < 80°
        const double dlat = (st::uniform_real(g) - 0.5) * 2e-5, dlon = (st::uniform_real(g) - 0.5) * 2e-5;   // tens of metres
        const double flat = earth * std::hypot(dlat, std::cos(lat + dlat / 2) * dlon);
        const double d = h(ll{lat, lon}, ll{lat + dlat, lon + dlon});
        EXPECT_NEAR(d, flat, 1e-6 * flat + 1e-9) << i;
    }
    // One metre north along a meridian: 1 / r radians
    EXPECT_NEAR(h(ll{0.3, 0.1}, ll{0.3 + 1 / earth, 0.1}), 1.0, 1e-9);
}

TEST(MetricHaversine, ACustomRadiusScalesTheDistance)
{
    const ll a = degrees(10, 20), b = degrees(-30, 75);
    EXPECT_NEAR(me::haversine{1.0}(ll{0, 0}, ll{0, pi / 2}), pi / 2, 1e-15);
    EXPECT_NEAR(me::haversine{1.0}(ll{0, 0}, ll{pi / 2, 0}), pi / 2, 1e-15);
    const double unit = me::haversine{1.0}(a, b);
    EXPECT_NEAR(me::haversine{6378137.0}(a, b), 6378137.0 * unit, 1e-6);
    EXPECT_NEAR(me::haversine{}(a, b), earth * unit, 1e-6);
    EXPECT_NEAR(me::haversine{3389500.0}(a, b), 3389500.0 * unit, 1e-6);   // Mars
}

TEST(MetricHaversine, IgnoresAThirdCoordinateAndMixesPointTypes)
{
    const me::haversine h{};
    const la::fixed_vector<double, 3> london_high{{51.5074 * pi / 180, -0.1278 * pi / 180, 10000.0}};
    const la::fixed_vector<double, 3> london_low{{51.5074 * pi / 180, -0.1278 * pi / 180, -50.0}};
    const ll paris = degrees(48.8566, 2.3522);
    const std::vector<float> paris_f{static_cast<float>(paris[0]), static_cast<float>(paris[1])};
    EXPECT_EQ(h(london_high, london_low), 0.0);
    EXPECT_EQ(h(london_high, paris), h(degrees(51.5074, -0.1278), paris));
    EXPECT_NEAR(h(london_high, paris_f), 343556.53488088306, 5.0);   // float coordinates: about a metre's resolution
}

TEST(MetricHaversine, IsAMetricOnRandomPoints)
{
    const me::haversine h{};
    st::wyrand g(11);
    const auto random_point = [&g] { return ll{std::asin(2 * st::uniform_real(g) - 1), (2 * st::uniform_real(g) - 1) * pi}; };
    for (int i = 0; i < 2000; ++i) {
        const ll a = random_point(), b = random_point(), c = random_point();
        const double ab = h(a, b), ba = h(b, a), ac = h(a, c), cb = h(c, b);
        EXPECT_GE(ab, 0.0);
        EXPECT_LE(ab, pi * earth);
        EXPECT_NEAR(ab, ba, 1e-6) << i;
        EXPECT_LE(ab, ac + cb + 1e-6) << i;
        EXPECT_EQ(h(a, a), 0.0);
        if (a != b) {
            EXPECT_GT(ab, 0.0);
        }
    }
}

} // namespace
