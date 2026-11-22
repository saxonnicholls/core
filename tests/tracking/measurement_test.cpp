//
//  measurement_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/tracking/measurement.hpp>: the three concepts on every model, the angle kinds, each
//  model's measurement against hand-computed geometry for a translated and rotated sensor, every analytic Jacobian
//  against central differences, the inverse round trips of the reversible models, range rate for moving targets and
//  sensors in the plane and in space, the sensor's own position and the pole (no NaN), and models in constant
//  expressions.
//

#include <snicholls/core/tracking/measurement.hpp>

#include <snicholls/core/geometry/angle.hpp>
#include <snicholls/core/geometry/coordinates.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/tracking/kalman.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>

namespace {

namespace tr = snicholls::tracking;
namespace la = snicholls::linalg;
namespace ge = snicholls::geometry;

// State (x, vx, y, vy) in the plane and (x, vx, y, vy, z, vz) in space
constexpr std::array<std::size_t, 2> pos2{0, 2};
constexpr std::array<std::size_t, 2> vel2{1, 3};
constexpr std::array<std::size_t, 3> pos3{0, 2, 4};
constexpr std::array<std::size_t, 3> vel3{1, 3, 5};

using lin = tr::linear_gaussian<double, 4, pos2>;
using brg = tr::bearing<double, 4, pos2>;
using br = tr::bearing_range<double, 4, pos2>;
using eb = tr::elevation_bearing<double, 6, pos3>;
using ebr = tr::elevation_bearing_range<double, 6, pos3>;
using brr2 = tr::bearing_range_rate<double, 4, pos2, vel2>;
using brr3 = tr::bearing_range_rate<double, 6, pos3, vel3>;

using v4 = la::fixed_vector<double, 4>;
using v6 = la::fixed_vector<double, 6>;

// The concepts
static_assert(tr::measurement_model<lin> && tr::linear_measurement_model<lin> && tr::reversible_measurement_model<lin>);
static_assert(tr::measurement_model<brg> && !tr::linear_measurement_model<brg> && !tr::reversible_measurement_model<brg>);
static_assert(tr::measurement_model<br> && !tr::linear_measurement_model<br> && tr::reversible_measurement_model<br>);
static_assert(tr::measurement_model<eb> && !tr::linear_measurement_model<eb> && !tr::reversible_measurement_model<eb>);
static_assert(tr::measurement_model<ebr> && !tr::linear_measurement_model<ebr> && tr::reversible_measurement_model<ebr>);
static_assert(tr::measurement_model<brr2> && tr::measurement_model<brr3> && !tr::linear_measurement_model<brr3> && !tr::reversible_measurement_model<brr2>);
// A model without kinds is not one
struct no_kinds {
    using value_type = double;
    static constexpr std::size_t state_dim = 2, measurement_dim = 1;
    la::fixed_vector<double, 1> measure(const la::fixed_vector<double, 2>& x) const { return {{x[0]}}; }
    la::fixed_matrix<double, 1, 2> jacobian(const la::fixed_vector<double, 2>&) const { return {{1, 0}}; }
    la::fixed_matrix<double, 1, 1> covariance() const { return {}; }
};
static_assert(!tr::measurement_model<no_kinds>);

// Dimensions and kinds
static_assert(lin::state_dim == 4 && lin::measurement_dim == 2 && lin::kinds == ge::all_linear<2>);
static_assert(brg::measurement_dim == 1 && brg::kinds[0] == ge::angle_kind::bearing);
static_assert(br::kinds == ge::angle_kinds<2>{ge::angle_kind::bearing, ge::angle_kind::linear});
static_assert(eb::kinds == ge::angle_kinds<2>{ge::angle_kind::elevation, ge::angle_kind::bearing});
static_assert(ebr::kinds == ge::angle_kinds<3>{ge::angle_kind::elevation, ge::angle_kind::bearing, ge::angle_kind::linear});
static_assert(brr2::kinds == ge::angle_kinds<3>{ge::angle_kind::bearing, ge::angle_kind::linear, ge::angle_kind::linear} && brr2::d == 2 && brr3::d == 3);

// Every model is a trivially copyable aggregate
static_assert(std::is_trivially_copyable_v<lin> && std::is_trivially_copyable_v<brg> && std::is_trivially_copyable_v<br> && std::is_trivially_copyable_v<eb> &&
              std::is_trivially_copyable_v<ebr> && std::is_trivially_copyable_v<brr2> && std::is_trivially_copyable_v<brr3>);
static_assert(std::is_aggregate_v<lin> && std::is_aggregate_v<br> && std::is_aggregate_v<ebr> && std::is_aggregate_v<brr3>);

// noexcept for double
static_assert(noexcept(br{}.measure(std::declval<const v4&>())) && noexcept(br{}.jacobian(std::declval<const v4&>())) && noexcept(br{}.inverse({})));
static_assert(noexcept(ebr{}.jacobian(std::declval<const v6&>())) && noexcept(brr3{}.jacobian(std::declval<const v6&>())) && noexcept(lin::matrix()));

// In a constant expression: a target at (3, 4) from the origin is at range 5, bearing atan2(4, 3); H is built at
// compile time
static_assert(br{}.measure({{3, 0, 4, 0}})[1] == 5);
static_assert(lin::matrix() == la::fixed_matrix<double, 2, 4>{{1, 0, 0, 0, 0, 0, 1, 0}});
constexpr bool rate_at_compile_time()
{
    const auto z = brr2{}.measure({{3, 1, 4, 0}});
    return z[1] == 5 && z[2] > 0.6 - 1e-15 && z[2] < 0.6 + 1e-15 && z[0] > 0.927 && z[0] < 0.928;
}
static_assert(rate_at_compile_time());
constexpr bool inverse_at_compile_time()
{
    const ebr s{};
    const auto x = s.inverse({{0.3, -1.2, 40}});
    const auto z = s.measure(x);
    return z[2] > 40 - 1e-9 && z[2] < 40 + 1e-9 && z[0] > 0.3 - 1e-9 && z[0] < 0.3 + 1e-9;
}
static_assert(inverse_at_compile_time());

template <std::size_t R, std::size_t C>
void expect_near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol, const char* what = "")
{
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol) << what << " (" << i << ", " << j << ")";
}

template <typename S>
void expect_jacobian(const S& s, const la::fixed_vector<double, S::state_dim>& x, double tol, const char* what)
{
    const auto numeric = tr::numerical_jacobian<S::measurement_dim>([&](const auto& y) { return s.measure(y); }, x);
    expect_near(s.jacobian(x), numeric, tol, what);
}

template <std::size_t N>
bool finite(const la::fixed_matrix<double, N, 1>& v)
{
    for (const double e : v.elements)
        if (!std::isfinite(e))
            return false;
    return true;
}

const double deg = std::numbers::pi / 180;

TEST(TrackingMeasurementLinearGaussian, GathersTheMappedComponents)
{
    const lin s{{{2, 0.5, 0.5, 3}}};
    const v4 x{{1, 2, 3, 4}};
    EXPECT_EQ(s.measure(x), (la::fixed_vector<double, 2>{{1, 3}}));
    EXPECT_EQ(s.matrix() * x, s.measure(x));
    EXPECT_EQ(s.jacobian(x), lin::matrix());
    EXPECT_EQ(s.covariance(), (la::fixed_matrix<double, 2, 2>{{2, 0.5, 0.5, 3}}));
    // The inverse puts the measurement in the mapped components, zero elsewhere
    EXPECT_EQ(s.inverse({{7, 8}}), (v4{{7, 0, 8, 0}}));
    EXPECT_EQ(s.measure(s.inverse({{7, 8}})), (la::fixed_vector<double, 2>{{7, 8}}));
    // A map out of order measures in its own order
    constexpr std::array<std::size_t, 3> order{3, 0, 0};
    const tr::linear_gaussian<double, 4, order> t{};
    EXPECT_EQ(t.measure(x), (la::fixed_vector<double, 3>{{4, 1, 1}}));
}

TEST(TrackingMeasurementBearing, IsTheAngleInTheSensorsFrame)
{
    // A target at (11, 12) seen from (1, 2) is at 45 degrees in the world; a sensor yawed by 30 degrees sees 15
    brg s{};
    s.translation = {{1, 2}};
    const v4 x{{11, 0, 12, 0}};
    EXPECT_NEAR(s.measure(x)[0], 45 * deg, 1e-15);
    s.rotation = ge::sensor_rotation(0.0, 0.0, 30 * deg);
    EXPECT_NEAR(s.measure(x)[0], 15 * deg, 1e-15);
    s.noise = {{1e-4}};
    EXPECT_EQ(s.covariance()(0, 0), 1e-4);
    // Behind the sensor: bearing -165 degrees, inside [-pi, pi)
    EXPECT_NEAR(s.measure({{-9, 0, -8, 0}})[0], -165 * deg, 1e-14);
}

TEST(TrackingMeasurementBearingRange, IsBearingAndRangeAndReverses)
{
    br s{};
    s.translation = {{1, 2}};
    s.rotation = ge::sensor_rotation(0.0, 0.0, 30 * deg);
    s.noise = {{1e-4, 0, 0, 4}};
    const v4 x{{11, 5, 12, -1}};
    const auto z = s.measure(x);
    EXPECT_NEAR(z[0], 15 * deg, 1e-15);
    EXPECT_NEAR(z[1], std::sqrt(200.0), 1e-14);
    EXPECT_EQ(s.covariance(), (la::fixed_matrix<double, 2, 2>{{1e-4, 0, 0, 4}}));
    // inverse(measure(x)) recovers the position, velocities zero
    const auto back = s.inverse(z);
    EXPECT_NEAR(back[0], 11, 1e-13);
    EXPECT_NEAR(back[2], 12, 1e-13);
    EXPECT_EQ(back[1], 0.0);
    EXPECT_EQ(back[3], 0.0);
    // measure(inverse(z)) recovers the measurement
    const la::fixed_vector<double, 2> zz{{-2.5, 70}};
    expect_near(s.measure(s.inverse(zz)), zz, 1e-13, "round trip");
}

TEST(TrackingMeasurementElevationBearing, IsElevationAndBearingInSpace)
{
    // Relative (3, 4, 12): elevation asin(12 / 13), bearing atan2(4, 3); yawed by 10 degrees, bearing less 10
    eb s{};
    s.translation = {{1, 1, 1}};
    const v6 x{{4, 0, 5, 0, 13, 0}};
    EXPECT_NEAR(s.measure(x)[0], std::asin(12.0 / 13), 1e-15);
    EXPECT_NEAR(s.measure(x)[1], std::atan2(4.0, 3.0), 1e-15);
    s.rotation = ge::sensor_rotation(0.0, 0.0, 10 * deg);
    EXPECT_NEAR(s.measure(x)[0], std::asin(12.0 / 13), 1e-15);
    EXPECT_NEAR(s.measure(x)[1], std::atan2(4.0, 3.0) - 10 * deg, 1e-15);
    s.noise = {{1, 0, 0, 2}};
    EXPECT_EQ(s.covariance()(1, 1), 2.0);
}

TEST(TrackingMeasurementElevationBearingRange, IsElevationBearingAndRangeAndReverses)
{
    ebr s{};
    s.translation = {{1, 1, 1}};
    const v6 x{{4, 0, 5, 0, 13, 0}};
    EXPECT_NEAR(s.measure(x)[2], 13, 1e-14);
    // Rolled, pitched and yawed, the range is unchanged and the inverse still round-trips
    s.rotation = ge::sensor_rotation(5 * deg, -20 * deg, 130 * deg);
    s.noise = {{1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 1}};
    EXPECT_NEAR(s.measure(x)[2], 13, 1e-14);
    const auto back = s.inverse(s.measure(x));
    EXPECT_NEAR(back[0], 4, 1e-13);
    EXPECT_NEAR(back[2], 5, 1e-13);
    EXPECT_NEAR(back[4], 13, 1e-13);
    EXPECT_EQ(back[1] + back[3] + back[5], 0.0);
    const la::fixed_vector<double, 3> z{{-0.4, 2.9, 250}};
    expect_near(s.measure(s.inverse(z)), z, 1e-12, "round trip");
    EXPECT_EQ(s.covariance()(2, 2), 1.0);
}

TEST(TrackingMeasurementBearingRangeRate, RateIsTheRelativeVelocityAlongTheLineOfSight)
{
    // In the plane: relative position (3, 4), relative velocity (1, 0): rate 3/5
    brr2 s{};
    EXPECT_NEAR(s.measure({{3, 1, 4, 0}})[2], 0.6, 1e-15);
    EXPECT_NEAR(s.measure({{3, 1, 4, 0}})[1], 5, 1e-15);
    // A sensor moving with the target sees no rate
    s.velocity = {{1, 0}};
    EXPECT_NEAR(s.measure({{3, 1, 4, 0}})[2], 0.0, 1e-15);
    // Crossing at right angles to the line of sight: zero rate
    s.velocity = {};
    EXPECT_NEAR(s.measure({{3, -4, 4, 3}})[2], 0.0, 1e-15);
    // In space: relative (3, 4, 12), climbing at 1: rate 12/13; translated and yawed, the same
    brr3 t{};
    t.translation = {{1, 1, 1}};
    t.rotation = ge::sensor_rotation(0.0, 0.0, 40 * deg);
    const v6 x{{4, 0, 5, 0, 13, 1}};
    EXPECT_NEAR(t.measure(x)[2], 12.0 / 13, 1e-15);
    EXPECT_NEAR(t.measure(x)[1], 13, 1e-14);
    EXPECT_NEAR(t.measure(x)[0], std::atan2(4.0, 3.0) - 40 * deg, 1e-15);
    t.noise = {{1, 0, 0, 0, 2, 0, 0, 0, 3}};
    EXPECT_EQ(t.covariance()(2, 2), 3.0);
}

TEST(TrackingMeasurementJacobian, EveryModelMatchesCentralDifferences)
{
    const auto yaw = ge::sensor_rotation(0.0, 0.0, 30 * deg);
    const auto full = ge::sensor_rotation(5 * deg, -20 * deg, 130 * deg);
    expect_jacobian(lin{}, v4{{1, 2, 3, 4}}, 1e-10, "linear_gaussian");
    expect_jacobian(brg{{}, {{1, 2}}, yaw}, v4{{11, 5, 12, -1}}, 1e-9, "bearing");
    expect_jacobian(br{{}, {{1, 2}}, yaw}, v4{{11, 5, 12, -1}}, 1e-9, "bearing_range");
    expect_jacobian(br{{}, {{-30, 2}}, yaw}, v4{{-35, 5, -1.5, -1}}, 1e-9, "bearing_range near the cut");
    expect_jacobian(eb{{}, {{1, 1, 1}}, full}, v6{{4, 2, 5, -3, 13, 1}}, 1e-9, "elevation_bearing");
    expect_jacobian(ebr{{}, {{1, 1, 1}}, full}, v6{{4, 2, 5, -3, 13, 1}}, 1e-9, "elevation_bearing_range");
    expect_jacobian(ebr{{}, {{1, 1, 1}}, full}, v6{{-40, 2, 5, -3, -13, 1}}, 1e-9, "elevation_bearing_range below");
    expect_jacobian(brr2{{}, {{1, 2}}, yaw, {{0.5, -0.5}}}, v4{{11, 5, 12, -1}}, 1e-9, "bearing_range_rate 2-D");
    expect_jacobian(brr3{{}, {{1, 1, 1}}, ge::sensor_rotation(0.0, 0.0, 40 * deg), {{0.2, 0.1, 0}}}, v6{{4, 2, 5, -3, 13, 1}}, 1e-9, "bearing_range_rate 3-D");
}

TEST(TrackingMeasurementJacobian, BearingRangeClosedForm)
{
    // At relative (3, 4): d(phi) = (-4, 3) / 25, d(rho) = (3, 4) / 5, written into the mapped columns
    const auto j = br{}.jacobian({{3, 9, 4, 9}});
    expect_near(j, la::fixed_matrix<double, 2, 4>{{-4.0 / 25, 0, 3.0 / 25, 0, 3.0 / 5, 0, 4.0 / 5, 0}}, 1e-16);
}

TEST(TrackingMeasurementDegenerate, AtTheSensorsPositionAnglesAndRowsAreZero)
{
    const br s{{}, {{1, 2}}, ge::sensor_rotation(0.0, 0.0, 30 * deg)};
    const v4 at{{1, 3, 2, 4}};
    EXPECT_EQ(s.measure(at), (la::fixed_vector<double, 2>{{0, 0}}));
    EXPECT_EQ(s.jacobian(at), (la::fixed_matrix<double, 2, 4>{}));
    const brg sb{{}, {{1, 2}}};
    EXPECT_EQ(sb.measure(at)[0], 0.0);
    EXPECT_EQ(sb.jacobian(at), (la::fixed_matrix<double, 1, 4>{}));
    const ebr e{{}, {{1, 1, 1}}};
    const v6 at3{{1, 5, 1, 5, 1, 5}};
    EXPECT_EQ(e.measure(at3), (la::fixed_vector<double, 3>{}));
    EXPECT_EQ(e.jacobian(at3), (la::fixed_matrix<double, 3, 6>{}));
    const eb se{{}, {{1, 1, 1}}};
    EXPECT_EQ(se.jacobian(at3), (la::fixed_matrix<double, 2, 6>{}));
    const brr3 r{{}, {{1, 1, 1}}};
    EXPECT_TRUE(finite(r.measure(at3)));
    EXPECT_EQ(r.measure(at3)[2], 0.0);
    EXPECT_EQ(r.jacobian(at3), (la::fixed_matrix<double, 3, 6>{}));
    EXPECT_EQ(brr2{}.jacobian({}), (la::fixed_matrix<double, 3, 4>{}));
}

TEST(TrackingMeasurementDegenerate, AtThePoleElevationIsRightAngleAndBearingZero)
{
    const ebr e{};
    const v6 above{{0, 0, 0, 0, 7, 0}};
    const auto z = e.measure(above);
    EXPECT_NEAR(z[0], std::numbers::pi / 2, 1e-15);
    EXPECT_EQ(z[1], 0.0);
    EXPECT_EQ(z[2], 7.0);
    const auto j = e.jacobian(above);
    for (std::size_t c = 0; c < 6; ++c) {
        EXPECT_EQ(j(0, c), 0.0) << c;
        EXPECT_EQ(j(1, c), 0.0) << c;
    }
    EXPECT_EQ(j(2, 4), 1.0);
    EXPECT_NEAR(eb{}.measure({{0, 0, 0, 0, -7, 0}})[0], -std::numbers::pi / 2, 1e-15);
}

TEST(TrackingMeasurementTwoDimensional, UsesTheRotationsUpperLeftBlock)
{
    // A roll is ignored in the plane: only the upper-left 2 x 2 block of the rotation is used (as Stone Soup does)
    br a{}, b{};
    a.rotation = ge::sensor_rotation(0.0, 0.0, 30 * deg);
    b.rotation = a.rotation;
    b.rotation(2, 2) = -1;
    b.rotation(0, 2) = b.rotation(2, 0) = 0.5;
    const v4 x{{11, 5, 12, -1}};
    EXPECT_EQ(a.measure(x), b.measure(x));
}

} // namespace
