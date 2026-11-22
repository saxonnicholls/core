//
//  unscented_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/tracking/unscented.hpp>: Stone Soup's default scaling, the sigma points and their weights
//  (the mean weights sum to one, the points reproduce the mean and covariance), a caller's kappa, the cubature points,
//  the transform exact for linear functions and circular in angle components, the unscented and cubature filters
//  equal to the Kalman filter on linear models, a radar through the unscented transform near the bearing cut, every
//  documented error, and a whole unscented predict in a constant expression.
//

#include <snicholls/core/tracking/unscented.hpp>

#include <snicholls/core/geometry/angle.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/kalman.hpp>
#include <snicholls/core/tracking/measurement.hpp>
#include <snicholls/core/tracking/transition.hpp>

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

constexpr std::array<std::size_t, 2> pos2{0, 2};

using g2 = tr::gaussian<double, 2>;
using g4 = tr::gaussian<double, 4>;
using v4 = la::fixed_vector<double, 4>;
using m4 = la::fixed_matrix<double, 4, 4>;
using cv = tr::constant_velocity<double>;
using cv2 = tr::combined<cv, cv>;
using position = tr::linear_gaussian<double, 4, pos2>;
using radar = tr::bearing_range<double, 4, pos2>;

// Stone Soup's defaults
static_assert(tr::unscented_parameters<double>{}.alpha == 0.5 && tr::unscented_parameters<double>{}.beta == 2 && !tr::unscented_parameters<double>{}.kappa);
static_assert(std::is_trivially_copyable_v<tr::point_set<double, 4, 9>> && std::is_trivially_copyable_v<tr::transformed<double, 4, 2>>);
static_assert(noexcept(tr::make_sigma_points(std::declval<const g4&>())) && noexcept(tr::make_cubature_points(std::declval<const g4&>())));

// The weights in a constant expression: n = 2, kappa = 1, lambda = 0.25 * 3 - 2 = -1.25, n + lambda = 0.75
constexpr bool weights_at_compile_time()
{
    const auto s = tr::make_sigma_points(g2{{{0, 0}}, la::fixed_matrix<double, 2, 2>::identity()});
    double sum = 0;
    for (const double w : s->mean_weights)
        sum += w;
    return s->mean_weights[0] == -1.25 / 0.75 && s->mean_weights[1] == 0.5 / 0.75 && s->covariance_weights[0] == -1.25 / 0.75 + 2.75 && sum > 1 - 1e-15 && sum < 1 + 1e-15;
}
static_assert(weights_at_compile_time());

// A whole unscented predict of a linear model in a constant expression: exact, so F x and F P F^T + Q
constexpr bool predict_at_compile_time()
{
    const g2 prior{{{1, 2}}, la::fixed_matrix<double, 2, 2>::identity()};
    const auto p = tr::unscented_predict(prior, cv{1}, 1.0);
    const auto k = tr::predict(prior, cv{1}, 1.0);
    const double e0 = p->mean[0] - k.mean[0], e1 = p->covariance(0, 0) - k.covariance(0, 0);
    return e0 < 1e-12 && e0 > -1e-12 && e1 < 1e-12 && e1 > -1e-12;
}
static_assert(predict_at_compile_time());

template <std::size_t R, std::size_t C>
void expect_near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol, const char* what = "")
{
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol) << what << " (" << i << ", " << j << ")";
}

g4 some_state()
{
    return {{{10, 1, -5, 2}}, {{4, 1, 0.5, 0, 1, 2, 0, 0.3, 0.5, 0, 3, 0.8, 0, 0.3, 0.8, 1.5}}};
}

// The weighted mean and covariance of a point set, written out
template <std::size_t N, std::size_t P>
void expect_reproduces(const tr::point_set<double, N, P>& s, const tr::gaussian<double, N>& g, double tol)
{
    la::fixed_vector<double, N> m{};
    for (std::size_t p = 0; p < P; ++p)
        m += s.points[p] * s.mean_weights[p];
    expect_near(m, g.mean, tol, "mean");
    la::fixed_matrix<double, N, N> c{};
    for (std::size_t p = 0; p < P; ++p)
        c += la::outer(s.points[p] - g.mean, s.points[p] - g.mean) * s.covariance_weights[p];
    expect_near(c, g.covariance, tol, "covariance");
}

TEST(TrackingUnscentedSigmaPoints, StoneSoupsDefaultWeights)
{
    // n = 4, alpha = 0.5, beta = 2, kappa = 3 - n = -1: lambda = 0.25 * 3 - 4 = -3.25, n + lambda = 0.75
    const auto s = tr::make_sigma_points(some_state());
    ASSERT_TRUE(s.is_ok());
    EXPECT_DOUBLE_EQ(s->mean_weights[0], -3.25 / 0.75);
    EXPECT_DOUBLE_EQ(s->covariance_weights[0], -3.25 / 0.75 + 1 - 0.25 + 2);
    double sum = 0;
    for (std::size_t p = 0; p < 9; ++p) {
        sum += s->mean_weights[p];
        if (p > 0) {
            EXPECT_DOUBLE_EQ(s->mean_weights[p], 0.5 / 0.75) << p;
            EXPECT_DOUBLE_EQ(s->covariance_weights[p], 0.5 / 0.75) << p;
        }
    }
    EXPECT_NEAR(sum, 1.0, 1e-14);
}

TEST(TrackingUnscentedSigmaPoints, PointsAreTheMeanPlusAndMinusScaledCholeskyColumns)
{
    const g4 g = some_state();
    const auto s = tr::make_sigma_points(g);
    const auto l = *la::cholesky(g.covariance);
    EXPECT_EQ(s->points[0], g.mean);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t r = 0; r < 4; ++r) {
            EXPECT_NEAR(s->points[1 + i][r], g.mean[r] + std::sqrt(0.75) * l(r, i), 1e-14);
            EXPECT_NEAR(s->points[5 + i][r], g.mean[r] - std::sqrt(0.75) * l(r, i), 1e-14);
        }
    expect_reproduces(*s, g, 1e-13);
}

TEST(TrackingUnscentedSigmaPoints, TheCallersKappaAndAlpha)
{
    // n = 4, alpha = 1, kappa = 0: lambda = 0, the centre weighs 0 in the mean and beta = 2 in the covariance
    const g4 g = some_state();
    const auto s = tr::make_sigma_points(g, {1.0, 2.0, 0.0});
    EXPECT_EQ(s->mean_weights[0], 0.0);
    EXPECT_EQ(s->covariance_weights[0], 2.0);
    EXPECT_DOUBLE_EQ(s->mean_weights[1], 1.0 / 8);
    const auto l = *la::cholesky(g.covariance);
    EXPECT_NEAR(s->points[1][0], g.mean[0] + 2 * l(0, 0), 1e-14);   // sqrt(n + lambda) = 2
    expect_reproduces(*s, g, 1e-13);
    // kappa = 3 - n given explicitly is the default
    const auto d = tr::make_sigma_points(g), e = tr::make_sigma_points(g, {0.5, 2.0, -1.0});
    EXPECT_EQ(d->points, e->points);
    EXPECT_EQ(d->mean_weights, e->mean_weights);
    EXPECT_EQ(d->covariance_weights, e->covariance_weights);
}

TEST(TrackingUnscentedCubature, EqualWeightsAndScaledPoints)
{
    const g4 g = some_state();
    const auto s = tr::make_cubature_points(g);
    ASSERT_TRUE(s.is_ok());
    const auto l = *la::cholesky(g.covariance);
    for (std::size_t p = 0; p < 8; ++p) {
        EXPECT_EQ(s->mean_weights[p], 1.0 / 8);
        EXPECT_EQ(s->covariance_weights[p], 1.0 / 8);
    }
    EXPECT_NEAR(s->points[0][0], g.mean[0] + 2 * l(0, 0), 1e-14);   // sqrt(alpha n) = 2
    EXPECT_NEAR(s->points[4][0], g.mean[0] - 2 * l(0, 0), 1e-14);
    expect_reproduces(*s, g, 1e-13);
    // alpha = 2: points sqrt(8) out, covariance weights 1 / 16
    const auto t = tr::make_cubature_points(g, 2.0);
    EXPECT_EQ(t->covariance_weights[3], 1.0 / 16);
    EXPECT_NEAR(t->points[0][0], g.mean[0] + std::sqrt(8.0) * l(0, 0), 1e-14);
    expect_reproduces(*t, g, 1e-13);
}

TEST(TrackingUnscentedTransform, IsExactForLinearFunctions)
{
    const g4 g = some_state();
    const la::fixed_matrix<double, 2, 4> a{{1, 2, 0, -1, 0.5, 0, 3, 1}};
    const la::fixed_vector<double, 2> b{{7, -3}};
    const la::fixed_matrix<double, 2, 2> noise{{0.1, 0.02, 0.02, 0.2}};
    const auto f = [&](const v4& x) { return a * x + b; };
    const auto s = tr::make_sigma_points(g);
    const auto t = tr::point_transform<ge::all_linear<2>, 2>(*s, f, noise);
    expect_near(t.mean, a * g.mean + b, 1e-12);
    expect_near(t.covariance, a * g.covariance * la::transpose(a) + noise, 1e-12);
    expect_near(t.cross_covariance, g.covariance * la::transpose(a), 1e-12);
    const auto c = tr::make_cubature_points(g, 1.5);
    const auto u = tr::point_transform<ge::all_linear<2>, 2>(*c, f, noise);
    expect_near(u.mean, t.mean, 1e-12);
    expect_near(u.covariance, t.covariance, 1e-12);
    expect_near(u.cross_covariance, t.cross_covariance, 1e-12);
}

TEST(TrackingUnscentedTransform, AnglesAverageOnTheCircle)
{
    // Two points at bearings 3.1 and -3.1: the mean is pi (the cut), not 0, and the spread is 2 pi - 6.2 across it
    const tr::point_set<double, 1, 2> s{{la::fixed_vector<double, 1>{{3.1}}, la::fixed_vector<double, 1>{{-3.1}}}, {0.5, 0.5}, {0.5, 0.5}};
    constexpr ge::angle_kinds<1> bearing{ge::angle_kind::bearing};
    const auto t = tr::point_transform<bearing, 1>(s, [](const la::fixed_vector<double, 1>& x) { return x; }, la::fixed_matrix<double, 1, 1>{});
    EXPECT_NEAR(std::abs(t.mean[0]), std::numbers::pi, 1e-12);
    const double half = std::numbers::pi - 3.1;
    EXPECT_NEAR(t.covariance(0, 0), half * half, 1e-12);
    // As linear numbers the same points average to 0 with a spread of 3.1
    const auto u = tr::point_transform<ge::all_linear<1>, 1>(s, [](const la::fixed_vector<double, 1>& x) { return x; }, la::fixed_matrix<double, 1, 1>{});
    EXPECT_NEAR(u.mean[0], 0.0, 1e-15);
    EXPECT_NEAR(u.covariance(0, 0), 3.1 * 3.1, 1e-12);
}

TEST(TrackingUnscentedFilter, EqualsTheKalmanFilterOnLinearModels)
{
    const g4 prior = some_state();
    const cv2 model{{cv{0.2}, cv{0.4}}};
    const position s{{{0.5, 0.1, 0.1, 0.7}}};
    const la::fixed_vector<double, 2> z{{11, -4}};
    const auto kp = tr::predict(prior, model, 1.5);
    const auto ku = tr::update(kp, z, s);

    const auto up = tr::unscented_predict(prior, model, 1.5);
    ASSERT_TRUE(up.is_ok());
    expect_near(up->mean, kp.mean, 1e-12, "UKF predict");
    expect_near(up->covariance, kp.covariance, 1e-12, "UKF predict");
    const auto ump = tr::unscented_predict_measurement(*up, s);
    ASSERT_TRUE(ump.is_ok());
    const auto uu = tr::update<position::kinds>(*up, z, *ump);
    expect_near(uu->posterior.mean, ku->posterior.mean, 1e-12, "UKF update");
    expect_near(uu->posterior.covariance, ku->posterior.covariance, 1e-12, "UKF update");
    EXPECT_NEAR(uu->distance2, ku->distance2, 1e-12);

    const auto cp = tr::cubature_predict(prior, model, 1.5);
    expect_near(cp->mean, kp.mean, 1e-12, "CKF predict");
    expect_near(cp->covariance, kp.covariance, 1e-12, "CKF predict");
    const auto cmp = tr::cubature_predict_measurement(*cp, s);
    const auto cu = tr::update<position::kinds>(*cp, z, *cmp);
    expect_near(cu->posterior.mean, ku->posterior.mean, 1e-12, "CKF update");
    expect_near(cu->posterior.covariance, ku->posterior.covariance, 1e-12, "CKF update");

    // Other scalings are exact too
    const auto up2 = tr::unscented_predict(prior, model, 1.5, {1.0, 0.0, 1.0});
    expect_near(up2->covariance, kp.covariance, 1e-12, "UKF alpha 1");
    const auto cp2 = tr::cubature_predict(prior, model, 1.5, 0.5);
    expect_near(cp2->covariance, kp.covariance, 1e-12, "CKF alpha 0.5");
    const auto ump2 = tr::unscented_predict_measurement(kp, s, {1.0, 2.0, 0.0});
    const auto cmp2 = tr::cubature_predict_measurement(kp, s, 3.0);
    expect_near(ump2->covariance, cmp2->covariance, 1e-12, "measurement scalings");
}

TEST(TrackingUnscentedFilter, NonlinearRadarAgreesWithTheExtendedFilterWhenTight)
{
    // A small covariance far from the radar: linearisation is good, so UKF and EKF agree closely
    g4 p{{{1000, 5, 500, -2}}, m4::identity()};
    const radar r{{{1e-6, 0, 0, 1}}};
    const auto emp = tr::predict_measurement(p, r);
    const auto ump = tr::unscented_predict_measurement(p, r);
    const auto cmp = tr::cubature_predict_measurement(p, r);
    EXPECT_NEAR(ump->mean[0], emp.mean[0], 1e-6);
    EXPECT_NEAR(ump->mean[1], emp.mean[1], 1e-3);
    EXPECT_NEAR(cmp->mean[1], emp.mean[1], 1e-3);
    expect_near(ump->covariance, emp.covariance, 1e-6, "S");
    expect_near(ump->cross_covariance, emp.cross_covariance, 1e-6, "cross");
    // The range's mean is pushed out by the curvature: E[|p|] > |E[p]|
    EXPECT_GT(ump->mean[1], emp.mean[1]);
}

TEST(TrackingUnscentedFilter, RadarAcrossTheBearingCut)
{
    // The target is due west (bearing pi) and spread across the cut: the predicted bearing is near pi, not 0, and an
    // update with a measurement at -179.5 degrees moves it the short way
    g4 p{{{-100, 0, 0, 0}}, m4::identity() * 4};
    const radar r{{{1e-4, 0, 0, 1}}};
    const auto mp = tr::unscented_predict_measurement(p, r);
    ASSERT_TRUE(mp.is_ok());
    EXPECT_NEAR(std::abs(mp->mean[0]), std::numbers::pi, 1e-12);
    EXPECT_LT(mp->covariance(0, 0), 1e-3);   // about 4 / 100^2 plus R, not pi^2
    const double deg = std::numbers::pi / 180;
    const auto u = tr::update<radar::kinds>(p, la::fixed_vector<double, 2>{{-179.5 * deg, 100}}, *mp);
    ASSERT_TRUE(u.is_ok());
    EXPECT_NEAR(std::abs(u->innovation[0]), 0.5 * deg, 1e-9);
    EXPECT_LT(u->posterior.mean[2], 0);   // toward negative y
    EXPECT_LT(u->posterior.mean[0], 0);
}

TEST(TrackingUnscentedFilter, ConstantTurnThroughTheUnscentedPredict)
{
    // With a tiny covariance the motion is linear over the spread: the unscented and cubature predicts are the extended
    // one
    const tr::constant_turn<double> m{0.1, 0.1, 0.001};
    tr::gaussian<double, 5> g{{{0, 10, 0, 0, 0.2}}, la::fixed_matrix<double, 5, 5>::identity() * 1e-10};
    const auto p = tr::unscented_predict(g, m, 1.0);
    const auto c = tr::cubature_predict(g, m, 1.0);
    expect_near(p->mean, m.apply(g.mean, 1.0), 1e-9);
    expect_near(c->mean, m.apply(g.mean, 1.0), 1e-9);
    const auto e = tr::predict(g, m, 1.0);
    expect_near(p->covariance, e.covariance, 1e-12);
    expect_near(c->covariance, e.covariance, 1e-12);
}

TEST(TrackingUnscentedErrors, NotPositiveDefiniteCovarianceIsAnError)
{
    const g4 bad{{{1, 2, 3, 4}}, m4{}};
    const cv2 model{{cv{0.2}, cv{0.4}}};
    const position s{{{0.5, 0, 0, 0.5}}};
    EXPECT_EQ(tr::make_sigma_points(bad).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::make_cubature_points(bad).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::unscented_predict(bad, model, 1.0).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::cubature_predict(bad, model, 1.0).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::unscented_predict_measurement(bad, s).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::cubature_predict_measurement(bad, s).unwrap_err(), la::error::not_positive_definite);
}

} // namespace
