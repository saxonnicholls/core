//
//  kalman_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/tracking/kalman.hpp>: a scalar filter worked by hand, linear and extended predicts
//  against F P F^T + Q written out, the control-input predict, linear and linearised measurement predictions, the
//  update against the textbook formulas with an explicit inverse, Joseph and standard forms agreeing for the optimal
//  gain, the update from a measurement prediction, the bearing innovation wrapping across the cut, the information
//  form's round trip and its equivalence with the covariance update, Fisher information and GDOP for two orthogonal
//  bearing sensors, a two-step Rauch-Tung-Striebel smoother by hand, central differences, the alpha-beta filter,
//  every documented error, and a whole predict and update in a constant expression.
//

#include <snicholls/core/tracking/kalman.hpp>

#include <snicholls/core/geometry/angle.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
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

constexpr std::array<std::size_t, 1> first{0};
constexpr std::array<std::size_t, 2> pos2{0, 2};
constexpr std::array<std::size_t, 2> vel2{1, 3};

using g1 = tr::gaussian<double, 1>;
using g4 = tr::gaussian<double, 4>;
using v4 = la::fixed_vector<double, 4>;
using m4 = la::fixed_matrix<double, 4, 4>;
using cv = tr::constant_velocity<double>;
using cv2 = tr::combined<cv, cv>;
using scalar_sensor = tr::linear_gaussian<double, 1, first>;
using position = tr::linear_gaussian<double, 4, pos2>;
using radar = tr::bearing_range<double, 4, pos2>;

constexpr double two_pi = 2 * std::numbers::pi;
const double deg = std::numbers::pi / 180;

// Types and noexcept
static_assert(static_cast<int>(tr::covariance_form::standard) == 0 && static_cast<int>(tr::covariance_form::joseph) == 1);
static_assert(std::is_trivially_copyable_v<tr::kalman_update<double, 4, 2>> && std::is_trivially_copyable_v<tr::measurement_prediction<double, 4, 2>> &&
              std::is_trivially_copyable_v<tr::information_state<double, 4>>);
static_assert(noexcept(tr::predict(std::declval<const g4&>(), std::declval<const cv2&>(), 1.0)));
static_assert(noexcept(tr::predict(std::declval<const tr::gaussian<double, 5>&>(), std::declval<const tr::constant_turn<double>&>(), 1.0)));
static_assert(noexcept(tr::update(std::declval<const g4&>(), std::declval<const la::fixed_vector<double, 2>&>(), std::declval<const radar&>())));
static_assert(noexcept(tr::predict_measurement(std::declval<const g4&>(), std::declval<const position&>())));
static_assert(noexcept(tr::to_information(std::declval<const g4&>())) && noexcept(tr::from_information(std::declval<const tr::information_state<double, 4>&>())));
static_assert(noexcept(tr::gdop(std::declval<const m4&>())) && noexcept(tr::rts_smooth(std::declval<const g4&>(), std::declval<const g4&>(), std::declval<const g4&>(), std::declval<const m4&>())));
static_assert(noexcept(tr::alpha_beta_update<pos2, vel2>(std::declval<const v4&>(), std::declval<const la::fixed_vector<double, 2>&>(), 0.5, 0.1, 1.0)));

// The scalar filter by hand: random walk q = 1, prior N(0, 1), dt = 1 gives N(0, 2); z = 3 with R = 1 gives S = 3,
// K = 2/3, posterior N(2, 2/3), innovation 3, d^2 = 3
constexpr bool scalar_at_compile_time()
{
    const g1 prior{{{0}}, {{1}}};
    const auto p = tr::predict(prior, tr::random_walk<double>{1}, 1.0);
    const auto u = tr::update(p, la::fixed_vector<double, 1>{{3}}, scalar_sensor{{{1}}});
    const double v = u->posterior.covariance(0, 0);
    return p.covariance(0, 0) == 2 && u->posterior.mean[0] > 2 - 1e-14 && u->posterior.mean[0] < 2 + 1e-14 && v > 2.0 / 3 - 1e-14 && v < 2.0 / 3 + 1e-14 && u->distance2 > 3 - 1e-14 &&
           u->distance2 < 3 + 1e-14;
}
static_assert(scalar_at_compile_time());

// A whole 2-D constant-velocity predict and position update in a constant expression
constexpr bool cv_at_compile_time()
{
    const g4 prior{{{0, 1, 0, 1}}, m4::identity()};
    const auto p = tr::predict(prior, tr::combine(cv{0.1}, cv{0.1}), 1.0);
    const auto u = tr::update(p, la::fixed_vector<double, 2>{{1.5, 0.5}}, position{{{1, 0, 0, 1}}});
    return u.is_ok() && u->posterior.mean[0] > 1 && u->posterior.mean[0] < 1.5 && u->posterior.mean[2] > 0.5 && u->posterior.mean[2] < 1 &&
           u->posterior.covariance(0, 0) < p.covariance(0, 0);
}
static_assert(cv_at_compile_time());
static_assert(tr::alpha_beta_update<pos2, vel2>(v4{{1, 2, 3, 4}}, la::fixed_vector<double, 2>{{2, 1}}, 0.5, 0.2, 2.0) == v4{{1.5, 2.1, 2, 3.8}});

template <std::size_t R, std::size_t C>
void expect_near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol, const char* what = "")
{
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol) << what << " (" << i << ", " << j << ")";
}

// A predicted 2-D constant-velocity state with correlated uncertainty
g4 some_prediction()
{
    return {{{10, 1, -5, 2}}, {{4, 1, 0.5, 0, 1, 2, 0, 0.3, 0.5, 0, 3, 0.8, 0, 0.3, 0.8, 1.5}}};
}

TEST(TrackingKalmanScalar, OneStepByHand)
{
    const auto p = tr::predict(g1{{{0}}, {{1}}}, tr::random_walk<double>{1}, 1.0);
    EXPECT_EQ(p.mean[0], 0.0);
    EXPECT_EQ(p.covariance(0, 0), 2.0);
    const auto u = tr::update(p, la::fixed_vector<double, 1>{{3}}, scalar_sensor{{{1}}});
    ASSERT_TRUE(u.is_ok());
    EXPECT_DOUBLE_EQ(u->innovation[0], 3.0);
    EXPECT_DOUBLE_EQ(u->innovation_covariance(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(u->gain(0, 0), 2.0 / 3);
    EXPECT_DOUBLE_EQ(u->posterior.mean[0], 2.0);
    EXPECT_DOUBLE_EQ(u->posterior.covariance(0, 0), 2.0 / 3);
    EXPECT_DOUBLE_EQ(u->distance2, 3.0);
    EXPECT_NEAR(u->log_likelihood, -(3 + std::log(3.0) + std::log(two_pi)) / 2, 1e-14);
}

TEST(TrackingKalmanPredict, LinearIsFPFtPlusQ)
{
    const g4 prior = some_prediction();
    const cv2 model{{cv{0.2}, cv{0.4}}};
    const double dt = 1.5;
    const auto p = tr::predict(prior, model, dt);
    m4 f = m4::identity();
    f(0, 1) = f(2, 3) = dt;
    expect_near(p.mean, f * prior.mean, 1e-14);
    expect_near(p.covariance, f * prior.covariance * la::transpose(f) + model.covariance(dt), 1e-13);
    EXPECT_NEAR(p.mean[0], 10 + 1.5, 1e-14);
}

TEST(TrackingKalmanPredict, NonlinearIsLinearisedAtTheMean)
{
    const tr::constant_turn<double> model{0.1, 0.1, 0.01};
    tr::gaussian<double, 5> prior{{{0, 10, 0, 0, 0.2}}, la::fixed_matrix<double, 5, 5>::identity()};
    prior.covariance(4, 4) = 0.01;
    const auto p = tr::predict(prior, model, 2.0);
    EXPECT_EQ(p.mean, model.apply(prior.mean, 2.0));
    const auto j = model.jacobian(prior.mean, 2.0);
    expect_near(p.covariance, j * prior.covariance * la::transpose(j) + model.covariance(2.0), 1e-12);
}

TEST(TrackingKalmanPredict, ControlInputAddsBuAndItsUncertainty)
{
    // A known acceleration a over dt: B = (dt^2 / 2, dt) per axis
    const g4 prior = some_prediction();
    const cv2 model{{cv{0.2}, cv{0.4}}};
    const double dt = 2;
    const la::fixed_matrix<double, 4, 2> b{{dt * dt / 2, 0, dt, 0, 0, dt * dt / 2, 0, dt}};
    const la::fixed_vector<double, 2> u{{0.5, -1}};
    const auto plain = tr::predict(prior, model, dt);
    const auto exact = tr::predict(prior, model, dt, b, u);
    expect_near(exact.mean, plain.mean + v4{{1, 1, -2, -2}}, 1e-14);
    EXPECT_EQ(exact.covariance, plain.covariance);   // the default control covariance is zero
    const la::fixed_matrix<double, 2, 2> qu{{0.01, 0, 0, 0.04}};
    const auto noisy = tr::predict(prior, model, dt, b, u, qu);
    expect_near(noisy.covariance, plain.covariance + b * qu * la::transpose(b), 1e-14);
    EXPECT_NEAR(noisy.covariance(0, 0) - plain.covariance(0, 0), 0.01 * 4, 1e-14);
    // The control input with a nonlinear model
    const tr::gaussian<double, 5> p5{{{0, 1, 0, 1, 0.1}}, la::fixed_matrix<double, 5, 5>::identity()};
    const la::fixed_matrix<double, 5, 1> b5{{1, 0, 0, 0, 0}};
    const auto t = tr::predict(p5, tr::constant_turn<double>{}, 1.0, b5, la::fixed_vector<double, 1>{{3}});
    EXPECT_NEAR(t.mean[0], tr::constant_turn<double>{}.apply(p5.mean, 1.0)[0] + 3, 1e-14);
}

TEST(TrackingKalmanPredictMeasurement, LinearIsHxHPHtPlusRAndPHt)
{
    const g4 p = some_prediction();
    const position s{{{0.5, 0.1, 0.1, 0.7}}};
    const auto mp = tr::predict_measurement(p, s);
    // H picks rows and columns (0, 2)
    EXPECT_EQ(mp.mean, (la::fixed_vector<double, 2>{{10, -5}}));
    expect_near(mp.covariance, la::fixed_matrix<double, 2, 2>{{4 + 0.5, 0.5 + 0.1, 0.5 + 0.1, 3 + 0.7}}, 1e-15);
    expect_near(mp.cross_covariance, la::fixed_matrix<double, 4, 2>{{4, 0.5, 1, 0, 0.5, 3, 0, 0.8}}, 0);
}

TEST(TrackingKalmanPredictMeasurement, NonlinearIsLinearisedAtTheMean)
{
    const g4 p = some_prediction();
    const radar s{{{1e-4, 0, 0, 1}}, {{1, 1}}};
    const auto mp = tr::predict_measurement(p, s);
    const auto h = s.jacobian(p.mean);
    EXPECT_EQ(mp.mean, s.measure(p.mean));
    expect_near(mp.covariance, h * p.covariance * la::transpose(h) + s.covariance(), 1e-14);
    expect_near(mp.cross_covariance, p.covariance * la::transpose(h), 1e-14);
}

TEST(TrackingKalmanUpdate, MatchesTheTextbookFormulas)
{
    const g4 p = some_prediction();
    const position s{{{0.5, 0.1, 0.1, 0.7}}};
    const la::fixed_vector<double, 2> z{{11, -4}};
    const auto u = tr::update(p, z, s);
    ASSERT_TRUE(u.is_ok());
    const auto h = position::matrix();
    const auto sm = h * p.covariance * la::transpose(h) + s.covariance();
    const auto k = p.covariance * la::transpose(h) * *la::inverse(sm);
    const auto nu = z - h * p.mean;
    expect_near(u->innovation, nu, 1e-15);
    expect_near(u->innovation_covariance, sm, 1e-15);
    expect_near(u->gain, k, 1e-13);
    expect_near(u->posterior.mean, p.mean + k * nu, 1e-13);
    expect_near(u->posterior.covariance, (m4::identity() - k * h) * p.covariance, 1e-13);
    EXPECT_NEAR(u->distance2, *tr::squared_mahalanobis(nu, sm), 1e-13);
    EXPECT_NEAR(u->log_likelihood, *tr::log_likelihood(nu, sm), 1e-13);
    // Exactly symmetric
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_EQ(u->posterior.covariance(i, j), u->posterior.covariance(j, i));
}

TEST(TrackingKalmanUpdate, JosephAndStandardAgreeForTheOptimalGain)
{
    const g4 p = some_prediction();
    const position s{{{0.5, 0.1, 0.1, 0.7}}};
    const auto a = tr::update(p, la::fixed_vector<double, 2>{{11, -4}}, s);
    const auto b = tr::update(p, la::fixed_vector<double, 2>{{11, -4}}, s, tr::covariance_form::joseph);
    expect_near(a->posterior.covariance, b->posterior.covariance, 1e-12, "linear");
    EXPECT_EQ(a->posterior.mean, b->posterior.mean);
    // Extended: the Joseph form uses the same Jacobian, so the two agree too
    const radar r{{{1e-4, 0, 0, 1}}, {{1, 1}}};
    const auto z = r.measure(p.mean) + la::fixed_vector<double, 2>{{0.01, 0.5}};
    const auto c = tr::update(p, z, r), d = tr::update(p, z, r, tr::covariance_form::joseph);
    expect_near(c->posterior.covariance, d->posterior.covariance, 1e-12, "extended");
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_EQ(d->posterior.covariance(i, j), d->posterior.covariance(j, i));
}

TEST(TrackingKalmanUpdate, FromAMeasurementPredictionIsTheSame)
{
    const g4 p = some_prediction();
    const radar r{{{1e-4, 0, 0, 1}}, {{1, 1}}};
    const auto z = r.measure(p.mean) + la::fixed_vector<double, 2>{{0.01, 0.5}};
    const auto a = tr::update(p, z, r);
    const auto b = tr::update<radar::kinds>(p, z, tr::predict_measurement(p, r));
    ASSERT_TRUE(a.is_ok() && b.is_ok());
    EXPECT_EQ(a->posterior, b->posterior);
    EXPECT_EQ(a->gain, b->gain);
    EXPECT_EQ(a->distance2, b->distance2);
    EXPECT_EQ(a->log_likelihood, b->log_likelihood);
}

TEST(TrackingKalmanUpdate, BearingInnovationWrapsAcrossTheCut)
{
    // Predicted at -179 degrees, measured at +179: the innovation is -2 degrees, not 358
    g4 p{};
    p.mean = {{100 * std::cos(-179 * deg), 0, 100 * std::sin(-179 * deg), 0}};
    p.covariance = m4::identity();
    const radar r{{{1e-4, 0, 0, 1}}};
    const la::fixed_vector<double, 2> z{{179 * deg, 100}};
    const auto u = tr::update(p, z, r);
    ASSERT_TRUE(u.is_ok());
    EXPECT_NEAR(u->innovation[0], -2 * deg, 1e-12);
    EXPECT_NEAR(u->innovation[1], 0, 1e-12);
    // The posterior moves the short way, across the negative x axis, toward +179 degrees
    EXPECT_GT(u->posterior.mean[2], p.mean[2]);
    EXPECT_LT(u->posterior.mean[0], 0);
    EXPECT_LT(u->distance2, 2 * deg * 2 * deg / 1e-4);
    // The update from a measurement prediction wraps the same way
    const auto v = tr::update<radar::kinds>(p, z, tr::predict_measurement(p, r));
    EXPECT_NEAR(v->innovation[0], -2 * deg, 1e-12);
    // Without the kinds the same numbers are 358 degrees apart
    const auto w = tr::update<ge::all_linear<2>>(p, z, tr::predict_measurement(p, r));
    EXPECT_NEAR(w->innovation[0], 358 * deg, 1e-12);
}

TEST(TrackingKalmanUpdate, NotPositiveDefiniteInnovationCovarianceIsAnError)
{
    const g4 zero{};
    const position s{};   // R = 0 and P = 0: S = 0
    const la::fixed_vector<double, 2> z{{1, 1}};
    EXPECT_EQ(tr::update(zero, z, s).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::update(zero, z, s, tr::covariance_form::joseph).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::update<position::kinds>(zero, z, tr::predict_measurement(zero, s)).unwrap_err(), la::error::not_positive_definite);
    const position negative{{{-5, 0, 0, -5}}};   // S = P_xy - 5 I is indefinite
    EXPECT_TRUE(tr::update(some_prediction(), z, negative).is_err());
}

TEST(TrackingKalmanInformation, RoundTripsAndIsTheInverse)
{
    const g4 g = some_prediction();
    const auto i = tr::to_information(g);
    ASSERT_TRUE(i.is_ok());
    expect_near(i->matrix * g.covariance, m4::identity(), 1e-13);
    expect_near(i->vector, i->matrix * g.mean, 1e-13);
    const auto back = tr::from_information(*i);
    ASSERT_TRUE(back.is_ok());
    expect_near(back->mean, g.mean, 1e-12);
    expect_near(back->covariance, g.covariance, 1e-12);
}

TEST(TrackingKalmanInformation, UpdateIsTheCovarianceUpdate)
{
    const g4 p = some_prediction();
    const position s{{{0.5, 0.1, 0.1, 0.7}}};
    const la::fixed_vector<double, 2> z{{11, -4}};
    const auto ku = tr::update(p, z, s);
    const auto iu = tr::information_update(*tr::to_information(p), z, position::matrix(), s.covariance());
    ASSERT_TRUE(iu.is_ok());
    const auto g = tr::from_information(*iu);
    expect_near(g->mean, ku->posterior.mean, 1e-12);
    expect_near(g->covariance, ku->posterior.covariance, 1e-12);
    // Y gains H^T R^-1 H, which is fisher_information(H, R)
    expect_near(iu->matrix - tr::to_information(p)->matrix, *tr::fisher_information(position::matrix(), s.covariance()), 1e-12);
}

TEST(TrackingKalmanInformation, IndependentSensorsAdd)
{
    // Two position sensors one after the other, in either order, as the information sums
    const g4 p = some_prediction();
    const position a{{{0.5, 0, 0, 0.7}}}, b{{{2, 0.3, 0.3, 1}}};
    const la::fixed_vector<double, 2> za{{11, -4}}, zb{{10.5, -4.4}};
    const auto i0 = *tr::to_information(p);
    const auto ab = *tr::information_update(*tr::information_update(i0, za, position::matrix(), a.covariance()), zb, position::matrix(), b.covariance());
    const auto ba = *tr::information_update(*tr::information_update(i0, zb, position::matrix(), b.covariance()), za, position::matrix(), a.covariance());
    expect_near(ab.matrix, ba.matrix, 1e-13);
    expect_near(ab.vector, ba.vector, 1e-13);
    // The same as two Kalman updates
    const auto k = tr::update(tr::update(p, za, a)->posterior, zb, b)->posterior;
    const auto g = *tr::from_information(ab);
    expect_near(g.mean, k.mean, 1e-11);
    expect_near(g.covariance, k.covariance, 1e-11);
}

TEST(TrackingKalmanInformation, NotPositiveDefiniteMatricesAreErrors)
{
    EXPECT_EQ(tr::to_information(g4{}).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::from_information(tr::information_state<double, 4>{}).unwrap_err(), la::error::not_positive_definite);
    const auto i0 = *tr::to_information(some_prediction());
    EXPECT_EQ(tr::information_update(i0, la::fixed_vector<double, 2>{{1, 1}}, position::matrix(), la::fixed_matrix<double, 2, 2>{}).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::fisher_information(position::matrix(), la::fixed_matrix<double, 2, 2>{{1, 2, 2, 1}}).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::gdop(m4{}).unwrap_err(), la::error::not_positive_definite);
}

TEST(TrackingKalmanFisher, TwoOrthogonalBearingSensorsGiveGdopOfRootTwoTenSigma)
{
    // A target at the origin, bearing sensors at (-10, 0) and (0, -10), sigma^2 = 0.01. Bearing rows: from (-10, 0),
    // d(phi) = (-py, px) / r^2 = (0, 1/10); from (0, -10), (-1/10, 0). Each adds diag 1 / (100 sigma^2) on its axis, so
    // J = I (in (x, y)) and GDOP = sqrt(tr J^-1) = sqrt(2)
    using brg = tr::bearing<double, 2, std::array<std::size_t, 2>{0, 1}>;
    const brg a{{{0.01}}, {{-10, 0}}}, b{{{0.01}}, {{0, -10}}};
    const la::fixed_vector<double, 2> target{};
    const auto ja = tr::fisher_information(a.jacobian(target), a.covariance());
    const auto jb = tr::fisher_information(b.jacobian(target), b.covariance());
    ASSERT_TRUE(ja.is_ok() && jb.is_ok());
    expect_near(*ja, la::fixed_matrix<double, 2, 2>{{0, 0, 0, 1}}, 1e-15);
    expect_near(*jb, la::fixed_matrix<double, 2, 2>{{1, 0, 0, 0}}, 1e-15);
    EXPECT_NEAR(*tr::gdop(*ja + *jb), std::sqrt(2.0), 1e-14);
    // One bearing sensor alone does not determine the position
    EXPECT_EQ(tr::gdop(*ja).unwrap_err(), la::error::not_positive_definite);
    // Moved twice as far, each bearing carries a quarter of the information: GDOP doubles
    const brg c{{{0.01}}, {{-20, 0}}}, d{{{0.01}}, {{0, -20}}};
    EXPECT_NEAR(*tr::gdop(*tr::fisher_information(c.jacobian(target), c.covariance()) + *tr::fisher_information(d.jacobian(target), d.covariance())), 2 * std::sqrt(2.0), 1e-13);
}

TEST(TrackingKalmanSmoother, TwoStepScalarCaseByHand)
{
    // Filtered x0 ~ N(0, 1); predicted x1 ~ N(0, 2); z1 = 2, R = 1 gives filtered (= smoothed) x1 ~ N(4/3, 2/3).
    // C = P_f F / P_p = 1/2, so x0s = 0 + (4/3 - 0) / 2 = 2/3 and P0s = 1 + (2/3 - 2) / 4 = 2/3
    const g1 f0{{{0}}, {{1}}};
    const auto p1 = tr::predict(f0, tr::random_walk<double>{1}, 1.0);
    const auto s1 = tr::update(p1, la::fixed_vector<double, 1>{{2}}, scalar_sensor{{{1}}})->posterior;
    EXPECT_DOUBLE_EQ(s1.mean[0], 4.0 / 3);
    const auto s0 = tr::rts_smooth(f0, p1, s1, la::fixed_matrix<double, 1, 1>{{1}});
    ASSERT_TRUE(s0.is_ok());
    EXPECT_NEAR(s0->mean[0], 2.0 / 3, 1e-15);
    EXPECT_NEAR(s0->covariance(0, 0), 2.0 / 3, 1e-15);
}

TEST(TrackingKalmanSmoother, MatchesTheGainFormulaAndIsIdentityWithoutNewInformation)
{
    const g4 f = some_prediction();
    const cv2 model{{cv{0.2}, cv{0.4}}};
    const auto p = tr::predict(f, model, 1.0);
    const auto fm = model.matrix(1.0);
    // Without new information the smoothed state is the filtered one
    const auto same = tr::rts_smooth(f, p, p, fm);
    expect_near(same->mean, f.mean, 1e-13);
    expect_near(same->covariance, f.covariance, 1e-13);
    // Against C = P_f F^T P_p^-1 with an explicit inverse
    g4 next = p;
    next.mean += v4{{0.5, -0.2, 0.3, 0.1}};
    next.covariance = p.covariance * 0.5;
    const auto c = f.covariance * la::transpose(fm) * *la::inverse(p.covariance);
    const auto s = tr::rts_smooth(f, p, next, fm);
    expect_near(s->mean, f.mean + c * (next.mean - p.mean), 1e-12);
    expect_near(s->covariance, f.covariance + c * (next.covariance - p.covariance) * la::transpose(c), 1e-12);
    EXPECT_EQ(tr::rts_smooth(f, g4{}, next, fm).unwrap_err(), la::error::not_positive_definite);
}

TEST(TrackingKalmanNumericalJacobian, MatchesAnAnalyticDerivative)
{
    // f(x, y) = (x^2 y, sin y, e^x): J = [[2xy, x^2], [0, cos y], [e^x, 0]]
    const auto f = [](const la::fixed_vector<double, 2>& v) { return la::fixed_vector<double, 3>{{v[0] * v[0] * v[1], std::sin(v[1]), std::exp(v[0])}}; };
    const la::fixed_vector<double, 2> x{{1.5, -0.7}};
    const auto j = tr::numerical_jacobian<3>(f, x);
    expect_near(j, la::fixed_matrix<double, 3, 2>{{2 * 1.5 * -0.7, 2.25, 0, std::cos(-0.7), std::exp(1.5), 0}}, 1e-9);
    // Large components scale the step: still accurate relative to the value
    const la::fixed_vector<double, 2> big{{1e4, 2}};
    const auto jb = tr::numerical_jacobian<3>([](const la::fixed_vector<double, 2>& v) { return la::fixed_vector<double, 3>{{v[0] * v[0], v[1], v[0] * v[1]}}; }, big);
    EXPECT_NEAR(jb(0, 0), 2e4, 2e4 * 1e-9);
    EXPECT_NEAR(jb(2, 1), 1e4, 1e4 * 1e-9);
}

TEST(TrackingKalmanAlphaBeta, UpdatesPositionAndVelocityWithFixedGains)
{
    // r = z - p = (1, -2); p += 0.5 r; v += (0.2 / 2) r
    const v4 x{{1, 2, 3, 4}};
    const auto y = tr::alpha_beta_update<pos2, vel2>(x, la::fixed_vector<double, 2>{{2, 1}}, 0.5, 0.2, 2.0);
    EXPECT_DOUBLE_EQ(y[0], 1.5);
    EXPECT_DOUBLE_EQ(y[1], 2.1);
    EXPECT_DOUBLE_EQ(y[2], 2.0);
    EXPECT_DOUBLE_EQ(y[3], 3.8);
    // alpha = 1 jumps to the measurement; a measurement on the prediction changes nothing
    EXPECT_EQ((tr::alpha_beta_update<pos2, vel2>(x, la::fixed_vector<double, 2>{{2, 1}}, 1.0, 0.0, 1.0)), (v4{{2, 2, 1, 4}}));
    EXPECT_EQ((tr::alpha_beta_update<pos2, vel2>(x, la::fixed_vector<double, 2>{{1, 3}}, 0.5, 0.2, 1.0)), x);
}

TEST(TrackingKalmanFilter, ConvergesOnAConstantVelocityTarget)
{
    // Noise-free measurements of a target moving at (1, 2): the filter's velocity converges and its covariance shrinks
    g4 x{{{0, 0, 0, 0}}, m4::identity() * 100};
    const auto model = tr::combine(cv{0.01}, cv{0.01});
    const position s{{{0.25, 0, 0, 0.25}}};
    for (int k = 1; k <= 50; ++k) {
        x = tr::predict(x, model, 1.0);
        x = tr::update(x, la::fixed_vector<double, 2>{{1.0 * k, 2.0 * k}}, s)->posterior;
    }
    EXPECT_NEAR(x.mean[1], 1, 1e-3);
    EXPECT_NEAR(x.mean[3], 2, 1e-3);
    EXPECT_LT(x.covariance(0, 0), 0.25);
}

} // namespace
