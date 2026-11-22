//
//  transition_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/tracking/transition.hpp>: the concepts on every model, the constant-derivative models'
//  F and Q against their closed forms, Ornstein-Uhlenbeck's closed-form Q, Singer's Q against an exact integration
//  done independently in Python, Singer-approximate's parts, the known turn rate's matrix with its small-angle series,
//  constant turn's motion and its analytic Jacobian against central differences (at a turn, at zero and at tiny
//  rates), combined models' block structure, symmetric noise for negative steps, and models in constant expressions.
//

#include <snicholls/core/tracking/transition.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/tracking/kalman.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace {

namespace tr = snicholls::tracking;
namespace la = snicholls::linalg;

using cv = tr::constant_velocity<double>;
using ca = tr::constant_acceleration<double>;
using rw = tr::random_walk<double>;
using ou = tr::ornstein_uhlenbeck<double>;
using sg = tr::singer<double>;
using sa = tr::singer_approximate<double>;
using ktr = tr::known_turn_rate<double>;
using ct = tr::constant_turn<double>;

// The concepts: every model is a transition_model; constant_turn and a combination holding it are not linear
static_assert(tr::linear_transition<cv> && tr::linear_transition<ca> && tr::linear_transition<rw> && tr::linear_transition<ou> && tr::linear_transition<sg>);
static_assert(tr::linear_transition<sa> && tr::linear_transition<ktr> && tr::linear_transition<tr::constant_nth_derivative<double, 3>>);
static_assert(tr::transition_model<ct> && !tr::linear_transition<ct>);
static_assert(tr::linear_transition<tr::combined<cv, cv>> && tr::linear_transition<tr::combined<cv, ca, rw>>);
static_assert(tr::transition_model<tr::combined<ct, rw>> && !tr::linear_transition<tr::combined<ct, rw>>);

// A type with apply alone is not a model
struct apply_only {
    using value_type = double;
    static constexpr std::size_t dim = 2;
    la::fixed_vector<double, 2> apply(const la::fixed_vector<double, 2>& x, double) const { return x; }
};
static_assert(!tr::transition_model<apply_only>);

// Dimensions and layout: every model but combined is a trivially copyable aggregate of its parameters
static_assert(rw::dim == 1 && cv::dim == 2 && ca::dim == 3 && ou::dim == 2 && sg::dim == 3 && sa::dim == 3 && ktr::dim == 4 && ct::dim == 5);
static_assert(tr::combined<cv, ca, ct>::dim == 10 && std::is_same_v<tr::combined<cv, ca>::value_type, double>);
static_assert(std::is_trivially_copyable_v<cv> && std::is_trivially_copyable_v<sg> && std::is_trivially_copyable_v<sa> && std::is_trivially_copyable_v<ktr> &&
              std::is_trivially_copyable_v<ct>);
static_assert(std::is_aggregate_v<cv> && std::is_aggregate_v<sg> && std::is_aggregate_v<ct> && sizeof(sg) == 2 * sizeof(double));

// noexcept for double
static_assert(noexcept(cv{}.covariance(1.0)) && noexcept(cv{}.matrix(1.0)) && noexcept(sg{}.covariance(1.0)) && noexcept(ct{}.jacobian({}, 1.0)));
static_assert(noexcept(tr::combined<cv, cv>{}.matrix(1.0)) && noexcept(tr::combine(cv{}, cv{})));

// Models in constant expressions: closed forms that are exact in binary
static_assert(cv{2}.covariance(3)(0, 0) == 18 && cv{2}.covariance(3)(0, 1) == 9 && cv{2}.covariance(3)(1, 1) == 6);
static_assert(cv{}.matrix(0.5)(0, 1) == 0.5 && ca{}.matrix(2)(0, 2) == 2);
static_assert(tr::combine(cv{1}, rw{4}).covariance(2)(2, 2) == 8);
constexpr bool singer_at_compile_time()
{
    const auto q = sg{0.7, 0.4}.covariance(3);
    const double expected = 4.633050814082627;   // Python, exact integration (see SingerMatchesAnExactIntegration)
    return q(0, 0) > expected * (1 - 1e-12) && q(0, 0) < expected * (1 + 1e-12);
}
static_assert(singer_at_compile_time());
constexpr bool turn_at_compile_time()
{
    const ct m{};
    const auto y = m.apply({{0, 1, 0, 0, 0.5}}, 2);   // w dt = 1: the velocity turns by one radian
    return y[4] == 0.5 && y[1] > 0.54 && y[1] < 0.541 && y[3] > 0.841 && y[3] < 0.842;
}
static_assert(turn_at_compile_time());

template <std::size_t R, std::size_t C>
void expect_matrix_near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol, const char* what = "")
{
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol) << what << " (" << i << ", " << j << ")";
}

template <std::size_t N>
void expect_symmetric(const la::fixed_matrix<double, N, N>& a)
{
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            EXPECT_EQ(a(i, j), a(j, i)) << i << ", " << j;
}

TEST(TrackingTransitionConstantVelocity, MatrixAndNoiseAreTheClosedForms)
{
    const double q = 0.3, dt = 1.7;
    const cv m{q};
    expect_matrix_near(m.matrix(dt), la::fixed_matrix<double, 2, 2>{{1, dt, 0, 1}}, 0);
    expect_matrix_near(m.covariance(dt), la::fixed_matrix<double, 2, 2>{{q * dt * dt * dt / 3, q * dt * dt / 2, q * dt * dt / 2, q * dt}}, 1e-15);
    const la::fixed_vector<double, 2> x{{10, -2}};
    expect_matrix_near(m.apply(x, dt), la::fixed_vector<double, 2>{{10 - 2 * dt, -2}}, 1e-15);
    EXPECT_EQ(m.jacobian(x, dt), m.matrix(dt));
}

TEST(TrackingTransitionConstantNthDerivative, RandomWalkAndConstantAccelerationAreTheClosedForms)
{
    const double q = 2, dt = 0.5;
    EXPECT_EQ(rw{q}.matrix(dt)(0, 0), 1.0);
    EXPECT_EQ(rw{q}.covariance(dt)(0, 0), q * dt);
    const double t2 = dt * dt, t3 = t2 * dt, t4 = t3 * dt, t5 = t4 * dt;
    expect_matrix_near(ca{q}.matrix(dt), la::fixed_matrix<double, 3, 3>{{1, dt, t2 / 2, 0, 1, dt, 0, 0, 1}}, 0);
    expect_matrix_near(ca{q}.covariance(dt),
                       la::fixed_matrix<double, 3, 3>{{q * t5 / 20, q * t4 / 8, q * t3 / 6, q * t4 / 8, q * t3 / 3, q * t2 / 2, q * t3 / 6, q * t2 / 2, q * dt}}, 1e-15);
    // N = 3: F has dt^3 / 6 in the corner, Q(0, 0) = q dt^7 / (3! 3! 7)
    const tr::constant_nth_derivative<double, 3> jerk{q};
    EXPECT_DOUBLE_EQ(jerk.matrix(dt)(0, 3), t3 / 6);
    EXPECT_DOUBLE_EQ(jerk.covariance(dt)(0, 0), q * t5 * t2 / (36 * 7));
}

TEST(TrackingTransitionConstantNthDerivative, NegativeStepHasTheSameNoiseAndTheInverseMatrix)
{
    const ca m{1.3};
    EXPECT_EQ(m.covariance(-0.8), m.covariance(0.8));
    expect_matrix_near(m.matrix(-0.8) * m.matrix(0.8), la::fixed_matrix<double, 3, 3>::identity(), 1e-15);
    expect_symmetric(m.covariance(0.8));
    // A zero step: F = I, Q = 0
    EXPECT_EQ(m.matrix(0), (la::fixed_matrix<double, 3, 3>::identity()));
    EXPECT_EQ(m.covariance(0), (la::fixed_matrix<double, 3, 3>{}));
}

TEST(TrackingTransitionOrnsteinUhlenbeck, MatrixAndNoiseAreTheClosedForms)
{
    // F = [[1, (1 - e) / k], [0, e]], e = exp(-k dt); Q from integrating f(tau) = ((1 - e^-k tau) / k, e^-k tau)
    const double q = 0.7, k = 0.4, t = 3;
    const double e = std::exp(-k * t), e2 = std::exp(-2 * k * t);
    const ou m{q, k};
    expect_matrix_near(m.matrix(t), la::fixed_matrix<double, 2, 2>{{1, (1 - e) / k, 0, e}}, 1e-15);
    const double q00 = q / (k * k) * (t - 2 * (1 - e) / k + (1 - e2) / (2 * k));
    const double q01 = q / k * ((1 - e) / k - (1 - e2) / (2 * k));
    const double q11 = q * (1 - e2) / (2 * k);
    expect_matrix_near(m.covariance(t), la::fixed_matrix<double, 2, 2>{{q00, q01, q01, q11}}, 1e-13);
    // The same numbers from the Python integration (singer.py, the "ou" case)
    EXPECT_NEAR(m.covariance(t)(0, 0), 2.8112595785279462, 1e-13);
    EXPECT_NEAR(m.covariance(t)(1, 1), 0.7956217908717641, 1e-14);
    EXPECT_EQ(m.covariance(-t), m.covariance(t));
    EXPECT_EQ(m.covariance(0), (la::fixed_matrix<double, 2, 2>{}));
    EXPECT_EQ(m.jacobian({{1, 2}}, t), m.matrix(t));
    expect_matrix_near(m.apply({{1, 2}}, t), la::fixed_vector<double, 2>{{1 + 2 * (1 - e) / k, 2 * e}}, 1e-14);
}

TEST(TrackingTransitionSinger, MatrixIsTheClosedForm)
{
    // F's last column: ((e - 1 + k dt) / k^2, (1 - e) / k, e)
    const double k = 0.4, t = 3, e = std::exp(-k * t);
    expect_matrix_near(sg{1, k}.matrix(t), la::fixed_matrix<double, 3, 3>{{1, t, (e - 1 + k * t) / (k * k), 0, 1, (1 - e) / k, 0, 0, e}}, 1e-14);
}

TEST(TrackingTransitionSinger, NoiseMatchesAnExactIntegration)
{
    // Constants: singer.py, Q = q * integral of f f^T integrated term by term in closed form in 60-digit decimals
    // (an independent computation: no Gauss-Legendre, no core code)
    const la::fixed_matrix<double, 3, 3> a{{4.633050814082627, 3.4343153640235418, 1.0194621616008728, 3.4343153640235418, 2.8112595785279462, 1.0682208457047058,
                                            1.0194621616008728, 1.0682208457047058, 0.7956217908717641}};
    const auto qa = sg{0.7, 0.4}.covariance(3);
    for (std::size_t i = 0; i < 9; ++i)
        EXPECT_NEAR(qa.elements[i], a.elements[i], 1e-12 * std::abs(a.elements[i])) << i;
    expect_symmetric(qa);
    // k dt = 100: the integrand decays fast, so the rule uses its 64 panels
    const la::fixed_matrix<double, 3, 3> b{{0.0015524824, 0.00117612, 6e-06, 0.00117612, 0.001182, 0.0003, 6e-06, 0.0003, 0.015}};
    const auto qb = sg{1.5, 50}.covariance(2);
    for (std::size_t i = 0; i < 9; ++i)
        EXPECT_NEAR(qb.elements[i], b.elements[i], 1e-10 * std::abs(b.elements[i])) << i;
    // A slow decay, k = 0.001: F's corner cancels 1/k^2 terms in double, so agreement is about 1e-6 relative
    const la::fixed_matrix<double, 3, 3> c{{0.75874252636436401, 1.2643601656360888, 1.1233138913652301, 1.2643601656360888, 2.2474705209262016, 2.2466279512275431,
                                            1.1233138913652301, 2.2466279512275431, 2.9955044966270239}};
    const auto qc = sg{2, 0.001}.covariance(1.5);
    for (std::size_t i = 0; i < 9; ++i)
        EXPECT_NEAR(qc.elements[i], c.elements[i], 1e-6 * std::abs(c.elements[i])) << i;
}

TEST(TrackingTransitionSinger, LowerBlockIsOrnsteinUhlenbeckAndZeroStepIsZero)
{
    // Singer's velocity and acceleration are an Ornstein-Uhlenbeck process in (v, a)
    const sg m{0.7, 0.4};
    const auto qs = m.covariance(3);
    const auto qo = ou{0.7, 0.4}.covariance(3);
    expect_matrix_near(la::block<1, 1, 2, 2>(qs), qo, 1e-13);
    EXPECT_EQ(m.covariance(0), (la::fixed_matrix<double, 3, 3>{}));
    EXPECT_EQ(m.covariance(-3), qs);
    EXPECT_EQ(m.jacobian({}, 3), m.matrix(3));
    const la::fixed_vector<double, 3> x{{1, 2, 3}};
    EXPECT_EQ(m.apply(x, 3), m.matrix(3) * x);
}

TEST(TrackingTransitionSingerApproximate, SingersMatrixWithConstantAccelerationsNoise)
{
    const sa m{0.7, 0.4};
    EXPECT_EQ(m.matrix(3), (sg{0.7, 0.4}.matrix(3)));
    EXPECT_EQ(m.covariance(3), ca{0.7}.covariance(3));
    EXPECT_EQ(m.jacobian({}, 3), m.matrix(3));
    const la::fixed_vector<double, 3> x{{1, 2, 3}};
    EXPECT_EQ(m.apply(x, 3), m.matrix(3) * x);
}

// Stone Soup's KnownTurnRate matrix, written out
la::fixed_matrix<double, 4, 4> turn_matrix(double w, double dt)
{
    const double s = std::sin(w * dt), c = std::cos(w * dt);
    return {{1, s / w, 0, -(1 - c) / w, 0, c, 0, -s, 0, (1 - c) / w, 1, s / w, 0, s, 0, c}};
}

TEST(TrackingTransitionKnownTurnRate, MatrixIsTheClosedFormAndNoiseIsPerAxis)
{
    const ktr m{0.2, 0.5, 0.3};
    expect_matrix_near(m.matrix(2), turn_matrix(0.3, 2), 1e-15);
    expect_matrix_near(m.covariance(2), la::block_diagonal(cv{0.2}.covariance(2), cv{0.5}.covariance(2)), 0);
    EXPECT_EQ(m.jacobian({}, 2), m.matrix(2));
    const la::fixed_vector<double, 4> x{{0, 1, 0, 0}};
    EXPECT_EQ(m.apply(x, 2), m.matrix(2) * x);
    // The speed is kept and the velocity turns by w dt
    const auto y = m.apply(x, 2);
    EXPECT_NEAR(std::hypot(y[1], y[3]), 1.0, 1e-15);
    EXPECT_NEAR(std::atan2(y[3], y[1]), 0.6, 1e-15);
}

TEST(TrackingTransitionKnownTurnRate, ZeroAndTinyRatesFollowTheSeries)
{
    // w = 0 is constant velocity per axis
    const auto f0 = ktr{0, 0, 0}.matrix(2);
    expect_matrix_near(f0, la::block_diagonal(cv{}.matrix(2), cv{}.matrix(2)), 0);
    // Either side of the series threshold |w dt| = 1e-4 the matrix is the closed form, to the closed form's own
    // cancellation in (1 - cos a) / w (about 1e-16 / w)
    const double dt = 2;
    expect_matrix_near(ktr{0, 0, 0.99e-4 / dt}.matrix(dt), turn_matrix(0.99e-4 / dt, dt), 1e-11);
    expect_matrix_near(ktr{0, 0, 1.01e-4 / dt}.matrix(dt), turn_matrix(1.01e-4 / dt, dt), 1e-12);
    // A tiny rate: (1 - cos a) / w = dt a / 2 to O(a^3)
    const auto tiny = ktr{0, 0, 1e-9}.matrix(dt);
    EXPECT_NEAR(tiny(0, 3), -dt * 1e-9 * dt / 2, 1e-24);
    EXPECT_NEAR(tiny(0, 1), dt, 1e-15);
}

TEST(TrackingTransitionConstantTurn, MovesAsTheKnownTurnRateAndKeepsTheRate)
{
    const ct m{0.1, 0.2, 0.01};
    const la::fixed_vector<double, 5> x{{3, 4, -1, 2, 0.25}};
    const auto y = m.apply(x, 1.5);
    const auto f = turn_matrix(0.25, 1.5);
    const la::fixed_vector<double, 4> x4{{3, 4, -1, 2}};
    const auto y4 = f * x4;
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(y[i], y4[i], 1e-14) << i;
    EXPECT_EQ(y[4], 0.25);
    // Zero rate: straight line
    const auto z = m.apply({{3, 4, -1, 2, 0}}, 1.5);
    expect_matrix_near(z, la::fixed_vector<double, 5>{{3 + 6, 4, -1 + 3, 2, 0}}, 1e-15);
}

TEST(TrackingTransitionConstantTurn, NoiseIsPerAxisPlusTheRate)
{
    const ct m{0.1, 0.2, 0.01};
    const auto q = m.covariance(-1.5);
    expect_matrix_near(la::block<0, 0, 4, 4>(q), la::block_diagonal(cv{0.1}.covariance(1.5), cv{0.2}.covariance(1.5)), 0);
    EXPECT_DOUBLE_EQ(q(4, 4), 0.01 * 1.5);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_EQ(q(i, 4), 0.0);
}

void expect_turn_jacobian(const la::fixed_vector<double, 5>& x, double dt, double tol)
{
    const ct m{};
    const auto analytic = m.jacobian(x, dt);
    const auto numeric = tr::numerical_jacobian<5>([&](const la::fixed_vector<double, 5>& s) { return m.apply(s, dt); }, x);
    expect_matrix_near(analytic, numeric, tol, "constant_turn");
}

TEST(TrackingTransitionConstantTurn, JacobianMatchesCentralDifferences)
{
    expect_turn_jacobian({{3, 4, -1, 2, 0.25}}, 1.5, 1e-8);
    expect_turn_jacobian({{3, -40, -1, 25, -1.3}}, 0.7, 1e-7);
    // At zero rate and at tiny ones the series branch is used
    expect_turn_jacobian({{3, 4, -1, 2, 0}}, 1.5, 1e-8);
    expect_turn_jacobian({{3, 4, -1, 2, 1e-6}}, 1.5, 1e-8);
    expect_turn_jacobian({{3, 4, -1, 2, -3e-5}}, 2, 1e-8);
    // Just above the threshold central differences of the closed form lose digits ((1 - cos a) / w cancels), so the
    // check is looser here and the next test pins the analytic value to a series
    expect_turn_jacobian({{3, 4, -1, 2, 1.2e-4}}, 1, 1e-6);
}

// d/dw (sin(w t) / w) and d/dw ((1 - cos(w t)) / w) by their Taylor series, eight terms: exact in double for |w t| < 0.1
double dsw_series(double w, double t)
{
    double s = 0, term_t = t * t * t, term_w = w, f = 6;   // n = 1: -2 w t^3 / 3!
    for (int n = 1; n <= 8; ++n) {
        s += ((n % 2) ? -1.0 : 1.0) * 2 * n * term_w * term_t / f;
        term_t *= t * t;
        term_w *= w * w;
        f *= (2.0 * n + 2) * (2.0 * n + 3);
    }
    return s;
}
double dcw_series(double w, double t)
{
    double s = 0, term_t = t * t, term_w = 1, f = 2;   // n = 1: t^2 / 2!
    for (int n = 1; n <= 8; ++n) {
        s += ((n % 2) ? 1.0 : -1.0) * (2 * n - 1) * term_w * term_t / f;
        term_t *= t * t;
        term_w *= w * w;
        f *= (2.0 * n + 1) * (2.0 * n + 2);
    }
    return s;
}

TEST(TrackingTransitionConstantTurn, RateColumnMatchesTheSeriesEitherSideOfTheThreshold)
{
    const double vx = 4, vy = 2, dt = 1;
    for (const double w : {0.0, 1e-6, 0.9e-4, 1.2e-4, 1e-3, 0.05}) {
        const auto j = ct{}.jacobian({{0, vx, 0, vy, w}}, dt);
        const double dsw = dsw_series(w, dt), dcw = dcw_series(w, dt);
        // The closed form's (1 - cos a) / w^2 has a cancellation error of about 1e-16 / w^2, 1e-8 at the threshold; below
        // it the header's two-term series is off by its next term, w^3 dt^5 / 30
        const double tol = w > 1e-4 ? 3e-16 / (w * w) + 1e-14 : 1e-12;
        EXPECT_NEAR(j(0, 4), vx * dsw - vy * dcw, tol) << w;
        EXPECT_NEAR(j(2, 4), vx * dcw + vy * dsw, tol) << w;
    }
}

TEST(TrackingTransitionConstantTurn, JacobianAtZeroRateIsTheClosedForm)
{
    // d/dw of the position rows at w = 0: x gains -vy dt^2 / 2, y gains vx dt^2 / 2
    const double dt = 1.5, vx = 4, vy = 2;
    const auto j = ct{}.jacobian({{0, vx, 0, vy, 0}}, dt);
    EXPECT_NEAR(j(0, 4), -vy * dt * dt / 2, 1e-15);
    EXPECT_NEAR(j(1, 4), -dt * vy, 1e-15);
    EXPECT_NEAR(j(2, 4), vx * dt * dt / 2, 1e-15);
    EXPECT_NEAR(j(3, 4), dt * vx, 1e-15);
    EXPECT_EQ(j(4, 4), 1.0);
}

TEST(TrackingTransitionCombined, IsBlockDiagonal)
{
    const auto m = tr::combine(cv{0.5}, ca{0.2});
    static_assert(std::is_same_v<std::remove_const_t<decltype(m)>, tr::combined<cv, ca>>);
    const double dt = 1.2;
    expect_matrix_near(m.matrix(dt), la::block_diagonal(cv{0.5}.matrix(dt), ca{0.2}.matrix(dt)), 0);
    expect_matrix_near(m.covariance(dt), la::block_diagonal(cv{0.5}.covariance(dt), ca{0.2}.covariance(dt)), 0);
    expect_matrix_near(m.jacobian({}, dt), m.matrix(dt), 0);
    const la::fixed_vector<double, 5> x{{1, 2, 3, 4, 5}};
    const auto y = m.apply(x, dt);
    EXPECT_EQ(y, m.matrix(dt) * x);
    // Each block moved by its own model
    EXPECT_DOUBLE_EQ(y[0], 1 + 2 * dt);
    EXPECT_DOUBLE_EQ(y[2], 3 + 4 * dt + 5 * dt * dt / 2);
    // The models are held in state order
    EXPECT_EQ(std::get<0>(m.models).q, 0.5);
    EXPECT_EQ(std::get<1>(m.models).q, 0.2);
}

TEST(TrackingTransitionCombined, TwoDimensionalConstantVelocityIsXVxYVy)
{
    const tr::combined<cv, cv> m{{cv{1}, cv{2}}};
    const auto y = m.apply({{0, 1, 10, -1}}, 2);
    expect_matrix_near(y, la::fixed_vector<double, 4>{{2, 1, 8, -1}}, 0);
    EXPECT_EQ(m.covariance(2)(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(m.covariance(2)(2, 2), 2 * 8.0 / 3);
}

TEST(TrackingTransitionCombined, NonlinearBlockHasItsJacobian)
{
    const auto m = tr::combine(ct{0.1, 0.1, 0.01}, rw{3});
    const la::fixed_vector<double, 6> x{{3, 4, -1, 2, 0.25, 7}};
    const auto j = m.jacobian(x, 1.5);
    expect_matrix_near(la::block<0, 0, 5, 5>(j), ct{}.jacobian(la::block<0, 0, 5, 1>(x), 1.5), 0);
    EXPECT_EQ(j(5, 5), 1.0);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_EQ(j(i, 5), 0.0);
    const auto y = m.apply(x, 1.5);
    EXPECT_EQ(y[5], 7.0);
    expect_matrix_near(la::block<0, 0, 5, 1>(y), ct{}.apply(la::block<0, 0, 5, 1>(x), 1.5), 0);
    EXPECT_DOUBLE_EQ(m.covariance(1.5)(5, 5), 4.5);
    const auto numeric = tr::numerical_jacobian<6>([&](const la::fixed_vector<double, 6>& s) { return m.apply(s, 1.5); }, x);
    expect_matrix_near(j, numeric, 1e-8, "combined");
}

TEST(TrackingTransitionNoise, EveryModelsNoiseIsSymmetricPositiveSemiDefinite)
{
    const auto check = [](const auto& q) {
        constexpr std::size_t n = std::remove_cvref_t<decltype(q)>::rows;
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_GE(q(i, i), 0.0);
            for (std::size_t j = 0; j < n; ++j)
                EXPECT_EQ(q(i, j), q(j, i));
        }
        EXPECT_TRUE(la::cholesky(q + std::remove_cvref_t<decltype(q)>::identity() * 1e-12).is_ok());
    };
    check(cv{1}.covariance(2));
    check(ca{1}.covariance(2));
    check(ou{1, 0.5}.covariance(2));
    check(sg{1, 0.5}.covariance(2));
    check(sa{1, 0.5}.covariance(2));
    check(ktr{1, 2, 0.1}.covariance(2));
    check(ct{1, 2, 0.1}.covariance(2));
    check(tr::combine(cv{1}, sg{1, 0.5}).covariance(2));
}

} // namespace
