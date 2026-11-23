//
//  prox_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/optim/prox.hpp>. Each prox is checked by its optimality condition v - p in lambda df(p):
//  by the subgradient conditions for the nonsmooth norms, and for the smooth functions by the derivative of
//  lambda f(x) + (x - v)^2 / 2 at p, by central differences, together with a check that p beats its neighbours. Then
//  the closed forms, Lambert W against scipy, the Moreau envelopes against the gradient identity
//  grad M(v) = (v - prox(v)) / lambda by finite differences, the conjugate by Moreau's identity against known
//  conjugates, the errors, and constant evaluation.
//

#include <snicholls/core/optim/prox.hpp>

#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace op = snicholls::optim;
namespace st = snicholls::stats;

using vec = std::vector<double>;

double uniform(st::wyrand& g, double lo, double hi) { return lo + (hi - lo) * st::uniform_real(g); }

vec random_vector(st::wyrand& g, std::size_t n, double lo = -3, double hi = 3)
{
    vec v(n);
    for (double& x : v)
        x = uniform(g, lo, hi);
    return v;
}

double norm(const vec& v)
{
    double s = 0;
    for (const double x : v)
        s += x * x;
    return std::sqrt(s);
}

double sign(double x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }

constexpr double lambdas[] = {0.05, 0.5, 1.0, 3.0};

// For a separable smooth f: phi(x) = lambda f(x) + (x - v)^2 / 2 has zero derivative at p (central differences) and
// p is no worse than p +- h.
void check_smooth_scalar(const std::function<double(double)>& f, double v, double p, double lambda, double tol)
{
    auto phi = [&](double x) { return lambda * f(x) + (x - v) * (x - v) / 2; };
    const double h = 1e-6 * std::max(1.0, std::abs(p));
    const double derivative = (phi(p + h) - phi(p - h)) / (2 * h);
    EXPECT_NEAR(derivative, 0.0, tol) << "v " << v << " p " << p << " lambda " << lambda;
    EXPECT_LE(phi(p), phi(p + 1e-4 * std::max(1.0, std::abs(p))) + 1e-15);
    EXPECT_LE(phi(p), phi(p - 1e-4 * std::max(1.0, std::abs(p))) + 1e-15);
}

TEST(OptimProxL1, SoftThresholdSatisfiesTheSubgradientCondition)
{
    vec v{3, -0.2, 0.5, -4, 0};
    op::prox_l1(std::span<double>(v), 0.5);
    EXPECT_EQ(v, (vec{2.5, 0, 0, -3.5, 0}));
    st::wyrand g(1);
    for (const double lambda : lambdas)
        for (int t = 0; t < 20; ++t) {
            const vec v0 = random_vector(g, 9);
            vec p = v0;
            op::prox_l1(std::span<double>(p), lambda);
            for (std::size_t i = 0; i < p.size(); ++i) {
                if (p[i] != 0) {
                    EXPECT_NEAR(v0[i] - p[i], lambda * sign(p[i]), 1e-14);
                } else {
                    EXPECT_LE(std::abs(v0[i]), lambda);
                }
            }
        }
}

TEST(OptimProxL2, BlockSoftThresholdSatisfiesTheSubgradientCondition)
{
    vec v{3, 4};
    op::prox_l2(std::span<double>(v), 1.0);   // (1 - 1/5) (3, 4)
    EXPECT_DOUBLE_EQ(v[0], 2.4);
    EXPECT_DOUBLE_EQ(v[1], 3.2);
    vec small{0.3, 0.4};
    op::prox_l2(std::span<double>(small), 0.5);
    EXPECT_EQ(small, (vec{0, 0}));
    st::wyrand g(2);
    for (const double lambda : lambdas)
        for (int t = 0; t < 20; ++t) {
            const vec v0 = random_vector(g, 5);
            vec p = v0;
            op::prox_l2(std::span<double>(p), lambda);
            const double np = norm(p);
            if (np > 0) {
                for (std::size_t i = 0; i < p.size(); ++i) {
                    EXPECT_NEAR(v0[i] - p[i], lambda * p[i] / np, 1e-13);
                }
            } else {
                EXPECT_LE(norm(v0), lambda);
            }
        }
}

TEST(OptimProxGroupL2, ThresholdsEachGroupAndReportsSizesThatDoNotAddUp)
{
    vec v{3, 4, 0.1, -0.1, 6};
    const std::array<std::size_t, 3> sizes{2, 2, 1};
    ASSERT_TRUE(op::prox_group_l2(std::span<double>(v), std::span<const std::size_t>(sizes), 1.0));
    EXPECT_DOUBLE_EQ(v[0], 2.4);
    EXPECT_DOUBLE_EQ(v[1], 3.2);
    EXPECT_EQ(v[2], 0.0);
    EXPECT_EQ(v[3], 0.0);
    EXPECT_DOUBLE_EQ(v[4], 5.0);
    const std::array<std::size_t, 2> wrong{2, 2};
    vec u{1, 2, 3, 4, 5};
    EXPECT_EQ(op::prox_group_l2(std::span<double>(u), std::span<const std::size_t>(wrong), 1.0).unwrap_err(), op::error::dimension_mismatch);
    EXPECT_EQ(u, (vec{1, 2, 3, 4, 5}));
}

TEST(OptimProxLinf, SatisfiesTheSubgradientConditionWithAndWithoutWorkspace)
{
    // The subdifferential of |x|_inf at p != 0 is the convex hull of sign(p_i) e_i over the i with |p_i| = |p|_inf:
    // g = (v - p) / lambda has |g|_1 = 1, is 0 where |p_i| < |p|_inf, and has the sign of p elsewhere.
    st::wyrand g(3);
    for (const double lambda : lambdas)
        for (int t = 0; t < 30; ++t) {
            const vec v0 = random_vector(g, 1 + static_cast<std::size_t>(t % 8));
            vec p = v0, q = v0, ws(v0.size());
            ASSERT_TRUE(op::prox_linf(std::span<double>(p), lambda, std::span<double>(ws)));
            op::prox_linf(std::span<double>(q), lambda);
            double m = 0, l1 = 0, g1 = 0;
            for (std::size_t i = 0; i < p.size(); ++i) {
                EXPECT_NEAR(p[i], q[i], 1e-13);
                m = std::max(m, std::abs(p[i]));
                l1 += std::abs(v0[i]);
            }
            if (m == 0) {
                EXPECT_LE(l1, lambda + 1e-13);
                continue;
            }
            for (std::size_t i = 0; i < p.size(); ++i) {
                const double gi = (v0[i] - p[i]) / lambda;
                g1 += std::abs(gi);
                if (std::abs(p[i]) < m - 1e-12) {
                    EXPECT_NEAR(gi, 0.0, 1e-12);
                } else {
                    EXPECT_GE(gi * sign(p[i]), -1e-12);
                }
            }
            EXPECT_NEAR(g1, 1.0, 1e-12);
        }
    vec inside{0.1, -0.2}, ws(1);
    op::prox_linf(std::span<double>(inside), 1.0);
    EXPECT_EQ(inside, (vec{0, 0}));
    vec v{3, 1};
    EXPECT_EQ(op::prox_linf(std::span<double>(v), 1.0, std::span<double>(ws)).unwrap_err(), op::error::workspace_too_small);
    vec ws2(2);
    ASSERT_TRUE(op::prox_linf(std::span<double>(v), 1.0, std::span<double>(ws2)));   // clamp at tau = 2
    EXPECT_EQ(v, (vec{2, 1}));
    vec z{0.25, 0.25}, ws3(2);
    ASSERT_TRUE(op::prox_linf(std::span<double>(z), 1.0, std::span<double>(ws3)));
    EXPECT_EQ(z, (vec{0, 0}));
}

double huber(double x, double m) { return std::abs(x) <= m ? x * x : 2 * m * std::abs(x) - m * m; }

TEST(OptimProxHuber, MatchesTheClosedFormAndTheOptimalityCondition)
{
    vec v{0.5, 10, -10};
    op::prox_huber(std::span<double>(v), 1.0, 1.0);   // 0.5 / 3; 10 - 2; -10 + 2
    EXPECT_DOUBLE_EQ(v[0], 0.5 / 3);
    EXPECT_DOUBLE_EQ(v[1], 8);
    EXPECT_DOUBLE_EQ(v[2], -8);
    st::wyrand g(4);
    for (const double lambda : lambdas)
        for (const double m : {0.3, 1.0, 2.0})
            for (int t = 0; t < 20; ++t) {
                const double v0 = uniform(g, -8, 8);
                vec p{v0};
                op::prox_huber(std::span<double>(p), lambda, m);
                check_smooth_scalar([m](double x) { return huber(x, m); }, v0, p[0], lambda, 1e-7);
            }
}

TEST(OptimProxSmooth, SquaredL2QuadraticAndElasticNetClosedForms)
{
    st::wyrand g(5);
    for (const double lambda : lambdas)
        for (int t = 0; t < 20; ++t) {
            const double v0 = uniform(g, -5, 5);
            vec p{v0};
            op::prox_squared_l2(std::span<double>(p), lambda);
            check_smooth_scalar([](double x) { return x * x; }, v0, p[0], lambda, 1e-7);
            const double d = uniform(g, 0, 3), c = uniform(g, -2, 2);
            vec q{v0};
            const vec dv{d}, cv{c};
            ASSERT_TRUE(op::prox_quadratic(std::span<double>(q), dv, cv, lambda));
            check_smooth_scalar([d, c](double x) { return d * x * x / 2 + c * x; }, v0, q[0], lambda, 1e-7);
            // Elastic net: v - p in lambda (d|p| + mu p)
            const double mu = uniform(g, 0, 2);
            vec e{v0};
            op::prox_elastic_net(std::span<double>(e), lambda, mu);
            if (e[0] != 0) {
                EXPECT_NEAR(v0 - e[0], lambda * (sign(e[0]) + mu * e[0]), 1e-13);
            } else {
                EXPECT_LE(std::abs(v0), lambda);
            }
        }
    vec q{1, 2};
    const vec one{1};
    EXPECT_EQ(op::prox_quadratic(std::span<double>(q), one, one, 1.0).unwrap_err(), op::error::dimension_mismatch);
}

TEST(OptimProxIndicators, BoxAndOrthantAreTheirProjections)
{
    const double inf = std::numeric_limits<double>::infinity();
    vec v{-3, 0.5, 7};
    const vec l{-1, -inf, 0}, u{1, 0, inf};
    ASSERT_TRUE(op::prox_box(std::span<double>(v), l, u));
    EXPECT_EQ(v, (vec{-1, 0, 7}));
    vec w{-3, 0.5};
    op::prox_nonnegative(std::span<double>(w));
    EXPECT_EQ(w, (vec{0, 0.5}));
    EXPECT_EQ(op::prox_box(std::span<double>(w), l, u).unwrap_err(), op::error::dimension_mismatch);
}

TEST(OptimProxLambertW, MatchesScipy)
{
    // scipy.special.lambertw(x).real
    EXPECT_NEAR(op::detail::prox::w0_exp(std::log(1.0)), 0.5671432904097838, 1e-15);
    EXPECT_NEAR(op::detail::prox::w0_exp(std::log(10.0)), 1.7455280027406994, 1e-15);
    EXPECT_NEAR(op::detail::prox::w0_exp(std::log(std::exp(1.0))), 1.0, 1e-15);
    EXPECT_NEAR(op::detail::prox::w0_exp(std::log(1e-10)), 9.999999999e-11, 1e-24);
    // W(e^a) for a beyond exp's range: w + log w = a
    const double w = op::detail::prox::w0_exp(1e6);
    EXPECT_NEAR(w + std::log(w), 1e6, 1e-9);
    EXPECT_NEAR(op::detail::prox::w0_exp(-800.0), 0.0, 1e-300);
    EXPECT_EQ(op::detail::prox::w0_exp(std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity());
}

TEST(OptimProxNegativeEntropy, SatisfiesTheOptimalityConditionAcrossMagnitudes)
{
    st::wyrand g(6);
    for (const double lambda : {0.01, 0.5, 1.0, 10.0})
        for (int t = 0; t < 40; ++t) {
            const double v0 = uniform(g, -20, 20);
            vec p{v0};
            op::prox_negative_entropy(std::span<double>(p), lambda);
            if (v0 / lambda < -700) {   // the exact value is subnormal or below: documented underflow towards 0
                EXPECT_LT(p[0], 1e-300);
                continue;
            }
            ASSERT_GT(p[0], 0.0);
            // lambda (log p + 1) + p - v = 0, relative to the terms' size
            const double residual = lambda * (std::log(p[0]) + 1) + p[0] - v0;
            EXPECT_NEAR(residual, 0.0, 1e-12 * (std::abs(v0) + lambda * std::abs(std::log(p[0])) + 1)) << v0 << " " << lambda;
        }
    // Huge and very negative v: no overflow, still the root
    vec big{1e6};
    op::prox_negative_entropy(std::span<double>(big), 1.0);
    EXPECT_NEAR(std::log(big[0]) + 1 + big[0], 1e6, 1e-9);
    vec small{-50};
    op::prox_negative_entropy(std::span<double>(small), 1.0);
    EXPECT_NEAR(std::log(small[0]) + 1 + small[0], -50, 1e-12);
}

TEST(OptimProxLogBarrier, SatisfiesTheOptimalityConditionWithoutCancellation)
{
    st::wyrand g(7);
    for (const double lambda : lambdas)
        for (int t = 0; t < 30; ++t) {
            const double v0 = uniform(g, -10, 10);
            vec p{v0};
            op::prox_log_barrier(std::span<double>(p), lambda);
            ASSERT_GT(p[0], 0.0);
            EXPECT_NEAR(-lambda / p[0] + p[0] - v0, 0.0, 1e-12 * (std::abs(v0) + 1 + lambda / p[0]));
        }
    vec far{-1e8};   // p = lambda / |v| to first order: 1e-8, which (v + sqrt(v^2 + 4 lambda)) / 2 would lose
    op::prox_log_barrier(std::span<double>(far), 1.0);
    EXPECT_NEAR(far[0], 1e-8, 1e-22);
}

// grad M(v) = (v - prox(v)) / lambda, by central differences of the envelope
template <typename Envelope, typename Prox>
void check_envelope(Envelope envelope, Prox prox, const vec& v, double lambda)
{
    vec p = v;
    prox(std::span<double>(p), lambda);
    for (std::size_t i = 0; i < v.size(); ++i) {
        vec a = v, b = v;
        const double h = 1e-6;
        a[i] += h;
        b[i] -= h;
        const double derivative = (envelope(std::span<const double>(a), lambda) - envelope(std::span<const double>(b), lambda)) / (2 * h);
        EXPECT_NEAR(derivative, (v[i] - p[i]) / lambda, 1e-6) << i;
    }
}

TEST(OptimProxMoreauEnvelope, GradientIsTheProxResidualOverLambda)
{
    st::wyrand g(8);
    const double inf = std::numeric_limits<double>::infinity();
    const vec l{-1, -inf, 0, -2}, u{1, 0.5, inf, -1};
    for (const double lambda : lambdas)
        for (int t = 0; t < 10; ++t) {
            const vec v = random_vector(g, 4);
            check_envelope([](std::span<const double> x, double lam) { return op::moreau_envelope_l1(x, lam); },
                           [](std::span<double> x, double lam) { op::prox_l1(x, lam); }, v, lambda);
            check_envelope([](std::span<const double> x, double lam) { return op::moreau_envelope_l2(x, lam); },
                           [](std::span<double> x, double lam) { op::prox_l2(x, lam); }, v, lambda);
            check_envelope([](std::span<const double> x, double lam) { return op::moreau_envelope_squared_l2(x, lam); },
                           [](std::span<double> x, double lam) { op::prox_squared_l2(x, lam); }, v, lambda);
            check_envelope([](std::span<const double> x, double lam) { return op::moreau_envelope_huber(x, lam, 0.7); },
                           [](std::span<double> x, double lam) { op::prox_huber(x, lam, 0.7); }, v, lambda);
            check_envelope([&](std::span<const double> x, double lam) { return op::moreau_envelope_box(x, l, u, lam); },
                           [&](std::span<double> x, double) { ASSERT_TRUE(op::prox_box(x, l, u)); }, v, lambda);
            check_envelope([](std::span<const double> x, double lam) { return op::moreau_envelope_nonnegative(x, lam); },
                           [](std::span<double> x, double) { op::prox_nonnegative(x); }, v, lambda);
        }
    // Closed values: the envelope of |x| at 3 with lambda 1 is 3 - 1/2, at 0.5 it is 0.125
    EXPECT_DOUBLE_EQ(op::moreau_envelope_l1(std::span<const double>(vec{3, 0.5}), 1.0), 2.5 + 0.125);
    EXPECT_DOUBLE_EQ(op::moreau_envelope_l2(std::span<const double>(vec{3, 4}), 1.0), 4.5);
    EXPECT_DOUBLE_EQ(op::moreau_envelope_squared_l2(std::span<const double>(vec{3, 4}), 1.0), 25.0 / 3);
    EXPECT_DOUBLE_EQ(op::moreau_envelope_nonnegative(std::span<const double>(vec{-2, 4}), 1.0), 2.0);
}

TEST(OptimProxConjugate, MoreauIdentityGivesKnownConjugates)
{
    st::wyrand g(9);
    for (const double lambda : lambdas) {
        const vec v0 = random_vector(g, 6);
        vec ws(6);
        // f = |x|_1: f* is the indicator of the l-infinity unit ball, whose prox is the clamp to [-1, 1]
        vec p = v0;
        ASSERT_TRUE(op::prox_conjugate([](std::span<double> x, double t) noexcept { op::prox_l1(x, t); }, std::span<double>(p), lambda, std::span<double>(ws)));
        for (std::size_t i = 0; i < 6; ++i) {
            EXPECT_NEAR(p[i], std::clamp(v0[i], -1.0, 1.0), 1e-14);
        }
        // f = |x|^2: f*(y) = |y|^2 / 4, whose prox is v / (1 + lambda / 2)
        vec q = v0;
        ASSERT_TRUE(op::prox_conjugate([](std::span<double> x, double t) { op::prox_squared_l2(x, t); }, std::span<double>(q), lambda, std::span<double>(ws)));
        for (std::size_t i = 0; i < 6; ++i) {
            EXPECT_NEAR(q[i], v0[i] / (1 + lambda / 2), 1e-14);
        }
        // f = |x|_2: f* is the indicator of the unit l2 ball, whose prox is the projection onto it
        vec r = v0;
        ASSERT_TRUE(op::prox_conjugate([](std::span<double> x, double t) { op::prox_l2(x, t); }, std::span<double>(r), lambda, std::span<double>(ws)));
        const double n = norm(v0);
        for (std::size_t i = 0; i < 6; ++i) {
            EXPECT_NEAR(r[i], n > 1 ? v0[i] / n : v0[i], 1e-14);
        }
    }
    vec v{1, 2}, ws(1);
    EXPECT_EQ(op::prox_conjugate([](std::span<double> x, double t) { op::prox_l1(x, t); }, std::span<double>(v), 1.0, std::span<double>(ws)).unwrap_err(),
              op::error::workspace_too_small);
}

// ---- Constant evaluation and noexcept ----

constexpr bool compile_time()
{
    std::array<double, 3> a{3, -0.2, -4};
    op::prox_l1(std::span<double>(a), 0.5);
    std::array<double, 1> e{2.0};
    op::prox_negative_entropy(std::span<double>(e), 1.0);   // log p + 1 + p = 2: p = W(e) = 1
    std::array<double, 1> b{0.0};
    op::prox_log_barrier(std::span<double>(b), 4.0);   // sqrt(16) / 2
    std::array<double, 2> c{3, 1}, ws{};
    const bool linf_ok = op::prox_linf(std::span<double>(c), 1.0, std::span<double>(ws)).is_ok();
    std::array<double, 2> d{3, 4};
    op::prox_l2(std::span<double>(d), 1.0);
    auto near = [](double x, double y, double tol) { return x - y <= tol && y - x <= tol; };
    return a[0] == 2.5 && a[1] == 0 && a[2] == -3.5 && near(e[0], 1.0, 1e-14) && b[0] == 2.0 && linf_ok && c[0] == 2 && c[1] == 1 && near(d[0], 2.4, 1e-15)
           && near(op::moreau_envelope_l1(std::span<const double>(a), 1.0), 5.0, 1e-15);
}
static_assert(compile_time());

static_assert(noexcept(op::prox_l1(std::declval<std::span<double>>(), 1.0)));
static_assert(noexcept(op::prox_negative_entropy(std::declval<std::span<double>>(), 1.0)));
static_assert(noexcept(op::prox_linf(std::declval<std::span<float>>(), 1.0f)));
static_assert(noexcept(op::moreau_envelope_huber(std::declval<std::span<const double>>(), 1.0)));
inline constexpr auto nothrow_prox = [](std::span<double> x, double t) noexcept { op::prox_l1(x, t); };
inline constexpr auto throwing_prox = [](std::span<double> x, double t) { op::prox_l1(x, t); };
static_assert(noexcept(op::prox_conjugate(nothrow_prox, std::declval<std::span<double>>(), 1.0, std::declval<std::span<double>>())));
static_assert(!noexcept(op::prox_conjugate(throwing_prox, std::declval<std::span<double>>(), 1.0, std::declval<std::span<double>>())));

TEST(OptimProxFloat, WorksInFloat)
{
    std::array<float, 2> v{3, 4};
    op::prox_l2(std::span<float>(v), 1.0f);
    EXPECT_NEAR(v[0], 2.4f, 1e-6f);
    std::array<float, 1> e{2.0f};
    op::prox_negative_entropy(std::span<float>(e), 1.0f);
    EXPECT_NEAR(e[0], 1.0f, 1e-6f);
}

} // namespace
