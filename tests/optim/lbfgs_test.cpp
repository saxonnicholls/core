//
//  lbfgs_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/optim/lbfgs.hpp>: the history's ring order, capacity and curvature check, the fixed and
//  run-time forms agreeing; the two-loop recursion's secant equation and its steepest-descent start; lbfgs_minimize on
//  Rosenbrock in 2 and 10 dimensions, a convex quadratic (about n iterations), an ill-conditioned quadratic, the
//  value tolerance, the iteration limit, a wrong gradient, a short workspace; noexcept and constant evaluation.
//

#include <snicholls/core/optim/lbfgs.hpp>

#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace op = snicholls::optim;
namespace st = snicholls::stats;
using cspan = std::span<const double>;

// The extended Rosenbrock function (Moré, Garbow and Hillstrom): independent pairs, minimum 0 at all ones
double rosenbrock(cspan x, std::span<double> g)
{
    double f = 0;
    for (std::size_t i = 0; i + 1 < x.size(); i += 2) {
        const double a = x[i + 1] - x[i] * x[i], b = 1 - x[i];
        f += 100 * a * a + b * b;
        g[i] = -400 * x[i] * a - 2 * b;
        g[i + 1] = 200 * a;
    }
    return f;
}

// ½ xᵀ A x - bᵀ x for a dense SPD A
struct quadratic {
    std::size_t n;
    std::vector<double> a, b;
    double operator()(cspan x, std::span<double> g) const
    {
        double f = 0;
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0;
            for (std::size_t j = 0; j < n; ++j)
                s += a[i * n + j] * x[j];
            g[i] = s - b[i];
            f += x[i] * (s / 2 - b[i]);
        }
        return f;
    }
};

// Q diag(eig) Qᵀ, Q a Householder reflector
quadratic make_quadratic(const std::vector<double>& eig, std::uint64_t seed)
{
    const std::size_t n = eig.size();
    st::wyrand g(seed);
    std::vector<double> v(n), b(n), a(n * n, 0.0);
    double vv = 0;
    for (auto& e : v) {
        e = st::uniform_real(g) - 0.5;
        vv += e * e;
    }
    for (auto& e : b)
        e = 2 * st::uniform_real(g) - 1;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k) {
                const double qik = (i == k ? 1.0 : 0.0) - 2 * v[i] * v[k] / vv, qjk = (j == k ? 1.0 : 0.0) - 2 * v[j] * v[k] / vv;
                a[i * n + j] += qik * eig[k] * qjk;
            }
    return {n, a, b};
}

// noexcept follows the objective
constexpr auto nothrow_fg = [](cspan x, std::span<double> g) noexcept {
    g[0] = 2 * x[0];
    return x[0] * x[0];
};
constexpr auto throwing_fg = [](cspan x, std::span<double>) -> double { throw std::runtime_error(x.empty() ? "" : "x"); };
static_assert(noexcept(op::lbfgs_minimize<5>(nothrow_fg, std::span<double>{}, {}, {})));
static_assert(!noexcept(op::lbfgs_minimize<5>(throwing_fg, std::span<double>{}, {}, {})));
static_assert(op::lbfgs_workspace(10, 5) == 140);
static_assert(op::lbfgs_history<double, 4>::capacity == 4);

// Constant evaluation: minimise (x - 1)² + 10 (y + 2)²
constexpr bool minimises_at_compile_time()
{
    const auto fg = [](cspan x, std::span<double> g) {
        g[0] = 2 * (x[0] - 1);
        g[1] = 20 * (x[1] + 2);
        return (x[0] - 1) * (x[0] - 1) + 10 * (x[1] + 2) * (x[1] + 2);
    };
    std::array<double, 2> x{5, 5};
    std::array<double, op::lbfgs_workspace(2, 3)> work{};
    const auto r = op::lbfgs_minimize<3>(fg, std::span<double>(x), {1e-10}, work);
    return r.converged() && (x[0] - 1) * (x[0] - 1) < 1e-18 && (x[1] + 2) * (x[1] + 2) < 1e-18;
}
static_assert(minimises_at_compile_time());

// The fixed-dimension history in a constant expression
constexpr bool fixed_history_at_compile_time()
{
    op::lbfgs_history<double, 2, 2> h;
    const std::array<double, 2> s1{1, 0}, y1{2, 0}, s2{0, 1}, y2{0, 4}, s3{1, 1}, y3{1, 1};
    h.push(s1, y1);
    h.push(s2, y2);
    h.push(s3, y3);
    return h.size() == 2 && h.s(0)[1] == 1 && h.s(1)[0] == 1 && h.rho(1) == 0.5 && h.gamma() == 1;
}
static_assert(fixed_history_at_compile_time());

TEST(OptimLbfgsHistory, KeepsTheNewestMPairsOldestFirst)
{
    std::vector<double> sbuf(3 * 2), ybuf(3 * 2);
    op::lbfgs_history<double, 3> h(sbuf, ybuf, 2);
    EXPECT_TRUE(h.empty());
    EXPECT_EQ(h.dimension(), 2u);
    EXPECT_EQ(h.gamma(), 1.0);
    for (int k = 1; k <= 5; ++k) {
        const std::vector<double> s{double(k), 0}, y{double(k) * 2, 1};
        EXPECT_TRUE(h.push(s, y));
    }
    EXPECT_EQ(h.size(), 3u);
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_EQ(h.s(k)[0], double(k + 3));
        EXPECT_EQ(h.y(k)[0], double(2 * (k + 3)));
        EXPECT_EQ(h.rho(k), 1 / (double(k + 3) * double(2 * (k + 3))));
    }
    EXPECT_DOUBLE_EQ(h.gamma(), 50.0 / 101.0);   // sᵀy / yᵀy of (5, 0), (10, 1)
    h.clear();
    EXPECT_TRUE(h.empty());
    EXPECT_EQ(h.gamma(), 1.0);
}

TEST(OptimLbfgsHistory, RefusesPairsWithoutPositiveCurvature)
{
    std::vector<double> sbuf(4), ybuf(4);
    op::lbfgs_history<double, 2> h(sbuf, ybuf, 2);
    const std::vector<double> s{1, 0}, neg{-1, 0}, zero{0, 0}, orth{0, 1};
    EXPECT_FALSE(h.push(s, neg));
    EXPECT_FALSE(h.push(s, zero));
    EXPECT_FALSE(h.push(s, orth));
    EXPECT_TRUE(h.empty());
}

TEST(OptimLbfgsHistory, FixedAndRunTimeFormsGiveTheSameDirection)
{
    constexpr std::size_t n = 4;
    std::vector<double> sbuf(n * 3), ybuf(n * 3);
    op::lbfgs_history<double, 3> dyn(sbuf, ybuf, n);
    op::lbfgs_history<double, 3, n> fix;
    st::wyrand g(5);
    for (int k = 0; k < 5; ++k) {
        std::array<double, n> s{}, y{};
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = st::uniform_real(g) - 0.5;
            y[i] = s[i] * (1 + static_cast<double>(i)) + 0.01 * (st::uniform_real(g) - 0.5);
        }
        EXPECT_EQ(dyn.push(s, y), fix.push(s, y));
    }
    const std::array<double, n> grad{1, -2, 3, -4};
    std::array<double, n> d1{}, d2{};
    op::two_loop(dyn, grad, d1);
    op::two_loop(fix, grad, d2);
    EXPECT_EQ(d1, d2);
}

TEST(OptimLbfgsTwoLoop, SatisfiesTheSecantEquationForTheNewestPair)
{
    // H y = s for the newest pair (s, y): two_loop(y) = -s
    constexpr std::size_t n = 5;
    std::vector<double> sbuf(n * 4), ybuf(n * 4), d(n);
    op::lbfgs_history<double, 4> h(sbuf, ybuf, n);
    st::wyrand g(9);
    std::vector<double> s(n), y(n);
    for (int k = 0; k < 6; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = st::uniform_real(g) - 0.5;
            y[i] = (2 + static_cast<double>(i)) * s[i];
        }
        ASSERT_TRUE(h.push(s, y));
    }
    op::two_loop(h, cspan(y), std::span<double>(d));
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(d[i], -s[i], 1e-12);
}

TEST(OptimLbfgsTwoLoop, AnEmptyHistoryGivesSteepestDescent)
{
    std::vector<double> sbuf(6), ybuf(6), d(3);
    const op::lbfgs_history<double, 2> h(sbuf, ybuf, 3);
    const std::vector<double> grad{1, -2, 0.5};
    op::two_loop(h, cspan(grad), std::span<double>(d));
    EXPECT_EQ(d, (std::vector<double>{-1, 2, -0.5}));
}

TEST(OptimLbfgsMinimize, SolvesRosenbrockInTwoDimensions)
{
    std::vector<double> x{-1.2, 1}, work(op::lbfgs_workspace(2, 5));
    const auto r = op::lbfgs_minimize<5>(rosenbrock, std::span<double>(x), {1e-9}, work);
    EXPECT_EQ(r.status, op::lbfgs_status::gradient_converged);
    EXPECT_LT(r.iterations, 60u);
    EXPECT_NEAR(x[0], 1, 1e-8);
    EXPECT_NEAR(x[1], 1, 1e-8);
    EXPECT_LT(r.value, 1e-16);
    EXPECT_LE(r.gradient_norm, 1e-9);
}

TEST(OptimLbfgsMinimize, SolvesRosenbrockInTenDimensions)
{
    std::vector<double> x(10), work(op::lbfgs_workspace(10, 7));
    for (std::size_t i = 0; i < 10; ++i)
        x[i] = i % 2 == 0 ? -1.2 : 1;
    const auto r = op::lbfgs_minimize<7>(rosenbrock, std::span<double>(x), {1e-8}, work);
    EXPECT_TRUE(r.converged());
    EXPECT_LT(r.iterations, 200u);
    for (const double v : x)
        EXPECT_NEAR(v, 1, 1e-7);
    EXPECT_GE(r.evaluations, r.iterations);
}

TEST(OptimLbfgsMinimize, ConvergesInAboutNIterationsOnAConvexQuadratic)
{
    constexpr std::size_t n = 12;
    std::vector<double> eig(n);
    for (std::size_t i = 0; i < n; ++i)
        eig[i] = 1 + static_cast<double>(i);
    const auto q = make_quadratic(eig, 3);
    std::vector<double> x(n, 0.0), work(op::lbfgs_workspace(n, n));
    // with M = n and near-exact line searches L-BFGS is BFGS, which finishes in n steps on a quadratic
    op::lbfgs_options<double> o{1e-8};
    o.line_search.c2 = 1e-3;
    const auto r = op::lbfgs_minimize<n>(q, std::span<double>(x), o, work);
    EXPECT_TRUE(r.converged());
    EXPECT_LE(r.iterations, n + 2);
    // the solution solves A x = b
    std::vector<double> g(n);
    q(x, g);
    for (const double v : g)
        EXPECT_LT(std::abs(v), 1e-8);
    // with the default (loose) line search it still takes a small multiple of n
    std::vector<double> x2(n, 0.0);
    const auto r2 = op::lbfgs_minimize<n>(q, std::span<double>(x2), {1e-8}, work);
    EXPECT_TRUE(r2.converged());
    EXPECT_LE(r2.iterations, 3 * n);
}

// ½ (x - x*)ᵀ A (x - x*), computed from x - x* so that f keeps its relative accuracy near the minimum
struct shifted_quadratic {
    quadratic q;
    double operator()(cspan x, std::span<double> g) const
    {
        std::vector<double> e(q.n);
        for (std::size_t i = 0; i < q.n; ++i)
            e[i] = x[i] - q.b[i];
        double f = 0;
        for (std::size_t i = 0; i < q.n; ++i) {
            double s = 0;
            for (std::size_t j = 0; j < q.n; ++j)
                s += q.a[i * q.n + j] * e[j];
            g[i] = s;
            f += e[i] * s / 2;
        }
        return f;
    }
};

TEST(OptimLbfgsMinimize, SolvesAnIllConditionedQuadratic)
{
    // eigenvalues from 1 to 1e6, a rotated basis, the minimum at x* = b: slow with M = 10 (hundreds to thousands of
    // iterations, as reference implementations take), but it gets there, and more memory helps
    constexpr std::size_t n = 20;
    std::vector<double> eig(n);
    for (std::size_t i = 0; i < n; ++i)
        eig[i] = std::pow(10.0, 6.0 * static_cast<double>(i) / (n - 1));
    const shifted_quadratic f{make_quadratic(eig, 4)};
    std::vector<double> x(n, 0.0), x20(n, 0.0), work(op::lbfgs_workspace(n, 20));
    const auto r = op::lbfgs_minimize<10>(f, std::span<double>(x), {1e-6, 0, 20000}, work);
    EXPECT_EQ(r.status, op::lbfgs_status::gradient_converged);
    EXPECT_LE(r.gradient_norm, 1e-6);
    // ‖x - x*‖ ≤ ‖∇f‖₂ / λ_min
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], f.q.b[i], 1e-6 * std::sqrt(static_cast<double>(n)));
    const auto r20 = op::lbfgs_minimize<20>(f, std::span<double>(x20), {1e-6, 0, 20000}, work);
    EXPECT_TRUE(r20.converged());
    EXPECT_LT(r20.iterations, r.iterations);
}

TEST(OptimLbfgsMinimize, AGradientToleranceBelowRoundingEndsInALineSearchFailureAtTheBestPoint)
{
    // Σ xᵢ (λᵢ xᵢ / 2 - bᵢ) near its minimum, about -0.67, is rounded to about 1e-16, which hides the decrease left in
    // the stiff directions (‖g‖² / 2λ with λ = 1e6), so 1e-9 cannot be reached from values; the search stops at its
    // best point rather than looping
    constexpr std::size_t n = 20;
    std::vector<double> eig(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        eig[i] = std::pow(10.0, 6.0 * static_cast<double>(i) / (n - 1));
        b[i] = std::cos(static_cast<double>(i));
    }
    const auto fg = [&](cspan x, std::span<double> g) {
        double f = 0;
        for (std::size_t i = 0; i < n; ++i) {
            g[i] = eig[i] * x[i] - b[i];
            f += x[i] * (eig[i] * x[i] / 2 - b[i]);
        }
        return f;
    };
    std::vector<double> x(n, 0.0), work(op::lbfgs_workspace(n, 20)), g(n);
    const auto r = op::lbfgs_minimize<20>(fg, std::span<double>(x), {1e-9, 0, 100000}, work);
    EXPECT_EQ(r.status, op::lbfgs_status::line_search_failed);
    EXPECT_LT(r.gradient_norm, 1e-4);
    EXPECT_EQ(fg(x, g), r.value);
}

TEST(OptimLbfgsMinimize, StopsOnTheValueToleranceAndTheIterationLimit)
{
    std::vector<double> x{-1.2, 1}, work(op::lbfgs_workspace(2, 5));
    auto r = op::lbfgs_minimize<5>(rosenbrock, std::span<double>(x), {0, 1e-3, 1000}, work);
    EXPECT_EQ(r.status, op::lbfgs_status::value_converged);
    EXPECT_TRUE(r.converged());
    x = {-1.2, 1};
    r = op::lbfgs_minimize<5>(rosenbrock, std::span<double>(x), {1e-12, 0, 3}, work);
    EXPECT_EQ(r.status, op::lbfgs_status::iteration_limit);
    EXPECT_EQ(r.iterations, 3u);
    EXPECT_FALSE(r.converged());
    // the value reported is f at the returned x
    std::vector<double> g(2);
    EXPECT_EQ(rosenbrock(x, g), r.value);
}

TEST(OptimLbfgsMinimize, ReportsALineSearchFailureOnAWrongGradient)
{
    // the gradient's sign is flipped: no step along "-∇f" decreases f
    const auto fg = [](cspan x, std::span<double> g) {
        g[0] = -2 * x[0];
        return x[0] * x[0];
    };
    std::vector<double> x{1}, work(op::lbfgs_workspace(1, 3));
    const auto r = op::lbfgs_minimize<3>(fg, std::span<double>(x), {}, work);
    EXPECT_EQ(r.status, op::lbfgs_status::line_search_failed);
    EXPECT_EQ(x[0], 1.0);
    EXPECT_EQ(r.value, 1.0);
}

TEST(OptimLbfgsMinimize, RejectsAShortWorkspaceAndStartsAtAMinimum)
{
    std::vector<double> x{3, 4}, work(op::lbfgs_workspace(2, 5) - 1);
    auto r = op::lbfgs_minimize<5>(rosenbrock, std::span<double>(x), {}, work);
    EXPECT_EQ(r.status, op::lbfgs_status::invalid_input);
    EXPECT_EQ(x, (std::vector<double>{3, 4}));
    x = {1, 1};
    work.resize(op::lbfgs_workspace(2, 5));
    r = op::lbfgs_minimize<5>(rosenbrock, std::span<double>(x), {}, work);
    EXPECT_EQ(r.status, op::lbfgs_status::gradient_converged);
    EXPECT_EQ(r.iterations, 0u);
    EXPECT_EQ(r.evaluations, 1u);
}

TEST(OptimLbfgsMinimize, WorksInFloat)
{
    const auto fg = [](std::span<const float> x, std::span<float> g) {
        g[0] = 2 * (x[0] - 1);
        g[1] = 8 * (x[1] + 1);
        return (x[0] - 1) * (x[0] - 1) + 4 * (x[1] + 1) * (x[1] + 1);
    };
    std::vector<float> x{3, 3}, work(op::lbfgs_workspace(2, 4));
    const auto r = op::lbfgs_minimize<4>(fg, std::span<float>(x), {1e-4f}, work);
    EXPECT_TRUE(r.converged());
    EXPECT_NEAR(x[0], 1.0f, 1e-4f);
    EXPECT_NEAR(x[1], -1.0f, 1e-4f);
}

} // namespace
