//
//  line_search_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/optim/line_search.hpp>: Armijo backtracking accepts the first step with sufficient
//  decrease, contracts by the factor given, survives NaN, and reports a non-descent direction and the evaluation
//  limit; strong Wolfe meets both conditions (on a quadratic in two evaluations by the cubic, on Rosenbrock along
//  many directions, after expanding a short first step and zooming a long one), leaves the returned point in the
//  trial storage, reports step_at_maximum on an unbounded function and the evaluation limit with its best point;
//  noexcept and constant evaluation.
//

#include <snicholls/core/optim/line_search.hpp>

#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace op = snicholls::optim;
namespace st = snicholls::stats;
using cspan = std::span<const double>;

// f(x) = ½ Σ cᵢ xᵢ², gradient cᵢ xᵢ
struct quadratic {
    std::vector<double> c;
    std::size_t* calls = nullptr;
    double operator()(cspan x) const
    {
        if (calls)
            ++*calls;
        double f = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            f += c[i] * x[i] * x[i] / 2;
        return f;
    }
    double operator()(cspan x, std::span<double> g) const
    {
        for (std::size_t i = 0; i < x.size(); ++i)
            g[i] = c[i] * x[i];
        return (*this)(x);
    }
};

double rosenbrock(cspan x, std::span<double> g)
{
    const double a = x[1] - x[0] * x[0], b = 1 - x[0];
    g[0] = -400 * x[0] * a - 2 * b;
    g[1] = 200 * a;
    return 100 * a * a + b * b;
}

double dot(cspan a, cspan b)
{
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

// noexcept follows the callable
constexpr auto nothrow_f = [](cspan x) noexcept { return x[0] * x[0]; };
constexpr auto nothrow_fg = [](cspan x, std::span<double> g) noexcept {
    g[0] = 2 * x[0];
    return x[0] * x[0];
};
constexpr auto throwing_fg = [](cspan x, std::span<double>) -> double { throw std::runtime_error(x.empty() ? "" : "x"); };
static_assert(noexcept(op::backtracking_armijo(nothrow_f, cspan{}, cspan{}, 0.0, -1.0, std::span<double>{})));
static_assert(noexcept(op::strong_wolfe(nothrow_fg, cspan{}, cspan{}, 0.0, -1.0, std::span<double>{}, std::span<double>{})));
static_assert(!noexcept(op::strong_wolfe(throwing_fg, cspan{}, cspan{}, 0.0, -1.0, std::span<double>{}, std::span<double>{})));

// Constant evaluation: φ(α) = (α - 3)² - 9 along d = 1 from x = 0 (f(x) = (x - 3)²): the cubic finds α = 3 exactly
constexpr double wolfe_step_at_compile_time()
{
    const auto fg = [](cspan x, std::span<double> g) {
        g[0] = 2 * (x[0] - 3);
        return (x[0] - 3) * (x[0] - 3);
    };
    const std::array<double, 1> x{0}, d{1};
    std::array<double, 1> t{}, tg{};
    const auto r = op::strong_wolfe(fg, x, d, 9.0, -6.0, t, tg, {1, 1e-4, 0.1, 2, 1e10, 40});
    return r.satisfied() ? r.step : -1;
}
static_assert(wolfe_step_at_compile_time() > 2.99 && wolfe_step_at_compile_time() < 3.01);
constexpr double armijo_step_at_compile_time()
{
    const std::array<double, 1> x{0}, d{1};
    std::array<double, 1> t{};
    const auto r = op::backtracking_armijo([](cspan v) { return (v[0] - 1) * (v[0] - 1); }, x, d, 1.0, -2.0, t, {8, 1e-4, 0.5, 50});
    return r.satisfied() ? r.step : -1;
}
static_assert(armijo_step_at_compile_time() == 1);   // 8, 4, 2 fail; 1 decreases

TEST(OptimLineSearchArmijo, AcceptsTheFirstStepWithSufficientDecrease)
{
    std::size_t calls = 0;
    const quadratic f{{1, 10}, &calls};
    const std::vector<double> x{1, 1}, d{-1, -10};   // -∇f
    std::vector<double> t(2);
    const double fx = f(cspan(x)), gd = -(1 + 100);
    calls = 0;
    const auto r = op::backtracking_armijo(f, x, d, fx, gd, t);
    EXPECT_TRUE(r.satisfied());
    // α = 1 overshoots along the stiff coordinate (f = 405 > 5.5); shorter steps are tried until one decreases enough
    EXPECT_LE(r.value, fx + 1e-4 * r.step * gd);
    EXPECT_EQ(r.evaluations, calls);
    // the previous step tried (twice as long) did not decrease enough
    std::vector<double> t2(2);
    for (std::size_t i = 0; i < 2; ++i)
        t2[i] = x[i] + 2 * r.step * d[i];
    if (r.step < 1) {
        EXPECT_GT(f(cspan(t2)), fx + 1e-4 * 2 * r.step * gd);
    }
    for (std::size_t i = 0; i < 2; ++i)
        EXPECT_EQ(t[i], x[i] + r.step * d[i]);
}

TEST(OptimLineSearchArmijo, ContractsByTheShrinkFactor)
{
    // f(x) = x², from x = 1 along d = -1 with first step 64 and contraction 1/4: the steps 64, 16, 4, 1
    const auto f = [](cspan v) { return v[0] * v[0]; };
    const std::vector<double> x{1}, d{-1};
    std::vector<double> t(1);
    const auto r = op::backtracking_armijo(f, x, d, 1.0, -2.0, t, {64, 1e-4, 0.25, 50});
    EXPECT_TRUE(r.satisfied());
    EXPECT_EQ(r.step, 1.0);   // 64, 16, 4 fail; 1 lands on 0
    EXPECT_EQ(r.evaluations, 4u);
    EXPECT_EQ(r.value, 0.0);
}

TEST(OptimLineSearchArmijo, StepsBackOutOfARegionWhereFIsNotANumber)
{
    const auto f = [](cspan v) { return v[0] < 0 ? std::numeric_limits<double>::quiet_NaN() : (v[0] - 0.5) * (v[0] - 0.5); };
    const std::vector<double> x{1}, d{-1};
    std::vector<double> t(1);
    const auto r = op::backtracking_armijo(f, x, d, 0.25, -1.0, t, {4, 1e-4, 0.5, 50});
    EXPECT_TRUE(r.satisfied());
    EXPECT_FALSE(std::isnan(r.value));
    EXPECT_EQ(r.step, 0.5);
}

TEST(OptimLineSearchArmijo, ReportsANonDescentDirectionAndTheEvaluationLimit)
{
    const auto f = [](cspan v) { return v[0]; };
    const std::vector<double> x{1}, d{1};
    std::vector<double> t{42};
    auto r = op::backtracking_armijo(f, x, d, 1.0, 1.0, t);
    EXPECT_EQ(r.status, op::line_search_status::not_descent_direction);
    EXPECT_EQ(r.evaluations, 0u);
    EXPECT_EQ(r.step, 0.0);
    EXPECT_EQ(t[0], 42.0);
    r = op::backtracking_armijo(f, x, d, 1.0, std::numeric_limits<double>::quiet_NaN(), t);
    EXPECT_EQ(r.status, op::line_search_status::not_descent_direction);
    // a wrong derivative (claimed descent, f rises along d): every step fails
    r = op::backtracking_armijo(f, x, d, 1.0, -1.0, t, {1, 1e-4, 0.5, 6});
    EXPECT_EQ(r.status, op::line_search_status::evaluation_limit);
    EXPECT_EQ(r.evaluations, 6u);
    EXPECT_EQ(r.step, 1.0 / 32);
    EXPECT_EQ(t[0], 1 + 1.0 / 32);
}

TEST(OptimLineSearchWolfe, FindsTheExactStepOnAQuadraticInTwoEvaluations)
{
    // φ(α) = ½ Σ cᵢ (xᵢ + α dᵢ)²: α = 1 overshoots, the cubic through both ends is φ itself, so its minimiser (21/73,
    // inside the safeguarded part of the bracket) is exact
    const quadratic f{{1, 2, 4}};
    const std::vector<double> x{1, 1, 1}, d{-1, -2, -4};
    std::vector<double> t(3), tg(3), g(3);
    const double fx = f(cspan(x), std::span<double>(g));
    const double gd = dot(g, d);
    const auto r = op::strong_wolfe(f, x, d, fx, gd, t, tg, {1, 1e-4, 1e-6, 2, 1e10, 40});
    EXPECT_TRUE(r.satisfied());
    EXPECT_EQ(r.evaluations, 2u);
    const double exact = -gd / (1 + 8 + 64);   // -gᵀd / dᵀCd
    EXPECT_NEAR(r.step, exact, 1e-12 * exact);
    EXPECT_LE(std::abs(r.derivative), 1e-6 * std::abs(gd));
}

TEST(OptimLineSearchWolfe, MeetsBothConditionsOnRosenbrockAlongManyDirections)
{
    st::wyrand gen(17);
    for (int trial = 0; trial < 200; ++trial) {
        const std::vector<double> x{3 * st::uniform_real(gen) - 1.5, 3 * st::uniform_real(gen) - 1};
        std::vector<double> g(2), t(2), tg(2), d(2);
        const double fx = rosenbrock(x, g);
        // -∇f, scaled anywhere from 1e-4 to 1e2, rotated by up to 80 degrees
        const double angle = (st::uniform_real(gen) - 0.5) * 2.8, scale = std::pow(10.0, -4 + 6 * st::uniform_real(gen));
        d[0] = -scale * (std::cos(angle) * g[0] - std::sin(angle) * g[1]);
        d[1] = -scale * (std::sin(angle) * g[0] + std::cos(angle) * g[1]);
        const double gd = dot(g, d);
        if (!(gd < 0))
            continue;
        const op::wolfe_options<double> o{};
        const auto r = op::strong_wolfe(rosenbrock, x, d, fx, gd, t, tg, o);
        ASSERT_TRUE(r.satisfied()) << trial << " status " << int(r.status);
        EXPECT_LE(r.value, fx + o.c1 * r.step * gd) << trial;
        EXPECT_LE(std::abs(r.derivative), -o.c2 * gd) << trial;
        // the trial storage holds the returned point and its gradient
        std::vector<double> g2(2);
        for (std::size_t i = 0; i < 2; ++i)
            EXPECT_EQ(t[i], x[i] + r.step * d[i]);
        EXPECT_EQ(rosenbrock(t, g2), r.value);
        EXPECT_EQ(g2, tg);
        EXPECT_EQ(dot(tg, d), r.derivative);
    }
}

TEST(OptimLineSearchWolfe, ExpandsAShortFirstStepAndZoomsALongOne)
{
    const quadratic f{{2}};
    const std::vector<double> x{10}, d{-1};
    std::vector<double> t(1), tg(1);
    // exact step 10; a first step of 1e-3 must grow by doubling until it brackets
    auto r = op::strong_wolfe(f, x, d, 100.0, -20.0, t, tg, {1e-3, 1e-4, 0.1, 2, 1e10, 40});
    EXPECT_TRUE(r.satisfied());
    EXPECT_NEAR(r.step, 10, 1.0);
    EXPECT_GT(r.evaluations, 10u);
    // a first step of 1e4 overshoots and is zoomed back
    r = op::strong_wolfe(f, x, d, 100.0, -20.0, t, tg, {1e4, 1e-4, 0.1, 2, 1e10, 40});
    EXPECT_TRUE(r.satisfied());
    EXPECT_NEAR(r.step, 10, 1.0);
}

TEST(OptimLineSearchWolfe, ReportsAStepAtTheMaximumOnAnUnboundedFunction)
{
    const auto fg = [](cspan x, std::span<double> g) {
        g[0] = -1;
        return -x[0];
    };
    const std::vector<double> x{0}, d{1};
    std::vector<double> t(1), tg(1);
    const auto r = op::strong_wolfe(fg, x, d, 0.0, -1.0, t, tg, {1, 1e-4, 0.9, 10, 1e6, 40});
    EXPECT_EQ(r.status, op::line_search_status::step_at_maximum);
    EXPECT_EQ(r.step, 1e6);
    EXPECT_EQ(t[0], 1e6);
}

TEST(OptimLineSearchWolfe, ReportsTheEvaluationLimitWithTheBestPointInTrial)
{
    // a curvature condition no step can meet (c₂ tiny, φ' never small on a kinked |α - 1| + α/2): the limit, then
    // the best point with sufficient decrease
    const auto fg = [](cspan x, std::span<double> g) {
        const double a = x[0] - 1;
        g[0] = (a > 0 ? 1.0 : -1.0) + 0.5;
        return std::abs(a) + 0.5 * x[0];
    };
    const std::vector<double> x{0}, d{1};
    std::vector<double> t(1), tg(1);
    const auto r = op::strong_wolfe(fg, x, d, 1.0, -0.5, t, tg, {0.3, 1e-4, 1e-3, 2, 1e10, 8});
    EXPECT_NE(r.status, op::line_search_status::satisfied);
    EXPECT_LE(r.evaluations, 9u);
    EXPECT_GT(r.step, 0.0);
    EXPECT_LE(r.value, 1.0 + 1e-4 * r.step * -0.5);
    EXPECT_EQ(t[0], r.step);
    std::vector<double> g2(1);
    EXPECT_EQ(fg(t, g2), r.value);
    // a non-descent direction evaluates nothing
    const auto nd = op::strong_wolfe(fg, x, d, 1.0, 0.5, t, tg);
    EXPECT_EQ(nd.status, op::line_search_status::not_descent_direction);
    EXPECT_EQ(nd.evaluations, 0u);
}

TEST(OptimLineSearchWolfe, ShortensAStepIntoARegionWhereFIsNotANumber)
{
    const auto fg = [](cspan x, std::span<double> g) {
        if (x[0] > 2) {
            g[0] = std::numeric_limits<double>::quiet_NaN();
            return std::numeric_limits<double>::quiet_NaN();
        }
        g[0] = 2 * (x[0] - 1);
        return (x[0] - 1) * (x[0] - 1);
    };
    const std::vector<double> x{0}, d{1};
    std::vector<double> t(1), tg(1);
    const auto r = op::strong_wolfe(fg, x, d, 1.0, -2.0, t, tg, {16, 1e-4, 0.5, 2, 1e10, 40});
    EXPECT_TRUE(r.satisfied());
    EXPECT_LE(r.step, 2.0);
    EXPECT_FALSE(std::isnan(r.value));
}

TEST(OptimLineSearchWolfe, WorksInFloat)
{
    const auto fg = [](std::span<const float> x, std::span<float> g) {
        g[0] = 2 * (x[0] - 3);
        return (x[0] - 3) * (x[0] - 3);
    };
    const std::vector<float> x{0}, d{1};
    std::vector<float> t(1), tg(1);
    const auto r = op::strong_wolfe(fg, x, d, 9.0f, -6.0f, t, tg);
    EXPECT_TRUE(r.satisfied());
}

} // namespace
