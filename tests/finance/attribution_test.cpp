//
//  attribution_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/attribution.hpp>: each greek's term, the factor split equals
//  the exact Shapley attribution of the Taylor model, a Black-Scholes call explained, and risk
//  contributions equal to the Shapley attribution of variance.
//

#include <snicholls/core/finance/attribution.hpp>

#include <snicholls/core/game/change.hpp>
#include <snicholls/core/game/shapley.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace {

namespace finance = snicholls::finance;
namespace game = snicholls::game;

constexpr finance::greeks<> straddle{.delta = 0.1, .gamma = 0.05, .vega = 30, .vanna = 2, .volga = 40, .theta = -10, .rho = 1};
constexpr finance::market_move<> move{.spot = 4, .vol = 0.02, .time = 0.5, .rate = 0.001};
constexpr auto pnl = finance::explain(straddle, move, 5.0);

static_assert(pnl.delta == 0.1 * 4 && pnl.gamma == 0.5 * 0.05 * 16 && pnl.theta == -5);
static_assert(straddle.long_volatility() && !(straddle * -1.0).long_volatility());
static_assert((straddle + straddle).vega == 60);
static_assert(std::is_trivially_copyable_v<finance::greeks<>> && sizeof(finance::greeks<>) == 7 * sizeof(double));

TEST(GreeksExplain, EveryTermAndTheResidual)
{
    EXPECT_DOUBLE_EQ(pnl.vega, 30 * 0.02);
    EXPECT_DOUBLE_EQ(pnl.volga, 0.5 * 40 * 0.02 * 0.02);
    EXPECT_DOUBLE_EQ(pnl.vanna, 2 * 4 * 0.02);
    EXPECT_DOUBLE_EQ(pnl.rho, 0.001);
    EXPECT_NEAR(pnl.explained, pnl.delta + pnl.gamma + pnl.vega + pnl.vanna + pnl.volga + pnl.theta + pnl.rho, 1e-15);
    EXPECT_NEAR(pnl.unexplained, 5.0 - pnl.explained, 1e-15);
    EXPECT_NEAR(pnl.spot, pnl.delta + pnl.gamma + pnl.vanna / 2, 1e-15);
    EXPECT_NEAR(pnl.volatility, pnl.vega + pnl.volga + pnl.vanna / 2, 1e-15);
    EXPECT_NEAR(pnl.spot + pnl.volatility + pnl.time + pnl.rates, pnl.explained, 1e-14);
}

TEST(GreeksExplain, TheFactorSplitIsTheExactShapleyAttributionOfTheTaylorModel)
{
    const auto model = finance::as_taylor_model(straddle);
    const auto taylor = [&model](const std::array<double, 4>& x) {
        double v = model.carry;
        for (std::size_t i = 0; i < 4; ++i) {
            v += model.gradient[i] * x[i];
            for (std::size_t j = 0; j < 4; ++j)
                v += 0.5 * model.hessian[i][j] * x[i] * x[j];
        }
        return v;
    };
    const std::array<double, 4> moves{move.spot, move.vol, move.time, move.rate};
    const auto shapley = game::attribute_change<4>(taylor, std::array<double, 4>{}, moves);
    EXPECT_NEAR(shapley.values[0], pnl.spot, 1e-12);
    EXPECT_NEAR(shapley.values[1], pnl.volatility, 1e-12);
    EXPECT_NEAR(shapley.values[2], pnl.time, 1e-12);
    EXPECT_NEAR(shapley.values[3], pnl.rates, 1e-12);
}

// Black-Scholes, to see the explain against a real revaluation
double normal_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
double call(double s, double k, double vol, double t, double r)
{
    const double d1 = (std::log(s / k) + (r + vol * vol / 2) * t) / (vol * std::sqrt(t));
    return s * normal_cdf(d1) - k * std::exp(-r * t) * normal_cdf(d1 - vol * std::sqrt(t));
}

TEST(GreeksExplain, ABlackScholesCallIsExplainedUpToTheThirdOrderTerms)
{
    const double s = 100, k = 100, vol = 0.2, t = 0.5, r = 0.01, h = 1e-3;
    const auto price = [&](double ds, double dv, double dt, double dr) { return call(s + ds, k, vol + dv, t - dt, r + dr); };
    finance::greeks<> g;   // by central differences
    g.delta = (price(h, 0, 0, 0) - price(-h, 0, 0, 0)) / (2 * h);
    g.gamma = (price(h, 0, 0, 0) - 2 * price(0, 0, 0, 0) + price(-h, 0, 0, 0)) / (h * h);
    const double hv = 1e-4;
    g.vega = (price(0, hv, 0, 0) - price(0, -hv, 0, 0)) / (2 * hv);
    g.volga = (price(0, hv, 0, 0) - 2 * price(0, 0, 0, 0) + price(0, -hv, 0, 0)) / (hv * hv);
    g.vanna = (price(h, hv, 0, 0) - price(h, -hv, 0, 0) - price(-h, hv, 0, 0) + price(-h, -hv, 0, 0)) / (4 * h * hv);
    g.theta = (price(0, 0, 1e-5, 0) - price(0, 0, -1e-5, 0)) / 2e-5;
    g.rho = (price(0, 0, 0, 1e-5) - price(0, 0, 0, -1e-5)) / 2e-5;
    const finance::market_move<> day{.spot = 1.5, .vol = 0.005, .time = 1.0 / 365, .rate = 0};
    const double actual = price(day.spot, day.vol, day.time, day.rate) - price(0, 0, 0, 0);
    const auto e = finance::explain(g, day, actual);
    // Second order leaves under 0.5% unexplained, and a tenth of what the first-order terms alone leave
    const double first_order_residual = actual - (e.delta + e.vega + e.theta + e.rho);
    EXPECT_LT(std::abs(e.unexplained), 5e-3 * std::abs(actual));
    EXPECT_LT(std::abs(e.unexplained), 0.1 * std::abs(first_order_residual));
    EXPECT_GT(e.delta, 0);
    EXPECT_GT(e.gamma, 0);   // long gamma
    EXPECT_TRUE(g.long_volatility());
    EXPECT_LT(e.theta, 0);   // time decay
}

TEST(RiskContributions, AreTheShapleyAttributionOfVarianceAndSumToTheTotal)
{
    const std::array<std::array<double, 3>, 3> cov{{{0.04, 0.006, -0.002}, {0.006, 0.09, 0.01}, {-0.002, 0.01, 0.0225}}};
    const std::array<double, 3> w{0.5, 0.3, 0.2};
    const auto rc = finance::risk_contributions(cov, w);
    const auto variance_of = [&](game::coalition s) {
        double v = 0;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                if (game::contains(s, i) && game::contains(s, j))
                    v += w[i] * cov[i][j] * w[j];
        return v;
    };
    const auto shapley = game::shapley<3>(variance_of);
    double to_vol = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(rc.to_variance[i], shapley.values[i], 1e-15);
        to_vol += rc.to_volatility[i];
    }
    EXPECT_NEAR(rc.variance, shapley.total, 1e-15);
    EXPECT_NEAR(to_vol, rc.volatility, 1e-15);
    EXPECT_EQ(finance::risk_contributions(cov, std::array<double, 3>{}).to_volatility[0], 0.0);
}

} // namespace
