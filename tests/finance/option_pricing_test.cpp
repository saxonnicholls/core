//
//  option_pricing_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/option_pricing.hpp>: Hull's worked examples (Black-Scholes, an index option
//  with a dividend yield, the greeks of a 20-week call, the five-step American put tree), put-call parity, the greeks
//  against central differences of the price, the tree's convergence to the closed form, American against European,
//  and float. Most of it is proved by the compiler.
//

#include <snicholls/core/finance/option_pricing.hpp>

#include <snicholls/core/math/constexpr_math.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace {

namespace finance = snicholls::finance;
namespace m = snicholls::math;
using finance::exercise_style;
using finance::option_kind;
using option = finance::vanilla_option<>;

// ---- Hull, Options, Futures and Other Derivatives ------------------------------------------------------------------
// Black-Scholes: S 42, K 40, r 10%, sigma 20%, six months: call 4.76, put 0.81
constexpr option hull_call{.kind = option_kind::call, .spot = 42, .strike = 40, .expiry = 0.5, .rate = 0.10, .vol = 0.20};
constexpr option hull_put{.kind = option_kind::put, .spot = 42, .strike = 40, .expiry = 0.5, .rate = 0.10, .vol = 0.20};
static_assert(m::near(finance::black_scholes(hull_call), 4.759422392871535, 1e-12));
static_assert(m::near(finance::black_scholes(hull_put), 0.8085993729000958, 1e-12));

// An index option with a dividend yield: S 930, K 900, r 8%, q 3%, sigma 20%, two months: 51.83
constexpr option index_call{.spot = 930, .strike = 900, .expiry = 2.0 / 12, .rate = 0.08, .yield = 0.03, .vol = 0.20};
static_assert(m::near(finance::black_scholes(index_call), 51.83295679649086, 1e-11));

// The greeks of a 20-week call: S 49, K 50, r 5%, sigma 20%: delta 0.522, gamma 0.066, vega 12.1, theta -4.31, rho 8.91
constexpr option twenty_weeks{.spot = 49, .strike = 50, .expiry = 0.3846, .rate = 0.05, .vol = 0.20};
constexpr auto hull_greeks = finance::black_scholes_greeks(twenty_weeks);
static_assert(m::near(finance::black_scholes(twenty_weeks), 2.4004610869656666, 1e-12));
static_assert(m::near(hull_greeks.delta, 0.5216016339715761, 1e-12) && m::near(hull_greeks.gamma, 0.06554537725247868, 1e-12));
static_assert(m::near(hull_greeks.vega, 12.105242754243845, 1e-11) && m::near(hull_greeks.theta, -4.305389964546105, 1e-11));
static_assert(m::near(hull_greeks.rho, 8.906574098800945, 1e-11));

// The five-step CRR tree for an American put: S 50, K 50, r 10%, sigma 40%, five months: 4.49
constexpr option five_months{.kind = option_kind::put, .spot = 50, .strike = 50, .expiry = 5.0 / 12, .rate = 0.10, .vol = 0.40};
constexpr auto hull_tree = finance::binomial<5>(five_months);
static_assert(m::near(hull_tree.value, 4.488458534725914, 1e-12));

// ---- Put-call parity: c - p = S e^{-qT} - K e^{-rT} ----------------------------------------------------------------
template <typename T>
constexpr T parity_gap(finance::vanilla_option<T> o)
{
    o.kind = option_kind::call;
    const T call = finance::black_scholes(o);
    o.kind = option_kind::put;
    return call - finance::black_scholes(o) - (o.spot * m::exp(-o.yield * o.expiry) - o.strike * m::exp(-o.rate * o.expiry));
}
static_assert(m::abs(parity_gap(hull_call)) < 1e-13 && m::abs(parity_gap(index_call)) < 1e-12);
static_assert(m::abs(parity_gap(option{.spot = 80, .strike = 120, .expiry = 3, .rate = 0.02, .yield = 0.05, .vol = 0.6})) < 1e-12);

// The greeks of a call and a put with the same inputs differ only by the forward's: gamma and vega equal
constexpr auto index_put_greeks = finance::black_scholes_greeks(option{.kind = option_kind::put, .spot = 930, .strike = 900, .expiry = 2.0 / 12,
                                                                       .rate = 0.08, .yield = 0.03, .vol = 0.20});
constexpr auto index_call_greeks = finance::black_scholes_greeks(index_call);
static_assert(m::near(index_call_greeks.delta - index_put_greeks.delta, m::exp(-0.03 * 2.0 / 12), 1e-14));
static_assert(index_call_greeks.gamma == index_put_greeks.gamma && index_call_greeks.vega == index_put_greeks.vega);

// ---- The tree: convergence to the closed form, and early exercise ----------------------------------------------------
constexpr option atm_put{.kind = option_kind::put, .spot = 100, .strike = 100, .expiry = 1, .rate = 0.05, .vol = 0.20};
constexpr double atm_put_bs = 5.573526022256971;
static_assert(m::near(finance::black_scholes(atm_put), atm_put_bs, 1e-12));
static_assert(m::abs(finance::binomial<50>(atm_put, exercise_style::european).value - atm_put_bs) < 0.05);
static_assert(m::abs(finance::binomial<200>(atm_put).value - 6.086382749916101) < 1e-10);   // the American premium, about 6.09

// The layout: a value, delta, gamma and one boundary per step; trivially copyable, nothing else
static_assert(std::is_trivially_copyable_v<finance::binomial_price<64>> && sizeof(finance::binomial_price<64>) == (3 + 64) * sizeof(double));
static_assert(std::is_trivially_copyable_v<option> && sizeof(finance::vanilla_option<float>) == 7 * sizeof(float));

TEST(OptionPricingBlackScholes, TheGreeksAreTheCentralDifferencesOfThePrice)
{
    const option cases[] = {twenty_weeks, index_call, hull_put, atm_put,
                            option{.kind = option_kind::put, .spot = 80, .strike = 120, .expiry = 3, .rate = 0.02, .yield = 0.05, .vol = 0.6}};
    for (const option& o : cases) {
        const auto g = finance::black_scholes_greeks(o);
        const auto price = [&](double ds, double dvol, double dt, double dr) {
            option b = o;
            b.spot += ds;
            b.vol += dvol;
            b.expiry -= dt;   // calendar time passing
            b.rate += dr;
            return finance::black_scholes(b);
        };
        const double h = 1e-4 * o.spot, hv = 1e-4, ht = 1e-5, hr = 1e-5;
        EXPECT_NEAR(g.delta, (price(h, 0, 0, 0) - price(-h, 0, 0, 0)) / (2 * h), 1e-7);
        EXPECT_NEAR(g.gamma, (price(h, 0, 0, 0) - 2 * price(0, 0, 0, 0) + price(-h, 0, 0, 0)) / (h * h), 1e-5);
        EXPECT_NEAR(g.vega, (price(0, hv, 0, 0) - price(0, -hv, 0, 0)) / (2 * hv), 1e-5 * std::abs(g.vega) + 1e-7);
        EXPECT_NEAR(g.vanna, (price(h, hv, 0, 0) - price(h, -hv, 0, 0) - price(-h, hv, 0, 0) + price(-h, -hv, 0, 0)) / (4 * h * hv), 1e-4);
        EXPECT_NEAR(g.volga, (price(0, hv, 0, 0) - 2 * price(0, 0, 0, 0) + price(0, -hv, 0, 0)) / (hv * hv), 1e-3 * std::abs(g.volga) + 1e-3);
        EXPECT_NEAR(g.theta, (price(0, 0, ht, 0) - price(0, 0, -ht, 0)) / (2 * ht), 1e-5 * std::abs(g.theta) + 1e-6);
        EXPECT_NEAR(g.rho, (price(0, 0, 0, hr) - price(0, 0, 0, -hr)) / (2 * hr), 1e-5 * std::abs(g.rho) + 1e-6);
    }
}

TEST(OptionPricingBinomial, AEuropeanTreeConvergesToBlackScholes)
{
    const double e50 = std::abs(finance::binomial<50>(atm_put, exercise_style::european).value - atm_put_bs);
    const double e500 = std::abs(finance::binomial<500>(atm_put, exercise_style::european).value - atm_put_bs);
    EXPECT_LT(e500, 0.005);
    EXPECT_LT(e500 * 5, e50);   // the error falls as 1/steps
    const auto call = finance::binomial<500>(index_call, exercise_style::european);
    EXPECT_NEAR(call.value, finance::black_scholes(index_call), 0.05);
    const auto g = finance::black_scholes_greeks(index_call);
    EXPECT_NEAR(call.delta, g.delta, 0.01);
    EXPECT_NEAR(call.gamma, g.gamma, 0.001);
}

TEST(OptionPricingBinomial, EarlyExerciseIsWorthSomethingOnlyWhereTheoryHasIt)
{
    // A call on a stock paying nothing is never exercised early: American and European agree to the last bit
    constexpr option call{.spot = 100, .strike = 95, .expiry = 1, .rate = 0.05, .vol = 0.25};
    constexpr auto american = finance::binomial<100>(call), european = finance::binomial<100>(call, exercise_style::european);
    static_assert(american.value == european.value);
    for (const double b : american.boundary)
        EXPECT_EQ(b, 0.0);
    // With a dividend yield it is, from the top of the tree down
    option paying = call;
    paying.yield = 0.08;
    const auto early = finance::binomial<100>(paying), late = finance::binomial<100>(paying, exercise_style::european);
    EXPECT_GT(early.value, late.value);
    EXPECT_GT(early.boundary[99], paying.strike);
    // A put's boundary is below the strike and rises towards expiry
    constexpr auto put = finance::binomial<200>(atm_put);
    EXPECT_EQ(put.boundary[0], 0.0);
    EXPECT_GT(put.boundary[199], put.boundary[100]);
    EXPECT_LT(put.boundary[199], atm_put.strike);
    EXPECT_GT(put.value, finance::binomial<200>(atm_put, exercise_style::european).value);
}

TEST(OptionPricingTypes, FloatAndLongDoubleAgreeWithDouble)
{
    constexpr finance::vanilla_option<float> f{.kind = option_kind::put, .spot = 42, .strike = 40, .expiry = 0.5f, .rate = 0.10f, .vol = 0.20f};
    constexpr finance::vanilla_option<long double> l{.kind = option_kind::put, .spot = 42, .strike = 40, .expiry = 0.5L, .rate = 0.10L, .vol = 0.20L};
    static_assert(std::is_same_v<decltype(finance::black_scholes(f)), float>);
    EXPECT_NEAR(finance::black_scholes(f), 0.8085993729000958, 1e-5);
    EXPECT_NEAR(static_cast<double>(finance::black_scholes(l)), 0.8085993729000958, 1e-12);
    EXPECT_NEAR(finance::black_scholes_greeks(f).delta, finance::black_scholes_greeks(hull_put).delta, 1e-5);
    EXPECT_NEAR(finance::binomial<5>(finance::vanilla_option<float>{.kind = option_kind::put, .spot = 50, .strike = 50, .expiry = 5.0f / 12,
                                                                      .rate = 0.10f, .vol = 0.40f}).value,
                4.488458534725914, 1e-4);
    EXPECT_EQ(f.intrinsic(30), 10.0f);
    EXPECT_EQ(hull_call.intrinsic(30), 0.0);
}

} // namespace
