//
//  attribution_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/attribution.hpp>: a day's P&L on a long straddle explained by the greeks and by factor, and a portfolio's risk by holding. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/finance/attribution.hpp>

#include <array>
#include <cmath>
#include <cstdio>

int main()
{
    std::printf("snicholls-core attribution demo\n");
    namespace finance = snicholls::finance;
    bool ok = true;

    //! [greeks]
    // A book of two positions in one underlying: greeks add, scaled by quantity.
    constexpr finance::greeks<> call{.delta = 0.55, .gamma = 0.04, .vega = 0.20, .vanna = 0.01, .volga = 0.30, .theta = -0.02, .rho = 0.25};
    constexpr finance::greeks<> put{.delta = -0.45, .gamma = 0.04, .vega = 0.20, .vanna = 0.01, .volga = 0.30, .theta = -0.015, .rho = -0.24};
    constexpr auto straddle = call * 100.0 + put * 100.0;

    // Spot up 2, implied volatility up 1.5 points, one day, rates flat; the P&L from full revaluation.
    constexpr finance::market_move<> today{.spot = 2.0, .vol = 1.5, .time = 1.0, .rate = 0.0};
    constexpr auto why = finance::explain(straddle, today, 162.4);
    //! [greeks]

    std::printf("  delta %+.2f gamma %+.2f vega %+.2f vanna %+.2f volga %+.2f theta %+.2f: explained %+.2f of %+.2f, %+.2f unexplained\n",
                why.delta, why.gamma, why.vega, why.vanna, why.volga, why.theta, why.explained, why.actual, why.unexplained);
    std::printf("  by factor: spot %+.2f, volatility %+.2f (%s vol), time %+.2f\n", why.spot, why.volatility,
                straddle.long_volatility() ? "long" : "short", why.time);
    ok &= std::abs(why.spot + why.volatility + why.time + why.rates - why.explained) < 1e-12;

    constexpr std::array<std::array<double, 2>, 2> cov{{{0.04, 0.01}, {0.01, 0.09}}};
    const auto risk = finance::risk_contributions(cov, std::array{0.6, 0.4});
    std::printf("  volatility %.4f: %.4f from the first holding, %.4f from the second\n", risk.volatility, risk.to_volatility[0],
                risk.to_volatility[1]);
    ok &= std::abs(risk.to_volatility[0] + risk.to_volatility[1] - risk.volatility) < 1e-15;

    std::printf(ok ? "attribution demo: ok\n" : "attribution demo: FAILED\n");
    return ok ? 0 : 1;
}
