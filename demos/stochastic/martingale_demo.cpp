//
//  martingale_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stochastic/martingale.hpp>: is a spread mean-reverting? Its Doob
//  decomposition under an AR(1) forecast, a strategy's P&L as a martingale transform, and the two
//  tests. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/stochastic/martingale.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core martingale demo\n");
    namespace st = snicholls::stochastic;
    bool ok = true;

    // A mean-reverting spread: each day it closes 20% of the gap to zero, plus noise
    std::vector<double> spread{2.0};
    std::uint64_t x = 7;
    for (int t = 0; t < 500; ++t) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        spread.push_back(0.8 * spread.back() + (static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5) * 0.4);
    }

    //! [martingale]
    // The model's forecast of each next value; the Doob decomposition splits the path into what it
    // predicted (the compensator: the reversion a strategy can harvest) and what it could not.
    std::vector<double> forecast;
    for (std::size_t t = 0; t + 1 < spread.size(); ++t)
        forecast.push_back(0.8 * spread[t]);
    const st::doob_parts parts = st::doob_decomposition(spread, forecast);

    // Fade the spread: hold minus its last value. The P&L is the martingale transform.
    std::vector<double> holding{0.0};
    for (std::size_t t = 0; t + 1 < spread.size(); ++t)
        holding.push_back(-spread[t]);
    const std::vector<double> pnl = st::martingale_transform(holding, spread);

    // Is the spread a martingale? Its increments have memory, so no.
    std::vector<double> moves;
    for (std::size_t t = 1; t < spread.size(); ++t)
        moves.push_back(spread[t] - spread[t - 1]);
    const st::test_result memory = st::ljung_box(moves, 5);
    //! [martingale]

    std::printf("  compensator %.2f, martingale part %.2f, fade P&L %.2f, Ljung-Box p %.2g\n", parts.compensator.back(),
                parts.martingale.back(), pnl.back(), memory.p_value);
    ok &= pnl.back() > 0 && memory.p_value < 0.01;

    std::printf(ok ? "martingale demo: ok\n" : "martingale demo: FAILED\n");
    return ok ? 0 : 1;
}
