//
//  iterate_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/iterate.hpp>: a compounding schedule, a running P&L and an implied
//  yield found by iteration, at compile time. Exits non-zero if a value is not the one documented.
//

#include <snicholls/core/iterate.hpp>

#include <array>
#include <cstdio>

int main()
{
    std::printf("snicholls-core iterate demo\n");
    bool ok = true;

    //! [iterate]
    // NestList: a balance compounding at 5% a year for five years.
    constexpr auto balances = snicholls::nest_list<5>([](double b) { return b * 1.05; }, 1000.0);

    // FoldList: the running P&L of a week's trades.
    constexpr auto running = snicholls::fold_list([](double total, double pnl) { return total + pnl; }, 0.0,
                                                  std::array{120.0, -40.0, 15.0, 60.0, -5.0});

    // FixedPoint: the yield y with 100 = 105 / (1 + y), by iterating y = 105 / 100 - 1 to a fixed point.
    constexpr auto yield = snicholls::fixed_point([](double y) { return 0.5 * y + 0.5 * (105.0 / 100.0 - 1.0); }, 0.0, 200);
    static_assert(yield.converged);
    //! [iterate]

    std::printf("  after 5 years %.2f, week's P&L %.0f, yield %.4f in %zu steps\n", balances[5], running[5], yield.value, yield.iterations);
    ok &= running[5] == 150.0 && yield.value > 0.0499 && yield.value < 0.0501;

    std::printf(ok ? "iterate demo: ok\n" : "iterate demo: FAILED\n");
    return ok ? 0 : 1;
}
