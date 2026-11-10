//
//  snell_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stochastic/snell.hpp>: when to close a mean-reverting position on a
//  five-state chain, with the stopping table computed by the compiler. Exits non-zero if the
//  table is not the one documented.
//

#include <snicholls/core/stochastic/snell.hpp>

#include <array>
#include <cstdio>

int main()
{
    std::printf("snicholls-core snell demo\n");
    namespace st = snicholls::stochastic;
    bool ok = true;

    //! [snell]
    // A spread in five states, -2 .. 2, drifting back towards 0. Closing in state s pays s; ten days to go.
    constexpr std::array<std::array<double, 5>, 5> drift_to_zero{{{0.5, 0.5, 0, 0, 0},
                                                                  {0.1, 0.4, 0.5, 0, 0},
                                                                  {0, 0.25, 0.5, 0.25, 0},
                                                                  {0, 0, 0.5, 0.4, 0.1},
                                                                  {0, 0, 0, 0.5, 0.5}}};
    constexpr auto close_now = [](std::size_t, std::size_t s) { return static_cast<double>(s) - 2.0; };
    constexpr auto plan = st::snell_envelope<5, 10>(drift_to_zero, close_now);

    // Computed by the compiler: at the top, take the profit; anywhere below, waiting is worth more.
    static_assert(plan.stops(0, 4) && !plan.stops(0, 2));
    //! [snell]

    std::printf("  stop at day 0 in state:");
    for (std::size_t s = 0; s < 5; ++s)
        std::printf(" %d=%s", static_cast<int>(s) - 2, plan.stops(0, s) ? "close" : "hold");
    std::printf("\n  value of holding the right in state 0: %.3f\n", plan.value[0][2]);
    ok &= plan.value[0][2] > 0;

    std::printf(ok ? "snell demo: ok\n" : "snell demo: FAILED\n");
    return ok ? 0 : 1;
}
