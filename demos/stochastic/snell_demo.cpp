//
//  snell_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stochastic/snell.hpp>: when to close a mean-reverting position on a
//  five-state chain, with the stopping table computed by the compiler. Exits non-zero if the
//  table is not the one documented; then an American put on a recombining binomial lattice, in O(steps) state.
//

#include <snicholls/core/stochastic/snell.hpp>

#include <snicholls/core/math/constexpr_math.hpp>

#include <array>
#include <cstddef>
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

    //! [lattice]
    // A recombining lattice: node j of step t moves to j (down) or j + 1 (up), the same stencil at every node, so one
    // row of storage carries the whole backward induction. A 64-step American put, S = K = 100, 4% a year, 20% vol.
    constexpr std::size_t steps = 64;
    constexpr double dt = 1.0 / steps, up = snicholls::math::exp(0.2 * snicholls::math::sqrt(dt));
    constexpr double p = (snicholls::math::exp(0.04 * dt) - 1 / up) / (up - 1 / up);
    constexpr auto put = [](std::size_t t, std::size_t j) {
        const double s = 100 * snicholls::math::pow(up, 2 * static_cast<double>(j) - static_cast<double>(t));
        return s < 100 ? 100 - s : 0.0;
    };
    constexpr auto tree = st::lattice_envelope<steps>(std::array{1 - p, p}, put, snicholls::math::exp(-0.04 * dt));

    // Computed by the compiler: worth more than exercising now, and exercised early near expiry from the bottom node up
    static_assert(tree.value > put(0, 0) && !tree.exercises(0) && tree.exercises(steps - 1) && tree.first_exercise[steps - 1] == 0);
    //! [lattice]

    std::printf("  64-step American put: %.4f; exercised at step 63 up to node %zu of 63\n", tree.value, tree.last_exercise[steps - 1]);
    ok &= tree.value > 6.3 && tree.value < 6.5 && tree.last_exercise[steps - 1] < steps / 2;

    std::printf(ok ? "snell demo: ok\n" : "snell demo: FAILED\n");
    return ok ? 0 : 1;
}
