//
//  owen_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/game/owen.hpp>: six factors in three groups, where two substitutes in one group would be short-changed by plain Shapley. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/game/owen.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

int main()
{
    std::printf("snicholls-core owen demo\n");
    namespace game = snicholls::game;
    bool ok = true;

    //! [owen]
    // Factors 0 and 1 are two duration measures that hedge the same risk: either earns 4, both still 4.
    // Factors 2..3 are credit, 4..5 equity, each adding its own amount.
    const auto pnl = [](game::coalition s) {
        const double duration = (game::contains(s, 0) || game::contains(s, 1)) ? 4.0 : 0.0;
        return duration + 1.0 * game::contains(s, 2) + 2.0 * game::contains(s, 3) + 0.5 * game::contains(s, 4) + 0.25 * game::contains(s, 5);
    };
    constexpr std::array<game::coalition, 3> groups{0b000011, 0b001100, 0b110000};   // rates, credit, equity

    const auto owen = game::owen<6>(pnl, groups);
    const auto by_group = game::shapley<3>(game::quotient_game(pnl, groups));
    //! [owen]

    std::printf("  rates %.2f (%.2f + %.2f)  credit %.2f  equity %.2f\n", by_group.values[0], owen.values[0], owen.values[1], by_group.values[1],
                by_group.values[2]);
    ok &= owen.values[0] == 2.0 && owen.values[1] == 2.0 && by_group.values[0] == 4.0;
    ok &= std::abs(owen.unattributed()) < 1e-12;

    std::printf(ok ? "owen demo: ok\n" : "owen demo: FAILED\n");
    return ok ? 0 : 1;
}
