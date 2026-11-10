//
//  coalition_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/game/coalition.hpp>: coalitions as bit strings, a game as a table of 2^N values built at compile time, and the enumeration orders. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/game/coalition.hpp>

#include <cstdio>
#include <string>

int main()
{
    std::printf("snicholls-core coalition demo\n");
    namespace game = snicholls::game;
    bool ok = true;

    //! [coalitions]
    // Three factors: rates, credit, equity. A coalition is a bit string, so the eight are 0 .. 7.
    constexpr game::coalition rates_and_equity = game::singleton(0) | game::singleton(2);
    static_assert(game::coalition_size(rates_and_equity) == 2 && !game::contains(rates_and_equity, 1));

    // A game maps 2^N to a value; this one is known at compile time, so its table is too.
    constexpr auto pnl = [](game::coalition s) { return 1.0 * game::contains(s, 0) + 2.0 * game::contains(s, 1) + 0.5 * game::coalition_size(s); };
    constexpr auto table = game::make_table<3>(pnl);
    static_assert(table[0b111] == 1.0 + 2.0 + 1.5);

    // Gray-code order changes one factor per step: revalue by moving one input at a time.
    std::string path;
    game::for_each_coalition_gray(3, [&path](game::coalition s, std::size_t) { path += std::to_string(s) + ' '; });
    //! [coalitions]

    std::printf("  gray order %s\n", path.c_str());
    ok &= path == "0 1 3 2 6 7 5 4 ";
    std::size_t pairs = 0;
    game::for_each_coalition_of_size(5, 2, [&pairs](game::coalition) { ++pairs; });
    ok &= pairs == game::binomial(5, 2);

    std::printf(ok ? "coalition demo: ok\n" : "coalition demo: FAILED\n");
    return ok ? 0 : 1;
}
