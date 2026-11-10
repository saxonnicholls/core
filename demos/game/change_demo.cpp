//
//  change_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/game/change.hpp>: a book revalued after the market moved, its change split exactly by the factors that moved. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/game/change.hpp>

#include <array>
#include <cmath>
#include <cstdio>

int main()
{
    std::printf("snicholls-core change demo\n");
    namespace game = snicholls::game;
    bool ok = true;

    //! [change]
    // A book worth f(spot, vol, rate): long a quantity of spot, long volatility, short rates, with a cross effect.
    const auto book = [](const std::array<double, 3>& m) { return 100 * m[0] + 5000 * m[1] * m[0] / 100 - 2000 * m[2]; };
    const std::array<double, 3> yesterday{100.0, 0.20, 0.050};
    const std::array<double, 3> today{102.0, 0.22, 0.051};

    // 2^3 revaluations, every cross effect shared fairly, nothing left over.
    const auto why = game::attribute_change<3>(book, yesterday, today);
    //! [change]

    std::printf("  P&L %+.2f: spot %+.2f, vol %+.2f, rates %+.2f\n", why.total - why.baseline, why.values[0], why.values[1], why.values[2]);
    ok &= std::abs(why.unattributed()) < 1e-9;

    std::printf(ok ? "change demo: ok\n" : "change demo: FAILED\n");
    return ok ? 0 : 1;
}
