//
//  shapley_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/game/shapley.hpp>: the tax-aware example's shape from Moehle, Boyd and Ang, attributed by every method, exactly and by sampling. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/game/shapley.hpp>

#include <cmath>
#include <cstdio>
#include <random>

int main()
{
    std::printf("snicholls-core shapley demo\n");
    namespace game = snicholls::game;
    bool ok = true;

    //! [shapley]
    // Bacon's return attribution (the paper's table 3): country allocation is feature 0, stock
    // selection feature 1, and each backtest's return is a table entry indexed by the features on.
    constexpr std::array<double, 4> returns{6.4, 5.2, 9.4, 8.3};
    const auto backtests = game::from_table(returns);

    const auto shapley = game::shapley<2>(backtests);          // fair and full
    const auto brinson = game::one_at_a_time<2>(backtests);    // leaves 0.1 unattributed
    const auto off_the_top = game::sequential<2>(backtests);   // full, but favours feature 0

    // Ten features and an expensive backtest: sample orders, and see the error of each estimate.
    std::mt19937_64 rng(2021);
    const auto quadratic = [](game::coalition s) { return static_cast<double>(game::coalition_size(s) * game::coalition_size(s)); };
    const auto estimate = game::shapley_sampled<10>(game::memoized(quadratic), rng, {.permutations = 200});
    //! [shapley]

    std::printf("  shapley       country %+.2f  stock %+.2f  unattributed %+.2f\n", shapley.values[0], shapley.values[1], shapley.unattributed());
    std::printf("  one at a time country %+.2f  stock %+.2f  unattributed %+.2f\n", brinson.values[0], brinson.values[1], brinson.unattributed());
    std::printf("  sequential    country %+.2f  stock %+.2f  unattributed %+.2f\n", off_the_top.values[0], off_the_top.values[1],
                off_the_top.unattributed());
    std::printf("  sampled: feature 0 gets %.3f +/- %.3f of 100 after %zu evaluations\n", estimate.values[0], estimate.standard_errors[0],
                estimate.evaluations);
    ok &= std::abs(shapley.values[0] + 1.15) < 1e-12 && std::abs(shapley.values[1] - 3.05) < 1e-12;
    ok &= std::abs(brinson.unattributed() - 0.1) < 1e-12 && std::abs(off_the_top.values[1] - 3.1) < 1e-12;
    ok &= std::abs(estimate.values[0] - 10.0) < 1e-9;   // symmetric: an equal tenth, exactly, whatever the orders

    std::printf(ok ? "shapley demo: ok\n" : "shapley demo: FAILED\n");
    return ok ? 0 : 1;
}
