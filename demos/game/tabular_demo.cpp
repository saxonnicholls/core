//
//  tabular_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/game/tabular.hpp>: a credit model explained at one borrower against a background of others. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/game/shapley.hpp>
#include <snicholls/core/game/tabular.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core tabular demo\n");
    namespace game = snicholls::game;
    bool ok = true;

    //! [tabular]
    // Any model over rows of features: here a score from leverage, coverage, and their interaction.
    const auto model = [](std::span<const double> x) { return 2.0 * x[0] - 1.0 * x[1] + 0.5 * x[0] * x[1]; };
    const std::array<double, 2> borrower{3.0, 1.0};
    const std::vector<std::array<double, 2>> others{{1.0, 2.0}, {2.0, 2.0}, {1.0, 3.0}};

    // Why did this borrower score what it did, compared with the others?
    const auto explained = game::shapley<2>(game::interventional(model, borrower, others));
    //! [tabular]

    std::printf("  score %.3f against %.3f on average: leverage %+.3f, coverage %+.3f\n", explained.total, explained.baseline, explained.values[0],
                explained.values[1]);
    ok &= std::abs(explained.unattributed()) < 1e-12 && std::abs(explained.total - model(borrower)) < 1e-12;

    std::printf(ok ? "tabular demo: ok\n" : "tabular demo: FAILED\n");
    return ok ? 0 : 1;
}
