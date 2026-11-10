//
//  properties_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/game/properties.hpp>: a convex game, whose Shapley attribution is in its core, and a diversified one. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/game/properties.hpp>

#include <cmath>
#include <cstdio>

int main()
{
    std::printf("snicholls-core properties demo\n");
    namespace game = snicholls::game;
    bool ok = true;

    //! [properties]
    // Three signals whose value grows with each other: the payoff is convex in them.
    const auto convex = [](game::coalition s) { const double k = static_cast<double>(game::coalition_size(s)); return k * k; };
    // Three risk factors that diversify: risk grows less than additively.
    const auto diversified = [](game::coalition s) { return std::sqrt(static_cast<double>(game::coalition_size(s))); };

    const bool complements = game::is_supermodular<3>(convex);
    const bool substitutes = game::is_submodular<3>(diversified);
    const auto attribution = game::shapley<3>(convex);
    const bool stable = game::in_core<3>(convex, attribution.values, 1e-12);   // Shapley 1971
    const auto where = game::find_supermodularity_violation<3>(diversified);
    //! [properties]

    std::printf("  convex: supermodular %d, Shapley in the core %d\n", complements, stable);
    std::printf("  diversified: submodular %d, first concave step at players %zu and %zu\n", substitutes, where->first, where->second);
    ok &= complements && substitutes && stable && where.has_value();

    std::printf(ok ? "properties demo: ok\n" : "properties demo: FAILED\n");
    return ok ? 0 : 1;
}
