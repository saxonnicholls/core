//
//  subsets_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/combinatorics/subsets.hpp>: every pair of hedges from a list, and
//  every combination of three scenario shocks, at compile time. Exits non-zero if a count is not
//  the one documented.
//

#include <snicholls/core/combinatorics/subsets.hpp>

#include <array>
#include <cstdio>
#include <string_view>

int main()
{
    std::printf("snicholls-core subsets demo\n");
    bool ok = true;

    //! [subsets]
    // Subsets[hedges, {2}]: every pair of hedges to test together, in order, computed by the compiler.
    constexpr std::array<std::string_view, 4> hedges{"puts", "vix", "gold", "duration"};
    constexpr auto pairs = snicholls::subsets<2>(hedges);
    static_assert(pairs.size() == 6 && pairs[5][0] == "gold" && pairs[5][1] == "duration");

    // Tuples[shocks, 3]: every path of three daily shocks, down, flat or up.
    constexpr auto paths = snicholls::tuples<3>(std::array{-1, 0, 1});
    static_assert(paths.size() == 27);
    //! [subsets]

    std::printf("  %zu pairs of hedges, %zu shock paths; first pair %.*s + %.*s\n", pairs.size(), paths.size(),
                static_cast<int>(pairs[0][0].size()), pairs[0][0].data(), static_cast<int>(pairs[0][1].size()), pairs[0][1].data());
    ok &= pairs[0][0] == "puts" && pairs[0][1] == "vix";

    std::printf(ok ? "subsets demo: ok\n" : "subsets demo: FAILED\n");
    return ok ? 0 : 1;
}
