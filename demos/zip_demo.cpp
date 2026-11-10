//
//  zip_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/zip.hpp>: reading and writing several ranges together. Exits
//  non-zero if a result is not the one documented.
//

#include <snicholls/core/zip.hpp>

#include <cstdio>
#include <ranges>
#include <vector>

int main()
{
    std::printf("snicholls-core zip demo\n");
    bool ok = true;

    //! [zip]
    // A structured binding, one name per range; the walk stops at the shortest.
    const std::vector<double> prices{189.25, 420.50, 15.75};
    const std::vector<int> sizes{100, 10};
    double notional = 0;
    for (auto [price, size] : snicholls::zip(prices, sizes))
        notional += price * size;

    // The names are references: writing changes the original.
    std::vector<int> filled{0, 0};
    for (auto [fill, size] : snicholls::zip(filled, sizes))
        fill = size / 2;

    // A range with itself, one along: adjacent pairs.
    int rises = 0;
    for (auto [before, after] : snicholls::zip(prices, prices | std::views::drop(1)))
        rises += after > before ? 1 : 0;
    //! [zip]

    std::printf("  notional %.2f, filled %d and %d, %d rise\n", notional, filled[0], filled[1], rises);
    ok &= notional == 189.25 * 100 + 420.50 * 10;
    ok &= filled == std::vector<int>{50, 5};
    ok &= rises == 1;

    std::printf(ok ? "zip demo: ok\n" : "zip demo: FAILED\n");
    return ok ? 0 : 1;
}
