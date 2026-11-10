//
//  md_array_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/md_array.hpp>: a scenario x tenor x strike grid with
//  compile-time strides. Exits non-zero if a cell is not the one documented.
//

#include <snicholls/core/containers/md_array.hpp>

#include <cstdio>

int main()
{
    std::printf("snicholls-core md_array demo\n");
    //! [md_array]
    // 3 scenarios x 4 tenors x 5 strikes, one block, strides known to the compiler.
    snicholls::containers::md_array<double, 3, 4, 5> pnl{};
    snicholls::containers::md_array<double, 3, 4, 5>::for_each_index([&](auto s, auto t, auto k) {
        pnl(s(), t(), k()) = 10.0 * static_cast<double>(s) - static_cast<double>(t) + 0.5 * static_cast<double>(k);
    });
    //! [md_array]
    std::printf("  scenario 2, tenor 3, strike 4: %.1f\n", pnl(2, 3, 4));
    const bool ok = pnl(2, 3, 4) == 19.0;
    std::printf(ok ? "md_array demo: ok\n" : "md_array demo: FAILED\n");
    return ok ? 0 : 1;
}
