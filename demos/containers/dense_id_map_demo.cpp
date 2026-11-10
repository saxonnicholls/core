//
//  dense_id_map_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/dense_id_map.hpp>.
//

// Demo: snicholls::containers::dense_id_map - sequential exchange ids map to pointers with one array lookup; the window
// slides up as old ids leave, and outlying ids fall back to a hash map. Exits non-zero on a mismatch.

#include <cstdio>

#include <snicholls/core/containers/dense_id_map.hpp>

int main()
{
    int failures = 0;
    auto expect = [&](bool ok, const char* what) {
        std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
        if (!ok) ++failures;
    };
    static int orders[1024];
    std::printf("dense_id_map<int*>\n");
    //! [dense]
    snicholls::containers::dense_id_map<int*> ids(1024);
    for (std::uint64_t id = 5'000'000; id < 6'000'000; ++id) {   // a day of ITCH order refs
        ids.insert(id, &orders[id % 1024]);
        if (id >= 5'000'512) ids.erase(id - 512);                  // ~512 live orders at any time
    }
    ids.insert(42, &orders[0]);                                    // an outlier id: hash fallback
    //! [dense]
    std::printf("  %zu live ids, window %zu slots, %zu in the fallback\n", ids.size(), ids.window(), ids.outliers());
    expect(ids.size() == 513 && ids.window() <= 4096, "the window slid with the live ids");
    expect(*ids.find(5'999'999) == &orders[5'999'999 % 1024] && ids.contains(42), "window and fallback lookups");
    expect(!ids.insert(42, &orders[1]) && ids.erase(42) && !ids.contains(42), "duplicate refused, outlier erased");
    ids.clear();
    ids.reserve(100);
    expect(ids.empty(), "clear");
    return failures == 0 ? 0 : 1;
}
