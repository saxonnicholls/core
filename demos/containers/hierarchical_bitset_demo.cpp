//
//  hierarchical_bitset_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/hierarchical_bitset.hpp>.
//

// Demo: snicholls::containers::hierarchical_bitset - occupancy of a price ladder; the next non-empty level after the best
// empties is a bit scan per summary level. Exits non-zero if a result is not as printed.

#include <cstdio>

#include <snicholls/core/containers/hierarchical_bitset.hpp>

namespace {
int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-56s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}
} // namespace

int main()
{
    using snicholls::containers::hierarchical_bitset;
    std::printf("hierarchical_bitset over 1,000,000 ticks (four levels)\n");
    //! [bits]
    hierarchical_bitset occupied(1'000'000);
    occupied.set(10);
    occupied.set(500'000);
    occupied.set(999'999);
    const auto next_ask = occupied.find_next(11);        // 500000: skips empty words via the summaries
    const auto next_bid = occupied.find_prev(499'999);   // 10
    //! [bits]
    expect(next_ask == 500'000 && next_bid == 10, "find_next / find_prev across empty ranges");
    occupied.reset(500'000);
    expect(occupied.find_next(11) == 999'999 && occupied.test(10) && !occupied.test(500'000), "reset, test");
    expect(occupied.find_next(1'000'000) == hierarchical_bitset::npos && occupied.any(), "past the end: npos");
    occupied.clear();
    expect(!occupied.any() && occupied.find_prev(999'999) == hierarchical_bitset::npos, "clear");
    occupied.resize(64);
    expect(occupied.size() == 64, "resize");
    return failures == 0 ? 0 : 1;
}
