//
//  partition_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/combinatorics/partition.hpp>: Wolfram's Partition on a price series at compile time, and every grouping of four factors. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/combinatorics/partition.hpp>

#include <array>
#include <cstdint>
#include <cstdio>

int main()
{
    std::printf("snicholls-core partition demo\n");
    bool ok = true;

    //! [partition]
    // Bars of two, and every window of three: Partition[list, 2] and Partition[list, 3, 1].
    constexpr std::array prices{100, 101, 99, 102, 104, 103};
    constexpr auto bars = snicholls::partition<2>(prices);
    constexpr auto windows = snicholls::partition<3, 1>(prices);
    static_assert(bars.size() == 3 && windows.size() == 4 && windows[3] == std::array{102, 104, 103});

    // Every way to group four factors (15, the Bell number), as the masks game::owen takes.
    std::size_t groupings = 0;
    snicholls::for_each_set_partition<4>([&](const std::array<std::size_t, 4>& block, std::size_t blocks) {
        const std::array<std::uint64_t, 4> groups = snicholls::blocks_as_masks(block, blocks);
        groupings += groups[0] != 0 ? 1 : 0;
    });
    //! [partition]

    std::printf("  %zu bars, %zu windows, %zu groupings of four factors\n", bars.size(), windows.size(), groupings);
    ok &= groupings == snicholls::bell_v<4>;

    std::printf(ok ? "partition demo: ok\n" : "partition demo: FAILED\n");
    return ok ? 0 : 1;
}
