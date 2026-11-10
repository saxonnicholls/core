//
//  combinatorics/partition_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/combinatorics/partition.hpp>: the Wolfram reference's forms at compile time, Bell
//  numbers, and every set partition once.
//

#include <snicholls/core/combinatorics/partition.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace {

using snicholls::partition;
using snicholls::partition_cyclic;
using snicholls::partition_padded;

constexpr std::array letters{'a', 'b', 'c', 'd', 'e', 'f'};
using pair = std::array<char, 2>;
using triple = std::array<char, 3>;

// Partition[{a, b, c, d, e, f}, 2] and Partition[..., 3, 1]
static_assert(partition<2>(letters) == std::array{pair{'a', 'b'}, pair{'c', 'd'}, pair{'e', 'f'}});
static_assert(partition<3, 1>(letters) == std::array{triple{'a', 'b', 'c'}, triple{'b', 'c', 'd'}, triple{'c', 'd', 'e'}, triple{'d', 'e', 'f'}});
// An incomplete sublist is dropped: Partition[{a, b, c, d, e}, 2]
static_assert(partition<2>(std::array{'a', 'b', 'c', 'd', 'e'}).size() == 2);
// Longer than the list: nothing
static_assert(partition<7>(letters).empty());
// Partition[{a, b, c, d, e}, 2, 2, {1, 1}, x]
static_assert(partition_padded<2>(std::array{'a', 'b', 'c', 'd', 'e'}, 'x') == std::array{pair{'a', 'b'}, pair{'c', 'd'}, pair{'e', 'x'}});
// Partition[{a, b, c, d}, 3, 1, {1, 1}]: cyclic
static_assert(partition_cyclic<3, 1>(std::array{'a', 'b', 'c', 'd'}) ==
              std::array{triple{'a', 'b', 'c'}, triple{'b', 'c', 'd'}, triple{'c', 'd', 'a'}, triple{'d', 'a', 'b'}});
static_assert(partition_padded<2>(std::array<char, 0>{}, 'x').empty());

// Bell numbers
static_assert(snicholls::bell_v<0> == 1 && snicholls::bell_v<1> == 1 && snicholls::bell_v<3> == 5 && snicholls::bell_v<5> == 52);
static_assert(snicholls::bell_v<25> == 4638590332229999353ULL);

consteval std::size_t count_partitions_of_four()
{
    std::size_t count = 0;
    snicholls::for_each_set_partition<4>([&count](const std::array<std::size_t, 4>&, std::size_t) noexcept { ++count; });
    return count;
}
static_assert(count_partitions_of_four() == 15);

TEST(SetPartitions, ComeInRestrictedGrowthOrderWithTheirBlockCounts)
{
    std::vector<std::string> seen;
    snicholls::for_each_set_partition<3>([&](const std::array<std::size_t, 3>& b, std::size_t blocks) {
        seen.push_back(std::to_string(b[0]) + std::to_string(b[1]) + std::to_string(b[2]) + ":" + std::to_string(blocks));
    });
    EXPECT_EQ(seen, (std::vector<std::string>{"000:1", "001:2", "010:2", "011:2", "012:3"}));
}

TEST(SetPartitions, EveryPartitionOfSevenOnceAsMasks)
{
    std::set<std::vector<std::uint64_t>> distinct;
    snicholls::for_each_set_partition<7>([&](const std::array<std::size_t, 7>& b, std::size_t blocks) {
        const auto masks = snicholls::blocks_as_masks(b, blocks);
        std::uint64_t all = 0;
        for (std::size_t k = 0; k < blocks; ++k) {
            EXPECT_EQ(all & masks[k], 0u);
            all |= masks[k];
        }
        EXPECT_EQ(all, 0x7Fu);
        distinct.insert(std::vector<std::uint64_t>(masks.begin(), masks.begin() + static_cast<std::ptrdiff_t>(blocks)));
    });
    EXPECT_EQ(distinct.size(), snicholls::bell_v<7>);
}

TEST(SetPartitions, NoElementsIsOneEmptyPartition)
{
    int calls = 0;
    snicholls::for_each_set_partition<0>([&](const std::array<std::size_t, 0>&, std::size_t blocks) {
        ++calls;
        EXPECT_EQ(blocks, 0u);
    });
    EXPECT_EQ(calls, 1);
}

} // namespace
