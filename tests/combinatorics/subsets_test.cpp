//
//  subsets_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/combinatorics/subsets.hpp>: Wolfram's Subsets and Tuples examples,
//  counts, the edges, and the binomial's exact overflow boundary.
//

#include <snicholls/core/combinatorics/subsets.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <set>
#include <vector>

namespace {

using pair = std::array<char, 2>;

// Subsets[{a, b, c}, {2}]
static_assert(snicholls::subsets<2>(std::array{'a', 'b', 'c'}) == std::array{pair{'a', 'b'}, pair{'a', 'c'}, pair{'b', 'c'}});
static_assert(snicholls::subsets<0>(std::array{'a', 'b'}).size() == 1 && snicholls::subsets<3>(std::array{'a', 'b'}).empty());
// Tuples[{0, 1}, 2]
static_assert(snicholls::tuples<2>(std::array{0, 1}) == std::array{std::array{0, 0}, std::array{0, 1}, std::array{1, 0}, std::array{1, 1}});
static_assert(snicholls::tuple_count_v<3, 4> == 81 && snicholls::tuple_count_v<0, 0> == 1 && snicholls::tuple_count_v<0, 2> == 0);
static_assert(snicholls::combination_count_v<52, 5> == 2598960);   // poker hands
static_assert(snicholls::index_combinations<5, 3>()[9] == std::array<std::size_t, 3>{2, 3, 4});

// The binomial's exact boundary: C(67, 33) fits 64 bits, C(68, 34) does not
static_assert(snicholls::checked_binomial(67, 33) == 14226520737620288370ULL);
static_assert(!snicholls::checked_binomial(68, 34).has_value());
static_assert(snicholls::checked_binomial(5, 7) == 0u && snicholls::checked_binomial(0, 0) == 1u);

TEST(Subsets, EverySubsetBySizeThenOrder)
{
    std::vector<std::uint64_t> masks;
    snicholls::for_each_subset<3>([&](std::uint64_t m) { masks.push_back(m); });
    // Subsets[{a, b, c}]: {}, {a}, {b}, {c}, {a, b}, {a, c}, {b, c}, {a, b, c}
    EXPECT_EQ(masks, (std::vector<std::uint64_t>{0, 1, 2, 4, 3, 5, 6, 7}));
}

TEST(Subsets, CombinationsAreDistinctIncreasingAndComplete)
{
    std::set<std::array<std::size_t, 4>> seen;
    snicholls::for_each_combination<10, 4>([&](const std::array<std::size_t, 4>& r) {
        EXPECT_TRUE(r[0] < r[1] && r[1] < r[2] && r[2] < r[3]);
        seen.insert(r);
    });
    EXPECT_EQ(seen.size(), 210u);
}

} // namespace
