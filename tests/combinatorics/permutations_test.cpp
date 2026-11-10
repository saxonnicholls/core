//
//  combinatorics/permutations_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/combinatorics/permutations.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using snicholls::index_permutations;
using snicholls::permutation_count_v;
using snicholls::permutations;
using row3 = std::array<std::size_t, 3>;
using row2 = std::array<std::size_t, 2>;

// Counts
static_assert(permutation_count_v<0> == 1 && permutation_count_v<1> == 1 && permutation_count_v<5> == 120);
static_assert(permutation_count_v<3, 2> == 6 && permutation_count_v<3, 0> == 1 && permutation_count_v<2, 3> == 0);
static_assert(permutation_count_v<20> == 2432902008176640000ULL || sizeof(std::size_t) < 8);
static_assert(snicholls::permutation_count<4, 2>::value == 12);

// The exact table, in lexicographic order
constexpr auto three = index_permutations<3>();
static_assert(three == std::array<row3, 6>{row3{0, 1, 2}, row3{0, 2, 1}, row3{1, 0, 2}, row3{1, 2, 0}, row3{2, 0, 1}, row3{2, 1, 0}});
constexpr auto three_two = index_permutations<3, 2>();
static_assert(three_two == std::array<row2, 6>{row2{0, 1}, row2{0, 2}, row2{1, 0}, row2{1, 2}, row2{2, 0}, row2{2, 1}});

// The edges, which the original could not compile: N = 0, and R above N
static_assert(index_permutations<0>().size() == 1 && index_permutations<0>()[0].empty());
static_assert(index_permutations<2, 3>().empty());
static_assert(index_permutations<2, 0>().size() == 1);
static_assert(index_permutations<1>()[0] == std::array<std::size_t, 1>{0});

// Values, by position; an equal value still gives its own row
constexpr auto values = permutations(std::array{7, 8, 9});
static_assert(values[0] == std::array{7, 8, 9} && values[5] == std::array{9, 8, 7});
constexpr auto chosen = permutations<2>(std::array{7, 8, 9});
static_assert(chosen.size() == 6 && chosen[5] == std::array{9, 8});
static_assert(permutations(std::array{1, 1}).size() == 2);
static_assert(permutations<3>(std::array{1, 2}).empty());
static_assert(permutations<0>(std::array{1, 2}).size() == 1);

// No default constructor needed, which the original required
struct no_default {
    int v;
    constexpr explicit no_default(int x) noexcept : v(x) {}
};
static_assert(!std::is_default_constructible_v<no_default>);
static_assert(permutations(std::array{no_default{1}, no_default{2}})[1][0].v == 2);

// Sizes: at compile time, on every compiler's default budget. The original failed at 6 on Clang.
static_assert(index_permutations<6>().size() == 720);
static_assert(index_permutations<6>()[719] == std::array<std::size_t, 6>{5, 4, 3, 2, 1, 0});

// constexpr iteration, with no table
consteval int sum_of_firsts() noexcept
{
    int sum = 0;
    snicholls::for_each_permutation<3>([&sum](const row3& row) noexcept { sum += static_cast<int>(row[0]); });
    return sum;
}
static_assert(sum_of_firsts() == 6);

// noexcept is accurate
static_assert(noexcept(index_permutations<2>()));
static_assert(noexcept(permutations(std::array{1, 2})));
static_assert(!noexcept(permutations(std::declval<const std::array<std::string, 2>&>())));
static_assert(noexcept(snicholls::for_each_permutation<2>([](const row2&) noexcept {})));
static_assert(!noexcept(snicholls::for_each_permutation<2>([](const row2&) {})));

TEST(Permutations, EveryRowAgreesWithNextPermutation)
{
    std::array<std::size_t, 7> expected{};
    std::iota(expected.begin(), expected.end(), std::size_t{0});
    std::size_t rows = 0;
    bool same = true;
    snicholls::for_each_permutation<7>([&](const std::array<std::size_t, 7>& row) {
        same = same && row == expected;
        std::next_permutation(expected.begin(), expected.end());
        ++rows;
    });
    EXPECT_TRUE(same);
    EXPECT_EQ(rows, 5040u);
}

TEST(Permutations, PartialRowsAreDistinctAndComplete)
{
    std::set<std::array<std::size_t, 3>> rows;
    snicholls::for_each_permutation<5, 3>([&](const row3& row) { rows.insert(row); });
    EXPECT_EQ(rows.size(), 60u);
    for (const row3& row : rows) {
        EXPECT_NE(row[0], row[1]);
        EXPECT_NE(row[1], row[2]);
        EXPECT_NE(row[0], row[2]);
    }
}

TEST(Permutations, ForEachWorksWhereATableWouldNotFit)
{
    std::size_t rows = 0;
    snicholls::for_each_permutation<9>([&](const std::array<std::size_t, 9>&) { ++rows; });
    EXPECT_EQ(rows, 362880u);
}

TEST(Permutations, ValuesNeedOnlyBeCopyable)
{
    const auto rows = permutations(std::array{std::string{"a"}, std::string{"b"}});
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[1][0], "b");
    EXPECT_EQ(rows[1][1], "a");
}

} // namespace
