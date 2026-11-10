//
//  constexpr_for_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/constexpr_for.hpp>. Ported from ts-moveables'
//  tests_parallel_for.cpp (the six constexpr_for / constexpr_nest /
//  unrolled_for tests) to GoogleTest, assertion for assertion, plus the
//  compile-time claims the header makes.
//

#include <snicholls/core/constexpr_for.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using snicholls::constexpr_for;
using snicholls::constexpr_nest;
using snicholls::unrolled_for;

// Usable in a constant expression
constexpr int sum_to(int)
{
    int s = 0;
    constexpr_for<0, 5>([&](auto I) { s += I; });
    return s;
}
static_assert(sum_to(0) == 10);

// The index type is decltype(Start)
constexpr bool index_types()
{
    bool ok = true;
    constexpr_for<0, 1>([&](auto I) { ok = ok && std::is_same_v<typename decltype(I)::value_type, int>; });
    constexpr_for<std::size_t{0}, std::size_t{1}>(
        [&](auto I) { ok = ok && std::is_same_v<typename decltype(I)::value_type, std::size_t>; });
    return ok;
}
static_assert(index_types());

// A negative step against unsigned bounds does not wrap
constexpr std::size_t count_down_unsigned()
{
    std::size_t n = 0;
    constexpr_for<std::size_t{3}, std::size_t{0}, -1>([&](auto) { ++n; });
    return n;
}
static_assert(count_down_unsigned() == 3);

TEST(ConstexprFor, ExpandsTheRangeAndAnEmptyRangeToNothing)
{
    int sum = 0;
    constexpr_for<0, 4>([&](auto I) { sum += I.value; });
    EXPECT_EQ(sum, 0 + 1 + 2 + 3);

    sum = 0;
    constexpr_for<4>([&](auto I) { sum += I.value; });   // [0, Count) spelling
    EXPECT_EQ(sum, 0 + 1 + 2 + 3);

    int calls = 0;
    constexpr_for<3, 3>([&](auto) { ++calls; });
    EXPECT_EQ(calls, 0);
}

TEST(ConstexprFor, HonoursAStepIncludingANegativeOne)
{
    int sum = 0;
    constexpr_for<0, 10, 2>([&](auto I) { sum += I; });  // 0 2 4 6 8
    EXPECT_EQ(sum, 20);

    std::vector<int> order;
    constexpr_for<5, 0, -1>([&](auto I) { order.push_back(I); });
    EXPECT_EQ(order, (std::vector<int>{5, 4, 3, 2, 1}));  // 0 is the open end

    int once = 0;                                         // a step that overshoots runs once
    constexpr_for<0, 3, 10>([&](auto) { ++once; });
    EXPECT_EQ(once, 1);
}

TEST(ConstexprFor, IndexIsUsableAsATemplateArgument)
{
    auto tup = std::make_tuple(1, 2.5, 3);
    double total = 0;
    constexpr_for<0, 3>([&](auto I) { total += static_cast<double>(std::get<I.value>(tup)); });
    EXPECT_EQ(total, 6.5);
}

TEST(ConstexprFor, TrailingArgumentsReachEveryIterationAsLvalues)
{
    int counter = 0;
    constexpr_for<0, 3>([](auto, int& c) { c += 2; }, counter);
    EXPECT_EQ(counter, 6);
}

TEST(ConstexprNest, UnrollsTwoThreeAndFourDeepWithConstantIndices)
{
    std::vector<std::pair<int, int>> visited;
    constexpr_nest<2, 3>([&](auto I, auto J) {
        visited.emplace_back(static_cast<int>(I.value), static_cast<int>(J.value));
    });
    const std::vector<std::pair<int, int>> expect_2d{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}};
    EXPECT_EQ(visited, expect_2d);                        // row-major, last varies fastest

    int count3 = 0, count4 = 0;
    constexpr_nest<2, 3, 4>([&](auto, auto, auto) { ++count3; });
    constexpr_nest<2, 2, 2, 2>([&](auto, auto, auto, auto) { ++count4; });
    EXPECT_EQ(count3, 2 * 3 * 4);
    EXPECT_EQ(count4, 16);

    int fixed = 0;
    constexpr_nest<2, 2>([&](auto I, auto J) {
        constexpr std::size_t flat = I.value * 2 + J.value;
        fixed += static_cast<int>(flat);
    });
    EXPECT_EQ(fixed, 0 + 1 + 2 + 3);
}

TEST(UnrolledFor, EveryElementOnceForEveryLengthTailIncluded)
{
    for (std::size_t n = 0; n < 40; ++n) {
        std::vector<int> hits(n, 0);
        unrolled_for<4>(std::size_t{0}, n, [&](std::size_t i) { ++hits[i]; });
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_EQ(hits[i], 1) << "n=" << n << " i=" << i;
    }

    int calls = 0;
    unrolled_for<1>(0, 5, [&](int) { ++calls; });         // Unroll 1 is a plain loop
    EXPECT_EQ(calls, 5);

    int inverted = 0;                                     // an inverted range does nothing
    unrolled_for<4>(9, 3, [&](int) { ++inverted; });
    EXPECT_EQ(inverted, 0);
}

} // namespace
