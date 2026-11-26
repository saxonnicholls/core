//
//  algorithm_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/algorithm.hpp>: every function at compile time and at run time, the empty and
//  one-element edges, overlap in copy and copy_backward, the noexcept contract, and agreement with <algorithm>.
//

#include <snicholls/core/algorithm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <list>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr std::array<int, 5> filled()
{
    std::array<int, 5> a{};
    snicholls::fill(a.begin(), a.end(), 7);
    return a;
}

constexpr std::array<int, 6> counted()
{
    std::array<int, 6> a{};
    snicholls::iota(a, 10);
    return a;
}

constexpr std::array<int, 5> reversed(std::array<int, 5> a)
{
    snicholls::reverse(a.begin(), a.end());
    return a;
}

// The table a C++17 consumer builds: the squares, filled and then overwritten from the front
constexpr std::array<int, 8> squares()
{
    std::array<int, 8> a{};
    snicholls::iota(a, 0);
    for (auto& x : a)
        x *= x;
    return a;
}

// copy_backward shifts right within one array: the overlap it exists for
constexpr std::array<int, 6> shifted_right()
{
    std::array<int, 6> a{1, 2, 3, 4, 5, 0};
    snicholls::copy_backward(a.begin(), a.begin() + 5, a.end());
    a[0] = 0;
    return a;
}

constexpr std::array<int, 6> shifted_left()
{
    std::array<int, 6> a{0, 1, 2, 3, 4, 5};
    snicholls::copy(a.begin() + 1, a.end(), a.begin());
    return a;
}

constexpr bool swaps()
{
    std::array<int, 3> a{1, 2, 3};
    std::array<int, 3> b{4, 5, 6};
    const auto end = snicholls::swap_ranges(a.begin(), a.end(), b.begin());
    return end == b.end() && a == std::array{4, 5, 6} && b == std::array{1, 2, 3};
}

constexpr bool fill_n_returns_end()
{
    std::array<int, 4> a{};
    const auto end = snicholls::fill_n(a.begin(), 3, 9);
    return end == a.begin() + 3 && a == std::array{9, 9, 9, 0} && snicholls::fill_n(a.begin(), 0, 1) == a.begin();
}

constexpr bool copy_n_copies()
{
    constexpr std::array<int, 4> from{1, 2, 3, 4};
    std::array<int, 4> to{};
    const auto end = snicholls::copy_n(from.begin(), 2, to.begin());
    return end == to.begin() + 2 && to == std::array{1, 2, 0, 0};
}

static_assert(filled() == std::array{7, 7, 7, 7, 7});
static_assert(counted() == std::array{10, 11, 12, 13, 14, 15});
static_assert(squares() == std::array{0, 1, 4, 9, 16, 25, 36, 49});
static_assert(reversed({1, 2, 3, 4, 5}) == std::array{5, 4, 3, 2, 1});
static_assert(shifted_right() == std::array{0, 1, 2, 3, 4, 5});
static_assert(shifted_left() == std::array{1, 2, 3, 4, 5, 5});
static_assert(swaps());
static_assert(fill_n_returns_end());
static_assert(copy_n_copies());
static_assert(snicholls::equal(std::array{1, 2, 3}, std::array{1, 2, 3}));
static_assert(!snicholls::equal(std::array{1, 2, 3}, std::array{1, 2, 4}));
static_assert(!snicholls::equal(std::array{1, 2, 3}, std::array{1, 2}), "ranges of different lengths are unequal");
static_assert(snicholls::equal(std::array<int, 0>{}, std::array<int, 0>{}), "two empty ranges are equal");

// noexcept exactly when the element operations are: int never throws, a std::string copy may
static_assert(noexcept(snicholls::fill(std::declval<int*>(), std::declval<int*>(), 0)));
static_assert(noexcept(snicholls::copy(std::declval<const int*>(), std::declval<const int*>(), std::declval<int*>())));
static_assert(!noexcept(snicholls::fill(std::declval<std::string*>(), std::declval<std::string*>(), std::string{})));
static_assert(!noexcept(snicholls::copy(std::declval<const std::string*>(), std::declval<const std::string*>(),
                                        std::declval<std::string*>())));
static_assert(noexcept(snicholls::reverse(std::declval<std::string*>(), std::declval<std::string*>())),
              "moving a std::string does not throw, so neither does reversing them");

TEST(AlgorithmFill, WritesEveryElementOfAVector)
{
    std::vector<double> v(1000, 1.0);
    snicholls::fill(v, 2.5);
    EXPECT_TRUE(std::all_of(v.begin(), v.end(), [](double x) { return x == 2.5; }));
}

TEST(AlgorithmFill, EmptyRangeIsANoOp)
{
    std::vector<int> v;
    snicholls::fill(v, 3);
    snicholls::fill(v.begin(), v.end(), 3);
    EXPECT_TRUE(v.empty());
}

TEST(AlgorithmFill, WorksThroughASpanAndACArray)
{
    int raw[4] = {};
    snicholls::fill(raw, 5);
    EXPECT_EQ(raw[0], 5);
    EXPECT_EQ(raw[3], 5);
    std::array<int, 6> a{};
    snicholls::fill(std::span<int>(a).subspan(2, 2), 1);
    EXPECT_EQ(a, (std::array<int, 6>{0, 0, 1, 1, 0, 0}));
}

TEST(AlgorithmFill, ClearsBytesLikeMemset)
{
    std::array<unsigned char, 48> buffer{};
    snicholls::fill(buffer, static_cast<unsigned char>(0xAB));
    snicholls::fill_n(buffer.begin() + 32, 16, static_cast<unsigned char>(0));
    EXPECT_EQ(buffer[31], 0xAB);
    EXPECT_EQ(buffer[32], 0);
    EXPECT_EQ(buffer[47], 0);
}

TEST(AlgorithmCopy, AgreesWithStdCopyOnStrings)
{
    const std::vector<std::string> from{"alpha", "beta", "", "a string long enough to allocate on the heap"};
    std::vector<std::string> ours(from.size()), theirs(from.size());
    snicholls::copy(from.begin(), from.end(), ours.begin());
    std::copy(from.begin(), from.end(), theirs.begin());
    EXPECT_EQ(ours, theirs);
}

TEST(AlgorithmCopy, WorksOverAListWhichIsNotContiguous)
{
    const std::list<int> from{3, 1, 4, 1, 5};
    std::vector<int> to(5);
    EXPECT_EQ(snicholls::copy(from.begin(), from.end(), to.begin()), to.end());
    EXPECT_EQ(to, (std::vector<int>{3, 1, 4, 1, 5}));
    std::vector<int> back(5);
    snicholls::copy_backward(from.begin(), from.end(), back.end());
    EXPECT_EQ(back, to);
}

TEST(AlgorithmCopy, CopyBackwardReturnsTheFirstElementWritten)
{
    std::array<int, 5> a{1, 2, 3, 0, 0};
    const auto first = snicholls::copy_backward(a.begin(), a.begin() + 3, a.end());
    EXPECT_EQ(first, a.begin() + 2);
    EXPECT_EQ(a, (std::array<int, 5>{1, 2, 1, 2, 3}));
}

TEST(AlgorithmIota, AgreesWithStdIota)
{
    std::vector<long> ours(100), theirs(100);
    snicholls::iota(ours.begin(), ours.end(), -50L);
    std::iota(theirs.begin(), theirs.end(), -50L);
    EXPECT_EQ(ours, theirs);
}

TEST(AlgorithmEqual, ThreeIteratorFormComparesThePrefix)
{
    const std::vector<int> a{1, 2, 3};
    const std::vector<int> longer{1, 2, 3, 4};
    EXPECT_TRUE(snicholls::equal(a.begin(), a.end(), longer.begin()));
    EXPECT_FALSE(snicholls::equal(a, longer));
    EXPECT_TRUE(snicholls::equal(a.begin(), a.begin(), longer.end())) << "an empty first range is equal to anything";
}

TEST(AlgorithmReverse, HandlesEmptyOneAndEvenLengths)
{
    std::vector<int> empty;
    snicholls::reverse(empty.begin(), empty.end());
    EXPECT_TRUE(empty.empty());
    std::vector<int> one{7};
    snicholls::reverse(one.begin(), one.end());
    EXPECT_EQ(one, std::vector<int>{7});
    std::vector<int> even{1, 2, 3, 4};
    snicholls::reverse(even.begin(), even.end());
    EXPECT_EQ(even, (std::vector<int>{4, 3, 2, 1}));
}

TEST(AlgorithmReverse, MovesMoveOnlyElements)
{
    std::array<std::unique_ptr<int>, 3> a{std::make_unique<int>(1), std::make_unique<int>(2), std::make_unique<int>(3)};
    snicholls::reverse(a.begin(), a.end());
    EXPECT_EQ(*a[0], 3);
    EXPECT_EQ(*a[1], 2);
    EXPECT_EQ(*a[2], 1);
}

TEST(AlgorithmSwapRanges, ExchangesStringsAndReturnsTheEndOfTheSecond)
{
    std::vector<std::string> a{"x", "y"};
    std::vector<std::string> b{"long enough to live on the heap, not in the small buffer", "z", "untouched"};
    const auto end = snicholls::swap_ranges(a.begin(), a.end(), b.begin());
    EXPECT_EQ(end, b.begin() + 2);
    EXPECT_EQ(a[0], "long enough to live on the heap, not in the small buffer");
    EXPECT_EQ(a[1], "z");
    EXPECT_EQ(b[0], "x");
    EXPECT_EQ(b[1], "y");
    EXPECT_EQ(b[2], "untouched");
}

} // namespace
