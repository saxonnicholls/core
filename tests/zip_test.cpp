//
//  zip_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/zip.hpp>: lengths, writing through, ownership, the range concepts,
//  and the layout contract.
//

#include <snicholls/core/zip.hpp>

#include <snicholls/core/acout.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <forward_list>
#include <list>
#include <memory>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using snicholls::zip;

using two_vectors = decltype(zip(std::declval<std::vector<int>&>(), std::declval<std::vector<double>&>()));

// The concepts
static_assert(std::ranges::view<two_vectors>);
static_assert(std::ranges::forward_range<two_vectors>);
static_assert(std::ranges::sized_range<two_vectors>);
static_assert(!std::ranges::common_range<two_vectors>);
static_assert(std::ranges::forward_range<const two_vectors>);
static_assert(std::ranges::forward_range<decltype(zip(std::declval<std::list<int>&>(), std::declval<std::vector<int>&>()))>);
static_assert(std::is_same_v<std::ranges::range_reference_t<two_vectors>, std::tuple<int&, double&>>);
static_assert(std::is_same_v<std::ranges::range_value_t<two_vectors>, snicholls::zip_value<int, double>>);
// Const ranges and move-only elements are ranges too, which C++20's std::tuple alone would not allow
using two_const = decltype(zip(std::declval<const std::vector<int>&>(), std::declval<const std::vector<int>&>()));
static_assert(std::ranges::forward_range<two_const> && std::ranges::viewable_range<two_const>);
static_assert(std::ranges::forward_range<decltype(zip(std::declval<std::vector<std::unique_ptr<int>>&>()))>);
static_assert(std::ranges::input_range<decltype(zip(std::declval<std::vector<bool>&>()))>);
static_assert(std::tuple_size_v<snicholls::zip_value<int, char>> == 2);

// Layout contract: a zip of two references is two pointers, an iterator two iterators
static_assert(sizeof(two_vectors) == 2 * sizeof(void*));
static_assert(sizeof(std::ranges::iterator_t<two_vectors>) == 2 * sizeof(std::vector<int>::iterator));

// A compile-time facility is tested at compile time
consteval int constexpr_dot()
{
    const std::array a{1, 2, 3};
    const std::array b{4, 5, 6};
    int sum = 0;
    for (auto [x, y] : zip(a, b))
        sum += x * y;
    return sum;
}
static_assert(constexpr_dot() == 32);

consteval int constexpr_write()
{
    std::array a{1, 2};
    const std::array b{10, 20};
    for (auto [x, y] : zip(a, b))
        x += y;
    return a[0] + a[1];
}
static_assert(constexpr_write() == 33);

TEST(Zip, WalksTogetherInOrder)
{
    const std::vector<int> ints{1, 2, 3};
    const std::vector<std::string> names{"a", "b", "c"};
    std::string seen;
    for (auto [i, name] : zip(ints, names))
        seen += std::to_string(i) + name;
    EXPECT_EQ(seen, "1a2b3c");
}

TEST(Zip, StopsAtTheShortestWhicheverItIs)
{
    const std::vector<int> three{1, 2, 3};
    const std::vector<int> one{9};
    const std::vector<int> none;
    EXPECT_EQ(std::ranges::distance(zip(three, one)), 1);
    EXPECT_EQ(std::ranges::distance(zip(one, three)), 1);
    EXPECT_EQ(std::ranges::distance(zip(three, one, three)), 1);
    EXPECT_EQ(zip(three, one).size(), 1u);
    EXPECT_EQ(zip(three, none).size(), 0u);
    EXPECT_TRUE(zip(three, none).empty());
    EXPECT_TRUE(zip(none, three).empty());
    EXPECT_FALSE(zip(three, one).empty());
}

TEST(Zip, OneRangeAndManyRanges)
{
    const std::vector<int> a{1, 2};
    int sum = 0;
    for (auto [x] : zip(a))
        sum += x;
    EXPECT_EQ(sum, 3);
    for (auto [p, q, r, s] : zip(a, a, a, a))
        sum += p + q + r + s;
    EXPECT_EQ(sum, 3 + 12);
}

TEST(Zip, WritesThroughToTheOriginals)
{
    std::vector<int> a{1, 2};
    std::vector<int> b{10, 20};
    for (auto [x, y] : zip(a, b)) {
        x += y;
        y = 0;
    }
    EXPECT_EQ(a, (std::vector<int>{11, 22}));
    EXPECT_EQ(b, (std::vector<int>{0, 0}));
}

TEST(Zip, TakesAnyMixOfRanges)
{
    const int raw[]{1, 2};
    const std::list<int> list{3, 4};
    std::forward_list<int> forward{5, 6};
    std::vector<bool> flags{true, false};   // a proxy reference
    int sum = 0;
    for (auto [r, l, f, flag] : zip(raw, list, forward, flags))
        sum += r + l + f + (flag ? 100 : 0);
    EXPECT_EQ(sum, 1 + 3 + 5 + 100 + 2 + 4 + 6);
    const std::string_view text = "ab";
    std::string joined;
    for (auto [c, i] : zip(text, std::views::iota(0)))   // unbounded: the other range decides
        joined += std::string(1, c) + std::to_string(i);
    EXPECT_EQ(joined, "a0b1");
}

TEST(Zip, TheAdjacentPairsOfARange)
{
    // The form a byte-pair encoder wants: a range with itself, one along
    const std::vector<int> ids{1, 2, 3};
    std::vector<std::pair<int, int>> pairs;
    for (auto [a, b] : zip(ids, ids | std::views::drop(1)))
        pairs.emplace_back(a, b);
    EXPECT_EQ(pairs, (std::vector<std::pair<int, int>>{{1, 2}, {2, 3}}));
}

TEST(Zip, AnRvalueIsOwnedAndAnLvalueIsReferredTo)
{
    std::vector<int> lvalue{1, 2};
    auto zipped = zip(lvalue, std::vector<int>{10, 20});
    lvalue[0] = 5;
    int sum = 0;
    for (auto [a, b] : zipped)
        sum += a + b;
    EXPECT_EQ(sum, 5 + 2 + 30);
}

TEST(Zip, MoveOnlyElementsWork)
{
    std::vector<std::unique_ptr<int>> owners;
    owners.push_back(std::make_unique<int>(4));
    const std::vector<int> scale{2};
    int product = 0;
    for (auto [p, s] : zip(owners, scale))
        product = *p * s;
    EXPECT_EQ(product, 8);
}

TEST(Zip, WorksWithRangesAlgorithmsAndAViewAdaptor)
{
    const std::vector<int> a{1, 2, 3};
    const std::vector<int> b{1, 5, 3};
    EXPECT_EQ(std::ranges::count_if(zip(a, b), [](auto&& t) { return std::get<0>(t) == std::get<1>(t); }), 2);
    int sum = 0;
    for (const int v : zip(a, b) | std::views::transform([](auto&& t) { return std::get<0>(t) * std::get<1>(t); }))
        sum += v;
    EXPECT_EQ(sum, 1 + 10 + 9);
}

TEST(Zip, ForwardIteratorsCanBeSavedAndCompared)
{
    const std::vector<int> a{1, 2};
    auto z = zip(a, a);
    auto first = z.begin();
    const auto saved = first++;
    EXPECT_EQ(saved, z.begin());
    EXPECT_NE(first, saved);
    EXPECT_EQ(std::get<0>(*first), 2);
    ++first;
    EXPECT_TRUE(first == z.end());
}

TEST(Zip, InputOnlyRangesStayInputOnly)
{
    std::istringstream in("1 2 3");
    auto words = std::views::istream<int>(in);
    const std::vector<int> scale{10, 20};
    auto z = zip(words, scale);
    static_assert(std::ranges::input_range<decltype(z)> && !std::ranges::forward_range<decltype(z)>);
    int sum = 0;
    for (auto [word, s] : z)
        sum += word * s;
    EXPECT_EQ(sum, 10 + 40);
}

TEST(Zip, PrintsThroughAcoutAsARangeOfTuples)
{
    const std::vector<int> a{1, 2};
    const std::vector<std::string> b{"x", "y"};
    EXPECT_EQ([&] {
        std::ostringstream os;
        snicholls::print_to(os, zip(a, b));
        return os.str();
    }(), R"([(1, "x"), (2, "y")])");
}

} // namespace
