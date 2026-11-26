//
//  list_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/list.hpp>: Data.List and Foldable over std::array and ranges, at compile time.
//

#include <snicholls/core/functional/list.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <list>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
namespace fn = snicholls::fn;
constexpr std::array xs{3, 1, 4, 1, 5};
template <class V> constexpr bool eq(const V& v, std::initializer_list<int> l) { return std::equal(v.begin(), v.end(), l.begin(), l.end()); }
static_assert(fn::map([](int x) { return x * 2; }, xs) == std::array{6, 2, 8, 2, 10});
static_assert(eq(fn::filter([](int x) { return x > 2; }, xs), {3, 4, 5}));
static_assert(fn::foldl1([](int a, int b) { return a - b; }, xs) == 3 - 1 - 4 - 1 - 5);
static_assert(fn::foldr1([](int a, int b) { return a - b; }, std::array{1, 2, 3}) == 1 - (2 - 3));
static_assert(fn::scanl(std::plus<>{}, 0, std::array{1, 2, 3}) == std::array{0, 1, 3, 6});
static_assert(fn::scanr(std::plus<>{}, 0, std::array{1, 2, 3}) == std::array{6, 5, 3, 0});
static_assert(fn::zip(std::array{1, 2}, std::array{'a', 'b', 'c'}) == std::array{std::pair{1, 'a'}, std::pair{2, 'b'}});
static_assert(fn::zip3(std::array{1}, std::array{2}, std::array{3})[0] == std::tuple{1, 2, 3});
static_assert(fn::unzip(std::array{std::pair{1, 'a'}}).second == std::array{'a'});
static_assert(fn::concat(std::array{std::array{1, 2}, std::array{3, 4}}) == std::array{1, 2, 3, 4});
static_assert(fn::concat_map([](int x) { return std::array{x, -x}; }, std::array{1, 2}) == std::array{1, -1, 2, -2});
static_assert(fn::take<2>(xs) == std::array{3, 1} && fn::drop<3>(xs) == std::array{1, 5});
static_assert(fn::split_at<1>(xs).first == std::array{3});
static_assert(eq(fn::span([](int x) { return x > 2; }, xs).first, {3}) && eq(fn::break_([](int x) { return x > 3; }, xs).first, {3, 1}));
static_assert(fn::reverse(std::array{1, 2, 3}) == std::array{3, 2, 1});
static_assert(fn::intersperse(0, std::array{1, 2, 3}) == std::array{1, 0, 2, 0, 3});
static_assert(fn::intercalate(std::array{0}, std::array{std::array{1, 2}, std::array{3, 4}}) == std::array{1, 2, 0, 3, 4});
static_assert(fn::group(std::array{1, 1, 2, 3, 3}).size() == 3);
static_assert(eq(fn::partition([](int x) { return x % 2; }, xs).first, {3, 1, 1, 5}));
static_assert(eq(fn::nub(xs), {3, 1, 4, 5}));
static_assert(fn::sort(xs) == std::array{1, 1, 3, 4, 5});
static_assert(fn::sort_on([](int x) { return -x; }, xs) == std::array{5, 4, 3, 1, 1});
static_assert(fn::lookup(2, std::array{std::pair{1, 'a'}, std::pair{2, 'b'}}) == std::optional<char>{'b'});
static_assert(fn::elem(4, xs) && !fn::elem(9, xs));
static_assert(fn::find([](int x) { return x > 3; }, xs) == std::optional<int>{4});
static_assert(fn::find_index([](int x) { return x > 3; }, xs) == std::optional<std::size_t>{2});
static_assert(fn::all([](int x) { return x > 0; }, xs) && !fn::any([](int x) { return x > 9; }, xs));
static_assert(fn::replicate<3>('x') == std::array{'x', 'x', 'x'});
static_assert(fn::transpose(std::array{std::array{1, 2, 3}, std::array{4, 5, 6}}) == std::array{std::array{1, 4}, std::array{2, 5}, std::array{3, 6}});
static_assert(fn::subsequences(std::array{1, 2, 3}).size() == 8 && eq(fn::subsequences(std::array{1, 2, 3})[5], {1, 3}));
static_assert(fn::permutations(std::array{1, 2, 3}).size() == 6);
static_assert(fn::length(xs) == 5 && fn::length(std::tuple{1, 'a'}) == 2 && fn::length(std::optional<int>{}) == 0 && !fn::null(xs));
static_assert(fn::sum(xs) == 14 && fn::product(xs) == 60 && fn::maximum(xs) == 5 && fn::minimum(xs) == 1);
static_assert(fn::fold_map([](int x) { return fn::max_of<int>{x}; }, xs).value == 5);
static_assert(fn::fst(std::pair{1, 'a'}) == 1 && fn::snd(std::pair{1, 'a'}) == 'a');
static_assert(fn::map_tuple([](auto x) { return x + 1; }, std::tuple{1, 2.5}) == std::tuple{2, 3.5});
static_assert(fn::take<0>(xs).empty() && fn::drop<5>(xs).empty());
static_assert(fn::intersperse(0, std::array<int, 0>{}).empty());
static_assert(fn::nub(std::array<int, 0>{}).empty());
static_assert(fn::sort_on([](const std::pair<int, char>& p) { return p.first; },
                          std::array{std::pair{2, 'a'}, std::pair{1, 'b'}, std::pair{2, 'c'}})[2].second == 'c');   // stable

TEST(FunctionalList, FoldsAndSearchesTakeRunTimeRanges)
{
    const std::vector<double> fills{10.5, 20.25, 5.0};
    const std::list<int> ids{7, 9};
    EXPECT_DOUBLE_EQ(fn::sum(fills), 35.75);
    EXPECT_DOUBLE_EQ(fn::maximum(fills), 20.25);
    EXPECT_EQ(fn::length(ids), 2U);
    EXPECT_TRUE(fn::elem(9, ids));
    EXPECT_EQ(fn::find_index([](double f) { return f < 6; }, fills), std::optional<std::size_t>(2));
}

TEST(FunctionalList, StringsSortAndGroup)
{
    const std::array<std::string, 4> desks{"rates", "fx", "fx", "credit"};
    EXPECT_EQ(fn::sort(desks)[0], "credit");
    EXPECT_EQ(fn::group(fn::sort(desks)).size(), 3U);
    EXPECT_EQ(fn::nub(desks).size(), 3U);
}

} // namespace
