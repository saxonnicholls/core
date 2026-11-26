//
//  list_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws for <snicholls/core/functional/list.hpp>, Data.List's equations checked by the compiler: map fusion and
//  identity, filter fusion, reverse and transpose involutions, take / drop / split_at, zip / unzip, concat and the
//  list monad, scans against folds, sort stability, nub idempotence, group and span reassembling the list.
//

#include <snicholls/core/functional/list.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fn = snicholls::fn;

constexpr std::array xs{3, 1, 4, 1, 5, 9, 2, 6};
constexpr auto inc = [](int x) { return x + 1; };
constexpr auto dbl = [](int x) { return 2 * x; };
constexpr auto even = [](int x) { return x % 2 == 0; };
constexpr auto small = [](int x) { return x < 5; };

template <class V, class A>
constexpr bool same(const V& v, const A& a)
{
    if (v.size() != a.size()) return false;
    for (std::size_t i = 0; i < v.size(); ++i)
        if (!(v[i] == a[i])) return false;
    return true;
}

// ---- Functor ------------------------------------------------------------------------------------------------------------

static_assert(fn::map(fn::identity, xs) == xs);                                                  // map id = id
static_assert(fn::map(inc, fn::map(dbl, xs)) == fn::map(fn::compose(inc, dbl), xs));            // map fusion

// ---- filter --------------------------------------------------------------------------------------------------------------

constexpr bool filter_fusion()   // filter p . filter q = filter (\x -> q x && p x)
{
    std::vector<int> twice;
    for (int x : fn::filter(small, xs))
        if (even(x)) twice.push_back(x);
    return same(twice, fn::filter([](int x) { return small(x) && even(x); }, xs));
}
static_assert(filter_fusion());
static_assert(fn::filter([](int) { return true; }, xs).size() == xs.size() && fn::filter([](int) { return false; }, xs).empty());
static_assert(fn::partition(even, xs).first.size() + fn::partition(even, xs).second.size() == xs.size());
static_assert(same(fn::partition(even, xs).first, fn::filter(even, xs)));

// ---- reverse, transpose ----------------------------------------------------------------------------------------------------

static_assert(fn::reverse(fn::reverse(xs)) == xs);
static_assert(fn::reverse(std::array<int, 0>{}).empty() && fn::reverse(std::array{7}) == std::array{7});
constexpr std::array<std::array<int, 3>, 2> m{{{1, 2, 3}, {4, 5, 6}}};
static_assert(fn::transpose(fn::transpose(m)) == m);

// ---- take, drop, split_at, replicate --------------------------------------------------------------------------------------

static_assert(fn::concat(std::array{fn::take<4>(xs), fn::drop<4>(xs)}) == xs);                  // take k ++ drop k = id
static_assert(fn::split_at<3>(xs).first == fn::take<3>(xs) && fn::split_at<3>(xs).second == fn::drop<3>(xs));
static_assert(fn::take<8>(xs) == xs && fn::drop<0>(xs) == xs && fn::drop<8>(xs).empty());
static_assert(fn::take<2>(fn::drop<3>(xs)) == std::array{1, 5});
static_assert(fn::replicate<0>(1).empty() && fn::length(fn::replicate<4>('a')) == 4);

// ---- zip, unzip ----------------------------------------------------------------------------------------------------------

constexpr std::array cs{'a', 'b', 'c'};
static_assert(fn::unzip(fn::zip(fn::take<3>(xs), cs)).first == fn::take<3>(xs));                // unzip . zip = id
static_assert(fn::unzip(fn::zip(fn::take<3>(xs), cs)).second == cs);
static_assert(fn::zip(xs, cs).size() == 3 && fn::zip(xs, std::array<char, 0>{}).empty());       // the shorter wins
static_assert(fn::zip_with(std::plus<>{}, xs, xs) == fn::map(dbl, xs));
static_assert(fn::zip_with([](int a, std::pair<int, char> p) { return a + p.first; }, xs, fn::zip(xs, cs)) == std::array{6, 2, 8});

// ---- concat and the list monad ---------------------------------------------------------------------------------------------

constexpr auto pm = [](int x) { return std::array{x, -x}; };
constexpr auto tw = [](int x) { return std::array{x, x * 10}; };
static_assert(fn::concat_map(pm, std::array{5}) == pm(5));                                       // return x >>= f = f x
static_assert(fn::concat_map([](int x) { return std::array{x}; }, xs) == xs);                    // m >>= return = m
static_assert(fn::concat_map(tw, fn::concat_map(pm, std::array{1, 2})) ==
              fn::concat_map([](int x) { return fn::concat_map(tw, pm(x)); }, std::array{1, 2}));   // associativity
static_assert(fn::concat(fn::map(pm, xs)) == fn::concat_map(pm, xs));

// ---- scans and folds -----------------------------------------------------------------------------------------------------

static_assert(fn::scanl(std::minus<>{}, 100, xs).back() == fn::foldl_of(std::minus<>{}, 100, xs));   // last . scanl = foldl
static_assert(fn::scanr(std::minus<>{}, 0, xs).front() == fn::foldr_of(std::minus<>{}, 0, xs));      // head . scanr = foldr
static_assert(fn::scanl(std::plus<>{}, 0, std::array<int, 0>{}) == std::array{0});
static_assert(fn::foldl1(std::plus<>{}, xs) == fn::sum(xs) && fn::foldr1(std::multiplies<>{}, xs) == fn::product(xs));
static_assert(fn::foldl1(std::minus<>{}, std::array{4}) == 4);                                   // one element: itself
static_assert(fn::maximum(xs) == fn::foldl1([](int a, int b) { return a < b ? b : a; }, xs));
static_assert(fn::fold_map([](int x) { return fn::sum_of<int>{x}; }, xs).value == fn::sum(xs));
static_assert(fn::sum(std::array<int, 0>{}) == 0 && fn::product(std::array<int, 0>{}) == 1);

// ---- sort, nub, group, span ------------------------------------------------------------------------------------------------

static_assert(fn::sort(fn::sort(xs)) == fn::sort(xs));                                           // idempotent
static_assert(fn::sort(fn::reverse(xs)) == fn::sort(xs));                                        // order-free
constexpr std::array<std::pair<int, char>, 4> ties{{{2, 'a'}, {1, 'b'}, {2, 'c'}, {1, 'd'}}};
static_assert(fn::sort_on(fn::fst, ties) == std::array<std::pair<int, char>, 4>{{{1, 'b'}, {1, 'd'}, {2, 'a'}, {2, 'c'}}});   // stable
constexpr bool nub_has_no_repeats()
{
    const auto n = fn::nub(xs);
    for (std::size_t i = 0; i < n.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (n[i] == n[j]) return false;
    return n.size() == 7 && n[0] == 3 && n[1] == 1;   // first occurrences, in order
}
static_assert(nub_has_no_repeats());
static_assert(fn::group(std::array{1, 1, 2, 2, 2, 3})[1].size() == 3);
constexpr bool group_concat_is_identity()
{
    std::vector<int> back;
    for (const auto& g : fn::group(xs))
        for (int x : g) back.push_back(x);
    return same(back, xs);
}
static_assert(group_concat_is_identity());
static_assert(fn::span(small, xs).first.size() + fn::span(small, xs).second.size() == xs.size());
static_assert(same(fn::span(small, xs).first, std::array{3, 1, 4, 1}) && same(fn::break_(small, xs).first, std::array<int, 0>{}));

// ---- searches ------------------------------------------------------------------------------------------------------------

static_assert(fn::find(even, xs) == std::optional<int>(4) && fn::find([](int x) { return x > 100; }, xs) == std::nullopt);
static_assert(fn::find_index(even, xs) == std::optional<std::size_t>(2));
static_assert(fn::elem(9, xs) == fn::any([](int x) { return x == 9; }, xs));
static_assert(fn::all(even, std::array<int, 0>{}) && !fn::any(even, std::array<int, 0>{}));     // vacuous truth
static_assert(fn::lookup('c', fn::zip(cs, xs)) == std::optional<int>(4) && fn::lookup('z', fn::zip(cs, xs)) == std::nullopt);
static_assert(fn::null(std::array<int, 0>{}) && fn::null(std::optional<int>()) && !fn::null(std::optional<int>(1)));

// ---- combinatorics -------------------------------------------------------------------------------------------------------

static_assert(fn::subsequences(std::array{1, 2, 3, 4}).size() == 16);
static_assert(fn::permutations(std::array{1, 2, 3, 4}).size() == 24);
static_assert(fn::intercalate(std::array{0, 0}, std::array{std::array{1}, std::array{2}, std::array{3}}) == std::array{1, 0, 0, 2, 0, 0, 3});

TEST(FunctionalListLaws, OverARunTimeVectorTheSearchesAgreeWithTheArrayForms)
{
    const std::vector<int> v(xs.begin(), xs.end());
    EXPECT_EQ(fn::find(even, v), fn::find(even, xs));
    EXPECT_EQ(fn::sum(v), fn::sum(xs));
    EXPECT_EQ(fn::maximum(v), 9);
    EXPECT_EQ(fn::minimum(v), 1);
    EXPECT_EQ(fn::length(v), xs.size());
    const std::vector<std::pair<std::string, int>> book{{"bid", 99}, {"ask", 101}};
    EXPECT_EQ(fn::lookup(std::string("ask"), book), std::optional<int>(101));
}

} // namespace
