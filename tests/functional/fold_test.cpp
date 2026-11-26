//
//  fold_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/fold.hpp>: foldl, foldr and scan over packs; foldl_of and foldr_of over
//  ranges and tuples; unfold and iterate_n; at compile time first.
//

#include <snicholls/core/functional/fold.hpp>

#include <gtest/gtest.h>

#include <array>
#include <list>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fn = snicholls::fn;

constexpr auto minus = [](auto a, auto b) noexcept { return a - b; };
constexpr auto throwing_minus = [](int a, int b) { return a - b; };
constexpr auto dbl = [](int x) noexcept { return 2 * x; };
constexpr auto fib_step = [](const std::pair<long, long>& s) {
    return std::pair{s.first, std::pair{s.second, s.first + s.second}};
};

// ---- packs -----------------------------------------------------------------------------------------------------------

static_assert(fn::foldl(minus, 10, 1, 2, 3) == ((10 - 1) - 2) - 3);
static_assert(fn::foldr(minus, 10, 1, 2, 3) == 1 - (2 - (3 - 10)));
static_assert(fn::foldl(minus, 10) == 10 && fn::foldr(minus, 10) == 10);   // empty pack: the start
static_assert(fn::foldl(minus, 10, 0.5) == 9.5);                           // the accumulator changes type
static_assert(std::is_same_v<decltype(fn::foldl(minus, 10, 0.5)), double>);
static_assert(fn::scan(minus, 10, 1, 2, 3) == std::array{10, 9, 7, 4});
static_assert(fn::scan(minus, 10) == std::array{10});
static_assert(fn::scan(minus, 10.0, 1, 2) == std::array{10.0, 9.0, 7.0});

// ---- sequences -------------------------------------------------------------------------------------------------------

static_assert(fn::foldl_of(minus, 10, std::array{1, 2, 3}) == 4);
static_assert(fn::foldr_of(minus, 10, std::array{1, 2, 3}) == 1 - (2 - (3 - 10)));
static_assert(fn::foldl_of(minus, 10, std::array<int, 0>{}) == 10);
static_assert(fn::foldl_of(minus, 10, std::tuple{1, 2.5}) == 6.5);         // a tuple is folded as its pack
static_assert(fn::foldr_of(minus, 10, std::pair{1, 2}) == 1 - (2 - 10));
static_assert(fn::foldl_of(minus, 0, std::tuple<>{}) == 0);

// ---- unfold and iterate_n --------------------------------------------------------------------------------------------

static_assert(fn::unfold<8>(std::pair{0L, 1L}, fib_step) == std::array{0L, 1L, 1L, 2L, 3L, 5L, 8L, 13L});
static_assert(fn::unfold<0>(std::pair{0L, 1L}, fib_step).empty());
static_assert(fn::unfold<3>(1, [](int s) { return std::tuple{s * 10, s + 1}; }) == std::array{10, 20, 30});
static_assert(fn::iterate_n(dbl, 10)(1) == 1024);
static_assert(fn::iterate_n(dbl, 0)(5) == 5);
static_assert(fn::iterate_n(dbl, 1, 3) == 8);

// ---- noexcept --------------------------------------------------------------------------------------------------------

static_assert(noexcept(fn::foldl(minus, 1, 2, 3)));
static_assert(!noexcept(fn::foldl(throwing_minus, 1, 2, 3)));
static_assert(noexcept(fn::foldr(minus, 1, 2, 3)));
static_assert(noexcept(fn::scan(minus, 10, 1, 2, 3)));
static_assert(!noexcept(fn::scan(throwing_minus, 10, 1, 2, 3)));
static_assert(noexcept(fn::foldl_of(minus, 10, std::declval<const std::array<int, 3>&>())));
static_assert(!noexcept(fn::foldl_of(throwing_minus, 10, std::declval<const std::array<int, 3>&>())));
static_assert(noexcept(fn::iterate_n(dbl, 1, 3)));
static_assert(noexcept(fn::iterate_n(dbl, 3)(1)));
static_assert(noexcept(fn::unfold<2>(std::pair{0L, 1L}, [](const std::pair<long, long>& s) noexcept {
    return std::pair{s.first, s};
})));
static_assert(!noexcept(fn::unfold<2>(std::pair{0L, 1L}, fib_step)));

TEST(FunctionalFold, RangesFoldAtRunTime)
{
    const std::vector<int> v{1, 2, 3};
    const std::list<int> l{1, 2, 3};
    EXPECT_EQ(fn::foldl_of(minus, 10, v), 4);
    EXPECT_EQ(fn::foldr_of(minus, 10, l), 1 - (2 - (3 - 10)));
    EXPECT_EQ(fn::foldl_of(minus, 10, std::vector<int>{}), 10);
}

TEST(FunctionalFold, TheAccumulatorIsMovedFromStepToStep)
{
    const std::vector<std::string> words{"bid", "ask", "mid"};
    const auto joined = fn::foldl_of([](std::string acc, const std::string& w) { return std::move(acc) + w + ";"; },
                                     std::string(), words);
    EXPECT_EQ(joined, "bid;ask;mid;");
}

TEST(FunctionalFold, FoldrBuildsFromTheRight)
{
    const auto s = fn::foldr([](const char* w, std::string acc) { return w + ("(" + acc + ")"); }, std::string("z"),
                             "a", "b");
    EXPECT_EQ(s, "a(b(z))");
}

TEST(FunctionalFold, UnfoldCallsTheStepExactlyNTimes)
{
    int calls = 0;
    const auto v = fn::unfold<4>(0, [&](int s) { ++calls; return std::pair{s * s, s + 1}; });
    EXPECT_EQ(v, (std::array{0, 1, 4, 9}));
    EXPECT_EQ(calls, 4);
}

} // namespace
