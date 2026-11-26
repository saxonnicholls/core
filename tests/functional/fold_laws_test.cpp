//
//  fold_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws for <snicholls/core/functional/fold.hpp>: empty and single folds, the last of a scan is the fold, foldr of
//  cons rebuilds the list, fold fusion, the duality of foldl and foldr for associative operations, unfold and
//  iterate_n at their boundaries, all at compile time.
//

#include <snicholls/core/functional/fold.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <functional>
#include <list>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace fn = snicholls::fn;

constexpr auto plus = [](auto a, auto b) { return a + b; };
constexpr auto minus = [](auto a, auto b) { return a - b; };

// ---- boundaries -------------------------------------------------------------------------------------------------------

static_assert(fn::foldl(minus, 7) == 7 && fn::foldr(minus, 7) == 7);                     // no elements: the seed
static_assert(fn::foldl(minus, 7, 2) == 5 && fn::foldr(minus, 7, 2) == -5);              // one element
static_assert(fn::scan(plus, 7) == std::array{7});
static_assert(fn::foldl_of(plus, 0, std::array<int, 0>{}) == 0 && fn::foldr_of(plus, 0, std::tuple<>{}) == 0);
static_assert(fn::unfold<0>(0, [](int s) { return std::pair{s, s + 1}; }).empty());
static_assert(fn::iterate_n([](int x) { return x * 2; }, 0)(5) == 5);                     // zero times is identity

// ---- laws ---------------------------------------------------------------------------------------------------------------

// The last element of a scan is the fold.
static_assert(fn::scan(minus, 100, 1, 2, 3).back() == fn::foldl(minus, 100, 1, 2, 3));

// foldl over an associative operation with its identity equals foldr.
static_assert(fn::foldl(plus, 0, 1, 2, 3, 4) == fn::foldr(plus, 0, 1, 2, 3, 4));
static_assert(fn::foldl(minus, 0, 1, 2, 3) != fn::foldr(minus, 0, 1, 2, 3));              // not for subtraction

// Fold fusion: h(foldl(f, z, xs)) == foldl(g, h(z), xs) when h(f(a, x)) == g(h(a), x). Here h doubles.
constexpr auto h = [](int a) { return 2 * a; };
static_assert(h(fn::foldl_of(plus, 1, std::array{1, 2, 3})) ==
              fn::foldl_of([](int a, int x) { return a + 2 * x; }, h(1), std::array{1, 2, 3}));

// foldr with cons and nil rebuilds the list (here into a number read digit by digit from the right).
constexpr auto cons_digit = [](int d, std::pair<int, int> acc) { return std::pair{acc.first + d * acc.second, acc.second * 10}; };
static_assert(fn::foldr_of(cons_digit, std::pair{0, 1}, std::array{1, 2, 3}).first == 123);

// Over a tuple each step may change type: a string-free heterogeneous sum.
static_assert(fn::foldl_of(plus, 0, std::tuple{1, 2.5, 3.0f}) == 6.5);
static_assert(fn::foldr_of([](auto x, auto acc) { return acc * 10 + x; }, 0, std::tuple{1, 2L, short{3}}) == 321);

// iterate_n composes: n then m applications is n + m.
constexpr auto twice = [](long x) { return 2 * x; };
static_assert(fn::iterate_n(twice, 3)(fn::iterate_n(twice, 4)(1L)) == fn::iterate_n(twice, 7)(1L));
static_assert(fn::iterate_n(twice, 1L, 62) == (1L << 62));

// unfold: Fibonacci and the powers of two, as std::arrays.
constexpr auto fibs = fn::unfold<10>(std::pair{0, 1}, [](std::pair<int, int> s) { return std::pair{s.first, std::pair{s.second, s.first + s.second}}; });
static_assert(fibs == std::array{0, 1, 1, 2, 3, 5, 8, 13, 21, 34});

TEST(FunctionalFoldLaws, RangesOfEveryKindFoldTheSame)
{
    const std::vector<int> v{1, 2, 3, 4};
    const std::list<int> l{1, 2, 3, 4};
    EXPECT_EQ(fn::foldl_of(minus, 0, v), fn::foldl_of(minus, 0, l));
    EXPECT_EQ(fn::foldr_of(minus, 0, v), fn::foldr_of(minus, 0, l));
    EXPECT_EQ(fn::foldr_of(minus, 0, v), 1 - (2 - (3 - (4 - 0))));
}

TEST(FunctionalFoldLaws, FoldLeftBuildsAStringInOrderAndFoldRightInReverse)
{
    const std::vector<std::string> words{"a", "b", "c"};
    EXPECT_EQ(fn::foldl_of(plus, std::string(), words), "abc");
    EXPECT_EQ(fn::foldr_of([](const std::string& w, const std::string& acc) { return acc + w; }, std::string(), words), "cba");
}

} // namespace
