//
//  recursion_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws for <snicholls/core/functional/recursion.hpp>: cata is foldr, hylo is a fold of an ana without the list,
//  ana stops at its bound, para sees the rest, and a trampoline runs to any depth: classic recursions (digits,
//  Euclid, Ackermann, binary, the McCarthy 91 function) as schemes, at compile time.
//

#include <snicholls/core/functional/fold.hpp>
#include <snicholls/core/functional/recursion.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace {

namespace fn = snicholls::fn;

constexpr std::array xs{1, 2, 3, 4};
constexpr auto minus = [](int x, int acc) { return x - acc; };

// ---- cata is foldr ---------------------------------------------------------------------------------------------------

static_assert(fn::cata(minus, 0, xs) == fn::foldr_of(minus, 0, xs));
static_assert(fn::cata(minus, 9, std::array<int, 0>{}) == 9);

// ---- ana: digits of a number, least significant first -----------------------------------------------------------------

constexpr auto digits = [](long long n) {   // long long: long is 32 bits on Windows
    return n == 0 ? std::nullopt : std::optional{std::pair{static_cast<int>(n % 10), n / 10}}; };
static_assert(fn::ana<20>(digits, 9'876LL).size() == 4 && fn::ana<20>(digits, 9'876LL)[0] == 6);
static_assert(fn::ana<2>(digits, 9'876LL).size() == 2);                                          // the bound stops it
static_assert(fn::ana<5>(digits, 0LL).empty());

// ---- hylo = cata . ana, fused ------------------------------------------------------------------------------------------

constexpr int digit_sum_hylo(long long n) { return fn::hylo([](int acc, int d) { return acc + d; }, 0, digits, n); }
constexpr int digit_sum_two_pass(long long n)
{
    const auto ds = fn::ana<20>(digits, n);
    int s = 0;
    for (int d : ds) s += d;
    return s;
}
static_assert(digit_sum_hylo(9'876'543'210LL) == 45 && digit_sum_hylo(9'876'543'210LL) == digit_sum_two_pass(9'876'543'210LL));

// Binary representation by hylo: the bits unfolded, folded back into a decimal-looking number.
constexpr auto bits = [](unsigned n) { return n == 0 ? std::nullopt : std::optional{std::pair{n % 2, n / 2}}; };
static_assert(fn::ana<32>(bits, 6u).size() == 3);
static_assert(fn::hylo([](unsigned acc, unsigned) { return acc + 1; }, 0u, bits, 255u) == 8);   // the bit length

// ---- para: each element with what follows it ---------------------------------------------------------------------------

// Count the inversions: for each x, how many later elements are smaller.
constexpr std::array perm{3, 1, 4, 2};
static_assert(fn::para([](int x, auto rest, int acc) {
                  for (int y : rest) acc += y < x ? 1 : 0;
                  return acc;
              }, 0, perm) == 3);

// ---- trampoline ----------------------------------------------------------------------------------------------------------

constexpr auto gcd = [](std::pair<long, long> s) -> fn::bounce<std::pair<long, long>, long> {
    if (s.second == 0) return fn::done(s.first);
    return fn::more(std::pair{s.second, s.first % s.second});
};
static_assert(fn::trampoline(gcd, std::pair{1071L, 462L}) == 21 && fn::trampoline(gcd, std::pair{7L, 0L}) == 7);

// The McCarthy 91 function, its two recursive calls flattened into a counter of pending calls.
constexpr auto mc91 = [](std::pair<int, int> s) -> fn::bounce<std::pair<int, int>, int> {
    auto [n, pending] = s;
    if (n > 100 && pending == 1) return fn::done(n - 10);
    if (n > 100) return fn::more(std::pair{n - 10, pending - 1});
    return fn::more(std::pair{n + 11, pending + 1});
};
constexpr int mccarthy(int n) { return fn::trampoline(mc91, std::pair{n, 1}); }
static_assert(mccarthy(1) == 91 && mccarthy(50) == 91 && mccarthy(100) == 91 && mccarthy(101) == 91 && mccarthy(150) == 140);

TEST(FunctionalRecursionLaws, ATrampolineGoesAMillionDeepWithoutTheStack)
{
    const auto countdown = [](long n) -> fn::bounce<long, long> {
        if (n == 0) return fn::done(-1L);
        return fn::more(n - 1);
    };
    EXPECT_EQ(fn::trampoline(countdown, 1'000'000L), -1);
}

TEST(FunctionalRecursionLaws, CataOverAVectorAgreesWithTheArray)
{
    const std::vector<int> v(xs.begin(), xs.end());
    EXPECT_EQ(fn::cata(minus, 0, v), fn::cata(minus, 0, xs));
}

} // namespace
