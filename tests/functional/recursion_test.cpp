//
//  recursion_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/recursion.hpp>: cata, ana, hylo, para and the trampoline.
//

#include <snicholls/core/functional/recursion.hpp>

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace {

namespace fn = snicholls::fn;

static_assert(fn::cata([](int x, int acc) { return x - acc; }, 0, std::array{1, 2, 3}) == 2);
constexpr auto countdown = [](int n) { return n > 0 ? std::optional{std::pair{n, n - 1}} : std::nullopt; };
static_assert(fn::ana<10>(countdown, 3).size() == 3 && fn::ana<10>(countdown, 3)[0] == 3);
static_assert(fn::ana<2>(countdown, 3).size() == 2);
static_assert(fn::hylo([](int acc, int x) { return acc + x; }, 0, countdown, 100) == 5050);
static_assert(fn::para([](int x, auto rest, int acc) { return acc + x * static_cast<int>(rest.size()); }, 0, std::array{5, 6, 7}) == 5 * 2 + 6);
constexpr auto collatz = [](std::pair<long, int> s) -> fn::bounce<std::pair<long, int>, int> {
  if (s.first == 1) return fn::done(s.second);
  return fn::more(std::pair{s.first % 2 ? 3 * s.first + 1 : s.first / 2, s.second + 1});
};
static_assert(fn::trampoline(collatz, std::pair{27L, 0}) == 111);
static_assert(fn::ana<5>(countdown, 0).empty());
static_assert(fn::hylo([](int acc, int x) { return acc + x; }, 0, countdown, 0) == 0);

TEST(FunctionalRecursion, ATrampolineRunsAMillionStepsInConstantStack)
{
    const auto count = [](long n) -> fn::bounce<long, long> {
        if (n == 0) return fn::done(1'000'000L);
        return fn::more(n - 1);
    };
    EXPECT_EQ(fn::trampoline(count, 1'000'000L), 1'000'000L);
}

TEST(FunctionalRecursion, CataAndParaOverAVector)
{
    const std::vector<int> v{1, 2, 3};
    EXPECT_EQ(fn::cata([](int x, int acc) { return x + 2 * acc; }, 0, v), 1 + 2 * (2 + 2 * 3));
    EXPECT_EQ(fn::para([](int, auto rest, int acc) { return acc + static_cast<int>(rest.size()); }, 0, v), 3);
}

} // namespace
