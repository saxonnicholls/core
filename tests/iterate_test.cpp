//
//  iterate_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/iterate.hpp>: Wolfram's Table, Nest, NestList, NestWhile, FixedPoint
//  and FoldList, at compile time.
//

#include <snicholls/core/iterate.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>

namespace {

constexpr auto twice = [](int x) { return 2 * x; };

static_assert(snicholls::table<4>([](std::size_t i) { return i * i; }) == std::array<std::size_t, 4>{0, 1, 4, 9});
static_assert(snicholls::nest(twice, 1, 10) == 1024);
static_assert(snicholls::nest_list<3>(twice, 1) == std::array{1, 2, 4, 8});
static_assert(snicholls::nest_while(twice, 1, [](int x) { return x < 100; }, 50) == 128);
static_assert(snicholls::nest_while(twice, 1, [](int) { return true; }, 3) == 8);   // the limit
// FoldList[Plus, 0, {1, 2, 3}] = {0, 1, 3, 6}
static_assert(snicholls::fold_list([](int a, int b) { return a + b; }, 0, std::array{1, 2, 3}) == std::array{0, 1, 3, 6});
// FixedPoint of Collatz-like halving on integers reaches 0
static_assert(snicholls::fixed_point([](int x) { return x / 2; }, 100, 64).value == 0);
static_assert(snicholls::fixed_point([](int x) { return x / 2; }, 100, 64).converged);
static_assert(!snicholls::fixed_point([](int x) { return x + 1; }, 0, 5).converged);

TEST(Iterate, FixedPointWithAToleranceFindsTheSquareRootOfTwo)
{
    const auto r = snicholls::fixed_point([](double x) { return (x + 2 / x) / 2; }, 1.0, 50,
                                          [](double a, double b) { return std::abs(a - b) < 1e-15; });
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.value, std::sqrt(2.0), 1e-15);
    EXPECT_LT(r.iterations, 10u);
}

} // namespace
