//
//  tabulate_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws for <snicholls/core/functional/tabulate.hpp>: a table agrees with its function at every index, tabulating a
//  table changes nothing, and dynamic programming reads earlier entries of a table the compiler built.
//

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/tabulate.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

namespace fn = snicholls::fn;

constexpr std::uint64_t catalan_direct(std::size_t n)
{
    std::uint64_t c = 1;   // C(n) = C(2n, n) / (n + 1), built incrementally
    for (std::size_t k = 0; k < n; ++k) c = c * 2 * (2 * k + 1) / (k + 2);
    return c;
}

constexpr auto catalan = fn::tabulate<30>(catalan_direct);

template <std::size_t N, class T, class F>
constexpr bool agrees(const T& t, F f)
{
    for (std::size_t i = 0; i < N; ++i)
        if (t(i) != f(i)) return false;
    return true;
}

static_assert(agrees<30>(catalan, catalan_direct));                                  // tabulate f = f on its domain
static_assert(agrees<30>(fn::tabulate<30>(catalan), catalan_direct));              // idempotent
static_assert(catalan(0) == 1 && catalan(1) == 1 && catalan(5) == 42 && catalan(29) == 1'002'242'216'651'368ULL);
static_assert(fn::compose([](std::uint64_t c) { return c % 7; }, catalan)(5) == 0);   // a table composes like a function

// Dynamic programming: partitions of n, p(n) by Euler's pentagonal recurrence over a table built row by row.
constexpr std::array<std::int64_t, 61> partitions = [] {
    std::array<std::int64_t, 61> p{};
    p[0] = 1;
    for (int n = 1; n <= 60; ++n)
        for (int k = 1;; ++k) {
            const int g1 = k * (3 * k - 1) / 2, g2 = k * (3 * k + 1) / 2;
            if (g1 > n) break;
            const std::int64_t sign = k % 2 == 1 ? 1 : -1;
            p[static_cast<std::size_t>(n)] += sign * p[static_cast<std::size_t>(n - g1)];
            if (g2 <= n) p[static_cast<std::size_t>(n)] += sign * p[static_cast<std::size_t>(n - g2)];
        }
    return p;
}();
constexpr auto p = fn::tabulate<61>([](std::size_t n) { return partitions[n]; });
static_assert(p(5) == 7 && p(10) == 42 && p(60) == 966'467);

TEST(FunctionalTabulateLaws, ATableIsReadAtRunTimeWithoutCallingTheFunction)
{
    volatile std::size_t i = 12;
    EXPECT_EQ(catalan(i), 208'012u);
    EXPECT_EQ(p(i), 77);
}

} // namespace
