//
//  tabulate_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/tabulate.hpp>: a compile-time table of a function of an index.
//

#include <snicholls/core/functional/tabulate.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace {

namespace fn = snicholls::fn;

constexpr std::uint64_t choose(std::size_t n, std::size_t k)
{
    std::uint64_t r = 1;
    for (std::size_t i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return r;
}

constexpr auto squares = fn::tabulate<10>([](std::size_t i) { return i * i; });
constexpr auto row20 = fn::tabulate<21>([](std::size_t k) { return choose(20, k); });

static_assert(squares(0) == 0 && squares(3) == 9 && squares(9) == 81);
static_assert(row20(10) == 184756);
static_assert(row20(0) == 1 && row20(20) == 1);
static_assert(std::is_same_v<decltype(squares(1)), std::size_t>);
static_assert(noexcept(squares(1)));
static_assert(std::is_trivially_copyable_v<decltype(squares)>);
static_assert(sizeof(squares) == 10 * sizeof(std::size_t));     // the table and nothing else
constexpr auto empty_table = fn::tabulate<0>([](std::size_t i) { return i; });   // builds; nothing to look up
static_assert(std::is_trivially_copyable_v<decltype(empty_table)>);
static_assert(noexcept(fn::tabulate<3>([](std::size_t i) noexcept { return i; })));
static_assert(!noexcept(fn::tabulate<3>([](std::size_t i) { return i; })));

TEST(FunctionalTabulate, CallsTheFunctionOnlyWhenBuilt)
{
    int calls = 0;
    const auto t = fn::tabulate<5>([&](std::size_t i) { ++calls; return static_cast<int>(i) * 2; });
    EXPECT_EQ(calls, 5);
    EXPECT_EQ(t(4) + t(4) + t(1), 18);
    EXPECT_EQ(calls, 5);
}

TEST(FunctionalTabulate, AThrowingFunctionThrowsWhenBuilt)
{
    auto build = [] {
        return fn::tabulate<3>([](std::size_t i) -> int {
            if (i == 2) throw std::domain_error("no value at 2");
            return static_cast<int>(i);
        });
    };
    EXPECT_THROW(static_cast<void>(build()), std::domain_error);
}

} // namespace
