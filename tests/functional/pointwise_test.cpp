//
//  pointwise_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/pointwise.hpp>: functions added, scaled, compared and combined point by point.
//

#include <snicholls/core/functional/pointwise.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

namespace fn = snicholls::fn;

constexpr fn::pointwise sq{[](int x) { return x * x; }};
constexpr fn::pointwise id{[](int x) { return x; }};
constexpr fn::pointwise even{[](int x) { return x % 2 == 0; }};
constexpr fn::pointwise big{[](int x) { return x > 10; }};

// ---- arithmetic ------------------------------------------------------------------------------------------------------

static_assert((sq + id)(3) == 12 && (sq - id)(3) == 6 && (sq * id)(3) == 27 && (sq / id)(3) == 3);
static_assert((-sq)(3) == -9 && (2 * sq + 1)(3) == 19 && (sq - 1)(4) == 15 && (100 / id)(4) == 25);
static_assert(((sq + id) * (sq - id))(3) == 72);
static_assert(std::is_same_v<decltype((sq + 0.5)(1)), double>);

// Many arguments: each operand is called with them all.
constexpr fn::pointwise first{[](int a, int) { return a; }};
constexpr fn::pointwise second{[](int, int b) { return b; }};
static_assert((first * 10 + second)(4, 2) == 42);

// ---- comparisons and logic -------------------------------------------------------------------------------------------

static_assert((sq > id)(3) && !(sq > id)(1) && (sq >= id)(1) && (sq < 10)(3) && !(sq <= 8)(3));
static_assert((even && big)(12) && !(even && big)(8) && (even || big)(13) && !(even || big)(9));
static_assert((!even)(3) && !(!even)(4));
static_assert(((sq > 50) || (even && !big))(8) && !((sq > 50) || (even && !big))(7));
static_assert(std::is_same_v<decltype((sq < id)(1)), bool>);

// ---- the contract ----------------------------------------------------------------------------------------------------

static_assert(sizeof(sq + id) == 1);   // stateless operands take no storage
static_assert(std::is_trivially_copyable_v<decltype((sq + id) * (sq - id))>);
static_assert(std::is_trivially_copyable_v<decltype(2.5 * sq)> && sizeof(2.5 * sq) == sizeof(double));
constexpr fn::pointwise nothrow_sq{[](int x) noexcept { return x * x; }};
constexpr auto nothrow_sum = nothrow_sq + nothrow_sq;
constexpr auto nothrow_test = (nothrow_sq > 1) && !(nothrow_sq > 50);
constexpr auto mixed_sum = sq + nothrow_sq;
static_assert(noexcept(nothrow_sum(1)) && noexcept(nothrow_test(1)));
static_assert(!noexcept(mixed_sum(1)));
static_assert(!std::is_invocable_v<decltype(sq + id), const char*>);   // the expression decides what a call accepts

// `==` is not lifted: comparing two pointwise functions is not a new function.
template <class A, class B>
concept lifts_equality = requires(A a, B b) { a == b; };
static_assert(!lifts_equality<decltype(sq), decltype(id)>);

TEST(FunctionalPointwise, LogicalOperatorsShortCircuit)
{
    int right_calls = 0;
    const fn::pointwise counted{[&](int) { ++right_calls; return true; }};
    EXPECT_FALSE((even && counted)(3));
    EXPECT_TRUE((even || counted)(4));
    EXPECT_EQ(right_calls, 0);
    EXPECT_TRUE((even && counted)(4));
    EXPECT_EQ(right_calls, 1);
}

TEST(FunctionalPointwise, ArgumentsAreNotMovedFromTwice)
{
    const fn::pointwise length{[](std::string s) { return s.size(); }};
    EXPECT_EQ((length + length)(std::string("abcd")), 8u);
}

TEST(FunctionalPointwise, AThrowingOperandPropagates)
{
    const fn::pointwise boom{[](int) -> int { throw std::range_error("x"); }};
    EXPECT_THROW(static_cast<void>((sq + boom)(1)), std::range_error);
}

} // namespace
