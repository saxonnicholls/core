//
//  curry_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/curry.hpp>: curry, curry<N>, uncurry, partial and partial_right at
//  compile time; move-only arguments, reference semantics and noexcept at run time. The compile-fail case for a
//  generic lambda without an arity is tests/compile_fail/fn_curry_generic_lambda.cpp.
//

#include <snicholls/core/functional/curry.hpp>

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

namespace fn = snicholls::fn;

constexpr int digits(int a, int b, int c) { return a * 100 + b * 10 + c; }
constexpr auto generic_digits = [](auto a, auto b, auto c) { return a * 100 + b * 10 + c; };
constexpr auto nothrow_add = [](int a, int b) noexcept { return a + b; };

struct account {
    int balance;
    constexpr int after(int debit) const noexcept { return balance - debit; }
};

// ---- curry: every way of splitting the arguments -----------------------------------------------------------------

static_assert(fn::curry(digits)(1)(2)(3) == 123);
static_assert(fn::curry(digits)(1, 2)(3) == 123);
static_assert(fn::curry(digits)(1)(2, 3) == 123);
static_assert(fn::curry(digits)(1, 2, 3) == 123);
static_assert(fn::curry(&digits)(1)(2)(3) == 123);
static_assert(fn::curry(nothrow_add)(1)(2) == 3);
static_assert(fn::curry(&account::after)(account{100})(30) == 70);    // the object is the first argument
static_assert(fn::curry<3>(generic_digits)(1)(2)(3) == 123);
static_assert(fn::curry<3>(generic_digits)(1, 2)(3) == 123);
static_assert(fn::curry<0>([] { return 7; })() == 7);
static_assert(fn::curry([] { return 7; })() == 7);

// A partial application is a value that can be kept and reused.
constexpr auto hundreds = fn::curry(digits)(1);
static_assert(hundreds(2)(3) == 123 && hundreds(4, 5) == 145);

// Too many arguments is not invocable; a wrong type is not invocable at the last call.
static_assert(!std::is_invocable_v<decltype(fn::curry(digits)), int, int, int, int>);
static_assert(!std::is_invocable_v<decltype(fn::curry(digits)(1)(2)), const char*>);
static_assert(!std::is_invocable_v<decltype(fn::curry(digits))>);                   // no arguments: nothing to do

// noexcept: the final call follows f; a partial application follows the copies.
static_assert(noexcept(fn::curry(nothrow_add)(1)(2)));
static_assert(!noexcept(fn::curry(digits)(1)(2)(3)));
static_assert(noexcept(fn::curry(digits)(1)));
static_assert(!noexcept(fn::curry<2>(std::plus<>{})(std::string("a"))));

// Layout: trivially copyable when f and the arguments are.
static_assert(std::is_trivially_copyable_v<decltype(fn::curry(digits)(1)(2))>);
static_assert(sizeof(fn::curry(nothrow_add)(1)) == sizeof(int));

// ---- uncurry, partial, partial_right -----------------------------------------------------------------------------

constexpr auto hand_curried = [](int a) { return [a](int b) { return [a, b](int c) { return digits(a, b, c); }; }; };
static_assert(fn::uncurry(hand_curried)(1, 2, 3) == 123);
static_assert(fn::uncurry(fn::curry(digits))(1, 2, 3) == 123);       // uncurry undoes curry
static_assert(!std::is_invocable_v<decltype(fn::uncurry(hand_curried))>);
static_assert(fn::partial(digits, 1)(2, 3) == 123);
static_assert(fn::partial(digits, 1, 2)(3) == 123);
static_assert(fn::partial(digits, 1, 2, 3)() == 123);
static_assert(fn::partial(digits)(1, 2, 3) == 123);
static_assert(fn::partial_right(digits, 3)(1, 2) == 123);
static_assert(fn::partial_right(digits, 2, 3)(1) == 123);
static_assert(noexcept(fn::partial(nothrow_add, 1)(2)));
static_assert(!noexcept(fn::partial(digits, 1)(2, 3)));
static_assert(!std::is_invocable_v<decltype(fn::partial(digits, 1)), int>);

TEST(FunctionalCurry, MoveOnlyArgumentsMoveThroughAnRvalueChain)
{
    auto take = fn::curry([](std::unique_ptr<int> p, std::unique_ptr<int> q) { return *p + *q; });
    auto half = std::move(take)(std::make_unique<int>(3));
    static_assert(!std::is_copy_constructible_v<decltype(half)>);
    EXPECT_EQ(std::move(half)(std::make_unique<int>(4)), 7);
}

TEST(FunctionalCurry, StoredArgumentsAreCopiesUnlessPassedWithStdRef)
{
    auto bump = fn::curry([](int& target, int by) { target += by; return target; });
    int x = 1;
    auto on_copy = bump(x);
    EXPECT_EQ(on_copy(10), 11);
    EXPECT_EQ(x, 1);   // the copy changed, not x
    auto on_x = bump(std::ref(x));
    EXPECT_EQ(on_x(10), 11);
    EXPECT_EQ(x, 11);
}

TEST(FunctionalCurry, APartialApplicationOutlivesItsArguments)
{
    auto greet = [] {
        std::string who = "desk";
        return fn::partial([](const std::string& a, const std::string& b) { return a + " " + b; }, who);
    }();
    EXPECT_EQ(greet("open"), "desk open");
}

TEST(FunctionalCurry, PartialRightBindsTrailingStrings)
{
    const auto suffix = fn::partial_right(std::plus<>{}, std::string(" USD"));
    EXPECT_EQ(suffix(std::string("100")), "100 USD");
}

TEST(FunctionalCurry, AThrowingFinalCallPropagates)
{
    auto risky = fn::curry([](int a, int b) -> int {
        if (b == 0) throw std::domain_error("divide by zero");
        return a / b;
    });
    EXPECT_EQ(risky(6)(3), 2);
    EXPECT_THROW(static_cast<void>(risky(6)(0)), std::domain_error);
}

} // namespace
