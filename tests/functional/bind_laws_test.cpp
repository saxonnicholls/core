//
//  bind_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws for <snicholls/core/functional/bind.hpp>: bind with every placeholder in order is the function, reordering
//  placeholders is flip, nested binds are composition, and binding every argument is a constant.
//

#include <snicholls/core/functional/bind.hpp>
#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/curry.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>

namespace {

namespace fn = snicholls::fn;
using namespace snicholls::fn::placeholders;

constexpr int digits(int a, int b, int c) { return 100 * a + 10 * b + c; }
constexpr auto sq = [](int x) { return x * x; };
constexpr auto inc = [](int x) { return x + 1; };

static_assert(fn::bind(digits, _1, _2, _3)(1, 2, 3) == digits(1, 2, 3));                 // the identity binding
static_assert(fn::bind(digits, _2, _1, _3)(1, 2, 3) == fn::flip(digits)(1, 2, 3));       // swapping is flip
static_assert(fn::bind(digits, _2, _3, _1)(1, 2, 3) == fn::rotate_args(digits)(1, 2, 3)); // rotating
static_assert(fn::bind(digits, 7, _1, _2)(8, 9) == fn::partial(digits, 7)(8, 9));       // leading values are partial
static_assert(fn::bind(digits, _1, _2, 9)(7, 8) == fn::partial_right(digits, 9)(7, 8)); // trailing ones, partial_right
static_assert(fn::bind(digits, 1, 2, 3)() == 123 && fn::bind(digits, 1, 2, 3)(42, "ignored") == 123);   // a constant
static_assert(fn::bind(digits, _1, _1, _1)(4) == 444);                                    // repetition: W combinator
static_assert(fn::bind(inc, fn::bind(sq, _1))(5) == fn::compose(inc, sq)(5));            // nesting is composition
static_assert(fn::bind(std::plus<>{}, fn::bind(sq, _1), fn::bind(sq, _2))(3, 4) == 25);  // S-like fan-out
static_assert(fn::bind(digits, fn::arg<2>, fn::arg<2>, fn::arg<1>)(1, 2) == 221);         // arg<N> is _N
static_assert(fn::bind(digits, _9, _1, _5)(1, 2, 3, 4, 5, 6, 7, 8, 9) == 915);            // the ninth placeholder

static_assert(!std::is_invocable_v<decltype(fn::bind(digits, _1, _2, _3)), int, int>);   // _3 missing
static_assert(std::is_trivially_copyable_v<decltype(fn::bind(digits, 1, _1, _2))>);
constexpr auto nothrow_add = [](int a, int b) noexcept { return a + b; };
constexpr auto bound_nothrow = fn::bind(nothrow_add, _2, 1);
static_assert(noexcept(bound_nothrow(0, 2)));
constexpr auto bound_throwing = fn::bind(digits, _2, 1, 1);
static_assert(!noexcept(bound_throwing(0, 2)));

TEST(FunctionalBindLaws, APlaceholderForwardsAnRvalueAndAValueIsMovedFromAnRvalueClosure)
{
    const auto take = [](std::string a, std::string b) { return a + b; };
    std::string s = "abc";
    auto b = fn::bind(take, std::string("x"), _1);
    EXPECT_EQ(b(std::move(s)), "xabc");
    EXPECT_EQ(b(std::string("y")), "xy");            // the stored value survives an lvalue call
    EXPECT_EQ(std::move(b)(std::string("z")), "xz");
}

} // namespace
