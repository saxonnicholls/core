//
//  curry_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws for <snicholls/core/functional/curry.hpp>: curry and uncurry are inverse, every grouping of the arguments
//  gives the same call, partial and partial_right agree with curry, sections are partial applications, and a
//  partly applied function is an ordinary value that composes.
//

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/curry.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace {

namespace fn = snicholls::fn;

constexpr int poly(int a, int b, int c, int d) { return 1000 * a + 100 * b + 10 * c + d; }
constexpr auto cpoly = fn::curry(poly);

// ---- every grouping of the four arguments is the same call ------------------------------------------------------------

static_assert(cpoly(1)(2)(3)(4) == 1234);
static_assert(cpoly(1, 2)(3)(4) == 1234 && cpoly(1)(2, 3)(4) == 1234 && cpoly(1)(2)(3, 4) == 1234);
static_assert(cpoly(1, 2)(3, 4) == 1234 && cpoly(1, 2, 3)(4) == 1234 && cpoly(1)(2, 3, 4) == 1234);
static_assert(cpoly(1, 2, 3, 4) == 1234);

// ---- curry and uncurry are inverse ------------------------------------------------------------------------------------

static_assert(fn::uncurry(cpoly)(1, 2, 3, 4) == poly(1, 2, 3, 4));                       // uncurry . curry = id
constexpr auto hand = [](int a) { return [a](int b) { return [a, b](int c) { return a * 100 + b * 10 + c; }; }; };
static_assert(fn::curry<3>(fn::uncurry(hand))(1)(2)(3) == hand(1)(2)(3));               // curry . uncurry = id

// ---- partial application agrees with currying --------------------------------------------------------------------------

static_assert(fn::partial(poly, 1)(2, 3, 4) == cpoly(1)(2, 3, 4));
static_assert(fn::partial(fn::partial(poly, 1), 2)(3, 4) == fn::partial(poly, 1, 2)(3, 4));   // partials nest
static_assert(fn::partial_right(poly, 4)(1, 2, 3) == 1234 && fn::partial_right(fn::partial_right(poly, 4), 3)(1, 2) == 1234);
static_assert(fn::partial_right(fn::partial(poly, 1), 4)(2, 3) == 1234);                 // both ends bound

// ---- sections ------------------------------------------------------------------------------------------------------------

static_assert(fn::lsection(std::minus<>{}, 10)(3) == 7);   // (10 -)
static_assert(fn::rsection(std::minus<>{}, 10)(3) == -7);  // (- 10)

// ---- a partly applied function is a value: it composes -------------------------------------------------------------------

constexpr double convert(double rate, double x) { return x * rate; }
constexpr double add(double spread, double x) { return x + spread; }
constexpr auto quote = fn::pipe(fn::curry(convert)(2.0), fn::curry(add)(0.5));
static_assert(quote(10.0) == 20.5);
static_assert(std::is_trivially_copyable_v<decltype(quote)>);   // two function pointers and two doubles

// ---- generic lambdas and member pointers ---------------------------------------------------------------------------------

constexpr auto generic = [](auto a, auto b, auto c) { return a + b + c; };
static_assert(fn::curry<3>(generic)(1)(2.5)(3) == 6.5);
struct account {
    int balance;
    constexpr int after(int spend) const { return balance - spend; }
};
static_assert(fn::curry(&account::after)(account{100})(1) == 99);

TEST(FunctionalCurryLaws, BoundArgumentsAreCopiedOnceAndMovedFromAnRvalueClosure)
{
    const auto join = [](std::string a, std::string b) { return a + b; };
    auto hello = fn::curry<2>(join)(std::string("hello, "));
    EXPECT_EQ(hello(std::string("world")), "hello, world");
    EXPECT_EQ(hello(std::string("there")), "hello, there");   // the bound string survives a call through an lvalue
    EXPECT_EQ(std::move(hello)(std::string("moved")), "hello, moved");
}

TEST(FunctionalCurryLaws, AMoveOnlyArgumentCanBeBoundAndMovedThrough)
{
    const auto deref = [](std::unique_ptr<int> p, int add) { return *p + add; };
    auto bound = fn::partial(deref, std::make_unique<int>(40));
    EXPECT_EQ(std::move(bound)(2), 42);
}

} // namespace
