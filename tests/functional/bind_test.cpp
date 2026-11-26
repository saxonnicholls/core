//
//  bind_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/bind.hpp>: placeholders, values and nested binds at compile time;
//  forwarding and noexcept at run time.
//

#include <snicholls/core/functional/bind.hpp>

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace {

namespace fn = snicholls::fn;
using namespace snicholls::fn::placeholders;

constexpr int digits(int a, int b, int c) { return a * 100 + b * 10 + c; }
constexpr auto nothrow_sub = [](int a, int b) noexcept { return a - b; };

static_assert(fn::bind(digits, _1, _2, _3)(1, 2, 3) == 123);
static_assert(fn::bind(digits, _3, _2, _1)(1, 2, 3) == 321);
static_assert(fn::bind(digits, _1, _1, _1)(7) == 777);              // repeated
static_assert(fn::bind(digits, 9, _1, 8)(5) == 958);                // values and placeholders
static_assert(fn::bind(digits, 1, 2, 3)() == 123);                  // values only
static_assert(fn::bind(digits, _1, 0, 0)(1, 99, 99) == 100);        // unused arguments are ignored
static_assert(fn::bind(digits, fn::arg<2>, fn::arg<1>, 0)(1, 2) == 210);
static_assert(fn::arg<4>.position == 4 && _9.position == 9);
static_assert(fn::bind(nothrow_sub, fn::bind(nothrow_sub, _1, _2), _3)(10, 3, 2) == 5);   // nested: (10 - 3) - 2
static_assert(fn::bind(std::minus<>{}, _2, _1)(1, 10) == 9);

// A placeholder beyond the arguments given is not invocable.
static_assert(!std::is_invocable_v<decltype(fn::bind(digits, _3, _1, _2)), int, int>);
static_assert(std::is_invocable_v<decltype(fn::bind(digits, _3, _1, _2)), int, int, int>);

static_assert(noexcept(fn::bind(nothrow_sub, _2, _1)(1, 2)));
static_assert(!noexcept(fn::bind(digits, _1, _1, _1)(1)));
static_assert(std::is_trivially_copyable_v<decltype(fn::bind(digits, _1, 2, _2))>);
static_assert(sizeof(fn::bind(nothrow_sub, _1, 2)) == sizeof(int));   // the placeholder and the lambda are empty

TEST(FunctionalBind, APlaceholderForwardsAnRvalue)
{
    auto sink = fn::bind([](std::unique_ptr<int> p, int k) { return *p * k; }, _1, 3);
    EXPECT_EQ(sink(std::make_unique<int>(5)), 15);
}

TEST(FunctionalBind, APlaceholderForwardsAnLvalueByReference)
{
    int x = 1;
    fn::bind([](int& target, int by) { target += by; }, _1, 4)(x);
    EXPECT_EQ(x, 5);
}

TEST(FunctionalBind, BoundValuesAreCopiesThatOutliveTheirSource)
{
    auto label = [] {
        std::string ccy = "EUR";
        return fn::bind(std::plus<>{}, _1, ccy);
    }();
    EXPECT_EQ(label(std::string("12.50 ")), "12.50 EUR");
}

TEST(FunctionalBind, AMoveOnlyBoundValueMovesOutOfAnRvalueBind)
{
    auto once = fn::bind([](std::unique_ptr<int> p) { return *p; }, std::make_unique<int>(42));
    static_assert(!std::is_invocable_v<decltype(once)&>);   // an lvalue bind passes the value as an lvalue
    EXPECT_EQ(std::move(once)(), 42);
}

} // namespace
