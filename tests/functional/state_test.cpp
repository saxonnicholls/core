//
//  state_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/state.hpp>: Identity, Const, Reader, Writer and State, run at compile time.
//

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/state.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace {
namespace fn = snicholls::fn;
// State: a counter that returns the old value and increments.
constexpr auto tick = fn::mbind(fn::get_state, [](int n) { return fn::mthen(fn::put_state(n + 1), fn::pure<fn::state<>>(n)); });
static_assert(fn::run_state(tick, 5) == std::pair{5, 6});
constexpr auto two = fn::mbind(tick, [](int a) { return fn::fmap([a](int b) { return a * 10 + b; }, tick); });
static_assert(fn::run_state(two, 1) == std::pair{12, 3});
static_assert(fn::exec_state(fn::modify([](int s) { return s * 2; }), 4) == 8);
static_assert(fn::eval_state(fn::gets([](int s) { return s + 1; }), 4) == 5);
static_assert(fn::eval_state(fn::ap(fn::pure<fn::state<>>([](int x) { return -x; }), fn::get_state), 3) == -3);
// Reader
constexpr auto r = fn::mbind(fn::ask, [](int env) { return fn::asks([env](int e) { return env + e; }); });
static_assert(fn::run_reader(r, 2) == 4);
static_assert(fn::run_reader(fn::local([](int e) { return e * 10; }, fn::ask), 2) == 20);
static_assert(fn::run_reader(fn::fmap([](int x) { return x + 1; }, fn::ask), 2) == 3);
static_assert(fn::run_reader(fn::pure<fn::reader<>>(7), 2) == 7);
// Writer
using W = fn::sum_of<int>;
constexpr auto w = fn::mbind(fn::tell(W{2}), [](std::monostate) { return fn::writer<W, int>{5, W{3}}; });
static_assert(w.value == 5 && w.log.value == 5);
static_assert(fn::pure<fn::writer<W>>(1).log.value == 0);
static_assert(fn::fmap([](int x) { return x * 2; }, fn::writer<W, int>{4, W{1}}).value == 8);
// Identity, Const
static_assert(fn::mbind(fn::identity_of<int>{3}, [](int x) { return fn::identity_of<int>{x + 1}; }).value == 4);
static_assert(fn::fmap([](int x) { return x; }, fn::const_of<int>{3}).value == 3);
static_assert(fn::ap(fn::const_of<W>{W{1}}, fn::const_of<W>{W{2}}).value.value == 3);

static_assert(std::is_trivially_copyable_v<decltype(two)>);

TEST(FunctionalState, AWriterKeepsAnAuditTrailInAString)
{
    const auto step = [](int qty) {
        return fn::mbind(fn::tell(std::string("fill ") + std::to_string(qty) + ";"),
                         [qty](std::monostate) { return fn::writer<std::string, int>{qty, std::string()}; });
    };
    const auto total = fn::mbind(step(3), [&](int a) { return fn::fmap([a](int b) { return a + b; }, step(4)); });
    EXPECT_EQ(total.value, 7);
    EXPECT_EQ(fn::exec_writer(total), "fill 3;fill 4;");
    EXPECT_EQ(fn::run_writer(total).first, 7);
}

TEST(FunctionalState, StateThreadsACashBalanceThroughFills)
{
    const auto fill = [](double cost) {
        return fn::mbind(fn::get_state, [cost](double cash) {
            return fn::mthen(fn::put_state(cash - cost), fn::pure<fn::state<>>(cash >= cost));
        });
    };
    const auto day = fn::mbind(fill(40.0), [&](bool a) { return fn::fmap([a](bool b) { return a && b; }, fill(70.0)); });
    const auto [ok, cash] = fn::run_state(day, 100.0);
    EXPECT_FALSE(ok);
    EXPECT_DOUBLE_EQ(cash, -10.0);
    EXPECT_DOUBLE_EQ(fn::exec_state(day, 200.0), 90.0);
}

} // namespace
