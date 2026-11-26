//
//  state_rpn_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/state.hpp>: the State monad as a counter handing out fresh ids and as the
//  stack of a reverse-Polish calculator, Reader for pricing configuration, and Writer for an audit summary kept in a
//  monoid. Pure values all of them, run by the compiler in static_asserts. Exits non-zero if a result is not the one
//  documented.
//

#include <snicholls/core/functional/fold.hpp>
#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/state.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

namespace fn = snicholls::fn;

//! [fresh]
// fresh :: State Int Int  -- the next id, and the counter moved on.
constexpr auto fresh = fn::mbind(fn::get_state, [](int n) { return fn::mthen(fn::put_state(n + 1), fn::pure<fn::state<>>(n)); });

// Three orders labelled with fresh ids, threading the counter without a variable in sight.
constexpr auto three_ids = fn::mbind(fresh, [](int a) {
    return fn::mbind(fresh, [a](int b) { return fn::fmap([a, b](int c) { return std::array{a, b, c}; }, fresh); });
});
//! [fresh]

//! [rpn]
// The machine's state: a small fixed stack.
struct stack {
    std::array<long, 8> v{};
    std::size_t n = 0;
};

// push x :: State Stack (); pop :: State Stack Long.
constexpr auto push(long x)
{
    return fn::modify([x](stack s) {
        s.v[s.n++] = x;
        return s;
    });
}
constexpr auto pop = fn::state{[](stack s) {
    const long top = s.v[--s.n];
    return std::pair{top, s};
}};

// binary op = do { b <- pop; a <- pop; push (a `op` b) }
constexpr auto binary(char op)
{
    return fn::mbind(pop, [op](long b) {
        return fn::mbind(pop, [op, b](long a) {
            return push(op == '+' ? a + b : op == '-' ? a - b : op == '*' ? a * b : a / b);
        });
    });
}

constexpr long to_number(std::string_view t)
{
    long n = 0;
    for (char c : t) n = n * 10 + (c - '0');
    return n;
}

// One token is one State program; the calculator is a left fold of their runs over the tokens.
constexpr stack step(stack s, std::string_view token)
{
    const bool number = token[0] >= '0' && token[0] <= '9';
    return number ? fn::exec_state(push(to_number(token)), s) : fn::exec_state(binary(token[0]), s);
}

template <std::size_t N>
constexpr long rpn(const std::array<std::string_view, N>& tokens)
{
    return fn::eval_state(pop, fn::foldl_of(step, stack{}, tokens));
}
//! [rpn]

//! [reader]
struct config {
    double fx;       // quote currency per base
    double fee_bp;   // fee in basis points
};
// cost :: Long -> Reader Config Double  -- the configuration is read, never passed by hand.
constexpr auto cost(long qty)
{
    return fn::mbind(fn::asks([](const config& c) { return c.fx; }), [qty](double fx) {
        return fn::asks([qty, fx](const config& c) { return static_cast<double>(qty) * fx * (1 + c.fee_bp / 10'000); });
    });
}
constexpr config live{1.25, 10};
//! [reader]

//! [writer]
// The audit summary is a monoid: (number of fills, total quantity, largest fill), combined by the Writer.
using audit = std::tuple<fn::sum_of<int>, fn::sum_of<long>, fn::max_of<long>>;
constexpr auto fill(long qty) { return fn::mthen(fn::tell(audit{{1}, {qty}, {qty}}), fn::pure<fn::writer<audit>>(qty)); }
constexpr auto day = fn::mbind(fill(300), [](long a) {
    return fn::mbind(fill(1200), [a](long b) { return fn::fmap([a, b](long c) { return a + b + c; }, fill(500)); });
});
//! [writer]
} // namespace

int main()
{
    std::printf("snicholls-core functional/state stack-machine demo\n");

    static_assert(fn::run_state(three_ids, 100) == std::pair{std::array{100, 101, 102}, 103});
    check(fn::eval_state(three_ids, 0) == std::array{0, 1, 2}, "fresh ids 0, 1, 2 from a counter at 0");
    check(fn::exec_state(fresh, 41) == 42, "the counter moves on by one");

    static_assert(rpn(std::array<std::string_view, 7>{"3", "4", "+", "2", "*", "7", "-"}) == 7);    // (3 + 4) * 2 - 7
    static_assert(rpn(std::array<std::string_view, 5>{"100", "8", "2", "/", "-"}) == 96);          // 100 - 8 / 2
    check(rpn(std::array<std::string_view, 3>{"6", "7", "*"}) == 42, "6 7 * is 42");
    check(fn::exec_state(fn::mthen(push(1), push(2)), stack{}).n == 2, "two pushes leave two values");
    check(fn::run_state(pop, fn::exec_state(push(9), stack{})).first == 9, "pop returns what was pushed");

    static_assert(fn::run_reader(cost(1000), live) == 1000 * 1.25 * 1.001);
    check(fn::run_reader(fn::local([](config c) { c.fee_bp = 0; return c; }, cost(1000)), live) == 1250.0,
          "local: the same program under a fee holiday");
    check(fn::run_reader(fn::fmap([](double c) { return c > 1000; }, cost(1000)), live), "fmap over a Reader");

    static_assert(day.value == 2000);
    static_assert(std::get<0>(day.log).value == 3 && std::get<1>(day.log).value == 2000 && std::get<2>(day.log).value == 1200);
    check(fn::run_writer(day).first == 2000 && std::get<2>(fn::exec_writer(day)).value == 1200, "three fills, 2000 in all, 1200 the largest");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
