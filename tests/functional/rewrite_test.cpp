//
//  rewrite_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/rewrite.hpp>: rules as values, applied once, to a normal form, and over trees.
//

#include <snicholls/core/functional/rewrite.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

namespace fn = snicholls::fn;

constexpr auto halve_even = fn::rule([](int x) { return x % 2 == 0 && x != 0; }, [](int x) { return x / 2; });
constexpr auto collatz_odd = fn::rule([](int x) { return x % 2 == 1 && x != 1; }, [](int x) { return 3 * x + 1; });
constexpr auto collatz = fn::rules(halve_even, collatz_odd);

// ---- rules ---------------------------------------------------------------------------------------------------------

static_assert(halve_even(8) == std::optional<int>(4) && halve_even(7) == std::nullopt && halve_even(0) == std::nullopt);
static_assert(collatz(7) == std::optional<int>(22) && collatz(8) == std::optional<int>(4) && collatz(1) == std::nullopt);
static_assert(fn::rules(halve_even, fn::rule([](int) { return true; }, [](int) { return -1; }))(8) == std::optional<int>(4));
static_assert(fn::rules(halve_even, fn::rule([](int) { return true; }, [](int) { return -1; }))(7) == std::optional<int>(-1));
static_assert(fn::except([](int x) { return x > 100; }, halve_even)(200) == std::nullopt);
static_assert(fn::except([](int x) { return x > 100; }, halve_even)(20) == std::optional<int>(10));
static_assert(std::is_trivially_copyable_v<decltype(collatz)>);

// ---- replace, replace_repeated ---------------------------------------------------------------------------------------

static_assert(fn::replace_once(8, halve_even) == 4 && fn::replace_once(7, halve_even) == 7);

constexpr auto from_27 = fn::replace_repeated(27, collatz, 1000);
static_assert(from_27.value == 1 && from_27.iterations == 111 && from_27.converged);   // the 111 steps of 27's orbit

constexpr auto cut_short = fn::replace_repeated(27, collatz, 10);
static_assert(cut_short.iterations == 10 && !cut_short.converged && cut_short.value != 1);

constexpr auto exactly_enough = fn::replace_repeated(8, collatz, 3);   // 8 -> 4 -> 2 -> 1, and then nothing fires
static_assert(exactly_enough.value == 1 && exactly_enough.iterations == 3 && exactly_enough.converged);
static_assert(fn::replace_repeated(1, collatz, 0).converged && fn::replace_repeated(2, collatz, 0).value == 2);
static_assert(!fn::replace_repeated(2, collatz, 0).converged);

// ---- trees -----------------------------------------------------------------------------------------------------------

// A small expression language: a number, or a sum or product of sub-expressions.
struct expr {
    enum class kind { num, add, mul } k = kind::num;
    int value = 0;
    std::vector<expr> args;

    friend constexpr bool operator==(const expr&, const expr&) = default;
};

constexpr expr num(int v) { return {expr::kind::num, v, {}}; }
constexpr expr add(std::vector<expr> a) { return {expr::kind::add, 0, std::move(a)}; }
constexpr expr mul(std::vector<expr> a) { return {expr::kind::mul, 0, std::move(a)}; }

constexpr auto children = [](const auto& g, expr e) {
    for (auto& a : e.args) a = g(a);
    return e;
};

constexpr bool all_numbers(const expr& e)
{
    for (const auto& a : e.args)
        if (a.k != expr::kind::num) return false;
    return true;
}

// Constant folding: an operation whose arguments are numbers becomes a number.
constexpr auto fold = fn::rule([](const expr& e) { return e.k != expr::kind::num && all_numbers(e); },
                               [](const expr& e) {
                                   int v = e.k == expr::kind::add ? 0 : 1;
                                   for (const auto& a : e.args) v = e.k == expr::kind::add ? v + a.value : v * a.value;
                                   return num(v);
                               });

// x * 0 is 0, whatever x is.
constexpr auto times_zero = fn::rule(
    [](const expr& e) {
        if (e.k != expr::kind::mul) return false;
        for (const auto& a : e.args)
            if (a == num(0)) return true;
        return false;
    },
    [](const expr&) { return num(0); });

constexpr expr sample() { return add({mul({num(2), num(3)}), add({num(1), num(4)}), num(5)}); }

static_assert(fn::rewrite_up(children, fold, sample()) == num(16));
static_assert(fn::transform_up(children, [](expr e) { return fn::replace_once(e, fold); }, sample()) == num(16));
// Built in a function, like sample(): GCC does not constant-evaluate nested braced vectors written inline here.
constexpr expr times_nothing() { return mul({add({num(1), num(2)}), num(0)}); }
static_assert(fn::rewrite_up(children, fn::rules(times_zero, fold), times_nothing()) == num(0));

// transform_down sees the parent before its children: renaming adds to muls everywhere, then folding.
constexpr expr flip(expr e)
{
    if (e.k == expr::kind::add) e.k = expr::kind::mul;
    return e;
}
static_assert(fn::rewrite_up(children, fold, fn::transform_down(children, flip, sample())) == num(6 * 4 * 5));

// Bottom-up order: the counter sees the leaves before the root.
TEST(FunctionalRewrite, TransformUpVisitsChildrenBeforeParents)
{
    std::vector<int> order;
    const expr e = add({num(1), mul({num(2), num(3)})});
    fn::transform_up(children, [&](expr x) { order.push_back(x.k == expr::kind::num ? x.value : -static_cast<int>(x.k)); return x; }, e);
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3, -2, -1}));

    order.clear();
    fn::transform_down(children, [&](expr x) { order.push_back(x.k == expr::kind::num ? x.value : -static_cast<int>(x.k)); return x; }, e);
    EXPECT_EQ(order, (std::vector<int>{-1, 1, -2, 2, 3}));
}

TEST(FunctionalRewrite, ARuleThatProducesARedexIsRewrittenAgain)
{
    // add(x) of one argument becomes x; then fold runs again on what that produced.
    const auto unwrap_single = fn::rule([](const expr& e) { return e.k == expr::kind::add && e.args.size() == 1; },
                                        [](const expr& e) { return e.args[0]; });
    const expr e = add({add({mul({num(2), num(5)})})});
    EXPECT_EQ(fn::rewrite_up(children, fn::rules(unwrap_single, fold), e), num(10));
}

TEST(FunctionalRewrite, ARuleThatThrowsPropagates)
{
    const auto bad = fn::rule([](int) { return true; }, [](int) -> int { throw std::domain_error("no"); });
    EXPECT_THROW(static_cast<void>(fn::replace_once(1, bad)), std::domain_error);
}

} // namespace
