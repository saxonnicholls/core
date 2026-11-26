//
//  combinators_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws and classic recursions for <snicholls/core/functional/combinators.hpp>: compose's identity and
//  associativity, pipe as reversed compose, the K and C combinators, fix (factorial, Fibonacci, Ackermann),
//  until, juxt, on, tap, when and unless, all at compile time.
//

#include <snicholls/core/functional/combinators.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fn = snicholls::fn;

constexpr auto inc = [](int x) { return x + 1; };
constexpr auto dbl = [](int x) { return 2 * x; };
constexpr auto sq = [](int x) { return x * x; };

// ---- compose and pipe: a category -----------------------------------------------------------------------------------

template <class F, class G>
constexpr bool same_on(F f, G g)
{
    for (int x = -20; x <= 20; ++x)
        if (f(x) != g(x)) return false;
    return true;
}

static_assert(same_on(fn::compose(fn::identity, sq), sq));                                     // left identity
static_assert(same_on(fn::compose(sq, fn::identity), sq));                                     // right identity
static_assert(same_on(fn::compose(fn::compose(inc, dbl), sq), fn::compose(inc, fn::compose(dbl, sq))));   // associativity
static_assert(same_on(fn::compose(inc, dbl, sq), fn::compose(fn::compose(inc, dbl), sq)));    // n-ary is the nesting
static_assert(same_on(fn::pipe(sq, dbl, inc), fn::compose(inc, dbl, sq)));                   // pipe is compose reversed
static_assert(same_on(fn::pipe(fn::identity), fn::identity));
static_assert(same_on([](int x) { return x | fn::then(sq) | fn::then(inc); }, fn::pipe(sq, inc)));   // x |> f |> g

// ---- K, C, application ------------------------------------------------------------------------------------------------

static_assert(fn::constant(7)(1, 2, 3) == 7 && fn::constant(7)() == 7);
static_assert(same_on(fn::compose(fn::constant(4), sq), fn::constant(4)));                    // const absorbs on the left
static_assert(fn::flip(fn::flip([](int a, int b) { return a - b; }))(10, 3) == 7);            // flip is an involution
static_assert(fn::call(sq, 9) == 81 && fn::call(fn::compose(inc, sq), 3) == 10);              // f $ x
static_assert(fn::apply(std::plus<>{})(std::pair{2, 3}) == 5 && fn::apply(fn::constant(1))(std::tuple{}) == 1);
static_assert(fn::juxt(inc, dbl, sq)(4) == std::tuple{5, 8, 16});                              // f &&& g &&& h
static_assert(fn::on(std::plus<>{}, sq)(3, 4) == 25);                                          // (+) `on` square
static_assert(fn::rotate_args([](int a, int b, int c) { return 100 * a + 10 * b + c; })(1, 2, 3) == 231);
static_assert(fn::swap_args([](int a, int b) { return a - b; })(1, 10) == 9);

// ---- conditionals ----------------------------------------------------------------------------------------------------

constexpr auto even = [](int x) { return x % 2 == 0; };
static_assert(fn::when(even, dbl)(4) == 8 && fn::when(even, dbl)(3) == 3);
static_assert(fn::unless(even, dbl)(4) == 4 && fn::unless(even, dbl)(3) == 6);
static_assert(fn::apply_when(true, inc)(1) == 2 && fn::apply_when(false, inc)(1) == 1);
static_assert(fn::bool_(10, 20, true) == 20 && fn::bool_(10, 20, false) == 10);
static_assert(fn::until([](int x) { return x > 1000; }, dbl)(3) == 1536);
static_assert(fn::until([](int) { return true; }, dbl)(3) == 3);                               // already there: no step

// Newton's method for the square root of 2, until the step is tiny.
constexpr double newton_sqrt2 = fn::until([](double x) { return (x * x - 2 > 0 ? x * x - 2 : 2 - x * x) < 1e-12; },
                                          [](double x) { return (x + 2 / x) / 2; })(1.0);
static_assert(newton_sqrt2 > 1.41421356237 && newton_sqrt2 < 1.41421356238);

// ---- fix: recursion without a name -----------------------------------------------------------------------------------

constexpr auto factorial = fn::fix([](auto self, std::uint64_t n) -> std::uint64_t { return n <= 1 ? 1 : n * self(n - 1); });
static_assert(factorial(0) == 1 && factorial(1) == 1 && factorial(20) == 2'432'902'008'176'640'000ULL);

constexpr auto fib = fn::fix([](auto self, int n) -> int { return n < 2 ? n : self(n - 1) + self(n - 2); });
static_assert(fib(0) == 0 && fib(1) == 1 && fib(20) == 6765);

constexpr auto ackermann = fn::fix([](auto self, int m, int n) -> int {
    if (m == 0) return n + 1;
    if (n == 0) return self(m - 1, 1);
    return self(m - 1, self(m, n - 1));
});
static_assert(ackermann(0, 0) == 1 && ackermann(1, 2) == 4 && ackermann(2, 3) == 9 && ackermann(3, 3) == 61);

// Mutual recursion through one fixed point over a tag.
constexpr auto parity = fn::fix([](auto self, bool want_even, int n) -> bool {
    if (n == 0) return want_even;
    return self(!want_even, n - 1);
});
static_assert(parity(true, 10) && !parity(true, 7) && parity(false, 7));

// ---- the closures' contract --------------------------------------------------------------------------------------------

static_assert(std::is_trivially_copyable_v<decltype(fn::compose(inc, dbl, sq))>);
static_assert(std::is_trivially_copyable_v<decltype(factorial)>);
static_assert(noexcept(fn::compose([](int x) noexcept { return x; }, [](int x) noexcept { return x; })(1)));
static_assert(!noexcept(fn::compose(inc, dbl)(1)));
static_assert(!std::is_invocable_v<decltype(fn::compose(inc, dbl)), const char*>);

TEST(FunctionalCombinatorsLaws, TapSeesTheValueAndPassesItOn)
{
    std::vector<int> seen;
    const auto logged = fn::pipe(sq, fn::tap([&](int x) { seen.push_back(x); }), inc);
    EXPECT_EQ(logged(3), 10);
    EXPECT_EQ(logged(4), 17);
    EXPECT_EQ(seen, (std::vector<int>{9, 16}));
}

TEST(FunctionalCombinatorsLaws, ComposeCallsEachFunctionOnceRightToLeft)
{
    std::string order;
    const auto a = [&](int x) { order += 'a'; return x; };
    const auto b = [&](int x) { order += 'b'; return x; };
    const auto c = [&](int x) { order += 'c'; return x; };
    EXPECT_EQ(fn::compose(a, b, c)(1), 1);
    EXPECT_EQ(order, "cba");
    order.clear();
    EXPECT_EQ(fn::pipe(a, b, c)(1), 1);
    EXPECT_EQ(order, "abc");
}

TEST(FunctionalCombinatorsLaws, AnExceptionStopsThePipeline)
{
    int after = 0;
    const auto boom = [](int) -> int { throw std::domain_error("boom"); };
    const auto p = fn::pipe(inc, boom, [&](int x) { ++after; return x; });
    EXPECT_THROW(static_cast<void>(p(1)), std::domain_error);
    EXPECT_EQ(after, 0);
}

TEST(FunctionalCombinatorsLaws, FixRecursesAtRunTimeToo)
{
    volatile int n = 25;
    EXPECT_EQ(fib(n), 75025);
    EXPECT_EQ(ackermann(2, n), 2 * 25 + 3);
}

} // namespace
