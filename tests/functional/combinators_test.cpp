//
//  combinators_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/combinators.hpp>: every combinator at compile time, then the
//  run-time properties (move-only parts, reference semantics, noexcept, layout).
//

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/pipeline/stages.hpp>

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fn = snicholls::fn;

constexpr auto inc = [](int x) { return x + 1; };
constexpr auto dbl = [](int x) { return x * 2; };
constexpr auto neg = [](int x) noexcept { return -x; };
constexpr auto sub = [](int a, int b) { return a - b; };
constexpr int square(int x) noexcept { return x * x; }

// ---- identity and constant ---------------------------------------------------------------------------------------

static_assert(fn::identity(3) == 3);
static_assert(std::is_same_v<decltype(fn::identity(3)), int>);            // an rvalue comes back as a value
static_assert(std::is_same_v<decltype(fn::identity(std::declval<int&>())), int&>);
static_assert(noexcept(fn::identity(3)));
static_assert(fn::constant(5)() == 5);
static_assert(fn::constant(5)(1, 2.0, "three") == 5);
static_assert(std::is_same_v<decltype(fn::constant(5)(1)), int>);

// ---- flip, compose, pipe, then -----------------------------------------------------------------------------------

static_assert(fn::flip(sub)(1, 10) == 9);
static_assert(fn::flip([](int a, int b, int c) { return a * 100 + b * 10 + c; })(1, 2, 3) == 213);
static_assert(fn::compose(inc)(1) == 2);
static_assert(fn::compose(inc, dbl)(3) == 7);                 // inc(dbl(3))
static_assert(fn::compose(inc, dbl, neg)(3) == -5);           // inc(dbl(neg(3)))
static_assert(fn::compose(inc, sub)(10, 3) == 8);             // the rightmost takes every argument
static_assert(fn::compose(inc, fn::compose(inc, dbl), inc)(1) == 6);
static_assert(fn::compose(&square, inc)(2) == 9);             // function pointers
static_assert(fn::pipe(inc)(1) == 2);
static_assert(fn::pipe(inc, dbl)(3) == 8);                    // dbl(inc(3))
static_assert(fn::pipe(neg, inc, dbl)(3) == -4);              // dbl(inc(neg(3)))
static_assert(fn::pipe(inc, dbl)(3) == fn::compose(dbl, inc)(3));
static_assert((3 | fn::then(inc)) == 4);
static_assert((3 | fn::then(inc) | fn::then(dbl)) == 8);
static_assert((3 | fn::then(fn::pipe(inc, dbl))) == 8);

// ---- on, apply, juxt ---------------------------------------------------------------------------------------------

static_assert(fn::on(std::plus<>{}, dbl)(1, 2) == 6);
static_assert(fn::on(std::less<>{}, neg)(1, 2) == false);
static_assert(fn::apply(sub)(std::tuple{10, 3}) == 7);
static_assert(fn::apply(sub)(std::pair{10, 3}) == 7);
static_assert(fn::apply(sub)(std::array{10, 3}) == 7);
static_assert(fn::apply([] { return 1; })(std::tuple<>{}) == 1);
static_assert(fn::juxt(inc, dbl, neg)(5) == std::tuple{6, 10, -5});
static_assert(fn::juxt(sub)(5, 2) == std::tuple{3});

// ---- fix ---------------------------------------------------------------------------------------------------------

constexpr auto factorial = fn::fix([](auto& self, int n) -> int { return n < 2 ? 1 : n * self(n - 1); });
static_assert(factorial(5) == 120);
static_assert(factorial(0) == 1);
constexpr auto fib = fn::fix([](const auto& self, int n) -> int { return n < 2 ? n : self(n - 1) + self(n - 2); });
static_assert(fib(10) == 55);

// ---- tap, when, unless -------------------------------------------------------------------------------------------

static_assert(fn::tap([](int) {})(4) == 4);
static_assert(fn::when([](int x) { return x > 0; }, dbl)(3) == 6);
static_assert(fn::when([](int x) { return x > 0; }, dbl)(-3) == -3);
static_assert(fn::unless([](int x) { return x > 0; }, dbl)(3) == 3);
static_assert(fn::unless([](int x) { return x > 0; }, dbl)(-3) == -6);

// ---- SFINAE-friendliness -----------------------------------------------------------------------------------------

static_assert(std::is_invocable_v<decltype(fn::compose(inc, dbl)), int>);
static_assert(!std::is_invocable_v<decltype(fn::compose(inc, dbl)), const char*>);
static_assert(!std::is_invocable_v<decltype(fn::compose(inc, dbl)), int, int>);
static_assert(!std::is_invocable_v<decltype(fn::flip(sub)), int>);
static_assert(!std::is_invocable_v<decltype(fn::apply(sub)), int>);
static_assert(!std::is_invocable_v<decltype(fn::tap(inc)), std::string>);
static_assert(!std::is_invocable_v<decltype(fn::when(inc, dbl)), std::string>);

// ---- noexcept propagation ----------------------------------------------------------------------------------------

static_assert(noexcept(fn::compose(neg, neg)(1)));
static_assert(!noexcept(fn::compose(neg, inc)(1)));
static_assert(noexcept(fn::compose(&square, neg)(1)));
static_assert(noexcept(fn::pipe(neg, neg, neg)(1)));
static_assert(noexcept(fn::juxt(neg, neg)(1)));
static_assert(!noexcept(fn::juxt(neg, inc)(1)));
static_assert(noexcept(fn::tap(neg)(1)));
static_assert(noexcept(fn::when(neg, neg)(1)));
static_assert(!noexcept(fn::when(neg, inc)(1)));
static_assert(noexcept(1 | fn::then(neg)));
static_assert(!noexcept(1 | fn::then(inc)));
static_assert(noexcept(fn::compose(neg, inc)));                                 // building copies empty lambdas
static_assert(!noexcept(fn::constant(std::string("x"))));                       // building copies a string

// ---- layout ------------------------------------------------------------------------------------------------------

static_assert(std::is_trivially_copyable_v<decltype(fn::compose(inc, dbl))>);
static_assert(std::is_trivially_copyable_v<decltype(fn::compose(&square, [k = 3](int x) { return x + k; }))>);
static_assert(sizeof(fn::compose(inc, dbl, neg)) == 1);                          // empty parts take no space
static_assert(sizeof(fn::compose(&square, inc)) == sizeof(int (*)(int)));
static_assert(sizeof(fn::juxt(inc, dbl, neg)) == 1);
static_assert(std::is_aggregate_v<decltype(fn::compose(inc, dbl))>);
static_assert(!std::is_trivially_copyable_v<decltype(fn::constant(std::string("x")))>);

TEST(FunctionalCombinators, ComposeMovesAMoveOnlyPartOutOfAnRvalueClosure)
{
    auto owner = fn::constant(std::make_unique<int>(7));
    static_assert(!std::is_copy_constructible_v<decltype(owner)>);
    std::unique_ptr<int> p = std::move(owner)();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p, 7);
}

TEST(FunctionalCombinators, AnLvalueClosureCopiesItsValueOut)
{
    const auto k = fn::constant(std::string("price"));
    EXPECT_EQ(k(1), "price");
    EXPECT_EQ(k(), "price");   // still there
}

TEST(FunctionalCombinators, ClosuresHoldCopiesSoTheyOutliveTheirArguments)
{
    auto make = [] {
        std::string unit = "bp";
        return fn::compose([unit](int x) { return std::to_string(x) + unit; }, dbl);
    };
    const auto f = make();
    EXPECT_EQ(f(21), "42bp");
}

TEST(FunctionalCombinators, StdRefReachesTheCallersObject)
{
    int calls = 0;
    auto count = [](int& n, int) { ++n; };
    auto counted = fn::tap(std::bind_front(count, std::ref(calls)));
    EXPECT_EQ(counted(1) + counted(2), 3);
    EXPECT_EQ(calls, 2);
}

TEST(FunctionalCombinators, IdentityAndTapReturnAnLvalueAsTheSameObject)
{
    std::vector<int> v{1, 2, 3};
    EXPECT_EQ(&fn::identity(v), &v);
    int seen = 0;
    auto& same = fn::tap([&](const std::vector<int>& x) { seen = static_cast<int>(x.size()); })(v);
    EXPECT_EQ(&same, &v);
    EXPECT_EQ(seen, 3);
}

TEST(FunctionalCombinators, WhenMovesAnRvalueThroughTheFunction)
{
    auto shout = fn::when([](const std::string& s) { return !s.empty(); }, [](std::string s) { return s + "!"; });
    EXPECT_EQ(shout(std::string("buy")), "buy!");
    EXPECT_EQ(shout(std::string()), "");
}

TEST(FunctionalCombinators, MutableStatefulPartsWorkThroughANonConstClosure)
{
    auto counter = fn::compose(dbl, [n = 0](int x) mutable { return x + ++n; });
    EXPECT_EQ(counter(0), 2);
    EXPECT_EQ(counter(0), 4);
    static_assert(!std::is_invocable_v<const decltype(counter)&, int>);
}

TEST(FunctionalCombinators, ThenAndThePipelineOperatorLiveTogether)
{
    using namespace snicholls::pipeline;
    std::vector<int> out;
    auto chain = transform([](int v) { return v | fn::then(inc); }) | [&](int v) { out.push_back(v); };
    chain(1);
    chain(2);
    EXPECT_EQ(out, (std::vector<int>{2, 3}));
}

TEST(FunctionalCombinators, FixRecursesAtRunTime)
{
    const auto gcd = fn::fix([](auto& self, long a, long b) -> long { return b == 0 ? a : self(b, a % b); });
    EXPECT_EQ(gcd(1071L, 462L), 21L);
}

} // namespace
