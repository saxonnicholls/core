//
//  overloaded_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/overloaded.hpp>: overloaded, first_match and fn_ptr.
//

#include <snicholls/core/overloaded.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <ostream>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

// ---- overloaded ------------------------------------------------------------------------------

// Deduction and aggregate shape, at compile time
constexpr auto twice = [](int x) { return 2 * x; };
constexpr auto negate = [](double x) { return -x; };
using both_t = decltype(snicholls::overloaded{twice, negate});
static_assert(std::is_same_v<both_t, snicholls::overloaded<std::remove_const_t<decltype(twice)>,
                                                           std::remove_const_t<decltype(negate)>>>);
static_assert(std::is_aggregate_v<both_t>);

// Usable in a constant expression
static_assert(snicholls::overloaded{twice, negate}(21) == 42);
static_assert(snicholls::overloaded{twice, negate}(1.5) == -1.5);

// Costs nothing: an overload set of empty lambdas is itself empty
static_assert(std::is_empty_v<both_t>);

TEST(OverloadedVisit, PicksTheMatchingAlternative)
{
    const auto describe = snicholls::overloaded{
        [](int) { return std::string{"int"}; },
        [](const std::string&) { return std::string{"string"}; },
        [](double) { return std::string{"double"}; },
    };
    std::variant<int, std::string, double> v = 7;
    EXPECT_EQ(std::visit(describe, v), "int");
    v = std::string{"x"};
    EXPECT_EQ(std::visit(describe, v), "string");
    v = 2.5;
    EXPECT_EQ(std::visit(describe, v), "double");
}

TEST(OverloadedVisit, GenericLambdaIsTheCatchAll)
{
    const auto f = snicholls::overloaded{
        [](int) { return 1; },
        [](const auto&) { return 0; },
    };
    EXPECT_EQ(f(5), 1);                        // exact non-template match wins
    EXPECT_EQ(f(std::string{"other"}), 0);     // everything else falls through
    EXPECT_EQ(f(2.0), 0);
}

TEST(OverloadedState, CapturesAreCopiedIntoTheOverloadSet)
{
    int hits = 0;
    auto f = snicholls::overloaded{
        [&hits](int) { ++hits; },
        [&hits](char) { hits += 10; },
    };
    f(1);
    f('c');
    EXPECT_EQ(hits, 11);

    auto copy = f;                             // copyable because each lambda is
    copy(2);
    EXPECT_EQ(hits, 12);
}

TEST(OverloadedState, MutableLambdasKeepTheirOwnState)
{
    auto counter = snicholls::overloaded{
        [n = 0](int) mutable { return ++n; },
    };
    EXPECT_EQ(counter(0), 1);
    EXPECT_EQ(counter(0), 2);
}

// ---- fn_ptr: function pointers as elements -----------------------------------------------------

constexpr int twice_fn(int n) { return 2 * n; }
constexpr int add_fn(int a, int b) noexcept { return a + b; }

// A function name or its address becomes a fn_ptr element, in both set types
using with_fn_t = decltype(snicholls::overloaded{twice_fn, negate});
static_assert(std::is_same_v<with_fn_t, snicholls::overloaded<snicholls::fn_ptr<int (*)(int)>,
                                                              std::remove_const_t<decltype(negate)>>>);
static_assert(std::is_same_v<decltype(snicholls::overloaded{&twice_fn, negate}), with_fn_t>);
static_assert(std::is_same_v<decltype(snicholls::first_match{twice_fn}),
                             snicholls::first_match<snicholls::fn_ptr<int (*)(int)>>>);
static_assert(std::is_aggregate_v<with_fn_t>);
static_assert(sizeof(with_fn_t) == sizeof(int (*)(int)));   // the pointer and nothing else

// Still constexpr, and overload resolution sees the pointer's exact signature
static_assert(snicholls::overloaded{twice_fn, negate}(21) == 42);
static_assert(snicholls::overloaded{twice_fn, negate}(1.5) == -1.5);
static_assert(snicholls::overloaded{twice_fn, add_fn}(1, 2) == 3);

// noexcept follows the pointer type
static_assert(noexcept(snicholls::overloaded{twice_fn, add_fn}(1, 2)));
static_assert(!noexcept(snicholls::overloaded{twice_fn, add_fn}(1)));

TEST(OverloadedFunctionPointer, CallsThroughThePointer)
{
    const auto f = snicholls::overloaded{twice_fn, [](const std::string& s) { return s.size(); }};
    EXPECT_EQ(f(4), 8);
    EXPECT_EQ(f(std::string{"abc"}), 3u);
    EXPECT_EQ(static_cast<const snicholls::fn_ptr<int (*)(int)>&>(f).target, &twice_fn);
}

TEST(OverloadedFunctionPointer, PointerToPointerParameterIsForwardedAsGiven)
{
    // A parameter taken by non-const reference reaches the function as that reference.
    struct local {
        static void bump(int& x) { ++x; }
    };
    const auto f = snicholls::overloaded{&local::bump};
    int x = 1;
    f(x);
    EXPECT_EQ(x, 2);
}

// ---- first_match ---------------------------------------------------------------------------------

template <typename T>
concept streamable = requires(std::ostream& os, const T& t) { os << t; };

constexpr auto chain = snicholls::first_match{
    [](std::integral auto) { return 1; },
    [](std::floating_point auto) { return 2; },
    [](const auto&) { return 3; },
};
using chain_t = std::remove_const_t<decltype(chain)>;

// Aggregate, empty, constexpr
static_assert(std::is_aggregate_v<chain_t>);
static_assert(std::is_empty_v<chain_t>);
static_assert(chain(1) == 1);
static_assert(chain(1u) == 1);
static_assert(chain(2.5) == 2);
static_assert(chain("x") == 3);

// SFINAE-friendly: no accepting callable is "not invocable", not an error
constexpr auto ints_only = snicholls::first_match{[](std::integral auto) { return 0; }};
static_assert(std::is_invocable_v<const decltype(ints_only)&, int>);
static_assert(!std::is_invocable_v<const decltype(ints_only)&, std::string>);
static_assert(!std::is_invocable_v<const snicholls::first_match<>&, int>);   // the empty set accepts nothing

// noexcept follows the chosen callable
constexpr auto mixed = snicholls::first_match{
    [](std::same_as<int> auto) noexcept { return 1; },
    [](std::same_as<double> auto) { return 2; },
};
static_assert(noexcept(mixed(1)));
static_assert(!noexcept(mixed(1.0)));

TEST(FirstMatch, OrderSettlesOverlappingConditions)
{
    // A std::string is both a range and streamable; the chain decides, not an ambiguity.
    const auto range_first = snicholls::first_match{
        [](const std::ranges::range auto&) { return std::string_view{"range"}; },
        [](const streamable auto&) { return std::string_view{"streamable"}; },
    };
    const auto streamable_first = snicholls::first_match{
        [](const streamable auto&) { return std::string_view{"streamable"}; },
        [](const std::ranges::range auto&) { return std::string_view{"range"}; },
    };
    const std::string s{"both"};
    EXPECT_EQ(range_first(s), "range");
    EXPECT_EQ(streamable_first(s), "streamable");
    EXPECT_EQ(range_first(std::vector<int>{1}), "range");   // only a range
    EXPECT_EQ(streamable_first(42), "streamable");          // only streamable
}

TEST(FirstMatch, AcceptsMeansInvocableSoConversionsCount)
{
    // The documented trap: [](bool) accepts an int, so it takes 1 when written first.
    constexpr auto loose = snicholls::first_match{
        [](bool) { return std::string_view{"bool"}; },
        [](int) { return std::string_view{"int"}; },
    };
    static_assert(loose(1) == "bool");
    static_assert(loose(true) == "bool");

    // The fix: say exactly what each branch takes.
    constexpr auto exact = snicholls::first_match{
        [](std::same_as<bool> auto) { return std::string_view{"bool"}; },
        [](std::same_as<int> auto) { return std::string_view{"int"}; },
    };
    static_assert(exact(1) == "int");
    static_assert(exact(true) == "bool");
}

TEST(FirstMatchVersusOverloaded, ConvertingConcreteBeatsGenericOnlyInFirstMatch)
{
    constexpr auto concrete = [](int) { return 1; };
    constexpr auto generic = [](const auto&) { return 0; };
    static_assert(snicholls::overloaded{concrete, generic}(short{1}) == 0);    // exact generic match wins
    static_assert(snicholls::first_match{concrete, generic}(short{1}) == 1);   // first that accepts wins
    static_assert(snicholls::overloaded{concrete, generic}(1) == 1);           // both agree on an exact int
    static_assert(snicholls::first_match{concrete, generic}(1) == 1);
}

TEST(FirstMatch, ReturnsWhatTheChosenCallableReturnsAsItReturnsIt)
{
    auto f = snicholls::first_match{[](int& v) -> int& { return v; }};
    int x = 1;
    static_assert(std::is_same_v<decltype(f(x)), int&>);
    f(x) = 5;
    EXPECT_EQ(x, 5);
}

TEST(FirstMatch, ConstnessOfTheSetSelectsTheCandidates)
{
    // A mutable lambda is a candidate through a non-const set only.
    auto f = snicholls::first_match{
        [n = 0](int) mutable { return ++n; },
        [](int) { return -1; },
    };
    EXPECT_EQ(f(0), 1);
    EXPECT_EQ(f(0), 2);
    EXPECT_EQ(std::as_const(f)(0), -1);
}

TEST(FirstMatch, ComposesWithOverloadedAndVisit)
{
    const auto f = snicholls::overloaded{
        snicholls::first_match{[](std::integral auto) { return 1; }, [](std::floating_point auto) { return 2; }},
        [](const std::string&) { return 3; },
    };
    EXPECT_EQ(f(1), 1);
    EXPECT_EQ(f(2.5), 2);
    EXPECT_EQ(f(std::string{}), 3);

    const auto g = snicholls::first_match{
        snicholls::overloaded{[](int) { return 1; }, [](double) { return 2; }},
        [](const auto&) { return 3; },
    };
    std::variant<int, double, std::string> v = 2.5;
    EXPECT_EQ(std::visit(g, v), 2);
    v = std::string{};
    EXPECT_EQ(std::visit(g, v), 3);
}

TEST(FirstMatch, CapturesAndFunctionPointersWorkAsInOverloaded)
{
    int hits = 0;
    auto f = snicholls::first_match{
        twice_fn,
        [&hits](const auto&) { ++hits; return 0; },
    };
    EXPECT_EQ(f(21), 42);
    EXPECT_EQ(f(std::string{}), 0);
    EXPECT_EQ(hits, 1);
}

TEST(FirstMatch, ReplacesAnIfConstexprChain)
{
    // The shape of a to_string written as one generic lambda with an if constexpr chain inside.
    const auto text = snicholls::first_match{
        [](std::string_view s) { return std::string{s}; },
        [](std::same_as<bool> auto b) { return std::string{b ? "true" : "false"}; },
        [](std::integral auto n) { return std::to_string(n); },
        [](const std::ranges::input_range auto& r) {
            std::string out{"["};
            for (const auto& e : r)
                out += (out.size() > 1 ? "," : "") + std::to_string(e);
            return out + "]";
        },
        [](const streamable auto& x) { std::ostringstream os; os << x; return os.str(); },
        [](const auto&) { return std::string{"?"}; },
    };
    struct opaque {};
    EXPECT_EQ(text("abc"), "abc");
    EXPECT_EQ(text(std::string{"s"}), "s");
    EXPECT_EQ(text(true), "true");
    EXPECT_EQ(text(42), "42");
    EXPECT_EQ(text(std::vector<int>{1, 2}), "[1,2]");
    EXPECT_EQ(text(2.5), "2.5");
    EXPECT_EQ(text(opaque{}), "?");
}

} // namespace
