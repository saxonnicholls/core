//
//  match_patterns_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  More patterns for <snicholls/core/functional/match.hpp>: guards tried in order, guards on every kind of case,
//  a catch-all after specific cases, nested matches, destructuring of larger aggregates and arrays, std::monostate,
//  and a variant that is a small sum type for an order book.
//

#include <snicholls/core/functional/match.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

namespace {

namespace fn = snicholls::fn;
using snicholls::err;
using snicholls::ok;

// ---- guards are tried in order; the first that holds runs ---------------------------------------------------------------

constexpr char grade(int x)
{
    return fn::match(x, fn::guard([](int v) { return v >= 90; }, [](int) { return 'A'; }),
                     fn::guard([](int v) { return v >= 80; }, [](int) { return 'B'; }),
                     fn::guard([](int v) { return v >= 90; }, [](int) { return 'Z'; }),   // never: shadowed by the first
                     fn::otherwise([](int) { return 'C'; }));
}
static_assert(grade(95) == 'A' && grade(90) == 'A' && grade(89) == 'B' && grade(80) == 'B' && grade(79) == 'C' && grade(-5) == 'C');

// ---- a sum type: an order-book event -----------------------------------------------------------------------------------

struct add {
    int id;
    int qty;
    double price;
};
struct cancel {
    int id;
};
struct trade {
    int buy;
    int sell;
    int qty;
};
using event = std::variant<std::monostate, add, cancel, trade>;

constexpr int volume(const event& e)
{
    return fn::match(e, [](std::monostate) { return 0; },
                     fn::guard([](const add& a) { return a.qty <= 0; }, [](const add&) { return -1; }),
                     // A case accepts what it can be called with, conversions included: `double` here would also take
                     // a trade's `int`, so the member's type is pinned with a constrained parameter.
                     fn::destructure([](int, int qty, std::same_as<double> auto) { return qty; }),   // add, by its members
                     [](const cancel&) { return 0; },
                     fn::destructure([](int, int, int qty) { return qty; }));    // trade, by its members
}
static_assert(volume(event{}) == 0);
static_assert(volume(event{add{1, 100, 99.5}}) == 100 && volume(event{add{1, 0, 99.5}}) == -1);
static_assert(volume(event{cancel{1}}) == 0 && volume(event{trade{1, 2, 40}}) == 40);

// ---- nested matches --------------------------------------------------------------------------------------------------------

using inner = std::variant<int, char>;
using outer = std::variant<inner, double>;
constexpr int depth(const outer& o)
{
    return fn::match(o, [](const inner& i) { return fn::match(i, [](std::same_as<int> auto) { return 2; }, [](char) { return 3; }); },
                     [](double) { return 1; });
}
static_assert(depth(outer{inner{1}}) == 2 && depth(outer{inner{'c'}}) == 3 && depth(outer{0.5}) == 1);

// ---- products: tuples, pairs, arrays, aggregates --------------------------------------------------------------------------

static_assert(fn::match(std::array{1, 2, 3}, fn::unpack([](int a, int b, int c) { return a * 100 + b * 10 + c; })) == 123);
static_assert(fn::match(std::pair{2, 'x'}, fn::guard([](const auto& p) { return p.first > 5; }, fn::unpack([](int, char) { return 0; })),
                        fn::unpack([](int n, char c) { return n + (c == 'x' ? 10 : 20); })) == 12);
struct quad {
    int a, b, c, d;
};
static_assert(fn::match(quad{1, 2, 3, 4}, fn::destructure([](int a, int b, int c, int d) { return a + b + c + d; })) == 10);
static_assert(fn::match(std::tuple{}, fn::unpack([] { return 7; })) == 7);

// ---- results ---------------------------------------------------------------------------------------------------------------

enum class e { stale, crossed };
using res = snicholls::result<int, e>;
constexpr int route(const res& r)
{
    return fn::match(r, fn::guard([](e x) { return x == e::stale; }, fn::on_err([](e) { return 0; })),
                     fn::on_err([](e) { return -1; }), fn::on_ok([](int v) { return v; }));
}
static_assert(route(ok(5)) == 5 && route(err(e::stale)) == 0 && route(err(e::crossed)) == -1);

// The result type is the first unguarded case's; later cases convert to it.
static_assert(std::is_same_v<decltype(fn::match(std::variant<int, const char*>{1}, [](int) { return 1L; }, [](const char*) { return 2; })), long>);
static_assert(fn::match(std::variant<int, const char*>{"x"}, [](int) { return 1L; }, [](const char*) { return 2; }) == 2L,
              "the second alternative's int is converted to the first's long");

TEST(FunctionalMatchPatterns, OverStringsAtRunTime)
{
    using msg = std::variant<std::string, int>;
    const auto label = [](const msg& m) {
        return fn::match(m, fn::guard([](const std::string& s) { return s.empty(); }, [](const std::string&) { return std::string("(empty)"); }),
                         [](const std::string& s) { return s; }, [](int n) { return std::to_string(n); });
    };
    EXPECT_EQ(label(msg{std::string()}), "(empty)");
    EXPECT_EQ(label(msg{std::string("hi")}), "hi");
    EXPECT_EQ(label(msg{3}), "3");
}

TEST(FunctionalMatchPatterns, OnlyTheChosenCaseRuns)
{
    int guard_calls = 0, body_calls = 0;
    const int r = fn::match(4, fn::guard([&](int) { ++guard_calls; return false; }, [&](int) { ++body_calls; return 1; }),
                            fn::otherwise([](int) { return 2; }));
    EXPECT_EQ(r, 2);
    EXPECT_EQ(guard_calls, 1);
    EXPECT_EQ(body_calls, 0);
}

} // namespace
