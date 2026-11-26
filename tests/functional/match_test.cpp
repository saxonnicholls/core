//
//  match_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/match.hpp>: guards, exhaustiveness, variants, results, tuples and aggregates.
//

#include <snicholls/core/functional/match.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <string>
#include <utility>
#include <variant>

namespace {
namespace fn = snicholls::fn;
using snicholls::result; using snicholls::ok; using snicholls::err;
using shape = std::variant<int, double, std::pair<int, int>>;
constexpr int area(const shape& s) {
  return fn::match(s,
      fn::guard([](int r) { return r < 0; }, [](std::same_as<int> auto) { return -1; }),
      [](std::same_as<int> auto r) { return 3 * r * r; },
      [](double) { return 0; },
      fn::unpack([](int w, int h) { return w * h; }));
}
static_assert(area(shape{2}) == 12 && area(shape{-2}) == -1 && area(shape{std::pair{2, 3}}) == 6 && area(shape{1.0}) == 0);
enum class e { bad };
constexpr int show(const result<int, e>& r) {
  return fn::match(r, fn::guard([](int v) { return v > 10; }, fn::on_ok([](int) { return 100; })), fn::on_ok([](int v) { return v; }), fn::on_err([](e) { return -1; }));
}
static_assert(show(ok(3)) == 3 && show(ok(30)) == 100 && show(err(e::bad)) == -1);
struct point { int x; int y; };
static_assert(fn::match(point{3, 4}, fn::destructure([](int x, int y) { return x * y; })) == 12);
static_assert(fn::match(5, fn::guard([](int x) { return x > 3; }, [](int) { return 'b'; }), fn::otherwise([](int) { return 's'; })) == 'b');
static_assert(fn::match(std::tuple{2, 3}, fn::unpack([](int a, int b) { return a + b; })) == 5);

TEST(FunctionalMatch, StringsAndGuardsAtRunTime)
{
    using event = std::variant<int, std::string>;
    const auto describe = [](const event& ev) {
        return fn::match(ev, fn::guard([](int q) { return q > 1000; }, [](std::same_as<int> auto) { return std::string("block"); }),
                         [](std::same_as<int> auto q) { return "fill " + std::to_string(q); },
                         [](const std::string& s) { return "note " + s; });
    };
    EXPECT_EQ(describe(event{50}), "fill 50");
    EXPECT_EQ(describe(event{5000}), "block");
    EXPECT_EQ(describe(event{std::string("halt")}), "note halt");
}

TEST(FunctionalMatch, AResultIsMatchedBySide)
{
    using parsed = snicholls::result<int, std::string>;
    const auto text = [](const parsed& r) {
        return fn::match(r, fn::on_ok([](int v) { return std::to_string(v); }), fn::on_err([](const std::string& e) { return "error: " + e; }));
    };
    EXPECT_EQ(text(ok(4)), "4");
    EXPECT_EQ(text(err(std::string("bad tick"))), "error: bad tick");
}

} // namespace
