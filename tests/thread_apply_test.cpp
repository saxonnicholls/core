//
//  thread_apply_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/thread_apply.hpp>: the Wolfram forms, per-element overload
//  resolution, repeats, moves, order, and the results.
//

#include <snicholls/core/thread_apply.hpp>

#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/overloaded.hpp>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using snicholls::thread_apply;
using snicholls::whole;

constexpr auto add = [](auto a, auto b) { return a + b; };

// Thread[f[{a, b}, {x, y}]], Thread[f[{a, b}, x]], Thread[f[{a, b}]], Thread[f[x]]
static_assert(thread_apply(add, std::tuple{1, 2.5}, std::tuple{10, 20}) == std::tuple{11, 22.5});
static_assert(thread_apply(add, std::tuple{1, 2}, 10) == std::tuple{11, 12});
static_assert(thread_apply(add, 10, std::tuple{1, 2}) == std::tuple{11, 12});
static_assert(thread_apply([](auto a) { return a * 2; }, std::tuple{1, 2.5}) == std::tuple{2, 5.0});
static_assert(thread_apply(add, 1, 2) == 3);
// Every tuple-like; a scalar repeated in the middle
static_assert(thread_apply(add, std::array{1, 2}, std::pair{10, 20}) == std::tuple{11, 22});
static_assert(thread_apply([](int a, int s, int b) { return a * s + b; }, std::array{1, 2}, 10, std::array{3, 4}) == std::tuple{13, 24});
// An empty tuple threads to nothing
static_assert(thread_apply(add, std::tuple<>{}, 1) == std::tuple<>{});
// whole() repeats a tuple rather than threading it
static_assert(thread_apply([](int a, const std::tuple<int, int>& t) { return a + std::get<1>(t); }, std::tuple{1, 2}, whole(std::tuple{0, 5})) == std::tuple{6, 7});
static_assert(thread_apply([](const std::array<int, 3>& t, int a) { return t[2] + a; }, whole(std::array{0, 0, 5}), std::array{1, 2}) == std::tuple{6, 7});
// One level only: a nested tuple reaches f whole
static_assert(thread_apply([](const std::tuple<int, int>& t) { return std::get<0>(t); }, std::tuple{std::tuple{1, 2}, std::tuple{3, 4}}) == std::tuple{1, 3});
// A result is by value, whatever f returns
inline constexpr int global_value = 7;
static_assert(std::is_same_v<decltype(thread_apply([](int) -> const int& { return global_value; }, std::tuple{1})), std::tuple<int>>);
// The plain form keeps what f returns
static_assert(std::is_same_v<decltype(thread_apply([](int) -> const int& { return global_value; }, 1)), const int&>);
// void
static_assert(std::is_void_v<decltype(thread_apply([](int) {}, std::tuple{1, 2}))>);
// Compile-time choice per element, by an overload set
constexpr auto kind = snicholls::overloaded{[](int) { return 'i'; }, [](double) { return 'd'; }, [](bool) { return 'b'; }};
static_assert(thread_apply(kind, std::tuple{1, 2.0, true}) == std::tuple{'i', 'd', 'b'});
// A member pointer
struct holder { int v; };
static_assert(thread_apply(&holder::v, std::tuple{holder{1}, holder{2}}) == std::tuple{1, 2});

TEST(ThreadApply, AnOverloadedSetChoosesForEachElementsType)
{
    std::string seen;
    thread_apply(snicholls::overloaded{
                     [&](int v, const std::string& tag) { seen += tag + "int" + std::to_string(v) + ' '; },
                     [&](const std::string& v, const std::string& tag) { seen += tag + "text" + v + ' '; },
                 },
                 std::tuple{1, std::string{"x"}, 2}, std::string{"#"});
    EXPECT_EQ(seen, "#int1 #textx #int2 ");
}

TEST(ThreadApply, FirstMatchAndConceptsChooseForEachElementToo)
{
    const auto described = thread_apply(snicholls::first_match{
                                            [](snicholls::integer auto) { return std::string{"integer"}; },
                                            [](snicholls::number auto) { return std::string{"number"}; },
                                            [](const auto&) { return std::string{"other"}; },
                                        },
                                        std::tuple{1, 2.5, std::string{"s"}});
    EXPECT_EQ(described, (std::tuple{std::string{"integer"}, std::string{"number"}, std::string{"other"}}));
}

TEST(ThreadApply, CallsAreMadeInOrder)
{
    std::vector<int> order;
    const auto ids = thread_apply([&](int v) { order.push_back(v); return v; }, std::tuple{1, 2, 3});
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(ids, (std::tuple{1, 2, 3}));
    order.clear();
    thread_apply([&](int v) { order.push_back(v); }, std::array{4, 5});
    EXPECT_EQ(order, (std::vector<int>{4, 5}));
}

TEST(ThreadApply, AnRvalueTuplesElementsAreMovedOutAndARepeatedArgumentNever)
{
    const std::string repeated(40, 'r');
    std::tuple<std::unique_ptr<int>, std::unique_ptr<int>> owners{std::make_unique<int>(1), std::make_unique<int>(2)};
    const auto sums = thread_apply([&](std::unique_ptr<int> p, const std::string& r) { return *p + static_cast<int>(r.size()); },
                                   std::move(owners), repeated);
    EXPECT_EQ(sums, (std::tuple{41, 42}));
    EXPECT_EQ(repeated.size(), 40u);
}

TEST(ThreadApply, AnLvalueTupleCanBeWrittenThrough)
{
    std::tuple<int, double> values{1, 2.5};
    thread_apply([](auto& v) { v = v + v; }, values);
    EXPECT_EQ(values, (std::tuple<int, double>{2, 5.0}));
}

TEST(ThreadApply, ACallableIsCalledAsAnLvalueAndKeepsItsState)
{
    int calls = 0;
    auto counting = [calls]() mutable { return ++calls; };
    const auto got = thread_apply([&](int) { return counting(); }, std::tuple{0, 0, 0});
    EXPECT_EQ(got, (std::tuple{1, 2, 3}));
}

// ---- outer ------------------------------------------------------------------------------------------
using snicholls::outer;
constexpr auto times = [](int a, int b) { return a * b; };
// Outer[Times, {1, 2}, {10, 20, 30}]: nested arrays, row-major
static_assert(outer(times, std::array{1, 2}, std::array{10, 20, 30}) == std::array{std::array{10, 20, 30}, std::array{20, 40, 60}});
// One list: f of each element
static_assert(outer([](int a) { return a + 1; }, std::array{1, 2}) == std::array{2, 3});
// Three lists: three levels
static_assert(outer([](int a, int b, int c) { return a * 100 + b * 10 + c; }, std::array{1, 2}, std::array{3}, std::array{4, 5})[1][0][1] == 235);
// Tuples: a tuple per level, and a branch per pair of types
constexpr auto describe = snicholls::overloaded{[](int, int) { return 'i'; }, [](int, double) { return 'm'; },
                                                [](double, int) { return 'm'; }, [](double, double) { return 'd'; }};
static_assert(outer(describe, std::tuple{1, 2.0}, std::tuple{3, 4.0}) == std::tuple{std::tuple{'i', 'm'}, std::tuple{'m', 'd'}});
// The results may differ in type across a tuple
static_assert(std::is_same_v<decltype(outer([](auto a, auto b) { return a + b; }, std::tuple{1, 2.0}, std::array{1, 2})),
                             std::tuple<std::array<int, 2>, std::array<double, 2>>>);
// An empty list: an empty tuple
static_assert(outer(times, std::array<int, 0>{}, std::array{1}) == std::tuple<>{});

TEST(Outer, CallsAreMadeInRowMajorOrder)
{
    std::vector<int> order;
    static_cast<void>(outer([&](int a, int b) { order.push_back(a * 10 + b); return 0; }, std::array{1, 2}, std::array{1, 2, 3}));
    EXPECT_EQ(order, (std::vector<int>{11, 12, 13, 21, 22, 23}));
}

TEST(Outer, ElementsAreNotMovedOutOfAnRvalueList)
{
    const auto lengths = outer([](const std::string& a, const std::string& b) { return a.size() + b.size(); },
                               std::array{std::string(30, 'a'), std::string(40, 'b')}, std::array{std::string(5, 'c')});
    EXPECT_EQ(lengths[0][0], 35u);
    EXPECT_EQ(lengths[1][0], 45u);
}

// Inner[Times, {1, 2, 3}, {4, 5, 6}, Plus] = 32
static_assert(snicholls::inner([](int a, int b) { return a * b; }, std::array{1, 2, 3}, std::array{4, 5, 6},
                               [](auto... xs) { return (xs + ...); }) == 32);
static_assert(snicholls::inner([](auto a, auto b) { return a * b; }, std::tuple{2, 0.5}, std::tuple{3, 4.0},
                               [](auto... xs) { return (0.0 + ... + xs); }) == 8.0);

} // namespace
