//
//  rust_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/rust.hpp>: Rust's Option methods over std::optional, std::cmp's ordering
//  helpers and Into.
//

#include <snicholls/core/functional/rust.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <compare>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

namespace option = snicholls::fn::rust::option;
namespace cmp = snicholls::fn::rust::cmp;
using snicholls::fn::rust::into;

constexpr std::optional<int> some(int x) { return x; }
constexpr std::optional<int> none{};
constexpr auto even = [](int x) { return x % 2 == 0; };
constexpr auto twice = [](int x) { return 2 * x; };

// ---- predicates and mapping ---------------------------------------------------------------------------------------

static_assert(option::is_some_and(some(4), even) && !option::is_some_and(some(3), even) && !option::is_some_and(none, even));
static_assert(option::is_none_or(none, even) && option::is_none_or(some(4), even) && !option::is_none_or(some(3), even));
static_assert(option::map_or(some(4), 0, twice) == 8 && option::map_or(none, -1, twice) == -1);
static_assert(std::is_same_v<decltype(option::map_or(some(4), 0L, twice)), long>);
static_assert(option::map_or_else(none, [] { return 7; }, twice) == 7 && option::map_or_else(some(1), [] { return 7; }, twice) == 2);
static_assert(noexcept(option::is_some_and(none, [](int) noexcept { return true; })));
static_assert(!noexcept(option::is_some_and(none, [](int) { return true; })));

// ---- to result -----------------------------------------------------------------------------------------------------

static_assert(option::ok_or(some(3), 'e').unwrap_or(0) == 3);
static_assert(option::ok_or(none, 'e').unwrap_err_unchecked() == 'e');
static_assert(std::is_same_v<decltype(option::ok_or(none, 'e')), snicholls::result<int, char>>);
static_assert(option::ok_or_else(none, [] { return 9L; }).unwrap_err_unchecked() == 9L);

// ---- filter, logic -------------------------------------------------------------------------------------------------

static_assert(option::filter(some(4), even) == some(4) && option::filter(some(3), even) == none && option::filter(none, even) == none);
static_assert(option::and_(some(1), std::optional<char>('a')) == std::optional<char>('a'));
static_assert(option::and_(none, std::optional<char>('a')) == std::nullopt);
static_assert(option::or_(some(1), some(2)) == some(1) && option::or_(none, some(2)) == some(2));
static_assert(option::xor_(some(1), none) == some(1) && option::xor_(none, some(2)) == some(2));
static_assert(option::xor_(some(1), some(2)) == none && option::xor_(none, none) == none);
static_assert(option::or_else(none, [] { return some(5); }) == some(5) && option::or_else(some(1), [] { return some(5); }) == some(1));

// ---- unwrapping ------------------------------------------------------------------------------------------------------

static_assert(option::unwrap_or_else(none, [] { return 3; }) == 3 && option::unwrap_or_else(some(1), [] { return 3; }) == 1);
static_assert(option::unwrap_or_default(none) == 0 && option::unwrap_or_default(some(6)) == 6);

// ---- zip, unzip, transpose -----------------------------------------------------------------------------------------

static_assert(option::zip(some(1), std::optional<char>('x')) == std::optional<std::pair<int, char>>({1, 'x'}));
static_assert(option::zip(none, std::optional<char>('x')) == std::nullopt);
static_assert(option::zip_with(some(2), some(5), std::plus<>{}) == some(7) && option::zip_with(some(2), none, std::plus<>{}) == none);
static_assert(option::unzip(std::optional<std::pair<int, char>>({1, 'x'})) == std::pair{some(1), std::optional<char>('x')});
static_assert(option::unzip(std::optional<std::pair<int, char>>()) == std::pair{none, std::optional<char>()});

using res = snicholls::result<int, char>;
static_assert(option::transpose(std::optional<res>()).unwrap_or(some(0)) == none);
static_assert(option::transpose(std::optional<res>(snicholls::ok(4))).unwrap_or(none) == some(4));
static_assert(option::transpose(std::optional<res>(snicholls::err('z'))).unwrap_err_unchecked() == 'z');
static_assert(std::is_same_v<decltype(option::transpose(std::optional<res>())), snicholls::result<std::optional<int>, char>>);

// ---- in place ------------------------------------------------------------------------------------------------------

constexpr bool in_place_works()
{
    std::optional<int> o = 3;
    const std::optional<int> taken = option::take(o);
    bool ok = taken == some(3) && !o.has_value();
    ok = ok && option::replace(o, 4) == none && o == some(4);
    ok = ok && option::replace(o, 5) == some(4) && o == some(5);
    ok = ok && option::take_if(o, even) == none && o == some(5);
    ok = ok && option::take_if(o, [](int& x) { return ++x == 6; }) == some(6) && !o.has_value();
    int& got = option::get_or_insert_with(o, [] { return 7; });
    ok = ok && got == 7;
    got = 8;
    ok = ok && option::get_or_insert(o, 1) == 8 && option::get_or_insert_with(o, [] { return 1; }) == 8;
    ok = ok && (option::insert(o, 2) = 3) == 3 && o == some(3);
    return ok;
}
static_assert(in_place_works());

// ---- cmp -----------------------------------------------------------------------------------------------------------

static_assert(cmp::then(std::strong_ordering::equal, std::strong_ordering::less) == std::strong_ordering::less);
static_assert(cmp::then(std::strong_ordering::greater, std::strong_ordering::less) == std::strong_ordering::greater);
static_assert(std::is_same_v<decltype(cmp::then(std::strong_ordering::equal, std::weak_ordering::less)), std::weak_ordering>);
static_assert(cmp::then_with(std::strong_ordering::equal, [] { return 1 <=> 2; }) < 0);
static_assert(cmp::reverse(std::strong_ordering::less) == std::strong_ordering::greater);
static_assert(cmp::reverse(std::partial_ordering::unordered) == std::partial_ordering::unordered);

struct order {
    int priority;
    double price;
};
struct only_less {
    int v;
    constexpr bool operator<(const only_less& o) const noexcept { return v < o.v; }
};

constexpr auto by_priority = cmp::comparing(&order::priority);
constexpr auto by_price = cmp::comparing(&order::price);
static_assert(by_priority(order{1, 2.0}, order{2, 1.0}) < 0);
static_assert(cmp::lexicographic(by_priority, by_price)(order{1, 2.0}, order{1, 1.0}) > 0);
static_assert(cmp::lexicographic(by_priority, cmp::reversed(by_price))(order{1, 2.0}, order{1, 1.0}) < 0);
static_assert(cmp::lexicographic(by_priority)(order{1, 2.0}, order{1, 1.0}) == 0);
static_assert(cmp::comparing([](const only_less& x) { return x; })(only_less{1}, only_less{2}) == std::weak_ordering::less);
static_assert(cmp::as_less(by_priority)(order{1, 0}, order{2, 0}) && !cmp::as_less(by_priority)(order{2, 0}, order{1, 0}));
static_assert(std::is_trivially_copyable_v<decltype(cmp::lexicographic(by_priority, cmp::reversed(by_price)))>);

constexpr bool sorts_by_two_keys()
{
    std::array<order, 4> xs{{{2, 1.0}, {1, 3.0}, {1, 2.0}, {2, 0.5}}};
    std::ranges::sort(xs, cmp::as_less(cmp::lexicographic(by_priority, cmp::reversed(by_price))));
    return xs[0].price == 3.0 && xs[1].price == 2.0 && xs[2].price == 1.0 && xs[3].price == 0.5;
}
static_assert(sorts_by_two_keys());

// ---- into ----------------------------------------------------------------------------------------------------------

static_assert(into<double>(1.5f) == 1.5 && into<long>(3) == 3L);
static_assert(std::is_same_v<decltype(into<double>(1.5f)), double>);
static_assert(std::is_invocable_v<decltype(into<long long>), int>);
static_assert(!std::is_invocable_v<decltype(into<int>), double>);        // narrowing, as Rust has no From<f64> for i32
static_assert(!std::is_invocable_v<decltype(into<double>), int>);        // narrowing by core's brace rules
static_assert(!std::is_invocable_v<decltype(into<int>), const char*>);   // no conversion at all
static_assert(!std::is_invocable_v<decltype(into<std::string>), int>);
static_assert(noexcept(into<double>(1.0f)));
static_assert(!noexcept(into<std::string>("x")));

TEST(FunctionalRustOption, MovesFromATemporaryAndCopiesFromAnLvalue)
{
    std::optional<std::unique_ptr<int>> p = std::make_unique<int>(5);
    auto moved = option::filter(std::move(p), [](const std::unique_ptr<int>& q) { return *q == 5; });
    ASSERT_TRUE(moved.has_value());
    EXPECT_EQ(**moved, 5);

    const std::optional<std::string> s = std::string("held");
    EXPECT_EQ(option::unwrap_or_default(s), "held");
    EXPECT_EQ(*s, "held");
    EXPECT_EQ(option::unwrap_or_default(std::optional<std::string>()), "");
}

TEST(FunctionalRustOption, InspectSeesTheValueAndReturnsTheOptional)
{
    int seen = 0;
    std::optional<int> o = 4;
    std::optional<int>& same = option::inspect(o, [&](int x) { seen = x; });
    EXPECT_EQ(&same, &o);
    EXPECT_EQ(seen, 4);
    const std::optional<int> copy = option::inspect(std::optional<int>(), [&](int) { seen = -1; });
    EXPECT_FALSE(copy.has_value());
    EXPECT_EQ(seen, 4);
    static_assert(std::is_same_v<decltype(option::inspect(std::optional<int>(), [](int) {})), std::optional<int>>);
}

TEST(FunctionalRustOption, LazyAlternativesRunOnlyWhenNeeded)
{
    int calls = 0;
    auto count = [&] { ++calls; return 0; };
    EXPECT_EQ(option::unwrap_or_else(some(1), count), 1);
    EXPECT_EQ(option::map_or_else(some(1), count, twice), 2);
    EXPECT_TRUE(option::ok_or_else(some(1), count).is_ok());
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(option::unwrap_or_else(none, count), 0);
    EXPECT_EQ(calls, 1);
}

TEST(FunctionalRustOption, AThrowingInitialiserLeavesTheOptionalEmpty)
{
    std::optional<int> o;
    EXPECT_THROW(static_cast<void>(option::get_or_insert_with(o, []() -> int { throw std::runtime_error("no"); })),
                 std::runtime_error);
    EXPECT_FALSE(o.has_value());
}

TEST(FunctionalRustCmp, ThenWithRunsOnlyOnATie)
{
    int calls = 0;
    auto tie_break = [&] { ++calls; return std::strong_ordering::less; };
    EXPECT_EQ(cmp::then_with(std::strong_ordering::greater, tie_break), std::strong_ordering::greater);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(cmp::then_with(std::strong_ordering::equal, tie_break), std::strong_ordering::less);
    EXPECT_EQ(calls, 1);
}

} // namespace
