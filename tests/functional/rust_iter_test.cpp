//
//  rust_iter_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/rust_iter.hpp>: Rust's slice and iterator methods that C++20 lacks.
//

#include <snicholls/core/functional/rust_iter.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace slice = snicholls::fn::rust::slice;
namespace iter = snicholls::fn::rust::iter;
using snicholls::fn::rust::slice_view;
using snicholls::fn::rust::step_view;

constexpr std::array<int, 5> xs{1, 2, 3, 4, 5};

constexpr int sum(std::span<const int> s)
{
    int t = 0;
    for (int x : s) t += x;
    return t;
}

// ---- windows -------------------------------------------------------------------------------------------------------

static_assert(slice::windows(xs, 2).size() == 4 && slice::windows(xs, 5).size() == 1 && slice::windows(xs, 6).size() == 0);
static_assert(sum(slice::windows(xs, 3)[0]) == 6 && sum(slice::windows(xs, 3)[2]) == 12);
static_assert(slice::windows(xs, 1).size() == 5 && slice::windows(xs, 1).remainder().empty());
static_assert(slice::windows(xs, 3).remainder().empty());
static_assert(std::is_same_v<decltype(slice::windows(xs, 3)[0]), std::span<const int>>);
static_assert(std::is_same_v<decltype(slice::windows_of<3>(xs)[0]), std::span<const int, 3>>);
static_assert(slice::windows_of<3>(xs).size() == 3 && slice::windows_of<3>(xs)[2][2] == 5);
static_assert(slice::windows(std::span<const int>(), 2).empty());

constexpr bool moving_sums_are_right()
{
    std::array<int, 3> out{};
    std::size_t i = 0;
    for (auto w : slice::windows(xs, 3)) out[i++] = sum(w);
    return i == 3 && out == std::array<int, 3>{6, 9, 12};
}
static_assert(moving_sums_are_right());

// ---- chunks --------------------------------------------------------------------------------------------------------

static_assert(slice::chunks(xs, 2).size() == 3 && slice::chunks(xs, 2)[2].size() == 1 && slice::chunks(xs, 2)[2][0] == 5);
static_assert(slice::chunks(xs, 5).size() == 1 && slice::chunks(xs, 7).size() == 1 && slice::chunks(xs, 7)[0].size() == 5);
static_assert(slice::chunks(xs, 2).remainder().empty());
static_assert(slice::chunks_exact(xs, 2).size() == 2 && slice::chunks_exact(xs, 2).remainder().size() == 1);
static_assert(slice::chunks_exact(xs, 2).remainder()[0] == 5);
static_assert(slice::chunks_exact(xs, 7).size() == 0 && slice::chunks_exact(xs, 7).remainder().size() == 5);
static_assert(slice::chunks_exact(xs, 5).remainder().empty());
static_assert(std::is_same_v<decltype(slice::chunks_exact_of<2>(xs)[0]), std::span<const int, 2>>);
static_assert(slice::chunks_exact_of<2>(xs)[1][1] == 4);
static_assert(slice::chunks(std::span<const int>(), 3).empty());

// The views are views: small, trivially copyable, and they compose with std::views.
static_assert(std::ranges::view<slice_view<const int>> && std::ranges::forward_range<slice_view<const int>>);
static_assert(std::ranges::view<step_view<int>> && std::ranges::forward_range<step_view<int>>);
static_assert(std::is_trivially_copyable_v<slice_view<const int>> && std::is_trivially_copyable_v<step_view<int>>);
static_assert(sizeof(slice_view<const int>) == sizeof(void*) + 4 * sizeof(std::size_t));
static_assert(sizeof(step_view<int>) == sizeof(void*) + 2 * sizeof(std::size_t));
static_assert(noexcept(slice::windows(xs, 2)) && noexcept(iter::step_by(xs, 2)));

constexpr bool composes_with_views()
{
    int t = 0;
    for (int s : slice::windows(xs, 2) | std::views::transform(sum)) t += s;
    return t == 3 + 5 + 7 + 9;
}
static_assert(composes_with_views());

// ---- dedup ---------------------------------------------------------------------------------------------------------

constexpr std::array<int, 7> runs{1, 1, 2, 2, 2, 1, 3};
static_assert(slice::dedup(runs).size() == 4 && slice::dedup(runs)[2] == 1 && slice::dedup(runs)[3] == 3);
static_assert(slice::dedup_by_key(runs, [](int x) { return x % 2; }).size() == 3);
static_assert(slice::dedup(std::array<int, 0>{}).empty());

// ---- enumerate, step_by --------------------------------------------------------------------------------------------

constexpr bool enumerate_pairs_index_and_element()
{
    std::size_t weighted = 0;
    for (auto [i, x] : iter::enumerate(xs)) weighted += i * static_cast<std::size_t>(x);
    return weighted == 0 * 1 + 1 * 2 + 2 * 3 + 3 * 4 + 4 * 5;
}
static_assert(enumerate_pairs_index_and_element());

constexpr bool enumerate_unsized()
{
    std::size_t last = 0;
    for (auto [i, x] : iter::enumerate(xs | std::views::filter([](int x) { return x % 2 == 1; }))) last = i * 10 + static_cast<std::size_t>(x);
    return last == 25;
}
static_assert(enumerate_unsized());

static_assert(iter::step_by(xs, 2).size() == 3 && iter::step_by(xs, 2)[2] == 5);
static_assert(iter::step_by(xs, 5).size() == 1 && iter::step_by(xs, 9).size() == 1 && iter::step_by(xs, 1).size() == 5);
static_assert(iter::step_by(std::span<const int>(), 2).empty());

constexpr bool step_by_writes_through()
{
    std::array<int, 5> ys{};
    for (int& y : iter::step_by(ys, 2)) y = 1;
    return ys == std::array<int, 5>{1, 0, 1, 0, 1};
}
static_assert(step_by_writes_through());

// ---- searches --------------------------------------------------------------------------------------------------------

constexpr auto half_if_even = [](int x) { return x % 2 == 0 ? std::optional<int>(x / 2) : std::nullopt; };
static_assert(iter::find_map(xs, half_if_even) == std::optional<int>(1));
static_assert(iter::find_map(std::array<int, 2>{1, 3}, half_if_even) == std::nullopt);

struct item {
    int key;
    int id;
};
constexpr std::array<item, 5> items{{{1, 0}, {3, 1}, {0, 2}, {3, 3}, {0, 4}}};
static_assert(iter::max_by_key(items, &item::key)->id == 3);   // the last of the greatest, as Rust
static_assert(iter::min_by_key(items, &item::key)->id == 2);   // the first of the least, as Rust
static_assert(!iter::max_by_key(std::array<item, 0>{}, &item::key).has_value());
static_assert(iter::max_by(items, [](const item& a, const item& b) { return a.key <=> b.key; })->id == 3);
static_assert(iter::min_by(items, [](const item& a, const item& b) { return a.key <=> b.key; })->id == 2);
static_assert(!iter::min_by(std::array<item, 0>{}, [](const item& a, const item& b) { return a.key <=> b.key; }).has_value());

static_assert(iter::nth(xs, 0) == std::optional<int>(1) && iter::nth(xs, 4) == std::optional<int>(5) && iter::nth(xs, 5) == std::nullopt);
static_assert(iter::last(xs) == std::optional<int>(5) && iter::last(std::array<int, 0>{}) == std::nullopt);

// ---- try_for_each --------------------------------------------------------------------------------------------------

constexpr snicholls::result<void, int> positive(int x)
{
    if (x > 0) return snicholls::ok();
    return snicholls::err(x);
}
static_assert(iter::try_for_each(xs, positive).is_ok());
static_assert(iter::try_for_each(std::array<int, 4>{1, -2, -3, 4}, positive).unwrap_err_unchecked() == -2);
static_assert(iter::try_for_each(std::array<int, 0>{}, positive).is_ok());
static_assert(iter::try_for_each(xs, [](int x) { return x < 4 ? std::optional<std::monostate>(std::monostate{}) : std::nullopt; }) == std::nullopt);

TEST(FunctionalRustIter, TryForEachStopsAtTheFirstFailure)
{
    std::vector<int> seen;
    const auto r = iter::try_for_each(std::vector<int>{3, 0, 5}, [&](int x) {
        seen.push_back(x);
        return positive(x);
    });
    EXPECT_TRUE(r.is_err());
    EXPECT_EQ(seen, (std::vector<int>{3, 0}));
}

TEST(FunctionalRustIter, ViewsOverAVectorReadItsElements)
{
    std::vector<std::string> words{"a", "b", "c", "d", "e"};
    std::string joined;
    for (auto chunk : slice::chunks(words, 2)) {
        for (const auto& w : chunk) joined += w;
        joined += '|';
    }
    EXPECT_EQ(joined, "ab|cd|e|");
    for (auto& w : iter::step_by(words, 2)) w += "!";
    EXPECT_EQ(words[0], "a!");
    EXPECT_EQ(words[1], "b");
    EXPECT_EQ(words[4], "e!");
}

TEST(FunctionalRustIter, MaxByKeyCallsTheKeyOncePerElement)
{
    int calls = 0;
    const auto best = iter::max_by_key(std::vector<int>{4, 9, 2}, [&](int x) { ++calls; return -x; });
    EXPECT_EQ(best, std::optional<int>(2));
    EXPECT_EQ(calls, 3);
}

TEST(FunctionalRustIter, AViewKeepsWorkingAfterItIsMoved)
{
    const std::vector<int> v{1, 2, 3, 4};
    auto w = slice::windows(v, 2);
    auto it = w.begin();
    const auto moved = std::move(w);
    EXPECT_EQ((*it)[1], 2);
    EXPECT_EQ(moved.size(), 3u);
}

} // namespace
