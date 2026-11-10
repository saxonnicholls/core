//
//  coalition_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/game/coalition.hpp>: every enumeration visits each coalition once,
//  in its order, at compile time; tables and the memoized game.
//

#include <snicholls/core/game/coalition.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <set>
#include <vector>

namespace {

namespace game = snicholls::game;
using game::coalition;

static_assert(game::grand_coalition(0) == 0 && game::grand_coalition(3) == 0b111 && game::grand_coalition(64) == ~coalition{0});
static_assert(game::contains(0b101, 2) && !game::contains(0b101, 1));
static_assert(game::with(0b001, 2) == 0b101 && game::without(0b101, 0) == 0b100);
static_assert(game::coalition_size(0b1011) == 3 && game::table_size(4) == 16);
static_assert(game::binomial(5, 2) == 10 && game::binomial(5, 6) == 0 && game::binomial(0, 0) == 1);
static_assert(game::binomial(64, 32) == 1832624140942590534ULL);

consteval std::size_t count_all(std::size_t n)
{
    std::size_t count = 0;
    game::for_each_coalition(n, [&count](coalition) noexcept { ++count; });
    return count;
}
static_assert(count_all(0) == 1 && count_all(5) == 32);

consteval std::size_t count_size(std::size_t n, std::size_t k)
{
    std::size_t count = 0;
    bool right_size = true;
    game::for_each_coalition_of_size(n, k, [&](coalition s) noexcept {
        ++count;
        right_size = right_size && game::coalition_size(s) == k;
    });
    return right_size ? count : 999;
}
static_assert(count_size(6, 3) == 20 && count_size(6, 0) == 1 && count_size(6, 6) == 1 && count_size(3, 4) == 0 && count_size(63, 1) == 63);

consteval std::size_t count_subsets(coalition s)
{
    std::size_t count = 0;
    game::for_each_subset_of(s, [&](coalition sub) noexcept { count += (sub & ~s) == 0 ? 1 : 1000; });
    return count;
}
static_assert(count_subsets(0b1011) == 8 && count_subsets(0) == 1);

consteval std::size_t count_supersets(coalition s, std::size_t n)
{
    std::size_t count = 0;
    game::for_each_superset_of(s, n, [&](coalition sup) noexcept { count += (sup & s) == s ? 1 : 1000; });
    return count;
}
static_assert(count_supersets(0b0101, 4) == 4 && count_supersets(0b1111, 4) == 1);

// A table at compile time: a game is a map from 2^N to its values
constexpr auto sizes = game::make_table<3>([](coalition s) { return game::coalition_size(s); });
static_assert(sizes == std::array<std::size_t, 8>{0, 1, 1, 2, 1, 2, 2, 3});

static_assert(game::game<decltype([](coalition) { return 1.0; })>);
static_assert(!game::game<decltype([](int, int) { return 1.0; })>);
static_assert(std::is_same_v<game::game_scalar_t<float>, float> && std::is_same_v<game::game_scalar_t<int>, double>);

TEST(Coalition, GrayCodeChangesOnePlayerAtATimeAndVisitsEveryCoalition)
{
    std::set<coalition> seen;
    coalition previous = 0;
    bool first = true;
    game::for_each_coalition_gray(6, [&](coalition s, std::size_t changed) {
        if (first) {
            EXPECT_EQ(s, 0u);
            EXPECT_EQ(changed, game::no_player);
            first = false;
        } else {
            EXPECT_EQ(std::popcount(s ^ previous), 1);
            EXPECT_EQ(s ^ previous, game::singleton(changed));
        }
        seen.insert(s);
        previous = s;
    });
    EXPECT_EQ(seen.size(), 64u);
}

TEST(Coalition, SizesComeOutInAscendingOrder)
{
    std::vector<coalition> pairs;
    game::for_each_coalition_of_size(4, 2, [&](coalition s) { pairs.push_back(s); });
    EXPECT_EQ(pairs, (std::vector<coalition>{0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100}));
}

TEST(Coalition, TabulateAndTableGameAgree)
{
    const auto table = game::tabulate(4, [](coalition s) { return static_cast<double>(s) * 0.5; });
    ASSERT_EQ(table.size(), 16u);
    const auto v = game::from_table(table);
    EXPECT_EQ(v.players(), 4u);
    EXPECT_EQ(v(0b1010), 5.0);
}

TEST(Coalition, MemoizedEvaluatesEachCoalitionOnce)
{
    int calls = 0;
    const game::memoized v([&calls](coalition s) {
        ++calls;
        return static_cast<double>(s);
    });
    EXPECT_EQ(v(3), 3.0);
    EXPECT_EQ(v(3), 3.0);
    EXPECT_EQ(v(1), 1.0);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(v.evaluations(), 2u);
}

} // namespace
