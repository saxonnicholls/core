//
//  owen_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/game/owen.hpp>: the two trivial groupings give Shapley, the groups get
//  the quotient game's Shapley values, a worked majority game, and 40 players without a table.
//

#include <snicholls/core/game/owen.hpp>

#include <gtest/gtest.h>

#include <array>
#include <random>
#include <vector>

namespace {

namespace game = snicholls::game;
using game::coalition;

constexpr auto majority = [](coalition s) { return game::coalition_size(s) >= 2 ? 1.0 : 0.0; };

static_assert(game::is_partition_of(3, std::array<coalition, 2>{0b011, 0b100}));
static_assert(!game::is_partition_of(3, std::array<coalition, 2>{0b011, 0b110}));   // overlap
static_assert(!game::is_partition_of(3, std::array<coalition, 1>{0b011}));          // misses one
static_assert(!game::is_partition_of(2, std::array<coalition, 2>{0b11, 0}));        // an empty group
static_assert(game::owen<3>(majority, std::array<coalition, 2>{0b011, 0b100}).values == std::array{0.5, 0.5, 0.0});

TEST(Owen, AGroupOfTwoCarriesAMajorityGame)
{
    // Shapley gives a third each; grouped, {0, 1} wins either order and player 2 never pivots
    const auto shapley = game::shapley<3>(majority);
    const auto owen = game::owen<3>(majority, std::array<coalition, 2>{0b011, 0b100});
    EXPECT_NEAR(shapley.values[2], 1.0 / 3.0, 1e-15);
    EXPECT_EQ(owen.values, (std::array{0.5, 0.5, 0.0}));
    EXPECT_EQ(owen.unattributed(), 0.0);
}

TEST(Owen, SingletonsAndOneGroupAreShapley)
{
    std::mt19937 g(4);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> table(32);
    for (double& x : table)
        x = u(g);
    const auto v = game::from_table(table);
    const auto shapley = game::shapley<5>(v);
    const auto singletons = game::owen<5>(v, std::array<coalition, 5>{1, 2, 4, 8, 16});
    const auto one_group = game::owen(5, v, std::array<coalition, 1>{0b11111});
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(singletons.values[i], shapley.values[i], 1e-12);
        EXPECT_NEAR(one_group.values[i], shapley.values[i], 1e-12);
    }
}

TEST(Owen, EachGroupGetsItsShapleyValueInTheGameBetweenGroups)
{
    std::mt19937 g(8);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> table(64);
    for (double& x : table)
        x = u(g);
    const auto v = game::from_table(table);
    const std::array<coalition, 3> groups{0b000111, 0b011000, 0b100000};
    const auto owen = game::owen<6>(v, groups);
    const auto between = game::shapley<3>(game::quotient_game(v, groups));
    EXPECT_NEAR(owen.values[0] + owen.values[1] + owen.values[2], between.values[0], 1e-12);
    EXPECT_NEAR(owen.values[3] + owen.values[4], between.values[1], 1e-12);
    EXPECT_NEAR(owen.values[5], between.values[2], 1e-12);
    EXPECT_NEAR(owen.unattributed(), 0, 1e-12);
}

TEST(Owen, FortyPlayersInFourGroupsNeedNoTable)
{
    // Additive, so every player's value is its own weight; evaluations counted
    std::array<double, 40> weight{};
    for (std::size_t i = 0; i < 40; ++i)
        weight[i] = static_cast<double>(i) - 13.5;
    std::size_t calls = 0;
    const auto v = [&](coalition s) {
        ++calls;
        double sum = 0;
        for (std::size_t i = 0; i < 40; ++i)
            sum += game::contains(s, i) ? weight[i] : 0;
        return sum;
    };
    const std::array<coalition, 4> groups{0x3FFull, 0x3FFull << 10, 0x3FFull << 20, 0x3FFull << 30};
    const auto owen = game::owen<40>(v, groups);
    for (std::size_t i = 0; i < 40; ++i)
        EXPECT_NEAR(owen.values[i], weight[i], 1e-9);
    EXPECT_EQ(calls, 2u + 4u * 8u * 1024u);   // 2 + sum over groups of 2^(m-1) 2^b
}

} // namespace
