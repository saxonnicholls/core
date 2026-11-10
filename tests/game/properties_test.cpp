//
//  properties_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/game/properties.hpp>: each property and its counterexample, and
//  Shapley's theorem that a convex game's Shapley value is in its core.
//

#include <snicholls/core/game/properties.hpp>

#include <gtest/gtest.h>

#include <array>

namespace {

namespace game = snicholls::game;
using game::coalition;

// x'Px on the players switched on: supermodular when every cross term is non-negative
template <std::size_t N>
constexpr auto quadratic(const std::array<std::array<double, N>, N>& p)
{
    return [p](coalition s) {
        double sum = 0;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                if (game::contains(s, i) && game::contains(s, j))
                    sum += p[i][j];
        return sum;
    };
}

constexpr auto complements = quadratic<3>({{{1, 0.5, 0.25}, {0.5, 2, 0}, {0.25, 0, 1}}});
constexpr auto substitutes = quadratic<3>({{{1, -0.25, -0.25}, {-0.25, 1, -0.25}, {-0.25, -0.25, 1}}});
constexpr auto linear = [](coalition s) { return static_cast<double>(s); };
constexpr auto majority = [](coalition s) { return game::coalition_size(s) >= 2 ? 1.0 : 0.0; };

static_assert(game::is_supermodular<3>(complements) && !game::is_submodular<3>(complements));
static_assert(game::is_submodular<3>(substitutes) && !game::is_supermodular<3>(substitutes));
static_assert(game::is_additive<3>(linear) && game::is_supermodular<3>(linear) && game::is_submodular<3>(linear));
static_assert(game::is_monotone<3>(complements) && game::is_superadditive<3>(complements));
static_assert(!game::is_superadditive<3>(substitutes));
static_assert(game::is_monotone<3>(majority) && !game::is_supermodular<3>(majority));

TEST(GameProperties, ACounterexampleSaysWhere)
{
    const auto bad = game::find_supermodularity_violation<3>(substitutes);
    ASSERT_TRUE(bad.has_value());
    EXPECT_EQ(bad->base, 0u);
    EXPECT_EQ(bad->first, 0u);
    EXPECT_EQ(bad->second, 1u);
    EXPECT_DOUBLE_EQ(bad->difference, -0.5);
    const auto falls = game::find_monotonicity_violation<2>([](coalition s) { return s == 0b11 ? 0.0 : static_cast<double>(s); });
    ASSERT_TRUE(falls.has_value());
    EXPECT_EQ(falls->base, 0b01u);
    EXPECT_EQ(falls->player, 1u);
    const auto apart = game::find_superadditivity_violation<3>(substitutes);
    ASSERT_TRUE(apart.has_value());
    EXPECT_LT(apart->shortfall, 0);
}

TEST(GameProperties, AConvexGamesShapleyValueIsInItsCore)
{
    const auto a = game::shapley<3>(complements);
    EXPECT_TRUE(game::in_core<3>(complements, a.values, 1e-12));
}

TEST(GameProperties, AMajorityGameHasNoCoreAndSaysWhoBlocks)
{
    const auto a = game::shapley<3>(majority);
    const auto blocked = game::find_core_violation<3>(majority, a.values, 1e-12);
    ASSERT_TRUE(blocked.has_value());
    EXPECT_EQ(game::coalition_size(blocked->blocking), 2u);   // two players earn 1 and are given 2/3
    EXPECT_NEAR(blocked->shortfall, -1.0 / 3.0, 1e-12);
    EXPECT_FALSE(game::in_core<3>(majority, std::array{0.5, 0.5, 0.5}, 1e-12));   // hands out too much
}

TEST(GameProperties, ToleranceAbsorbsRounding)
{
    const auto nearly = [](coalition s) { return s == 0b11 ? 1.0 - 1e-13 : (s == 0 ? 0.0 : 0.5); };
    EXPECT_FALSE(game::is_supermodular<2>(nearly));
    EXPECT_TRUE(game::is_supermodular<2>(nearly, 1e-12));
}

} // namespace
