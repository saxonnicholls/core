//
//  harsanyi_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/game/harsanyi.hpp>: the transforms invert each other, unanimity games
//  have one dividend, and the Shapley and Banzhaf values agree with game/shapley.hpp.
//

#include <snicholls/core/game/harsanyi.hpp>

#include <gtest/gtest.h>

#include <array>
#include <random>
#include <span>
#include <vector>

namespace {

namespace game = snicholls::game;
using game::coalition;

// One dividend of 1 at {0, 2}; nothing else
constexpr auto unanimity_02 = game::harsanyi_dividends<3>([](coalition s) { return (s & 0b101) == 0b101 ? 1.0 : 0.0; });
static_assert(unanimity_02 == std::array{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0});

// The transforms are exact on integers, and invert each other
consteval bool round_trip()
{
    std::array<long, 8> t{3, -1, 4, 1, -5, 9, 2, -6};
    const auto original = t;
    game::to_dividends(std::span<long>(t));
    game::from_dividends(std::span<long>(t));
    return t == original;
}
static_assert(round_trip());

TEST(Harsanyi, AQuadraticsDividendsAreItsDiagonalAndCrossTerms)
{
    // x'Px with P = [[1, 2], [2, 3]] on the features switched on
    const auto v = [](coalition s) {
        const double x0 = game::contains(s, 0) ? 1 : 0, x1 = game::contains(s, 1) ? 1 : 0;
        return x0 * x0 * 1 + 2 * 2 * x0 * x1 + 3 * x1 * x1;
    };
    const auto d = game::harsanyi_dividends<2>(v);
    EXPECT_EQ(d, (std::array{0.0, 1.0, 3.0, 4.0}));   // singletons P_ii, the pair 2 P_01
}

TEST(Harsanyi, ShapleyAndBanzhafFromDividendsAgreeWithTheDirectMethods)
{
    std::mt19937 g(17);
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> table(64);
    for (double& x : table)
        x = u(g);
    const auto direct = game::shapley<6>(game::from_table(table));
    const auto banzhaf = game::banzhaf<6>(game::from_table(table));
    const auto d = game::harsanyi_dividends(6, game::from_table(table));
    const auto via = game::shapley_from_dividends(std::span<const double>(d));
    const auto via_banzhaf = game::banzhaf_from_dividends(std::span<const double>(d));
    EXPECT_NEAR(via.baseline, direct.baseline, 1e-12);
    EXPECT_NEAR(via.total, direct.total, 1e-12);
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(via.values[i], direct.values[i], 1e-12);
        EXPECT_NEAR(via_banzhaf.values[i], banzhaf.values[i], 1e-12);
    }
}

} // namespace
