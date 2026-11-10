//
//  tabular_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/game/tabular.hpp>: a linear model's attribution in closed form, an
//  interaction shared equally, and any row and feature type.
//

#include <snicholls/core/game/tabular.hpp>

#include <snicholls/core/game/shapley.hpp>

#include <gtest/gtest.h>

#include <array>
#include <span>
#include <vector>

namespace {

namespace game = snicholls::game;

TEST(Tabular, ALinearModelGetsWeightTimesDistanceFromTheBackgroundMean)
{
    const std::array<double, 3> w{2.0, -1.0, 0.5};
    const auto model = [&w](std::span<const double> x) { return 10 + w[0] * x[0] + w[1] * x[1] + w[2] * x[2]; };
    const std::vector<double> instance{1, 2, 3};
    const std::vector<std::vector<double>> background{{0, 0, 0}, {2, 4, 6}, {1, 1, 0}};   // means 1, 5/3, 2
    const auto v = game::interventional(model, instance, background);
    EXPECT_EQ(v.players(), 3u);
    const auto a = game::shapley<3>(v);
    EXPECT_NEAR(a.values[0], 2.0 * (1 - 1), 1e-12);
    EXPECT_NEAR(a.values[1], -1.0 * (2 - 5.0 / 3.0), 1e-12);
    EXPECT_NEAR(a.values[2], 0.5 * (3 - 2), 1e-12);
    EXPECT_NEAR(a.total, model(instance), 1e-12);
    EXPECT_NEAR(a.unattributed(), 0, 1e-12);
}

TEST(Tabular, AnInteractionIsSharedEquallyWithOneBackgroundRow)
{
    const auto model = [](std::span<const int> x) { return static_cast<double>(x[0] * x[1]); };
    const std::array<int, 2> instance{3, 5};
    const std::array<std::array<int, 2>, 1> background{{{1, 1}}};
    const auto a = game::shapley<2>(game::interventional(model, instance, background));
    EXPECT_DOUBLE_EQ(a.values[0], 6.0);   // (3 - 1) / 2 + (15 - 5) / 2
    EXPECT_DOUBLE_EQ(a.values[1], 8.0);
}

} // namespace
