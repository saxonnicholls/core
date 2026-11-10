//
//  change_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/game/change.hpp>: a product's change shared exactly, at compile time
//  and at run time, with no residual.
//

#include <snicholls/core/game/change.hpp>

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace {

namespace game = snicholls::game;

constexpr auto product = [](const std::array<double, 2>& x) { return x[0] * x[1]; };
constexpr std::array<double, 2> before{1, 1};
constexpr std::array<double, 2> after{3, 5};
constexpr auto explained = game::attribute_change<2>(product, before, after);
static_assert(explained.baseline == 1 && explained.total == 15);
static_assert(explained.values[0] == 6 && explained.values[1] == 8);   // (2 + 10) / 2 and (4 + 12) / 2
static_assert(explained.unattributed() == 0);

TEST(Change, ARunTimeSizedPointWorksAlike)
{
    const auto f = [](const std::vector<double>& x) { return x[0] * x[1] + x[2]; };
    const std::vector<double> from{1, 1, 0};
    const std::vector<double> to{3, 5, 7};
    const auto a = game::attribute_change(f, from, to);
    ASSERT_EQ(a.values.size(), 3u);
    EXPECT_DOUBLE_EQ(a.values[0], 6);
    EXPECT_DOUBLE_EQ(a.values[1], 8);
    EXPECT_DOUBLE_EQ(a.values[2], 7);
    EXPECT_DOUBLE_EQ(a.unattributed(), 0);
}

TEST(Change, AnInputThatDidNotMoveGetsNothing)
{
    const auto f = [](const std::array<double, 3>& x) { return x[0] * x[1] * x[2]; };
    const auto a = game::attribute_change<3>(f, std::array{1.0, 2.0, 3.0}, std::array{2.0, 2.0, 4.0});
    EXPECT_DOUBLE_EQ(a.values[1], 0);
    EXPECT_NEAR(a.unattributed(), 0, 1e-12);
}

} // namespace
