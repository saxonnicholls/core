//
//  martingale_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stochastic/martingale.hpp>: the Doob decomposition of a drifting
//  walk at compile time, the transform's fairness, a Markov chain's decomposition, and the
//  tests against Python arch 8.0 and SciPy on the same data.
//

#include <snicholls/core/stochastic/martingale.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

namespace st = snicholls::stochastic;

// A walk with drift 0.5: X = 0, 1.5, 1, 2.5. Predicted each step: the last value plus 0.5.
consteval bool drifting_walk()
{
    const std::array<double, 4> x{0, 1.5, 1, 2.5};
    const std::array<double, 3> predicted{0.5, 2.0, 1.5};
    const auto parts = st::doob_decomposition(x, predicted);
    return parts.compensator == std::vector<double>{0, 0.5, 1.0, 1.5} && parts.martingale == std::vector<double>{0, 1.0, 0.0, 1.0};
}
static_assert(drifting_walk());

TEST(Martingale, TransformAndQuadraticVariation)
{
    const std::vector<double> x{100, 101, 99, 102};
    const std::vector<double> h{0, 2, -1, 3};   // h[k] held over the move into k
    EXPECT_EQ(st::martingale_transform(h, x), (std::vector<double>{0, 2, 4, 13}));
    EXPECT_EQ(st::quadratic_variation(x), (std::vector<double>{0, 1, 5, 14}));
}

TEST(Martingale, AMarkovChainPredictsItsOwnFunctions)
{
    const st::markov_chain chain(2, {0.9, 0.1, 0.5, 0.5});
    const std::vector<double> f{0.0, 10.0};
    const std::vector<std::size_t> path{0, 0, 1, 1};
    const auto parts = chain.doob(path, f);
    // predicted from state 0 is 1, from state 1 is 5
    EXPECT_DOUBLE_EQ(parts.compensator[3], (1 - 0) + (1 - 0) + (5 - 10));
    EXPECT_DOUBLE_EQ(parts.martingale[3], (0 - 1) + (10 - 1) + (10 - 5));
}

double w(std::uint64_t k) { return static_cast<double>((k * k * 40503ULL + k * 2654435761ULL + 12345ULL) % 4294967296ULL) / 4294967296.0 - 0.5; }

TEST(MartingaleTests, AgreeWithArchAndScipy)
{
    std::vector<double> inc(400);
    for (std::size_t t = 0; t < inc.size(); ++t)
        inc[t] = (t > 0 ? 0.3 * inc[t - 1] : 0.0) + w(t);
    std::vector<double> y{0.0};
    for (const double v : inc)
        y.push_back(y.back() + v);
    const auto vr = st::variance_ratio_test(y, 4);
    EXPECT_NEAR(vr.statistic, -4.999241713935683, 1e-9);
    EXPECT_NEAR(vr.p_value, 5.75562140614494e-07, 1e-12);
    const auto lb = st::ljung_box(inc, 5);
    EXPECT_NEAR(lb.statistic, 147.21539909449092, 1e-9);
    EXPECT_LT(lb.p_value, 1e-12);
}

TEST(MartingaleTests, ARandomWalkPasses)
{
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    std::vector<double> y{0.0}, steps;
    for (int i = 0; i < 2000; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        steps.push_back(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
        y.push_back(y.back() + steps.back());
    }
    EXPECT_GT(st::variance_ratio_test(y, 5).p_value, 0.01);
    EXPECT_GT(st::ljung_box(steps, 10).p_value, 0.01);
}

} // namespace
