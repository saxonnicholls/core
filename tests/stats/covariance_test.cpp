//
//  covariance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/covariance.hpp>: Ledoit and Wolf's shrinkage against
//  scikit-learn 1.9.1's LedoitWolf on the same data, and the sample covariance's definition.
//

#include <snicholls/core/stats/covariance.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

// The same numbers in C++ and in the Python that produced the references
double u(std::uint64_t k) { return static_cast<double>((k * 2654435761ULL) % 4294967296ULL) / 4294967296.0 - 0.5; }

std::vector<std::array<double, 8>> panel()
{
    std::vector<std::array<double, 8>> rows(60);
    for (std::uint64_t t = 0; t < 60; ++t)
        for (std::uint64_t j = 0; j < 8; ++j)
            rows[t][j] = 0.5 * u(1000 + t) + (0.2 + 0.1 * static_cast<double>(j)) * u(5000 + 97 * t + j);
    return rows;
}

TEST(LedoitWolf, AgreesWithScikitLearn)
{
    const auto lw = snicholls::stats::ledoit_wolf(panel());
    ASSERT_EQ(lw.n, 8u);
    EXPECT_NEAR(lw.shrinkage, 0.09409011666453788, 1e-13);
    EXPECT_NEAR(lw(0, 0), 0.02723034464991899, 1e-15);
    EXPECT_NEAR(lw(3, 7), 0.001904973198597772, 1e-15);
    EXPECT_NEAR(lw(7, 3), lw(3, 7), 0);
    EXPECT_NEAR(lw(7, 7), 0.07911773427638945, 1e-15);
}

TEST(LedoitWolf, ShrinksTheSampleTowardsTheMeanVariance)
{
    const auto rows = panel();
    const auto s = snicholls::stats::sample_covariance(rows);
    const auto lw = snicholls::stats::ledoit_wolf(rows);
    EXPECT_EQ(s.shrinkage, 0.0);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            EXPECT_NEAR(lw(i, j), (1 - lw.shrinkage) * s(i, j) + (i == j ? lw.shrinkage * lw.target : 0.0), 1e-16);
    EXPECT_TRUE(snicholls::stats::ledoit_wolf(std::vector<std::array<double, 3>>{}).values.empty());
}

TEST(SampleCovariance, IsDividedByTAboutTheMean)
{
    const std::vector<std::array<double, 2>> rows{{1, 2}, {3, 2}, {5, 8}};
    const auto s = snicholls::stats::sample_covariance(rows);
    EXPECT_DOUBLE_EQ(s(0, 0), 8.0 / 3.0);   // deviations -2, 0, 2
    EXPECT_DOUBLE_EQ(s(0, 1), 4.0);         // with -2, -2, 4: (4 + 0 + 8) / 3
    EXPECT_DOUBLE_EQ(s(1, 1), 8.0);
}

} // namespace
