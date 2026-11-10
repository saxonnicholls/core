//
//  bootstrap_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/bootstrap.hpp>: each resampler's defining property, the
//  agreement of indices and counts, the distribution's summaries, and the standard error of a
//  mean against its formula.
//

#include <snicholls/core/stats/bootstrap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace {

namespace stats = snicholls::stats;
using stats::wyrand;

std::vector<double> sample(std::size_t n, std::uint64_t seed)
{
    wyrand g(seed);
    std::vector<double> x(n);
    for (double& v : x)
        v = stats::uniform_real(g) * 2 - 1 + 0.05;   // mean 0.05, variance 1/3
    return x;
}

TEST(Bootstrap, IndexResamplesAreInRangeAndReproducible)
{
    std::vector<std::uint32_t> first;
    wyrand g1(1), g2(1);
    stats::for_each_resample(10, 3, g1, [&](std::span<const std::uint32_t> idx) {
        ASSERT_EQ(idx.size(), 10u);
        for (const std::uint32_t i : idx)
            ASSERT_LT(i, 10u);
        first.insert(first.end(), idx.begin(), idx.end());
    });
    std::vector<std::uint32_t> second;
    stats::for_each_resample(10, 3, g2, [&](std::span<const std::uint32_t> idx) { second.insert(second.end(), idx.begin(), idx.end()); });
    EXPECT_EQ(first, second);
}

TEST(Bootstrap, CountsAreTheHistogramOfTheSameDraws)
{
    wyrand g1(7), g2(7);
    std::vector<std::vector<std::uint32_t>> histograms;
    stats::for_each_resample(50, 20, g1, [&](std::span<const std::uint32_t> idx) {
        std::vector<std::uint32_t> h(50);
        for (const std::uint32_t i : idx)
            ++h[i];
        histograms.push_back(h);
    });
    std::size_t b = 0;
    stats::for_each_resample_counts(50, 20, g2, [&](std::span<const std::uint32_t> counts) {
        EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), 0u), 50u);
        EXPECT_TRUE(std::equal(counts.begin(), counts.end(), histograms[b++].begin()));
    });
    EXPECT_EQ(b, 20u);
}

TEST(Bootstrap, PoissonCountsHaveMeanAndVarianceOne)
{
    wyrand g(3);
    double sum = 0, sum2 = 0, zeros = 0, total = 0;
    stats::for_each_poisson_resample(1001, 200, g, [&](std::span<const std::uint32_t> counts) {
        for (const std::uint32_t c : counts) {
            sum += c;
            sum2 += double(c) * c;
            zeros += c == 0 ? 1 : 0;
            ++total;
        }
    });
    const double mean = sum / total, variance = sum2 / total - mean * mean;
    EXPECT_NEAR(mean, 1.0, 0.01);
    EXPECT_NEAR(variance, 1.0, 0.02);
    EXPECT_NEAR(zeros / total, std::exp(-1.0), 0.005);
}

TEST(Bootstrap, BalancedResamplesUseEachObservationExactlyBTimes)
{
    wyrand g(4);
    std::vector<int> uses(30);
    std::size_t resamples = 0;
    stats::for_each_balanced_resample(30, 17, g, [&](std::span<const std::uint32_t> idx) {
        ++resamples;
        for (const std::uint32_t i : idx)
            ++uses[i];
    });
    EXPECT_EQ(resamples, 17u);
    for (const int u : uses)
        EXPECT_EQ(u, 17);
}

TEST(Bootstrap, StationaryBlocksHaveTheirMeanLength)
{
    wyrand g(5);
    double breaks = 0, steps = 0;
    stats::for_each_stationary_resample(5000, 8.0, 20, g, [&](std::span<const std::uint32_t> idx) {
        for (std::size_t k = 1; k < idx.size(); ++k) {
            breaks += idx[k] != (idx[k - 1] + 1) % 5000 ? 1 : 0;
            ++steps;
        }
    });
    EXPECT_NEAR(steps / breaks, 8.0, 0.4);   // a restart can land on the next observation: slightly long
}

TEST(Bootstrap, MovingBlocksAreConsecutiveAndWithinTheData)
{
    wyrand g(6);
    stats::for_each_block_resample(20, 5, 10, g, [&](std::span<const std::uint32_t> idx) {
        ASSERT_EQ(idx.size(), 20u);
        for (std::size_t k = 0; k < 20; k += 5)
            for (std::size_t j = 1; j < 5; ++j)
                ASSERT_EQ(idx[k + j], idx[k] + j);
    });
}

TEST(BootstrapMoments, WeightedMomentsAreExact)
{
    const std::vector<double> x{1, 2, 4};
    const std::vector<std::uint32_t> counts{2, 0, 1};   // the resample 1, 1, 4
    const auto m = stats::moments_of(x, counts, 2.0);
    EXPECT_DOUBLE_EQ(m.weight, 3);
    EXPECT_DOUBLE_EQ(m.mean, 2);
    EXPECT_DOUBLE_EQ(m.variance, 3);   // ((1-2)^2 + (1-2)^2 + (4-2)^2) / 2
}

TEST(BootstrapMoments, TheStandardErrorOfAMeanIsSigmaOverRootN)
{
    const auto x = sample(1000, 11);
    wyrand g(12);
    for (const auto how : {stats::resampling::poisson, stats::resampling::multinomial}) {
        const auto d = stats::bootstrap_moments(x, 2000, g, [](const stats::weighted_moments& m) { return m.mean; }, how);
        const double sd = std::sqrt(stats::moments_of(x, std::vector<std::uint32_t>(x.size(), 1u), 0).variance);
        EXPECT_NEAR(d.standard_error(), sd / std::sqrt(1000.0), 0.08 * sd / std::sqrt(1000.0));
        EXPECT_NEAR(d.bias(), 0, 3 * d.standard_error() / std::sqrt(2000.0) + 1e-12);
        const auto [lo, hi] = d.percentile_interval(0.95);
        EXPECT_LT(lo, d.estimate);
        EXPECT_GT(hi, d.estimate);
    }
}

TEST(BootstrapAny, AMedianIsBootstrappedThroughAGatheredResample)
{
    const auto x = sample(201, 13);
    wyrand g(14);
    const auto median = [](std::span<const double> v) {
        std::vector<double> c(v.begin(), v.end());
        std::nth_element(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(c.size() / 2), c.end());
        return c[c.size() / 2];
    };
    const auto d = stats::bootstrap(x, 500, g, median);
    EXPECT_EQ(d.replicates.size(), 500u);
    EXPECT_GT(d.standard_error(), 0);
    const auto [lo, hi] = d.basic_interval(0.9);
    EXPECT_LT(lo, hi);
}

TEST(BootstrapDistribution, QuantilesInterpolateLikeRsTypeSeven)
{
    stats::bootstrap_distribution d;
    d.estimate = 2;
    d.replicates = {4, 1, 3, 2};
    EXPECT_DOUBLE_EQ(d.quantile(0), 1);
    EXPECT_DOUBLE_EQ(d.quantile(1), 4);
    EXPECT_DOUBLE_EQ(d.quantile(0.5), 2.5);
    EXPECT_DOUBLE_EQ(d.quantile(0.25), 1.75);
    EXPECT_DOUBLE_EQ(d.mean(), 2.5);
    EXPECT_DOUBLE_EQ(d.bias(), 0.5);
    EXPECT_TRUE(std::isnan(stats::bootstrap_distribution{}.standard_error()));
}

// ---- BCa and block lengths, against SciPy 1.15 and arch 8.0 on the same data ------------------------

double u(std::uint64_t k) { return static_cast<double>((k * 2654435761ULL) % 4294967296ULL) / 4294967296.0 - 0.5; }

double sample_variance(std::span<const double> v)
{
    double m = 0;
    for (const double x : v)
        m += x;
    m /= static_cast<double>(v.size());
    double s = 0;
    for (const double x : v)
        s += (x - m) * (x - m);
    return s / static_cast<double>(v.size() - 1);
}

TEST(BootstrapBca, AgreesWithScipy)
{
    std::vector<double> data(40);
    for (std::uint64_t k = 0; k < 40; ++k)
        data[k] = -std::log(u(9000 + k) + 0.5 + 1e-9);
    stats::bootstrap_distribution d;
    d.estimate = sample_variance(data);
    for (std::uint64_t b = 0; b < 300; ++b) {
        std::vector<double> resample(40);
        for (std::uint64_t k = 0; k < 40; ++k)
            resample[k] = data[(k * 7 + b * 13) % 40];
        d.replicates.push_back(sample_variance(resample) * (1 + 0.3 * u(20000 + b)));
    }
    EXPECT_NEAR(d.estimate, 0.8285918878576044, 1e-13);
    const auto jack = stats::jackknife(data, sample_variance);
    ASSERT_EQ(jack.size(), 40u);
    const auto [lo, hi] = d.bca_interval(0.95, jack);
    EXPECT_NEAR(lo, 0.7185997133554034, 1e-10);
    EXPECT_NEAR(hi, 0.9519165243335664, 1e-10);
}

TEST(BootstrapBlockLength, AgreesWithArch)
{
    std::vector<double> ar(500), structured(500);
    for (std::uint64_t k = 0; k < 500; ++k)
        structured[k] = u(k);
    ar[0] = structured[0];
    for (std::size_t t = 1; t < 500; ++t)
        ar[t] = 0.7 * ar[t - 1] + structured[t];
    const auto b = stats::optimal_block_length(ar);
    EXPECT_NEAR(b.stationary, 53.59488556526938, 1e-9);
    EXPECT_NEAR(b.circular, 61.350828834579836, 1e-9);
    const auto capped = stats::optimal_block_length(structured);   // a regular sequence: capped at ceil(min(3 sqrt n, n / 3))
    EXPECT_EQ(capped.stationary, 68.0);
    EXPECT_EQ(capped.circular, 68.0);
}

TEST(BootstrapBlockLength, AnIndependentSeriesNeedsShortBlocks)
{
    stats::wyrand g(5);
    std::vector<double> x(2000);
    for (double& v : x)
        v = stats::uniform_real(g);
    EXPECT_LT(stats::optimal_block_length(x).stationary, 5.0);
}

} // namespace
