//
//  sampling_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/sampling.hpp>: argmax and its ties, the inverse CDF at its boundaries and under
//  rounding, sampling from logits at temperatures 0, 1 and high against the softmax frequencies, top_k's order,
//  ties and short rows, top-k and top-p sampling only ever picking kept tokens and in the right proportions, the
//  same draw giving the same index, and compile-time use.
//

#include <snicholls/core/ml/sampling.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace st = snicholls::stats;

constexpr std::array<double, 5> logits{1.0, 3.0, 2.0, 3.0, -1.0};
static_assert(ml::argmax(logits) == 1);                          // the first of the equal maxima
static_assert(ml::sample_logits(logits, 0.0, 0.99) == 1);        // temperature 0 is greedy
static_assert(ml::top_k<2>(logits).index[0] == 1 && ml::top_k<2>(logits).index[1] == 3);
static_assert(ml::top_k<8>(logits).count == 5 && ml::top_k<8>(logits).index[4] == 4);
static_assert(ml::sample(std::array<double, 3>{0.25, 0.5, 0.25}, 0.0) == 0);
static_assert(ml::sample(std::array<double, 3>{0.25, 0.5, 0.25}, 0.25) == 1);
static_assert(ml::sample(std::array<double, 3>{0.25, 0.5, 0.25}, 0.75) == 2);

constexpr std::size_t top_p_at_compile_time()
{
    std::array<std::size_t, 5> scratch{};
    return ml::sample_top_p(logits, 1.0, 0.1, 0.5, scratch);   // p = 0.1 keeps only the most probable
}
static_assert(top_p_at_compile_time() == 1);

TEST(MlSampling, RoundingNeverReturnsAnIndexPastTheRow)
{
    // Probabilities that sum to just below 1, a draw just below 1, a trailing zero: the last nonzero index
    const std::array<float, 4> p{0.3f, 0.3f, 0.39999f, 0.0f};
    EXPECT_EQ(ml::sample(p, 0.99999994f), 2u);
}

TEST(MlSampling, SampleLogitsFollowsTheSoftmaxAtEachTemperature)
{
    const std::vector<double> x{0.5, 2.0, -1.0, 1.0, 0.0};
    for (const double t : {1.0, 0.5, 4.0}) {
        std::vector<double> scaled(x.size()), want(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            scaled[i] = x[i] / t;
        ml::softmax(scaled, want);
        st::wyrand g(42);
        std::vector<double> count(x.size(), 0.0);
        constexpr int draws = 200000;
        for (int k = 0; k < draws; ++k)
            count[ml::sample_logits(x, t, st::uniform_real(g))] += 1.0 / draws;
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(count[i], want[i], 0.005) << "temperature " << t << ", index " << i;
    }
}

TEST(MlSampling, TheSameDrawGivesTheSameIndex)
{
    const std::vector<float> x{0.1f, 0.4f, 0.2f, 0.9f, 0.3f};
    st::wyrand a(7), b(7);
    for (int k = 0; k < 100; ++k)
        EXPECT_EQ(ml::sample_logits(x, 1.0f, static_cast<float>(st::uniform_real(a))),
                  ml::sample_logits(x, 1.0f, static_cast<float>(st::uniform_real(b))));
}

TEST(MlSampling, TopKKeepsTheLargestInOrder)
{
    const std::vector<double> x{5, 1, 9, 7, 9, 3, 8};
    const auto best = ml::top_k<4>(x);
    ASSERT_EQ(best.count, 4u);
    EXPECT_EQ(best.index, (std::array<std::size_t, 4>{2, 4, 6, 3}));
    EXPECT_EQ(best.value, (std::array<double, 4>{9, 9, 8, 7}));
    const auto one = ml::top_k<1>(x);
    EXPECT_EQ(one.index[0], 2u);
    EXPECT_EQ(one.count, 1u);
}

TEST(MlSampling, TopKSamplesOnlyKeptTokensInProportion)
{
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    st::wyrand g(3);
    std::array<double, 6> count{};
    constexpr int draws = 100000;
    for (int k = 0; k < draws; ++k)
        count[ml::sample_top_k<2>(x, 1.0, st::uniform_real(g))] += 1.0 / draws;
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_EQ(count[i], 0.0);
    const double p5 = 1 / (1 + std::exp(-1.0));
    EXPECT_NEAR(count[5], p5, 0.005);
    EXPECT_NEAR(count[4], 1 - p5, 0.005);
}

TEST(MlSampling, TopPKeepsTheSmallestSetReachingP)
{
    // Probabilities 0.5, 0.25, 0.125, 0.125 (logits log2 apart): p = 0.7 keeps the first two
    const std::vector<double> x{std::log(0.125), std::log(0.5), std::log(0.125), std::log(0.25)};
    std::vector<std::size_t> scratch(x.size());
    st::wyrand g(11);
    std::array<double, 4> count{};
    constexpr int draws = 100000;
    for (int k = 0; k < draws; ++k)
        count[ml::sample_top_p(x, 1.0, 0.7, st::uniform_real(g), scratch)] += 1.0 / draws;
    EXPECT_EQ(count[0], 0.0);
    EXPECT_EQ(count[2], 0.0);
    EXPECT_NEAR(count[1], 2.0 / 3.0, 0.005);
    EXPECT_NEAR(count[3], 1.0 / 3.0, 0.005);
    // p = 1 keeps everything; temperature 0 is greedy
    EXPECT_EQ(ml::sample_top_p(x, 1.0, 1.0, 0.99, scratch), 2u);   // the last of the sorted: 0.125 at index 2, after index 0
    EXPECT_EQ(ml::sample_top_p(x, 0.0, 0.5, 0.99, scratch), 1u);
}

} // namespace
