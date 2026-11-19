//
//  logprobs_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/logprobs.hpp>: token_logprob is -cross_entropy to the bit and log(softmax) within
//  rounding, finite for an improbable target with large logits; the fused gradient and the rows' backward (log-prob
//  and entropy together) against central differences; the entropy against -sum p log p and log V for uniform logits;
//  the masked sums and means against a direct computation, prompt tokens contributing nothing, an empty response
//  giving 0 and no gradient; the backward through both steps to the logits leaving the prompt's logits untouched;
//  compile-time use and the noexcept claims.
//

#include <snicholls/core/ml/logprobs.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/softmax.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
constexpr auto& backend = snicholls::linalg::reference;

static_assert(noexcept(ml::token_logprob(std::array<float, 2>{}, 0)));
static_assert(noexcept(ml::token_entropy(std::array<double, 2>{})));
static_assert(noexcept(ml::sequence_logprobs<float>(backend, {}, {}, {}, ml::logprob_reduction::sum)));

constexpr bool at_compile_time()
{
    const std::array<double, 3> logits{1.0, 2.0, 3.0};
    const double lp = ml::token_logprob(logits, 2);
    const std::array<double, 4> token_logps{-1.0, -2.0, -3.0, -4.0};
    const std::array<std::uint8_t, 4> mask{0, 1, 1, 0};
    std::array<double, 2> sums{}, means{};
    ml::sequence_logprobs<double>(backend, token_logps, mask, sums, ml::logprob_reduction::sum);
    ml::sequence_logprobs<double>(backend, token_logps, mask, means, ml::logprob_reduction::mean);
    return lp < 0.0 && lp > -1.0 && sums[0] == -2.0 && sums[1] == -3.0 && means[0] == -2.0 && means[1] == -3.0
           && ml::response_length(mask) == 2;
}
static_assert(at_compile_time());

template <typename T>
std::vector<T> random_row(stats::wyrand& g, std::size_t n, double scale)
{
    std::vector<T> v(n);
    for (auto& x : v)
        x = static_cast<T>(scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

TEST(MlTokenLogprob, IsMinusCrossEntropyToTheBit)
{
    stats::wyrand g(1);
    for (int trial = 0; trial < 200; ++trial) {
        const auto row = random_row<float>(g, 37, 8.0);
        const std::size_t target = stats::uniform_index(g, row.size());
        const float lp = ml::token_logprob(row, target);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(lp), std::bit_cast<std::uint32_t>(-ml::cross_entropy(row, target)));
    }
}

TEST(MlTokenLogprob, IsLogOfSoftmaxAndFiniteForAnImprobableTarget)
{
    const std::vector<double> row{0.5, -1.25, 3.0, 2.0};
    std::vector<double> p(row.size());
    ml::softmax(row, p);
    for (std::size_t i = 0; i < row.size(); ++i)
        EXPECT_NEAR(ml::token_logprob(row, i), std::log(p[i]), 1e-15);
    const std::vector<float> big{1e4f, 0.0f, -1e4f};
    EXPECT_EQ(ml::token_logprob(big, 2), -2e4f);
    EXPECT_EQ(ml::token_logprob(big, 0), 0.0f);
}

TEST(MlTokenLogprob, FusedGradientIsCentralDifferences)
{
    stats::wyrand g(2);
    auto row = random_row<double>(g, 11, 3.0);
    const std::size_t target = 4;
    const double dlogp = -0.7;
    std::vector<double> grad(row.size());
    const double lp = ml::token_logprob_with_gradient(row, target, grad, dlogp);
    EXPECT_EQ(lp, ml::token_logprob(row, target));
    for (std::size_t i = 0; i < row.size(); ++i) {
        const double h = 1e-6, x = row[i];
        row[i] = x + h;
        const double up = dlogp * ml::token_logprob(row, target);
        row[i] = x - h;
        const double down = dlogp * ml::token_logprob(row, target);
        row[i] = x;
        EXPECT_NEAR(grad[i], (up - down) / (2 * h), 1e-8) << i;
    }
}

TEST(MlTokenEntropy, IsMinusSumPLogPAndLogVForUniformLogits)
{
    const std::vector<double> row{0.3, -2.0, 1.5, 0.0, 4.0};
    std::vector<double> p(row.size());
    ml::softmax(row, p);
    double h = 0;
    for (const double q : p)
        h -= q * std::log(q);
    EXPECT_NEAR(ml::token_entropy(row), h, 1e-14);
    const std::vector<double> flat(64, 2.5);
    EXPECT_NEAR(ml::token_entropy(flat), std::log(64.0), 1e-14);
    const std::vector<double> peaked{1e4, 0.0, 0.0};
    EXPECT_EQ(ml::token_entropy(peaked), 0.0);
}

TEST(MlTokenLogprobsRows, ForwardIsPerRowAndBackwardIsCentralDifferences)
{
    stats::wyrand g(3);
    constexpr std::size_t n = 4, v = 9;
    auto logits = random_row<double>(g, n * v, 4.0);
    const std::vector<std::uint32_t> targets{0, 8, 3, 5};
    const std::vector<double> dlogps{0.5, -1.0, 0.0, 2.0}, dents{0.25, 0.0, -0.5, 1.0};
    std::vector<double> logps(n), ents(n);
    ml::token_logprobs_rows<double>(backend, logits, targets, logps, ents);
    for (std::size_t t = 0; t < n; ++t) {
        const std::span<const double> row(logits.data() + t * v, v);
        EXPECT_EQ(logps[t], ml::token_logprob(row, targets[t]));
        EXPECT_EQ(ents[t], ml::token_entropy(row));
    }
    const auto objective = [&](const std::vector<double>& x) {
        std::vector<double> lp(n), en(n);
        ml::token_logprobs_rows<double>(backend, x, targets, lp, en);
        double s = 0;
        for (std::size_t t = 0; t < n; ++t)
            s += dlogps[t] * lp[t] + dents[t] * en[t];
        return s;
    };
    auto grad = logits;
    ml::token_logprobs_rows_backward<double>(backend, grad, targets, dlogps, dents);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        auto x = logits;
        const double h = 1e-6;
        x[i] += h;
        const double up = objective(x);
        x[i] -= 2 * h;
        const double down = objective(x);
        EXPECT_NEAR(grad[i], (up - down) / (2 * h), 1e-7) << i;
    }
    // Without entropies, the backward is token_logprob_with_gradient row by row, to the bit
    auto grad2 = logits;
    ml::token_logprobs_rows_backward<double>(backend, grad2, targets, dlogps);
    for (std::size_t t = 0; t < n; ++t) {
        std::vector<double> ref(v);
        ml::token_logprob_with_gradient(std::span<const double>(logits.data() + t * v, v), targets[t], ref, dlogps[t]);
        for (std::size_t i = 0; i < v; ++i)
            EXPECT_EQ(grad2[t * v + i], ref[i]);
    }
}

TEST(MlSequenceLogprobs, SumsAndMeansOverTheResponseOnly)
{
    const std::vector<double> lp{-1, -2, -3, -4, /**/ -5, -6, -7, -8, /**/ -9, -10, -11, -12};
    const std::vector<std::uint8_t> mask{0, 0, 1, 1, /**/ 0, 1, 1, 1, /**/ 0, 0, 0, 0};
    std::vector<double> sums(3), means(3);
    ml::sequence_logprobs<double>(backend, lp, mask, sums, ml::logprob_reduction::sum);
    ml::sequence_logprobs<double>(backend, lp, mask, means, ml::logprob_reduction::mean);
    EXPECT_EQ(sums, (std::vector<double>{-7, -21, 0}));
    EXPECT_EQ(means, (std::vector<double>{-3.5, -7, 0}));
    // The prompt's log-probabilities change nothing
    auto lp2 = lp;
    lp2[0] = lp2[1] = lp2[4] = -1e9;
    std::vector<double> sums2(3);
    ml::sequence_logprobs<double>(backend, lp2, mask, sums2, ml::logprob_reduction::sum);
    EXPECT_EQ(sums2, sums);
    // The backward: mask times dseq, over the length for a mean; nothing for an empty response
    std::vector<double> dtok(12, 99.0);
    ml::sequence_logprobs_backward<double>(backend, mask, std::vector<double>{2, 3, 5}, dtok,
                                                                                   ml::logprob_reduction::mean);
    EXPECT_EQ(dtok, (std::vector<double>{0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0}));
    ml::sequence_logprobs_backward<double>(backend, mask, std::vector<double>{2, 3, 5}, dtok,
                                                                                   ml::logprob_reduction::sum);
    EXPECT_EQ(dtok, (std::vector<double>{0, 0, 2, 2, 0, 3, 3, 3, 0, 0, 0, 0}));
}

TEST(MlSequenceLogprobs, TheBackwardToTheLogitsLeavesThePromptUntouchedAndIsCentralDifferences)
{
    stats::wyrand g(4);
    constexpr std::size_t rows = 2, len = 3, v = 5;
    auto logits = random_row<double>(g, rows * len * v, 2.0);
    const std::vector<std::uint32_t> targets{1, 2, 3, 4, 0, 1};
    const std::vector<std::uint8_t> mask{0, 1, 1, 0, 0, 1};
    const std::vector<double> dseq{1.5, -0.5};
    const auto objective = [&](const std::vector<double>& x) {
        std::vector<double> lp(rows * len), s(rows);
        ml::token_logprobs_rows<double>(backend, x, targets, lp);
        ml::sequence_logprobs<double>(backend, lp, mask, s, ml::logprob_reduction::mean);
        return dseq[0] * s[0] + dseq[1] * s[1];
    };
    std::vector<double> dtok(rows * len);
    ml::sequence_logprobs_backward<double>(backend, mask, dseq, dtok, ml::logprob_reduction::mean);
    auto grad = logits;
    ml::token_logprobs_rows_backward<double>(backend, grad, targets, dtok);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (mask[i / v] == 0)
            EXPECT_EQ(grad[i], 0.0) << i;
        auto x = logits;
        const double h = 1e-6;
        x[i] += h;
        const double up = objective(x);
        x[i] -= 2 * h;
        EXPECT_NEAR(grad[i], (up - objective(x)) / (2 * h), 1e-8) << i;
    }
}

} // namespace
