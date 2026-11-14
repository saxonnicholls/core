//
//  lm_head_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/lm_head.hpp>: the loss against the mean of the rows' cross-entropies, the gradients
//  against central differences in double, a tied head adding into the embedding's gradient, accumulation, no
//  targets, and compile-time use.
//

#include <snicholls/core/ml/lm_head.hpp>

#include <snicholls/core/ml/embedding.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

constexpr std::size_t vocab = 7, d = 4, n = 5;

std::vector<double> wave(std::size_t size, double a, double b)
{
    std::vector<double> v(size);
    for (std::size_t i = 0; i < size; ++i)
        v[i] = a * std::sin(0.53 * double(i) + b);
    return v;
}

const std::vector<std::size_t> targets{3, 0, 6, 3, 1};

constexpr bool compile_time()
{
    // Two classes, a hidden state that gives equal logits: the loss is ln 2
    const std::array<double, 4> head{1, 0, 1, 0};   // 2 x 2
    const std::array<double, 2> h{0, 1};
    const std::array<std::size_t, 1> target{1};
    std::array<double, 2> logits{};
    return m::near(ml::lm_head_loss(ml::dense_weights<double>{head, 2, 2}, h, target, logits), m::log(2.0), 1e-15);
}
static_assert(compile_time());

TEST(MlLmHead, LossIsTheMeanOfTheRowsCrossEntropies)
{
    const auto w = wave(vocab * d, 0.8, 0.1), hs = wave(n * d, 1.3, 0.7);
    std::vector<double> logits(vocab);
    const double loss = ml::lm_head_loss(ml::dense_weights<double>{w, vocab, d}, hs, targets, logits);
    double want = 0;
    for (std::size_t t = 0; t < n; ++t) {
        std::vector<double> z(vocab);
        for (std::size_t v = 0; v < vocab; ++v)
            for (std::size_t c = 0; c < d; ++c)
                z[v] += w[v * d + c] * hs[t * d + c];
        want += ml::logsumexp(z) - z[targets[t]];
    }
    EXPECT_NEAR(loss, want / n, 1e-14);
}

TEST(MlLmHead, TheTrainingFormGivesTheSameLossAndCentralDifferenceGradients)
{
    auto w = wave(vocab * d, 0.8, 0.1), hs = wave(n * d, 1.3, 0.7);
    std::vector<double> logits(vocab);
    auto loss = [&] { return ml::lm_head_loss(ml::dense_weights<double>{w, vocab, d}, hs, targets, logits); };
    std::vector<double> dhs(hs.size()), dw(w.size());
    const double l = ml::lm_head_loss(ml::dense_weights<double>{w, vocab, d}, hs, targets, logits, dhs, dw);
    EXPECT_EQ(l, loss());
    auto check = [&](std::vector<double>& p, const std::vector<double>& grad) {
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double keep = p[i], e = 1e-6;
            p[i] = keep + e;
            const double hi = loss();
            p[i] = keep - e;
            const double lo = loss();
            p[i] = keep;
            EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 1e-8) << i;
        }
    };
    check(hs, dhs);
    check(w, dw);
}

TEST(MlLmHead, ATiedHeadAddsIntoTheEmbeddingsGradient)
{
    // L = lm_head_loss(E, embed(E, tokens)): E is both the embedding and the head
    auto table = wave(vocab * d, 0.8, 0.1);
    const std::vector<std::size_t> tokens{1, 4, 4, 2, 0};
    std::vector<double> logits(vocab);
    auto loss = [&] {
        std::vector<double> hs(n * d);
        for (std::size_t t = 0; t < n; ++t)
            ml::embed(table, tokens[t], std::span<double>(hs).subspan(t * d, d));
        return ml::lm_head_loss(ml::dense_weights<double>{table, vocab, d}, hs, targets, logits);
    };
    std::vector<double> hs(n * d), dhs(n * d), dtable(table.size());
    for (std::size_t t = 0; t < n; ++t)
        ml::embed(table, tokens[t], std::span<double>(hs).subspan(t * d, d));
    ml::lm_head_loss(ml::dense_weights<double>{table, vocab, d}, hs, targets, logits, dhs, dtable);
    for (std::size_t t = 0; t < n; ++t)
        ml::embedding_backward(std::span<const double>(dhs).subspan(t * d, d), tokens[t], dtable);
    for (std::size_t i = 0; i < table.size(); ++i) {
        const double keep = table[i], e = 1e-6;
        table[i] = keep + e;
        const double hi = loss();
        table[i] = keep - e;
        const double lo = loss();
        table[i] = keep;
        EXPECT_NEAR(dtable[i], (hi - lo) / (2 * e), 1e-8) << i;
    }
}

TEST(MlLmHead, GradientsAccumulate)
{
    const auto w = wave(vocab * d, 0.8, 0.1), hs = wave(n * d, 1.3, 0.7);
    std::vector<double> logits(vocab), dhs1(hs.size()), dw1(w.size()), dhs2(hs.size()), dw2(w.size());
    const ml::dense_weights<double> head{w, vocab, d};
    ml::lm_head_loss(head, hs, targets, logits, dhs1, dw1);
    ml::lm_head_loss(head, hs, targets, logits, dhs2, dw2);
    ml::lm_head_loss(head, hs, targets, logits, dhs2, dw2);
    for (std::size_t i = 0; i < dw1.size(); ++i)
        EXPECT_NEAR(dw2[i], 2 * dw1[i], 1e-15);
    for (std::size_t i = 0; i < dhs1.size(); ++i)
        EXPECT_NEAR(dhs2[i], 2 * dhs1[i], 1e-15);
}

TEST(MlLmHead, NoTargetsIsZeroAndAddsNothing)
{
    const auto w = wave(vocab * d, 0.8, 0.1);
    std::vector<double> logits(vocab), dw(w.size());
    const ml::dense_weights<double> head{w, vocab, d};
    EXPECT_EQ(ml::lm_head_loss(head, std::span<const double>{}, {}, logits), 0.0);
    EXPECT_EQ(ml::lm_head_loss(head, std::span<const double>{}, {}, logits, std::span<double>{}, dw), 0.0);
    for (double g : dw)
        EXPECT_EQ(g, 0.0);
}

} // namespace
