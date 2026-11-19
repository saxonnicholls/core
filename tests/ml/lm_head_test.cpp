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

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

std::vector<float> fwave(std::size_t size, double a, double b)
{
    const auto w = wave(size, a, b);
    return {w.begin(), w.end()};
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

// The sequence's loss with the logits as one product through a backend: lm_head_loss's loss to the bit (the same
// products in the same order, the rows summed in order), and its gradients to rounding
TEST(MlLmHead, TheBatchedLossIsLmHeadLossAndItsGradientsAgree)
{
    constexpr std::size_t vocab = 7, d = 5, n = 4;
    std::vector<double> head(vocab * d), hs(n * d);
    for (std::size_t i = 0; i < head.size(); ++i)
        head[i] = 0.3 * std::sin(0.7 * double(i));
    for (std::size_t i = 0; i < hs.size(); ++i)
        hs[i] = std::cos(0.4 * double(i));
    const std::vector<std::uint32_t> t32{1, 6, 0, 3};
    const std::vector<std::size_t> t64(t32.begin(), t32.end());
    const ml::dense_weights<double> w{head, vocab, d};
    std::vector<double> logits(n * vocab), losses(n), dhs(n * d), dhead(vocab * d), row(vocab), dhs1(n * d), dhead1(vocab * d);
    const double batched = ml::lm_head_loss_sequence(snicholls::linalg::reference, w, std::span<const double>(hs), t32, std::span<double>(logits),
                                                     std::span<double>(losses), std::span<double>(dhs), std::span<double>(dhead));
    const double per_row = ml::lm_head_loss(w, hs, t64, row, dhs1, dhead1);
    EXPECT_EQ(batched, per_row);
    for (std::size_t i = 0; i < dhs.size(); ++i)
        EXPECT_NEAR(dhs[i], dhs1[i], 1e-14);
    for (std::size_t i = 0; i < dhead.size(); ++i)
        EXPECT_NEAR(dhead[i], dhead1[i], 1e-14);
    EXPECT_EQ(ml::lm_head_loss_sequence(snicholls::linalg::reference, w, std::span<const double>(), std::span<const std::uint32_t>(),
                                        std::span<double>(), std::span<double>(), std::span<double>(), std::span<double>(dhead)),
              0.0);
}

// A head padded past its classes (GPT-2's 50,257 in 50,304 rows): the unpadded head's loss and gradients to the bit,
// nothing added to the padding's rows
TEST(MlLmHead, APaddedHeadIsTheUnpaddedHeadAndItsPaddingGetsNothing)
{
    constexpr std::size_t vocab = 7, padded = 10, d = 5, n = 4;
    std::vector<double> head(padded * d), hs(n * d);
    for (std::size_t i = 0; i < head.size(); ++i)
        head[i] = 0.3 * std::sin(0.7 * double(i));
    for (std::size_t i = 0; i < hs.size(); ++i)
        hs[i] = std::cos(0.4 * double(i));
    const std::vector<std::uint32_t> targets{1, 6, 0, 3};
    const ml::dense_weights<double> wide{head, padded, d}, narrow{std::span<const double>(head).first(vocab * d), vocab, d};
    std::vector<double> lw(n * padded), ln(n * vocab), losses_w(n), losses_n(n), dhs_w(n * d), dhs_n(n * d), dhead_w(padded * d), dhead_n(vocab * d);
    const double padded_loss = ml::lm_head_loss_sequence(snicholls::linalg::reference, wide, std::span<const double>(hs), targets, std::span<double>(lw),
                                                         std::span<double>(losses_w), std::span<double>(dhs_w), std::span<double>(dhead_w), vocab);
    const double loss = ml::lm_head_loss_sequence(snicholls::linalg::reference, narrow, std::span<const double>(hs), targets, std::span<double>(ln),
                                                  std::span<double>(losses_n), std::span<double>(dhs_n), std::span<double>(dhead_n));
    EXPECT_EQ(padded_loss, loss);
    EXPECT_EQ(losses_w, losses_n);
    EXPECT_EQ(dhs_w, dhs_n);
    EXPECT_EQ(std::vector<double>(dhead_w.begin(), dhead_w.begin() + std::ptrdiff_t(vocab * d)), dhead_n);
    EXPECT_TRUE(std::all_of(dhead_w.begin() + std::ptrdiff_t(vocab * d), dhead_w.end(), [](double e) { return e == 0; }));
}

TEST(MlLmHead, AWeightAndADenominatorScaleTheLossAndItsGradients)
{
    const auto hs = fwave(4 * 8, 1.0, 0.3), head = fwave(5 * 8, 0.5, 1.2);
    const std::vector<std::uint32_t> targets{1, 4, 0, 2};
    std::vector<float> l1(4 * 5), l2(4 * 5), s1(4), s2(4), dh1(32), dh2(32), dw1(40), dw2(40);
    const ml::dense_weights<float> w{head, 5, 8};
    const float a = ml::lm_head_loss_sequence(snicholls::linalg::reference, w, hs, targets, l1, s1, dh1, dw1);
    const float b = ml::lm_head_loss_sequence(snicholls::linalg::reference, w, hs, targets, l2, s2, dh2, dw2, 0, 1.0f, 4);
    EXPECT_EQ(a, b);
    EXPECT_EQ(dh1, dh2);
    EXPECT_EQ(dw1, dw2);
    // A weight of 0.5 over a denominator of 8: the loss and every gradient a quarter of the mean's (exact in binary)
    std::vector<float> l3(4 * 5), s3(4), dh3(32), dw3(40);
    EXPECT_EQ(ml::lm_head_loss_sequence(snicholls::linalg::reference, w, hs, targets, l3, s3, dh3, dw3, 0, 0.5f, 8), a / 4);
    for (std::size_t i = 0; i < 32; ++i)
        EXPECT_FLOAT_EQ(dh3[i], dh1[i] / 4);
    EXPECT_EQ(s3, s1);   // the per-row losses are unweighted
}

} // namespace
