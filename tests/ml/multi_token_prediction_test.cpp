//
//  multi_token_prediction_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/multi_token_prediction.hpp>: the shifted ids; the combine against the RMS norms and
//  concatenation written out, and its backward pass against central differences; a module's loss against DeepSeek-V3's
//  eq. 24 (divided by T, weighted by lambda / D) computed with lm_head_loss; the padded positions changing nothing that
//  is scored; the whole module, its block a latent-attention block, against central differences in double for the
//  previous depth's hidden states, the shared embedding table and head, the projection, the norms and the block.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/latent_block.hpp>
#include <snicholls/core/ml/lm_head.hpp>
#include <snicholls/core/ml/multi_token_prediction.hpp>
#include <snicholls/core/ml/norm.hpp>

#include "../support/latent_attention_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::linalg::reference;
using snicholls::core_testing::block_problem;
using snicholls::core_testing::table;
using snicholls::core_testing::wave;

constexpr auto C = ml::tiny_mla;
constexpr std::size_t d = C.d_model, vocab = 7, n = 5, sequences = 2, rows = n * sequences, depth = 1;
constexpr double weight = 0.3;

TEST(MlMultiTokenPrediction, InputIdsShiftByTheDepthAndPadTheEnd)
{
    const std::array<std::uint32_t, 8> ids{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<std::uint32_t, 8> out{};
    ml::mtp_input_ids(ids, 2, 2, 0, out);
    EXPECT_EQ(out, (std::array<std::uint32_t, 8>{3, 4, 0, 0, 7, 8, 0, 0}));
    ml::mtp_input_ids(ids, 0, 1, 0, out);
    EXPECT_EQ(out, ids);
}

TEST(MlMultiTokenPrediction, CombineIsBothNormsConcatenatedAndItsBackwardMatchesCentralDifferences)
{
    auto h = wave<double>(3 * d, 1.0, 0.2), e = wave<double>(3 * d, 0.7, 1.9), wh = wave<double>(d, 0.3, 0.5), we = wave<double>(d, 0.3, 2.5);
    std::vector<double> c(6 * d), want(d);
    ml::mtp_combine_rows<C>(reference, h, e, wh, we, std::span<double>(c));
    for (std::size_t t = 0; t < 3; ++t) {
        ml::rms_norm(std::span<const double>(h).subspan(t * d, d), wh, want, ml::norm_eps_v<C, double>);
        for (std::size_t i = 0; i < d; ++i)
            EXPECT_EQ(c[2 * t * d + i], want[i]);
        ml::rms_norm(std::span<const double>(e).subspan(t * d, d), we, want, ml::norm_eps_v<C, double>);
        for (std::size_t i = 0; i < d; ++i)
            EXPECT_EQ(c[2 * t * d + d + i], want[i]);
    }
    const auto g = wave<double>(6 * d, 1.0, 0.8);
    std::vector<double> dh(3 * d), de(3 * d), dwh(d), dwe(d);
    ml::mtp_combine_rows_backward<C>(reference, g, h, e, wh, we, std::span<double>(dh), de, dwh, dwe);
    const auto loss = [&] {
        std::vector<double> y(6 * d);
        ml::mtp_combine_rows<C>(reference, h, e, wh, we, std::span<double>(y));
        double l = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            l += g[i] * y[i];
        return l;
    };
    const auto check = [&](std::vector<double>& x, const std::vector<double>& analytic) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double x0 = x[i];
            x[i] = x0 + 1e-6;
            const double up = loss();
            x[i] = x0 - 1e-6;
            const double down = loss();
            x[i] = x0;
            EXPECT_NEAR(analytic[i], (up - down) / 2e-6, 1e-8);
        }
    };
    check(h, dh);
    check(e, de);
    check(wh, dwh);
    check(we, dwe);
}

// One module over two sequences of five tokens, its block a latent-attention block, the head and table shared
struct mtp_problem
{
    block_problem<C, double> block{n, sequences};
    std::vector<double> hidden_prev = wave<double>(rows * d, 1.0, 0.4), table_rows = wave<double>(vocab * d, 1.0, 1.6),
                        head = wave<double>(vocab * d, 0.6, 2.8), hnorm = wave<double>(d, 0.2, 0.1), enorm = wave<double>(d, 0.2, 0.9),
                        onorm = wave<double>(d, 0.2, 1.7), projection = wave<double>(d * 2 * d, 0.3, 0.3);
    std::vector<std::uint32_t> ids{3, 1, 4, 1, 5, 2, 6, 5, 3, 5}, targets{1, 4, 1, 5, 0, 6, 5, 3, 5, 2}, shifted = std::vector<std::uint32_t>(rows);
    std::vector<double> embedded = std::vector<double>(rows * d), combined = std::vector<double>(2 * rows * d),
                        output = std::vector<double>(rows * d), head_input = std::vector<double>(rows * d);
    std::vector<double> logits = std::vector<double>(n * vocab), losses = std::vector<double>(rows), dhead_input = std::vector<double>(rows * d),
                        dhead = std::vector<double>(vocab * d), dhidden = std::vector<double>(rows * d), dembedded = std::vector<double>(rows * d),
                        dtable = std::vector<double>(vocab * d), dhnorm = std::vector<double>(d), denorm = std::vector<double>(d),
                        donorm = std::vector<double>(d), dprojection = std::vector<double>(d * 2 * d), wcombined = std::vector<double>(2 * rows * d),
                        woutput = std::vector<double>(rows * d);

    mtp_problem()
    {
        for (auto* v : {&hnorm, &enorm, &onorm})
            for (double& x : *v)
                x += 1;
    }
    [[nodiscard]] auto weights() const
    {
        return ml::mtp_weights<ml::latent_block_weights<ml::dense_weights<double>, double>, double>{hnorm, enorm, {projection, d, 2 * d},
                                                                                                    block.weights(), onorm};
    }
    [[nodiscard]] auto act() { return ml::mtp_activations<ml::latent_block_activations<double>, double>{combined, block.act(), output, head_input}; }
    [[nodiscard]] auto grads()
    {
        return ml::mtp_gradients<ml::latent_block_gradients<double>, double>{dhnorm, denorm, dprojection, block.grads(), donorm, wcombined, woutput};
    }
    // Forward and the loss; the loss's gradients land in dhead_input and dhead
    double run()
    {
        ml::mtp_input_ids(ids, depth, sequences, 0, shifted);
        ml::embed_rows(reference, std::span<const double>(table_rows), std::span<const double>{}, shifted, std::span<double>(embedded));
        ml::mtp_forward_sequence<C>(reference, weights(), hidden_prev, embedded, table<C, double>(), act(), sequences);
        std::fill(dhead_input.begin(), dhead_input.end(), 0.0);
        return ml::mtp_loss_sequence(reference, ml::dense_weights<double>{head, vocab, d}, head_input, targets, depth, logits, losses,
                                     dhead_input, dhead, weight, sequences);
    }
};

TEST(MlMultiTokenPrediction, TheLossIsEquationTwentyFourDividedByTheWholeLength)
{
    mtp_problem p;
    const double loss = p.run();
    // Each sequence's first n - 1 positions predict the token after the next: their mean cross-entropy, times their
    // count, over the batch's n · sequences positions, times lambda / D
    double want = 0;
    std::vector<double> logits(vocab);
    for (std::size_t s = 0; s < sequences; ++s) {
        std::vector<std::size_t> t;
        for (std::size_t i = 0; i + depth < n; ++i)
            t.push_back(p.targets[s * n + i + depth]);
        want += ml::lm_head_loss(ml::dense_weights<double>{p.head, vocab, d},
                                 std::span<const double>(p.head_input).subspan(s * n * d, (n - depth) * d), t, logits)
                * static_cast<double>(n - depth);
    }
    want = weight * want / static_cast<double>(rows);
    EXPECT_NEAR(loss, want, 1e-12);
    for (std::size_t s = 0; s < sequences; ++s) {   // the padded position is not scored and has no gradient
        EXPECT_EQ(p.losses[s * n + n - 1], 0.0);
        for (std::size_t i = 0; i < d; ++i)
            EXPECT_EQ(p.dhead_input[(s * n + n - 1) * d + i], 0.0);
    }
    // The padding's embedding changes nothing that is scored
    p.table_rows[0] += 1.5;   // id 0 is the pad (and a target, never an input here)
    EXPECT_EQ(p.run(), loss);
}

TEST(MlMultiTokenPrediction, ModuleBackwardMatchesCentralDifferences)
{
    mtp_problem p;
    p.run();
    ml::mtp_backward<C>(reference, p.weights(), p.hidden_prev, p.embedded, table<C, double>(), p.act(), p.dhead_input,
                        std::span<const double>{}, p.dhidden, p.dembedded, p.grads(), sequences);
    ml::embed_rows_backward(reference, std::span<const double>(p.dembedded), p.shifted, std::span<double>(p.dtable), std::span<double>{});
    const auto analytic = p;   // the gradients, before central differences overwrite the working spaces
    const auto check = [&](std::vector<double>& x, const std::vector<double>& g, const char* name) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double x0 = x[i];
            x[i] = x0 + 1e-6;
            const double up = p.run();
            x[i] = x0 - 1e-6;
            const double down = p.run();
            x[i] = x0;
            EXPECT_NEAR(g[i], (up - down) / 2e-6, 1e-8 * (1 + std::abs(g[i]))) << name << ' ' << i;
        }
    };
    check(p.hidden_prev, analytic.dhidden, "hidden_prev");
    check(p.table_rows, analytic.dtable, "embedding table");
    check(p.head, analytic.dhead, "head");
    check(p.projection, analytic.dprojection, "projection");
    check(p.hnorm, analytic.dhnorm, "hidden norm");
    check(p.enorm, analytic.denorm, "embedding norm");
    check(p.onorm, analytic.donorm, "output norm");
    check(p.block.norm1, analytic.block.g_norm1, "block norm1");
    check(p.block.attn.kv_down, analytic.block.attn.g_kv_down, "block kv_down");
    check(p.block.down, analytic.block.g_down, "block mlp down");
}

} // namespace
