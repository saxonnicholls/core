//
//  short_conv_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/short_conv.hpp>: the forward pass against a transcription of llama.cpp's
//  GGML_OP_SSM_CONV (commit 39039aa) to the bit, a sequence fed in pieces through its state, decoding token by token to
//  the bit, two sequences in one call, the backward pass against central differences in double (taps, bias, inputs,
//  initial state), the state's size and a forward pass at compile time.
//

#include <snicholls/core/ml/short_conv.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include "../support/mixer_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::core_testing::central_difference;
using snicholls::core_testing::uniform;
using snicholls::core_testing::weighted;
constexpr auto& ref = snicholls::linalg::reference;

static_assert(ml::short_conv_state_size(6, 4) == 18 && ml::short_conv_state_size(6, 1) == 0 && ml::short_conv_state_size(6, 0) == 0);

constexpr float compile_time_conv()
{
    const std::array<float, 2> w{0.5f, 2.0f};
    const std::array<float, 1> bias{1.0f};
    const std::array<float, 3> x{1, 2, 3};
    std::array<float, 3> y{};
    ml::short_conv_forward(ref, ml::short_conv_weights<float>{w, 1, 2, bias}, std::span<const float>(x), std::span<float>(y));
    return y[2];   // 0.5 · 2 + 2 · 3 + 1
}
static_assert(compile_time_conv() == 8.0f);

// ggml_compute_forward_ssm_conv_f32: conv_x is the state then the tokens, per channel; each output a dot of d_conv
// consecutive entries with the channel's taps, summed from 0.0f in tap order; the bias added after, as the graphs do
std::vector<float> ggml_ssm_conv(const std::vector<float>& state, const std::vector<float>& x, const std::vector<float>& w,
                                 const std::vector<float>& b, std::size_t ch, std::size_t nc, std::size_t n_t)
{
    const std::size_t ncs = nc - 1 + n_t;
    std::vector<float> conv_x(ncs * ch);   // {d_conv - 1 + n_t, d_inner}
    for (std::size_t c = 0; c < ch; ++c) {
        for (std::size_t r = 0; r + 1 < nc; ++r)
            conv_x[c * ncs + r] = state[r * ch + c];
        for (std::size_t t = 0; t < n_t; ++t)
            conv_x[c * ncs + nc - 1 + t] = x[t * ch + c];
    }
    std::vector<float> out(n_t * ch);
    for (std::size_t i2 = 0; i2 < n_t; ++i2)
        for (std::size_t i1 = 0; i1 < ch; ++i1) {
            const float* s = conv_x.data() + i2 + i1 * ncs;
            float sumf = 0.0f;
            for (std::size_t i0 = 0; i0 < nc; ++i0)
                sumf += s[i0] * w[i0 + i1 * nc];
            out[i2 * ch + i1] = sumf + b[i1];
        }
    return out;
}

TEST(MlShortConv, IsLlamaCppsSsmConvToTheBit)
{
    constexpr std::size_t ch = 5, K = 4, n = 7;
    const auto w = uniform<float>(ch * K, 1), b = uniform<float>(ch, 2), x = uniform<float>(n * ch, 3), st = uniform<float>((K - 1) * ch, 4);
    std::vector<float> y(n * ch), state = st;
    ml::short_conv_forward(ref, ml::short_conv_weights<float>{w, ch, K, b}, std::span<const float>(x), std::span<float>(y), 1,
                           std::span<float>(state));
    EXPECT_EQ(y, ggml_ssm_conv(st, x, w, b, ch, K, n));
    for (std::size_t r = 0; r + 1 < K; ++r)   // the state kept is the last K - 1 inputs
        for (std::size_t c = 0; c < ch; ++c)
            EXPECT_EQ(state[r * ch + c], x[(n - (K - 1) + r) * ch + c]);
}

TEST(MlShortConv, ASequenceInPiecesIsTheWholeSequence)
{
    constexpr std::size_t ch = 3, K = 4, n = 9;
    const auto w = uniform<double>(ch * K, 5), b = uniform<double>(ch, 6), x = uniform<double>(n * ch, 7);
    const ml::short_conv_weights<double> cw{w, ch, K, b};
    std::vector<double> whole(n * ch), state((K - 1) * ch, 0.0);
    ml::short_conv_forward(ref, cw, std::span<const double>(x), std::span<double>(whole));
    std::vector<double> pieces(n * ch);
    for (const auto& [from, to] : {std::pair<std::size_t, std::size_t>{0, 2}, {2, 3}, {3, 9}})   // shorter and longer than K - 1
        ml::short_conv_forward(ref, cw, std::span<const double>(x).subspan(from * ch, (to - from) * ch),
                               std::span<double>(pieces).subspan(from * ch, (to - from) * ch), 1, std::span<double>(state));
    EXPECT_EQ(pieces, whole);
}

TEST(MlShortConv, DecodingIsTheForwardPassToTheBit)
{
    constexpr std::size_t ch = 4, K = 3, n = 6;
    const auto w = uniform<float>(ch * K, 8), x = uniform<float>(n * ch, 9);
    const ml::short_conv_weights<float> cw{w, ch, K, {}};
    std::vector<float> y(n * ch), state(ml::short_conv_state_size(ch, K), 0.0f), step(ch);
    ml::short_conv_forward(ref, cw, std::span<const float>(x), std::span<float>(y));
    for (std::size_t t = 0; t < n; ++t) {
        ml::short_conv_step(cw, std::span<const float>(x).subspan(t * ch, ch), std::span<float>(step), std::span<float>(state));
        for (std::size_t c = 0; c < ch; ++c)
            EXPECT_EQ(step[c], y[t * ch + c]) << t << " " << c;
    }
}

TEST(MlShortConv, SequencesInOneCallAreSeparateCalls)
{
    constexpr std::size_t ch = 2, K = 4, n = 5;
    const auto w = uniform<double>(ch * K, 10), x = uniform<double>(2 * n * ch, 11);
    const ml::short_conv_weights<double> cw{w, ch, K, {}};
    std::vector<double> both(2 * n * ch);
    ml::short_conv_forward(ref, cw, std::span<const double>(x), std::span<double>(both), 2);
    for (std::size_t s = 0; s < 2; ++s) {
        std::vector<double> one(n * ch);
        ml::short_conv_forward(ref, cw, std::span<const double>(x).subspan(s * n * ch, n * ch), std::span<double>(one));
        for (std::size_t i = 0; i < one.size(); ++i)
            EXPECT_EQ(one[i], both[s * n * ch + i]);
    }
}

TEST(MlShortConv, TheBackwardPassMatchesCentralDifferences)
{
    constexpr std::size_t ch = 3, K = 4, n = 5, seqs = 2;
    auto w = uniform<double>(ch * K, 12), b = uniform<double>(ch, 13), x = uniform<double>(seqs * n * ch, 14),
         init = uniform<double>(seqs * (K - 1) * ch, 15);
    const auto wy = uniform<double>(seqs * n * ch, 16);
    const auto loss = [&] {
        std::vector<double> y(x.size()), st = init;
        ml::short_conv_forward(ref, ml::short_conv_weights<double>{w, ch, K, b}, std::span<const double>(x), std::span<double>(y), seqs,
                               std::span<double>(st));
        return weighted(y, wy);
    };
    std::vector<double> dx(x.size()), dw(w.size()), db(ch), dinit(init.size());
    ml::short_conv_backward(ref, ml::short_conv_weights<double>{w, ch, K, b}, std::span<const double>(x), std::span<const double>(wy),
                            std::span<double>(dx), std::span<double>(dw), std::span<double>(db), seqs, std::span<const double>(init),
                            std::span<double>(dinit));
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(dx[i], central_difference(x, i, loss), 1e-8) << "x " << i;
    for (std::size_t i = 0; i < w.size(); ++i)
        EXPECT_NEAR(dw[i], central_difference(w, i, loss), 1e-8) << "w " << i;
    for (std::size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(db[i], central_difference(b, i, loss), 1e-8) << "b " << i;
    for (std::size_t i = 0; i < init.size(); ++i)
        EXPECT_NEAR(dinit[i], central_difference(init, i, loss), 1e-8) << "initial " << i;
}

TEST(MlShortConv, OneTapIsAScaleAndNoState)
{
    const std::vector<float> w{2.0f, -1.0f}, x{1, 2, 3, 4};
    std::vector<float> y(4);
    ml::short_conv_forward(ref, ml::short_conv_weights<float>{w, 2, 1, {}}, std::span<const float>(x), std::span<float>(y));
    EXPECT_EQ(y, (std::vector<float>{2, -2, 6, -4}));
}

} // namespace
