//
//  bitnet_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/bitnet.hpp>: absmean weights and per-token absmax activations as BitNet's reference
//  defines them (ties to even, the 1e-5 floors, the clamps), their fake-quantised forms, the straight-through
//  estimator's gradients checked against the definition and against a finite difference of the surrogate,
//  ternary_dot, BitLinear against its float definition and against dequantise-then-multiply, the SubLN form, and
//  compile-time use.
//

#include <snicholls/core/ml/bitnet.hpp>

#include "quant_rows.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr bool compile_time()
{
    std::array<float, 6> w{0.9f, -0.2f, 0.05f, -1.4f, 0.0f, 0.45f};   // mean |w| = 0.5
    std::array<std::int8_t, 6> q{};
    const float gamma = ml::absmean_quantize(w, q);
    std::array<float, 4> x{1.0f, -0.5f, 0.25f, 0.0f};
    std::array<std::int8_t, 4> a{};
    const float s = ml::absmax_quantize(x, a);
    return gamma == 0.5f && q == std::array<std::int8_t, 6>{1, 0, 0, -1, 0, 1} && s == 127.0f
           && a == std::array<std::int8_t, 4>{127, -64, 32, 0}   // -63.5 and 31.75: ties to even, then nearest
           && ml::ternary_dot(std::span<const std::int8_t>(q).first(4), a) == 127;
}
static_assert(compile_time());

TEST(MlBitnet, AbsmeanQuantisationIsBitnetsWeightQuant)
{
    // s = 1 / max(mean |w|, 1e-5), q = round(w · s).clamp(-1, 1): 0.5 and 1.5 are ties, to even
    const std::vector<float> w{0.5f, -0.5f, 1.5f, -1.5f, 0.75f, 0.25f, 2.0f, -4.0f};
    std::vector<std::int8_t> q(w.size());
    double sum = 0;
    for (const float v : w)
        sum += std::fabs(v);
    const float gamma = ml::absmean_quantize(w, q);
    EXPECT_EQ(gamma, float(sum / 8));
    const float s = 1.0f / gamma;
    for (std::size_t i = 0; i < w.size(); ++i)
        EXPECT_EQ(q[i], std::int8_t(std::clamp(std::nearbyint(w[i] * s), -1.0f, 1.0f))) << i;
    std::vector<float> fake(w.size());
    EXPECT_EQ(ml::absmean_fake_quantize(w, fake), gamma);
    for (std::size_t i = 0; i < w.size(); ++i)
        EXPECT_EQ(fake[i], float(q[i]) / s);
}

TEST(MlBitnet, AZeroMatrixHasTheFloorScale)
{
    const std::vector<float> w(16, 0.0f);
    std::vector<std::int8_t> q(16, 7);
    EXPECT_EQ(ml::absmean_quantize(w, q), ml::bitnet_eps);
    EXPECT_EQ(q[5], 0);
    std::vector<std::int8_t> a(16, 7);
    EXPECT_EQ(ml::absmax_quantize(w, a), 127.0f / ml::bitnet_eps);
    EXPECT_EQ(a[9], 0);
}

TEST(MlBitnet, AbsmaxQuantisationIsBitLinearsQuantInput)
{
    const auto x = quant_rows::row(quant_rows::outliers, 300, 3);
    std::vector<std::int8_t> q(x.size());
    const float s = ml::absmax_quantize(x, q);
    float amax = 0;
    for (const float v : x)
        amax = std::fmax(amax, std::fabs(v));
    EXPECT_EQ(s, 127.0f / amax);
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_EQ(q[i], std::int8_t(std::clamp(std::nearbyint(x[i] * s), -128.0f, 127.0f))) << i;
    std::vector<float> fake(x.size());
    EXPECT_EQ(ml::absmax_fake_quantize(x, fake), s);
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_EQ(fake[i], float(q[i]) / s);
}

TEST(MlBitnet, FourBitActivationsUseSevenAndClampAtMinusEight)
{
    const std::vector<float> x{1.0f, -1.0f, 0.5f, -0.5f, 0.07f};
    std::vector<std::int8_t> q(x.size());
    EXPECT_EQ(ml::absmax_quantize<4>(x, q), 7.0f);
    EXPECT_EQ(q, (std::vector<std::int8_t>{7, -7, 4, -4, 0}));   // 3.5 to even: 4; 0.49 to 0
}

TEST(MlBitnet, ANanQuantisesToZero)
{
    const std::vector<float> x{1.0f, std::numeric_limits<float>::quiet_NaN()};
    std::vector<std::int8_t> q(2);
    ml::absmax_quantize(x, q);
    EXPECT_EQ(q[1], 0);
}

TEST(MlBitnet, TheIdentityEstimatorPassesTheGradientThrough)
{
    const auto x = quant_rows::row(quant_rows::heavy, 64, 4);
    const auto g = quant_rows::row(quant_rows::uniform, 64, 5);
    std::vector<std::int8_t> q(64);
    const float s = ml::absmax_quantize(x, q);
    std::vector<float> dx(64, 9.0f);
    ml::ste_backward(g, x, s, -128, 127, dx);
    EXPECT_EQ(dx, g);
}

TEST(MlBitnet, TheClippedEstimatorZeroesTheGradientWhereTheForwardClipped)
{
    // weights: x · s beyond ±1.5 is clipped to ±1
    const std::vector<float> w{0.1f, 1.4f, 1.6f, -1.49f, -1.51f, 3.0f};
    const std::vector<float> g{1, 2, 3, 4, 5, 6};
    std::vector<float> dx(6);
    ml::ste_backward(g, w, 1.0f, -1, 1, dx, ml::ste::clipped);
    EXPECT_EQ(dx, (std::vector<float>{1, 2, 0, 4, 0, 0}));
}

TEST(MlBitnet, TheEstimatorIsTheGradientOfTheSurrogate)
{
    // The surrogate x + (quant(x) - x).detach() has derivative 1: a finite difference of the surrogate's value with
    // the quantised part held fixed is the identity, which ste_backward gives, and a loss's gradient through it is dout
    const auto x = quant_rows::row(quant_rows::weights, 32, 6);
    std::vector<float> fake(32);
    const float gamma = ml::absmean_fake_quantize(x, fake);
    const auto loss = [&](const std::vector<float>& v) {   // L = sum c_i · (v_i + (fake_i - x_i)), c_i = i + 1
        double l = 0;
        for (std::size_t i = 0; i < 32; ++i)
            l += double(i + 1) * (double(v[i]) + (double(fake[i]) - double(x[i])));
        return l;
    };
    std::vector<float> dout(32), dx(32);
    for (std::size_t i = 0; i < 32; ++i)
        dout[i] = float(i + 1);
    ml::ste_backward(dout, x, 1.0f / gamma, -1, 1, dx);
    for (std::size_t i = 0; i < 32; i += 7) {
        auto up = x, down = x;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        const double numeric = (loss(up) - loss(down)) / (double(up[i]) - double(down[i]));
        EXPECT_NEAR(numeric, dx[i], 1e-3 * double(i + 1)) << i;
    }
}

TEST(MlBitnet, TernaryDotIsExact)
{
    std::vector<std::int8_t> t(1000), a(1000);
    quant_rows::stream s(3);
    long long exact = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        t[i] = std::int8_t(int(s.next() % 3) - 1);
        a[i] = std::int8_t(s.next() >> 56);
        exact += t[i] * a[i];
    }
    EXPECT_EQ(ml::ternary_dot(t, a), exact);
    std::vector<std::int8_t> top(1000, 1), low(1000, -128);
    EXPECT_EQ(ml::ternary_dot(top, low), -128000);
}

TEST(MlBitnet, BitLinearIsItsFloatDefinition)
{
    constexpr std::size_t rows = 12, cols = 64;
    const auto wf = quant_rows::row(quant_rows::weights, rows * cols, 7);
    std::vector<std::int8_t> t(rows * cols);
    const float gamma = ml::absmean_quantize(wf, t);
    const auto x = quant_rows::row(quant_rows::uniform, cols, 8);
    std::vector<float> y(rows);
    std::vector<std::int8_t> codes(cols);
    const ml::bitlinear_weights w{t, rows, cols, gamma};
    ml::bitlinear(w, x, y, codes);
    std::vector<std::int8_t> a(cols);
    const float s = ml::absmax_quantize(x, a);
    std::vector<float> xf(cols);
    ml::absmax_fake_quantize(x, xf);
    for (std::size_t i = 0; i < rows; ++i) {
        std::int32_t acc = 0;
        double ref = 0;
        for (std::size_t j = 0; j < cols; ++j) {
            acc += t[i * cols + j] * a[j];
            ref += double(t[i * cols + j]) * double(gamma) * double(xf[j]);   // F.linear(quant_input(x), weight_quant(W))
        }
        EXPECT_EQ(y[i], float(acc) / s * gamma) << i;
        EXPECT_NEAR(y[i], ref, 1e-5 * (std::fabs(ref) + 1)) << i;
    }
}

TEST(MlBitnet, BitLinearNormedIsRmsNormThenBitLinear)
{
    constexpr std::size_t rows = 5, cols = 32;
    const auto wf = quant_rows::row(quant_rows::uniform, rows * cols, 9);
    std::vector<std::int8_t> t(rows * cols);
    const float gamma = ml::absmean_quantize(wf, t);
    const ml::bitlinear_weights w{t, rows, cols, gamma};
    const auto x = quant_rows::row(quant_rows::heavy, cols, 10);
    const auto g = quant_rows::row(quant_rows::positive, cols, 11);
    std::vector<float> y(rows), expected(rows), normed(cols), scratch(cols);
    std::vector<std::int8_t> codes(cols);
    ml::bitlinear_normed(w, g, x, y, scratch, codes);
    ml::rms_norm(x, g, normed, ml::bitnet_eps);
    ml::bitlinear(w, normed, expected, codes);
    EXPECT_EQ(y, expected);
}

} // namespace
