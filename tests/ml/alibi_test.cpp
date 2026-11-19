//
//  alibi_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/alibi.hpp>: the slopes against the paper's geometric series and against ggml's float
//  formula (llama.cpp 39039aa, ggml/src/ggml-cpu/ops.cpp), the bias, attention with the bias against a two-pass softmax
//  in long double and against attend at a slope of 0 to the bit, the backward pass against central differences, and
//  compile-time slopes.
//

#include <snicholls/core/ml/alibi.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using ld = long double;

// ggml's slopes, as written there in float
float ggml_slope(std::uint32_t h, std::uint32_t n_head, float max_bias)
{
    const std::uint32_t n_head_log2 = 1u << std::uint32_t(std::floor(std::log2(double(n_head))));
    const float m0 = std::pow(2.0f, -(max_bias) / float(n_head_log2));
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / float(n_head_log2));
    return max_bias > 0.0f ? h < n_head_log2 ? std::pow(m0, float(h + 1)) : std::pow(m1, float(2 * (h - n_head_log2) + 1)) : 1.0f;
}

TEST(MlAlibi, SlopesAreThePapersSeries)
{
    // 8 heads: 1/2, 1/4, ..., 1/256 (Press et al., section 3)
    const auto s8 = ml::alibi_slopes<double, 8>();
    for (std::size_t h = 0; h < 8; ++h)
        EXPECT_EQ(s8[h], std::ldexp(1.0, -int(h + 1)));
    // 12 heads: the 8-head series, then the odd powers of 2^-0.5 (the 16-head series' odd terms)
    const auto s12 = ml::alibi_slopes<double, 12>();
    for (std::size_t h = 8; h < 12; ++h)
        EXPECT_NEAR(s12[h], std::pow(2.0, -0.5 * double(2 * (h - 8) + 1)), 1e-15) << h;
    EXPECT_EQ(ml::alibi_slope(3, 8, 0.0), 1.0);   // max_bias 0: ggml's "no ALiBi"
}

TEST(MlAlibi, SlopesAreGgmls)
{
    for (std::uint32_t heads : {1u, 2u, 5u, 8u, 12u, 16u, 32u, 40u, 64u, 71u})
        for (std::uint32_t h = 0; h < heads; ++h) {
            const float mine = ml::alibi_slope<float>(h, heads, 8.0f), theirs = ggml_slope(h, heads, 8.0f);
            EXPECT_NEAR(mine, theirs, 2 * std::nextafter(theirs, 1.0f) - 2 * theirs) << heads << " " << h;   // within an ulp
        }
}

TEST(MlAlibi, BiasIsMinusSlopeTimesTheDistance)
{
    EXPECT_EQ(ml::alibi_bias(0.25, 10, 7), -0.75);
    EXPECT_EQ(ml::alibi_bias(0.25, 7, 10), -0.75);
    EXPECT_EQ(ml::alibi_bias(0.25, 7, 7), -0.0);
}

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.61 * double(i) + b);
    return v;
}

TEST(MlAlibi, AttentionIsTheSoftmaxOfTheBiasedScores)
{
    constexpr std::size_t d = 8, n = 13;
    const auto q = wave(d, 1.0, 0.2), k = wave(n * d, 1.0, 1.1), v = wave(n * d, 1.0, 2.3);
    std::vector<double> y(d);
    const double slope = 0.125, scale = 0.35;
    const double lse = ml::attend_alibi(q, k, v, y, scale, slope, 20, 8);
    std::vector<ld> s(n);
    ld mx = -1e300L;
    for (std::size_t j = 0; j < n; ++j) {
        ld dot = 0;
        for (std::size_t i = 0; i < d; ++i)
            dot += ld(q[i]) * ld(k[j * d + i]);
        s[j] = ld(scale) * dot - ld(slope) * std::abs(20.0L - ld(8 + j));
        mx = std::max(mx, s[j]);
    }
    ld sum = 0;
    for (const ld x : s)
        sum += std::exp(x - mx);
    EXPECT_NEAR(lse, double(mx + std::log(sum)), 1e-13);
    for (std::size_t i = 0; i < d; ++i) {
        ld acc = 0;
        for (std::size_t j = 0; j < n; ++j)
            acc += std::exp(s[j] - mx) / sum * ld(v[j * d + i]);
        EXPECT_NEAR(y[i], double(acc), 1e-13) << i;
    }
}

TEST(MlAlibi, ASlopeOfZeroIsAttendToTheBit)
{
    constexpr std::size_t d = 16, n = 9;
    std::vector<float> q(d), k(n * d), v(n * d), a(d), b(d);
    for (std::size_t i = 0; i < k.size(); ++i) {
        k[i] = std::sin(float(i) * 0.37f);
        v[i] = std::cos(float(i) * 0.21f);
    }
    for (std::size_t i = 0; i < d; ++i)
        q[i] = 0.5f - float(i) * 0.05f;
    const float la = ml::attend(q, k, v, a, 0.25f);
    const float lb = ml::attend_alibi(q, k, v, b, 0.25f, 0.0f, 100, 3);
    EXPECT_EQ(a, b);
    EXPECT_EQ(la, lb);
}

TEST(MlAlibi, BackwardMatchesCentralDifferences)
{
    constexpr std::size_t d = 4, n = 6;
    auto q = wave(d, 1.0, 0.3), k = wave(n * d, 1.0, 0.9), v = wave(n * d, 1.0, 1.7);
    const auto c = wave(d, 1.0, 2.2);
    const double slope = 0.5, scale = 0.7;
    auto loss = [&] {
        std::vector<double> y(d);
        ml::attend_alibi(q, k, v, y, scale, slope, 3, 0);
        double l = 0;
        for (std::size_t i = 0; i < d; ++i)
            l += c[i] * y[i];
        return l;
    };
    std::vector<double> y(d), dq(d), dk(n * d), dv(n * d);
    const double lse = ml::attend_alibi(q, k, v, y, scale, slope, 3, 0);
    ml::attend_alibi_backward(q, k, v, y, c, lse, scale, slope, 3, 0, dq, dk, dv);
    auto check = [&](std::vector<double>& x, const std::vector<double>& g, const char* name) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double keep = x[i];
            x[i] = keep + 1e-6;
            const double up = loss();
            x[i] = keep - 1e-6;
            const double down = loss();
            x[i] = keep;
            EXPECT_NEAR(g[i], (up - down) / 2e-6, 1e-8) << name << " " << i;
        }
    };
    check(q, dq, "q");
    check(k, dk, "k");
    check(v, dv, "v");
}

constexpr auto mpt_slopes = ml::alibi_slopes<float, 8>();       // eight heads at compile time: 1/2 to 1/256
static_assert(snicholls::math::near(mpt_slopes[0], 0.5f, 1e-7f) && snicholls::math::near(mpt_slopes[7], 1.0f / 256.0f, 1e-9f));
constexpr auto bloom_slopes = ml::alibi_slopes<double, 16>();   // BLOOM-560M's 16 heads: 2^-0.5 to 2^-8
static_assert(snicholls::math::near(bloom_slopes[0], 0.70710678118654752, 1e-15) && snicholls::math::near(bloom_slopes[15], 1.0 / 256.0, 1e-14));
static_assert(ml::alibi_bias(1.0, 0, 5) == -5.0);

} // namespace
