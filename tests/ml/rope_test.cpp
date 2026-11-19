//
//  rope_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/rope.hpp>: the frequencies, a table built at compile time, the rotation in both
//  layouts against the formula, rope_inverse undoing rope, lengths kept, position 0 the identity, the relative-
//  position property of the dot product, and rope_inverse as rope's backward pass; the long-context rescalings (position
//  interpolation, NTK-aware, YaRN as its paper and as DeepSeek-V3's code compute it) against their formulas at sample
//  points, plain RoPE at a factor of 1, YaRN's temperature, and a table built at compile time.
//

#include <snicholls/core/ml/rope.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

constexpr ml::rope_table<float, 8, 4> small_table{};
static_assert(small_table.cos[0][0] == 1.0f && small_table.sin[0][1] == 0.0f);
static_assert(m::near<float>(small_table.cos[1][0], m::cos(1.0f), 1e-6f) && m::near<float>(small_table.sin[3][1], m::sin(0.03f), 1e-6f));
static_assert(std::is_trivially_copyable_v<ml::rope_table<float, 8, 4>> && sizeof(ml::rope_table<float, 8, 4>) == 8 * 4 * sizeof(float));
static_assert(ml::rope_frequency<double>(0, 64) == 1.0 && m::near(ml::rope_frequency<double>(16, 64), 0.01, 1e-15));

constexpr bool rotates_at_compile_time()
{
    std::array<double, 4> x{1, 2, 3, 4};
    std::array<double, 2> c{}, s{};
    ml::rope_angles(5, c, s);
    ml::rope(x, c, s);
    ml::rope_inverse(x, c, s);
    return m::near(x[0], 1.0, 1e-12) && m::near(x[3], 4.0, 1e-12);
}
static_assert(rotates_at_compile_time());

std::vector<double> head(std::size_t d, double phase)
{
    std::vector<double> v(d);
    for (std::size_t i = 0; i < d; ++i)
        v[i] = std::sin(0.9 * static_cast<double>(i) + phase);
    return v;
}

TEST(MlRope, AnglesArePositionTimesFrequency)
{
    std::array<double, 8> c{}, s{};
    ml::rope_angles(37, c, s, 500000.0);
    for (std::size_t i = 0; i < 8; ++i) {
        const double angle = 37.0 * std::pow(500000.0, -2.0 * static_cast<double>(i) / 16.0);
        EXPECT_NEAR(c[i], std::cos(angle), 1e-13);
        EXPECT_NEAR(s[i], std::sin(angle), 1e-13);
    }
}

TEST(MlRope, BothLayoutsRotateTheirPairsByTheFormula)
{
    const auto x = head(8, 0.3);
    std::array<double, 4> c{}, s{};
    ml::rope_angles(11, c, s);
    auto half = x, inter = x;
    ml::rope(half, c, s, ml::rope_layout::half);
    ml::rope(inter, c, s, ml::rope_layout::interleaved);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(half[i], x[i] * c[i] - x[i + 4] * s[i], 1e-15);
        EXPECT_NEAR(half[i + 4], x[i] * s[i] + x[i + 4] * c[i], 1e-15);
        EXPECT_NEAR(inter[2 * i], x[2 * i] * c[i] - x[2 * i + 1] * s[i], 1e-15);
        EXPECT_NEAR(inter[2 * i + 1], x[2 * i] * s[i] + x[2 * i + 1] * c[i], 1e-15);
    }
}

TEST(MlRope, InverseUndoesAndLengthsAreKept)
{
    for (const auto layout : {ml::rope_layout::half, ml::rope_layout::interleaved}) {
        const auto x = head(16, 1.1);
        std::array<double, 8> c{}, s{};
        ml::rope_angles(1000, c, s);
        auto y = x;
        ml::rope(y, c, s, layout);
        double nx = 0, ny = 0;
        for (std::size_t i = 0; i < 16; ++i) {
            nx += x[i] * x[i];
            ny += y[i] * y[i];
        }
        EXPECT_NEAR(nx, ny, 1e-12);
        ml::rope_inverse(y, c, s, layout);
        for (std::size_t i = 0; i < 16; ++i)
            EXPECT_NEAR(y[i], x[i], 1e-14);
    }
}

TEST(MlRope, PositionZeroIsTheIdentity)
{
    const auto x = head(8, 0.5);
    std::array<double, 4> c{}, s{};
    ml::rope_angles(0, c, s);
    auto y = x;
    ml::rope(y, c, s);
    EXPECT_EQ(y, x);
}

TEST(MlRope, TheDotProductDependsOnlyOnTheDistance)
{
    // q at 10 and k at 7 score as q at 103 and k at 100: the point of RoPE
    const auto q0 = head(16, 0.2), k0 = head(16, 2.0);
    const auto score = [&](std::size_t pq, std::size_t pk) {
        auto q = q0, k = k0;
        std::array<double, 8> c{}, s{};
        ml::rope_angles(pq, c, s);
        ml::rope(q, c, s);
        ml::rope_angles(pk, c, s);
        ml::rope(k, c, s);
        double d = 0;
        for (std::size_t i = 0; i < 16; ++i)
            d += q[i] * k[i];
        return d;
    };
    EXPECT_NEAR(score(10, 7), score(103, 100), 1e-12);
    EXPECT_NEAR(score(4, 4), score(500, 500), 1e-12);
}

TEST(MlRope, InverseIsTheBackwardPass)
{
    // L = g · rope(x); dL/dx by central differences equals rope_inverse(g)
    const auto x = head(8, 0.7), g = head(8, 1.9);
    std::array<double, 4> c{}, s{};
    ml::rope_angles(23, c, s);
    auto dx = g;
    ml::rope_inverse(dx, c, s);
    for (std::size_t i = 0; i < 8; ++i) {
        const auto loss = [&](double h) {
            auto y = x;
            y[i] += h;
            ml::rope(y, c, s);
            double l = 0;
            for (std::size_t k = 0; k < 8; ++k)
                l += g[k] * y[k];
            return l;
        };
        EXPECT_NEAR(dx[i], (loss(1e-6) - loss(-1e-6)) / 2e-6, 1e-8);
    }
}

// A long-context table, built by the compiler: YaRN as DeepSeek-V3 computes it, in float, with its temperature in the table
constexpr ml::rope_table<float, 4, 8> yarn_table{10000.0f, ml::rope_scaling<float>{ml::rope_scaling_kind::yarn_index_ramp, 40.0f, 4096, 32.0f,
                                                                                    1.0f, ml::yarn_attention_factor(40.0f)}};
static_assert(yarn_table.cos[0][0] > 1.0f && yarn_table.sin[0][3] == 0.0f);
static_assert(ml::yarn_ramp(0.5, 1.0, 32.0) == 0.0 && ml::yarn_ramp(40.0, 1.0, 32.0) == 1.0 && ml::yarn_ramp(16.5, 1.0, 32.0) == 0.5);
static_assert(ml::yarn_attention_factor(1.0) == 1.0 && ml::yarn_attention_factor(0.5) == 1.0);

TEST(MlRopeScaling, AFactorOfOneIsPlainRopeUnderEveryMethod)
{
    for (const auto kind : {ml::rope_scaling_kind::none, ml::rope_scaling_kind::linear, ml::rope_scaling_kind::ntk,
                            ml::rope_scaling_kind::yarn, ml::rope_scaling_kind::yarn_index_ramp}) {
        const ml::rope_scaling<double> s{kind, 1.0, 4096, 32.0, 1.0, 1.0};
        for (std::size_t i = 0; i < 32; ++i)
            EXPECT_EQ(ml::rope_scaled_frequency(i, 64, 10000.0, s), ml::rope_frequency<double>(i, 64, 10000.0));
    }
    // At compile time, and at run time with a base the compiler cannot see: with a constant base it may fold one
    // table's std::pow and call the platform's for the other (Apple's ARM64 powf differs from the folded value in the
    // last bit), which would compare the compiler's arithmetic with the library's, not the two tables
    static constexpr ml::rope_table<float, 16, 8> plain_ct{500000.0f}, scaled_ct{500000.0f, ml::rope_scaling<float>{}};
    static_assert(plain_ct.cos == scaled_ct.cos && plain_ct.sin == scaled_ct.sin);
    volatile float opaque = 500000.0f;
    const float base = opaque;
    const ml::rope_table<float, 16, 8> plain{base}, scaled{base, ml::rope_scaling<float>{}};
    EXPECT_EQ(plain.cos, scaled.cos);
    EXPECT_EQ(plain.sin, scaled.sin);
}

TEST(MlRopeScaling, LinearAndNtkAreThePapersFormulas)
{
    // Position interpolation: θ / s. NTK-aware (YaRN 3.1): the base becomes b · s^(|D| / (|D| - 2))
    constexpr std::size_t d = 64;
    const double b = 10000, s = 8;
    for (const std::size_t i : {0u, 1u, 7u, 20u, 31u}) {
        const double theta = std::pow(b, -2.0 * static_cast<double>(i) / d);
        EXPECT_NEAR(ml::rope_scaled_frequency(i, d, b, ml::rope_scaling<double>{ml::rope_scaling_kind::linear, s}), theta / s, 1e-15);
        const double b2 = b * std::pow(s, static_cast<double>(d) / (d - 2));
        EXPECT_NEAR(ml::rope_scaled_frequency(i, d, b, ml::rope_scaling<double>{ml::rope_scaling_kind::ntk, s}),
                    std::pow(b2, -2.0 * static_cast<double>(i) / d), 1e-14);
    }
}

TEST(MlRopeScaling, YarnIsThePapersNtkByPartsAtSamplePoints)
{
    // YaRN eqs. 10 to 13: r = L / λ, λ = 2π / θ; γ ramps from α to β; h(θ) = (1 - γ) θ / s + γ θ
    constexpr std::size_t d = 128, L = 4096;
    const double b = 10000, s = 16, alpha = 1, beta = 32;
    const ml::rope_scaling<double> sc{ml::rope_scaling_kind::yarn, s, L, beta, alpha, 1.0};
    int below = 0, between = 0, above = 0;
    for (std::size_t i = 0; i < d / 2; ++i) {
        const double theta = std::pow(b, -2.0 * static_cast<double>(i) / d);
        const double r = static_cast<double>(L) / (2 * std::numbers::pi / theta);
        const double g = r < alpha ? 0 : r > beta ? 1 : (r - alpha) / (beta - alpha);
        below += r < alpha;
        above += r > beta;
        between += r >= alpha && r <= beta;
        EXPECT_NEAR(ml::rope_scaled_frequency(i, d, b, sc), (1 - g) * theta / s + g * theta, 1e-15 * theta + 1e-18);
    }
    EXPECT_GT(below, 0);   // the sample covers every branch of the ramp
    EXPECT_GT(between, 0);
    EXPECT_GT(above, 0);
}

TEST(MlRopeScaling, YarnIndexRampIsDeepSeekV3sCode)
{
    // inference/model.py's precompute_freqs_cis, written out: the correction range floored and ceiled, the ramp over
    // the pair index, freqs / factor · (1 - smooth) + freqs · smooth
    constexpr std::size_t dim = 64, L = 4096;
    const double base = 10000, factor = 40, beta_fast = 32, beta_slow = 1;
    const auto cd = [&](double rot) { return dim * std::log(L / (rot * 2 * std::numbers::pi)) / (2 * std::log(base)); };
    const double low = std::max(std::floor(cd(beta_fast)), 0.0), high = std::min(std::ceil(cd(beta_slow)), dim - 1.0);
    EXPECT_EQ(low, 10.0);
    EXPECT_EQ(high, 23.0);
    const ml::rope_scaling<double> sc{ml::rope_scaling_kind::yarn_index_ramp, factor, L, beta_fast, beta_slow, 1.0};
    for (std::size_t i = 0; i < dim / 2; ++i) {
        const double f = 1 / std::pow(base, static_cast<double>(2 * i) / dim);
        const double ramp = std::clamp((static_cast<double>(i) - low) / (high - low), 0.0, 1.0), smooth = 1 - ramp;
        EXPECT_NEAR(ml::rope_scaled_frequency(i, dim, base, sc), f / factor * (1 - smooth) + f * smooth, 1e-15 * f);
    }
    // DeepSeek-V3's softmax scale gains mscale² with mscale = 0.1 ln 40 + 1
    EXPECT_NEAR(ml::yarn_attention_factor(40.0), 0.1 * std::log(40.0) + 1, 1e-15);
    EXPECT_NEAR(ml::yarn_attention_factor(40.0, 0.707), 0.0707 * std::log(40.0) + 1, 1e-15);
}

TEST(MlRopeScaling, AmplitudeScalesTheTableAndRopeInverseStaysTheBackwardPass)
{
    const double a = 1.25;
    const ml::rope_scaling<double> sc{ml::rope_scaling_kind::linear, 4.0, 0, 32.0, 1.0, a};
    std::array<double, 4> c{}, s{}, c1{}, s1{};
    ml::rope_angles(9, c, s, 10000.0, sc);
    ml::rope_angles(9, c1, s1, 10000.0, ml::rope_scaling<double>{ml::rope_scaling_kind::linear, 4.0});
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(c[i], a * c1[i], 1e-15);
        EXPECT_NEAR(c[i] * c[i] + s[i] * s[i], a * a, 1e-14);
    }
    // ⟨g, rope(x)⟩ = ⟨rope_inverse(g), x⟩: the scaled rotation's adjoint
    const auto x = head(8, 0.3), g = head(8, 2.2);
    auto y = x, dx = g;
    ml::rope(y, c, s);
    ml::rope_inverse(dx, c, s);
    double l = 0, r = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        l += g[i] * y[i];
        r += dx[i] * x[i];
    }
    EXPECT_NEAR(l, r, 1e-13);
}

} // namespace
