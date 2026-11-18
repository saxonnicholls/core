//
//  optim_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/optim.hpp>: AdamW against the formula in long double over several steps, its bias
//  correction near beta = 1 in the first steps against exact references, its first step's size, weight decay alone,
//  AdamW minimising a quadratic; both schedules at their corners; the global norm over shards and the clip
//  coefficient; Newton-Schulz carrying singular values near 1 and its documented range; the FLOPs of GPT-2 small;
//  and compile-time use.
//

#include <snicholls/core/ml/optim.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

static_assert(std::is_trivially_copyable_v<ml::adamw_config<float>> && sizeof(ml::adamw_config<float>) == 5 * sizeof(float));

constexpr bool adamw_at_compile_time()
{
    // The first step moves every parameter by about the learning rate, against its gradient's sign
    std::array<double, 2> p{1.0, -1.0}, m1{}, v1{};
    const std::array<double, 2> g{0.5, -3.0};
    ml::adamw_step(p, g, m1, v1, 1, {.learning_rate = 0.01});
    return m::near(p[0], 0.99, 1e-9) && m::near(p[1], -0.99, 1e-9);
}
static_assert(adamw_at_compile_time());
static_assert(ml::warmup_cosine<double>(0, 10, 100, 1.0, 0.1) == 0.1 && ml::warmup_cosine<double>(9, 10, 100, 1.0, 0.1) == 1.0);
static_assert(ml::warmup_warmdown<double>(50, 100, 10, 20, 0.0) == 1.0);
static_assert(ml::clip_coefficient(0.5, 1.0) == 1.0 && ml::newton_schulz_polynomial(0.0) == 0.0);

TEST(MlOptim, AdamWMatchesTheFormulaOverSeveralSteps)
{
    const ml::adamw_config<double> c{.learning_rate = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-8, .weight_decay = 0.1};
    std::vector<double> p{0.3, -0.7, 1.2}, mm(3), vv(3);
    std::vector<long double> rp(p.begin(), p.end()), rm(3), rv(3);
    for (std::uint64_t t = 1; t <= 5; ++t) {
        const std::vector<double> g{std::sin(double(t)), std::cos(double(t)), 0.01 * double(t)};
        ml::adamw_step(p, g, mm, vv, t, c);
        for (std::size_t i = 0; i < 3; ++i) {
            rm[i] = 0.9L * rm[i] + 0.1L * g[i];
            rv[i] = 0.95L * rv[i] + 0.05L * g[i] * g[i];
            const long double mh = rm[i] / (1 - std::pow(0.9L, t)), vh = rv[i] / (1 - std::pow(0.95L, t));
            rp[i] -= 3e-4L * (mh / (std::sqrt(vh) + 1e-8L) + 0.1L * rp[i]);
        }
    }
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(p[i], static_cast<double>(rp[i]), 1e-14);
}

TEST(MlOptim, BiasCorrectionDoesNotCancelInTheFirstSteps)
{
    // 1 - beta^t for beta near 1 and small t: 1 - pow(beta, t) loses the digits pow rounds away near 1 (before 0.14.0,
    // 56 and 420 units in the last place in float at t = 2 for beta = 0.999 and 0.9999). The float beta's powers up to
    // the third are exact in double, so double is an exact reference here.
    for (const float beta : {0.9f, 0.95f, 0.999f, 0.9999f})
        for (const std::uint64_t t : {1u, 2u, 3u}) {
            const double b = beta, want = 1 - std::pow(b, double(t));
            EXPECT_LE(std::abs(ml::adam_bias_correction(beta, t) - want) / want, 2 * std::numeric_limits<float>::epsilon()) << beta << " " << t;
        }
    // double at t = 2: 1 - b^2 = fma(-b, b, 1), rounded once
    for (const double b : {0.999, 0.9999, 0.99999}) {
        const double want = std::fma(-b, b, 1.0);
        EXPECT_LE(std::abs(ml::adam_bias_correction(b, 2) - want) / want, 2 * std::numeric_limits<double>::epsilon()) << b;
    }
    EXPECT_EQ(ml::adam_bias_correction(0.0, 1), 1.0);
    EXPECT_NEAR(ml::adam_bias_correction(0.999, 100000), 1.0, 1e-15);
    static_assert(snicholls::math::near(ml::adam_bias_correction(0.999, 2), 1.999e-3, 1e-15));
    static_assert(ml::adam_bias_correction(0.5, 1) == 0.5);
    static_assert(ml::adam_bias_correction(0.0f, 1) == 1.0f);
    static_assert(noexcept(ml::adam_bias_correction(0.9f, 1)) && noexcept(ml::adam_bias_correction(0.9, 1)));
    static_assert(std::same_as<decltype(ml::adam_bias_correction(0.9f, 1)), float>);
    // long double, against (1 - b)(1 + b + ... + b^(t-1)): 1 - b is exact and the sum has no cancellation
    for (const std::uint64_t t : {1u, 2u, 3u, 10u}) {
        const long double b = 0.9999L;
        long double sum = 0, power = 1;
        for (std::uint64_t k = 0; k < t; ++k, power *= b)
            sum += power;
        const long double want = (1 - b) * sum;
        EXPECT_LE(std::abs(ml::adam_bias_correction(b, t) - want) / want, 16 * std::numeric_limits<long double>::epsilon()) << t;
    }
    // step 0: no correction, 0
    EXPECT_EQ(ml::adam_bias_correction(0.9, 0), 0.0);

    // Through adamw_step in float: the update at t = 2 with beta2 = 0.999 and 0.9999, against double
    for (const float beta2 : {0.999f, 0.9999f})
        for (const std::uint64_t t : {1u, 2u, 3u}) {
            const ml::adamw_config<float> c{.learning_rate = 1.0f, .beta1 = 0.9f, .beta2 = beta2, .eps = 0.0f, .weight_decay = 0.0f};
            std::array<float, 1> p{0.0f}, mm{-0.0620489f}, vv{0.555658f};
            const float g = -1.23339f;
            const double m1 = 0.9 * double(mm[0]) + double(1.0f - 0.9f) * g;
            const double v1 = double(beta2) * double(vv[0]) + double(1.0f - beta2) * double(g) * g;
            const double want = -(m1 / (1 - std::pow(double(0.9f), double(t)))) / std::sqrt(v1 / (1 - std::pow(double(beta2), double(t))));
            ml::adamw_step(p, std::array<float, 1>{g}, mm, vv, t, c);
            EXPECT_LE(std::abs(p[0] - want) / std::abs(want), 8 * std::numeric_limits<float>::epsilon()) << beta2 << " " << t;
        }
}

TEST(MlOptim, WeightDecayAloneShrinksTowardZero)
{
    std::array<double, 1> p{2.0}, mm{}, vv{};
    ml::adamw_step(p, std::array<double, 1>{0.0}, mm, vv, 1, {.learning_rate = 0.1, .weight_decay = 0.5});
    EXPECT_DOUBLE_EQ(p[0], 2.0 - 0.1 * 0.5 * 2.0);
}

TEST(MlOptim, AdamWMinimisesAQuadratic)
{
    // f(x) = sum (x_i - i)^2 from zero: AdamW with a cosine schedule lands on the minimum
    std::vector<float> x(4, 0.0f), mm(4), vv(4), g(4);
    for (std::uint64_t t = 0; t < 2000; ++t) {
        for (std::size_t i = 0; i < 4; ++i)
            g[i] = 2.0f * (x[i] - float(i));
        ml::adamw_step(x, g, mm, vv, t + 1, {.learning_rate = ml::warmup_cosine<float>(t, 50, 2000, 0.05f, 0.0f)});
    }
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(x[i], float(i), 1e-3f);
}

TEST(MlOptim, WarmupCosineAtItsCorners)
{
    EXPECT_DOUBLE_EQ(ml::warmup_cosine<double>(4, 10, 110, 6e-4, 6e-5), 6e-4 * 5 / 10);
    EXPECT_NEAR(ml::warmup_cosine<double>(10, 10, 110, 6e-4, 6e-5), 6e-4, 1e-18);
    EXPECT_NEAR(ml::warmup_cosine<double>(60, 10, 110, 6e-4, 6e-5), (6e-4 + 6e-5) / 2, 1e-18);
    EXPECT_DOUBLE_EQ(ml::warmup_cosine<double>(110, 10, 110, 6e-4, 6e-5), 6e-5);
    EXPECT_DOUBLE_EQ(ml::warmup_cosine<double>(500, 10, 110, 6e-4, 6e-5), 6e-5);
}

TEST(MlOptim, WarmupWarmdownIsATrapezoid)
{
    EXPECT_DOUBLE_EQ(ml::warmup_warmdown<double>(0, 100, 4, 20, 0.1), 0.25);
    EXPECT_DOUBLE_EQ(ml::warmup_warmdown<double>(3, 100, 4, 20, 0.1), 1.0);
    EXPECT_DOUBLE_EQ(ml::warmup_warmdown<double>(80, 100, 4, 20, 0.1), 1.0);
    EXPECT_DOUBLE_EQ(ml::warmup_warmdown<double>(90, 100, 4, 20, 0.1), 0.5 + 0.5 * 0.1);
    EXPECT_DOUBLE_EQ(ml::warmup_warmdown<double>(100, 100, 4, 20, 0.1), 0.1);
    EXPECT_DOUBLE_EQ(ml::warmup_warmdown<double>(5, 100, 0, 0, 0.1), 1.0);   // no warm-up, no warm-down
}

TEST(MlOptim, GlobalNormOverShardsAndTheClipCoefficient)
{
    const std::vector<double> a{3.0, 0.0}, b{0.0, 4.0};
    const double norm = std::sqrt(ml::squared_norm(a) + ml::squared_norm(b));   // shards add
    EXPECT_DOUBLE_EQ(norm, 5.0);
    EXPECT_NEAR(ml::clip_coefficient(norm, 1.0), 1.0 / (5.0 + 1e-6), 1e-15);
    EXPECT_EQ(ml::clip_coefficient(0.999, 1.0), 1.0);
    EXPECT_EQ(ml::clip_coefficient(std::numeric_limits<double>::quiet_NaN(), 1.0), 1.0);
}

TEST(MlOptim, NewtonSchulzCarriesSingularValuesNearOne)
{
    for (const double s0 : {0.002, 0.05, 0.3, 0.7, 1.0}) {
        double s = s0;
        for (int k = 0; k < ml::muon_newton_schulz<double>::steps; ++k)
            s = ml::newton_schulz_polynomial(s);
        EXPECT_GT(s, 0.68) << s0;
        EXPECT_LT(s, 1.2) << s0;
    }
    EXPECT_NEAR(ml::newton_schulz_polynomial(1.0), 3.4445 - 4.7750 + 2.0315, 1e-15);
}

TEST(MlOptim, NewtonSchulzRangeIsTheDocumentedOne)
{
    // Five steps from [1.55e-3, 1] land in about [0.682, 1.2024]: a little outside [0.7, 1.2] on both sides, which
    // the documentation stated before 0.14.0. Below 1.55e-3 a value x reaches only about a^5 x.
    auto five = [](double s) {
        for (int k = 0; k < ml::muon_newton_schulz<double>::steps; ++k)
            s = ml::newton_schulz_polynomial(s);
        return s;
    };
    double lo = 2, hi = 0;
    for (int i = 0; i <= 200000; ++i) {
        const double s = five(1.55e-3 + (1 - 1.55e-3) * i / 200000.0);
        lo = std::min(lo, s);
        hi = std::max(hi, s);
    }
    EXPECT_GT(lo, 0.6818);
    EXPECT_LT(lo, 0.6819);
    EXPECT_GT(hi, 1.2023);
    EXPECT_LT(hi, 1.2025);
    EXPECT_LT(five(1.5e-3), 0.68);
    const double a5 = std::pow(ml::muon_newton_schulz<double>::a, 5);
    EXPECT_NEAR(five(1e-6) / 1e-6, a5, 1e-3 * a5);
}

TEST(MlOptim, FlopsOfGpt2Small)
{
    // GPT-2 124M: 85M non-embedding parameters, 12 layers of 12 heads of 64, context 1024
    const double f = ml::transformer_flops_per_token(85'000'000, 12, 12, 64, 1024);
    EXPECT_DOUBLE_EQ(f, 6.0 * 85e6 + 12.0 * 12 * 12 * 64 * 1024);
    EXPECT_NEAR(ml::model_flops_utilisation(f, 1e6, 989e12), f * 1e6 / 989e12, 1e-18);
}

// AdamW on a backend is adamw_step
TEST(MlOptim, AdamwRowsIsAdamwStep)
{
    std::vector<double> p{1, -2, 0.5}, g{0.3, -0.1, 2}, m(3), v(3), p2 = p, m2(3), v2(3);
    const ml::adamw_config<double> c{.learning_rate = 0.01, .weight_decay = 0.1};
    ml::adamw_rows(snicholls::linalg::reference, std::span<double>(p), std::span<const double>(g), std::span<double>(m), std::span<double>(v), 1, c);
    ml::adamw_step(p2, g, m2, v2, 1, c);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(m, m2);
    EXPECT_EQ(v, v2);
}

} // namespace
