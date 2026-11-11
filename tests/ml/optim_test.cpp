//
//  optim_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/optim.hpp>: AdamW against the formula in long double over several steps, its
//  first step's size, weight decay alone, AdamW minimising a quadratic; both schedules at their corners; the
//  global norm over shards and the clip coefficient; Newton-Schulz carrying singular values near 1; the FLOPs of
//  GPT-2 small; and compile-time use.
//

#include <snicholls/core/ml/optim.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
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

TEST(MlOptim, FlopsOfGpt2Small)
{
    // GPT-2 124M: 85M non-embedding parameters, 12 layers of 12 heads of 64, context 1024
    const double f = ml::transformer_flops_per_token(85'000'000, 12, 12, 64, 1024);
    EXPECT_DOUBLE_EQ(f, 6.0 * 85e6 + 12.0 * 12 * 12 * 64 * 1024);
    EXPECT_NEAR(ml::model_flops_utilisation(f, 1e6, 989e12), f * 1e6 / 989e12, 1e-18);
}

} // namespace
