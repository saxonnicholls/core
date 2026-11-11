//
//  activations_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/activations.hpp>: each activation against its textbook formula in long
//  double where the formula is safe, each derivative against a central difference, the tails where the
//  textbook formula overflows or cancels, NaN, float, and compile-time evaluation.
//

#include <snicholls/core/ml/activations.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;
using ld = long double;

// Compile time: every function is a constant expression
static_assert(ml::sigmoid(0.0) == 0.5 && ml::relu(-2.0) == 0.0 && ml::relu(3) == 3.0 && ml::relu_squared(3.0) == 9.0);
static_assert(ml::relu_derivative(2.0) == 1.0 && ml::relu_derivative(0.0) == 0.0 && ml::relu_squared_derivative(1.5) == 3.0);
static_assert(ml::gelu(0.0) == 0.0 && ml::gelu_tanh(0.0) == 0.0 && ml::silu(0.0) == 0.0 && ml::softcap(0.0, 15.0) == 0.0);
static_assert(m::near(ml::softplus(0.0), 0.6931471805599453, 1e-15) && m::near(ml::log_sigmoid(0.0), -0.6931471805599453, 1e-15));
static_assert(ml::logit(0.5) == 0.0 && ml::logit(0.0) == -m::inf && ml::logit(1.0) == m::inf && m::is_nan(ml::logit(1.5)));
static_assert(m::near(ml::gelu(1.0), 0.8413447460685429, 1e-15) && m::near(ml::gelu_tanh(1.0), 0.8411919906082768, 1e-15));
static_assert(ml::sigmoid_derivative(0.0) == 0.25 && ml::tanh_derivative_from_output(0.5) == 0.75);
static_assert(m::near(ml::gelu_derivative(0.0), 0.5, 1e-15) && m::near(ml::gelu_tanh_derivative(0.0), 0.5, 1e-15));
static_assert(m::near(ml::silu_derivative(0.0), 0.5, 1e-15) && ml::softcap_derivative(0.0, 15.0) == 1.0);

// A table a kernel reads, built by the compiler: GELU over [-4, 4] in float
constexpr auto gelu_table = [] {
    std::array<float, 33> t{};
    for (int i = 0; i <= 32; ++i)
        t[static_cast<std::size_t>(i)] = ml::gelu(-4.0f + 0.25f * static_cast<float>(i));
    return t;
}();
static_assert(gelu_table[16] == 0.0f);

ld sigmoid_ref(ld x) { return 1 / (1 + std::exp(-x)); }

TEST(MlActivations, AgreeWithTheTextbookFormulasWhereThoseAreSafe)
{
    for (int i = -400; i <= 400; ++i) {
        const double x = i / 40.0;   // [-10, 10]
        const ld X = x;
        const ld s = sigmoid_ref(X);
        EXPECT_NEAR(ml::sigmoid(x), static_cast<double>(s), 2e-16) << x;
        EXPECT_NEAR(ml::softplus(x), static_cast<double>(std::log(1 + std::exp(X))), 4e-15) << x;
        EXPECT_NEAR(ml::log_sigmoid(x), static_cast<double>(std::log(s)), 4e-15) << x;
        EXPECT_NEAR(ml::silu(x), static_cast<double>(X * s), 4e-15) << x;
        EXPECT_NEAR(ml::gelu(x), static_cast<double>(X * 0.5L * (1 + std::erf(X / std::sqrt(2.0L)))), 4e-15) << x;
        const ld u = std::sqrt(2 / 3.14159265358979323846264338327950288L) * (X + 0.044715L * X * X * X);
        EXPECT_NEAR(ml::gelu_tanh(x), static_cast<double>(0.5L * X * (1 + std::tanh(u))), 4e-15) << x;
        EXPECT_NEAR(ml::softcap(x, 3.0), static_cast<double>(3 * std::tanh(X / 3)), 4e-15) << x;
        EXPECT_EQ(ml::relu_squared(x), x > 0 ? x * x : 0.0);
    }
    for (const double p : {1e-12, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-12})
        EXPECT_NEAR(ml::logit(p), static_cast<double>(std::log(ld(p) / (1 - ld(p)))), 1e-12) << p;
}

// d/dx by a central difference in long double, which every derivative must match. Its rounding error is about
// eps |f| / h: where long double is only double (MSVC), a step of 1e-6 leaves ~1e-9 at |f| near 6, so the step
// there is 1e-5 (truncation stays below 1e-10)
template <typename F>
ld central(F f, ld x)
{
    const ld h = std::numeric_limits<ld>::digits > 53 ? 1e-6L : 1e-5L;
    return (f(x + h) - f(x - h)) / (2 * h);
}

TEST(MlActivations, DerivativesMatchACentralDifference)
{
    for (int i = -60; i <= 60; ++i) {
        const ld x = i / 10.0L + 0.013L;   // avoid 0, where relu's derivative jumps
        const double xd = static_cast<double>(x);
        EXPECT_NEAR(ml::sigmoid_derivative(xd), static_cast<double>(central([](ld v) { return ml::sigmoid(v); }, x)), 1e-9) << xd;
        EXPECT_NEAR(ml::silu_derivative(xd), static_cast<double>(central([](ld v) { return ml::silu(v); }, x)), 1e-9) << xd;
        EXPECT_NEAR(ml::gelu_derivative(xd), static_cast<double>(central([](ld v) { return ml::gelu(v); }, x)), 1e-9) << xd;
        EXPECT_NEAR(ml::gelu_tanh_derivative(xd), static_cast<double>(central([](ld v) { return ml::gelu_tanh(v); }, x)), 1e-9) << xd;
        EXPECT_NEAR(ml::relu_squared_derivative(xd), static_cast<double>(central([](ld v) { return ml::relu_squared(v); }, x)), 1e-9) << xd;
        EXPECT_NEAR(ml::relu_derivative(xd), static_cast<double>(central([](ld v) { return ml::relu(v); }, x)), 1e-9) << xd;
        EXPECT_NEAR(ml::softcap_derivative(xd, 2.5), static_cast<double>(central([](ld v) { return ml::softcap(v, 2.5L); }, x)), 1e-9) << xd;
        EXPECT_NEAR(ml::sigmoid(-xd), static_cast<double>(central([](ld v) { return ml::log_sigmoid(v); }, x)), 1e-9) << xd;
        EXPECT_NEAR(ml::sigmoid(xd), static_cast<double>(central([](ld v) { return ml::softplus(v); }, x)), 1e-9) << xd;
    }
}

TEST(MlActivations, TailsStayFiniteWhereTheNaiveFormulasFail)
{
    // e^1000 overflows: the naive sigmoid of -1000 is 1/(1+inf) = 0 and fine, but e^x/(1+e^x) at +1000 is inf/inf
    EXPECT_EQ(ml::sigmoid(1000.0), 1.0);
    EXPECT_EQ(ml::sigmoid(-1000.0), 0.0);
    EXPECT_GT(ml::sigmoid(-700.0), 0.0);                       // still representable, and kept
    EXPECT_EQ(ml::softplus(1000.0), 1000.0);                    // log(1 + e^1000) is inf
    EXPECT_EQ(ml::log_sigmoid(-1000.0), -1000.0);               // log(0) is -inf
    EXPECT_NEAR(ml::log_sigmoid(-40.0), -40.0 - std::exp(-40.0), 1e-15);
    EXPECT_EQ(ml::softplus(-1000.0), 0.0);
    EXPECT_GT(ml::softplus(-700.0), 0.0);
    // GELU's left tail: x Φ(x) for x = -30 is about -1.5e-196; 0.5 x (1 + erf) cancels to 0
    EXPECT_LT(ml::gelu(-30.0), 0.0);
    EXPECT_NEAR(ml::gelu(-30.0) / (-30.0 * 0.5 * std::erfc(30.0 / std::sqrt(2.0))), 1.0, 1e-12);
    EXPECT_LT(ml::gelu_tanh(-12.0), 0.0);                       // 1 + tanh(u) cancels to 0 near u = -19
    EXPECT_EQ(ml::gelu_tanh(1000.0), 1000.0);
    EXPECT_EQ(ml::silu(-1000.0), -0.0);
    EXPECT_EQ(ml::softcap(1e9, 15.0), 15.0);
    EXPECT_EQ(ml::softcap(-1e9, 15.0), -15.0);
    EXPECT_NEAR(ml::logit(ml::sigmoid(20.0)), 20.0, 1e-6);      // as far as the rounding of 1 - sigmoid(20) allows
    EXPECT_NEAR(ml::logit(ml::sigmoid(-30.0)), -30.0, 1e-12);   // the small side keeps every digit
    EXPECT_NEAR(ml::logit(1e-300), std::log(1e-300), 1e-12);
}

TEST(MlActivations, NanPropagates)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const double v : {ml::sigmoid(nan), ml::softplus(nan), ml::log_sigmoid(nan), ml::logit(nan), ml::relu(nan),
                           ml::relu_squared(nan), ml::silu(nan), ml::gelu(nan), ml::gelu_tanh(nan), ml::softcap(nan, 15.0),
                           ml::sigmoid_derivative(nan), ml::silu_derivative(nan), ml::gelu_derivative(nan),
                           ml::gelu_tanh_derivative(nan), ml::softcap_derivative(nan, 15.0)})
        EXPECT_TRUE(std::isnan(v));
}

TEST(MlActivations, FloatComputesInFloatAndMatchesDouble)
{
    static_assert(std::is_same_v<decltype(ml::gelu(1.0f)), float> && std::is_same_v<decltype(ml::sigmoid(1)), double>);
    for (int i = -40; i <= 40; ++i) {
        const float x = static_cast<float>(i) / 4.0f;
        EXPECT_NEAR(ml::gelu(x), static_cast<float>(ml::gelu(static_cast<double>(x))), 1e-6f) << x;
        EXPECT_NEAR(ml::sigmoid(x), static_cast<float>(ml::sigmoid(static_cast<double>(x))), 1e-7f) << x;
        EXPECT_NEAR(ml::gelu_tanh(x), static_cast<float>(ml::gelu_tanh(static_cast<double>(x))), 1e-6f) << x;
    }
    // The compile-time table agrees with the run-time function
    for (int i = 0; i <= 32; ++i)
        EXPECT_NEAR(gelu_table[static_cast<std::size_t>(i)], ml::gelu(-4.0f + 0.25f * static_cast<float>(i)), 1e-6f);
}

} // namespace
