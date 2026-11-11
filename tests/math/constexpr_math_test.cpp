//
//  constexpr_math_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/math/constexpr_math.hpp>: compile-time results against <cmath> across
//  each function's range, the run-time path's identity with <cmath>, the normal distribution's
//  round trip and tails; float, long double, integer and mixed arguments, and a number type of the
//  caller's with its own math_limits.
//

#include <snicholls/core/math/constexpr_math.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string_view>
#include <type_traits>

namespace {

namespace m = snicholls::math;

static_assert(m::abs(-2.5) == 2.5 && m::floor(-1.5) == -2.0 && m::trunc(-1.5) == -1.0 && m::round(2.5) == 3.0);
static_assert(m::sqrt(4.0) == 2.0 && m::pow(2.0, 10.0) == 1024.0 && m::exp(0.0) == 1.0 && m::log(1.0) == 0.0);
static_assert(m::near(m::exp(1.0), m::e_v, 1e-15) && m::near(m::log(m::e_v), 1.0, 1e-15));
static_assert(m::near(m::sin(m::pi / 6), 0.5, 1e-15) && m::near(m::cos(m::pi / 3), 0.5, 1e-15));
static_assert(m::near(m::atan(1.0), m::pi / 4, 1e-15) && m::near(m::atan2(1.0, -1.0), 3 * m::pi / 4, 1e-15));
static_assert(m::normal_cdf(0.0) == 0.5 && m::near(m::normal_cdf(1.959963984540054), 0.975, 1e-14));
static_assert(m::near(m::normal_quantile(0.975), 1.959963984540054, 1e-14) && m::normal_quantile(0.5) == 0.0);
static_assert(m::normal_quantile(0.0) == -m::inf && m::is_nan(m::normal_quantile(1.5)));
static_assert(m::pow(0.0, -1.0) == m::inf);   // a pole, not a division by zero

// Every compile-time kernel against <cmath>, by calling the kernel at run time through a constant table
template <typename F, typename G>
void agree(F ours, G theirs, double from, double to, double tol, const char* name)
{
    for (int i = 0; i <= 400; ++i) {
        const double x = from + (to - from) * i / 400.0;
        const double want = theirs(x);
        const double got = ours(x);
        ASSERT_TRUE(m::near(got, want, tol)) << name << "(" << x << "): " << got << " against " << want;
    }
}

TEST(ConstexprMath, CompileTimeKernelsAgreeWithCmath)
{
    namespace d = snicholls::math::detail;
    agree(d::exp_ct<double>, [](double x) { return std::exp(x); }, -700, 700, 1e-14, "exp");
    agree(d::log_ct<double>, [](double x) { return std::log(x); }, 1e-300, 1e300, 1e-15, "log");
    agree(d::sqrt_ct<double>, [](double x) { return std::sqrt(x); }, 0, 1e6, 1e-15, "sqrt");
    agree(d::sin_ct<double>, [](double x) { return std::sin(x); }, -100, 100, 1e-14, "sin");
    agree(d::cos_ct<double>, [](double x) { return std::cos(x); }, -100, 100, 1e-14, "cos");
    agree(d::atan_ct<double>, [](double x) { return std::atan(x); }, -50, 50, 1e-15, "atan");
    agree(d::erf_ct<double>, [](double x) { return std::erf(x); }, -6, 6, 1e-14, "erf");
    agree(d::erfc_ct<double>, [](double x) { return std::erfc(x); }, -6, 26, 1e-12, "erfc");
    agree(d::tanh_ct<double>, [](double x) { return std::tanh(x); }, -25, 25, 1e-14, "tanh");
}

// expm1 and log1p exist for their accuracy near 0, where an absolute tolerance would hide a cancellation
TEST(ConstexprMath, Expm1AndLog1pKeepTheirRelativeAccuracyNearZero)
{
    namespace d = snicholls::math::detail;
    for (const double x : {1e-300, -1e-300, 1e-12, -1e-12, 1e-6, -3e-4, 0.1, -0.3, 0.5, -0.5, 0.99, -0.99, 1.0, -1.0, 2.0, -20.0, 300.0}) {
        EXPECT_LE(std::abs(d::expm1_ct(x) - std::expm1(x)), 4e-16 * std::abs(std::expm1(x))) << "expm1(" << x << ")";
        if (x > -1) {
            EXPECT_LE(std::abs(d::log1p_ct(x) - std::log1p(x)), 4e-16 * std::abs(std::log1p(x))) << "log1p(" << x << ")";
        }
    }
    for (const double x : {1e-20, 1e-9, 0.25, 1e10, 1e300})
        EXPECT_LE(std::abs(d::log1p_ct(x) - std::log1p(x)), 4e-16 * std::abs(std::log1p(x))) << "log1p(" << x << ")";
    EXPECT_EQ(m::expm1(0.3), std::expm1(0.3));   // run time: <cmath>
    EXPECT_EQ(m::log1p(0.3), std::log1p(0.3));
}

static_assert(m::expm1(0.0) == 0.0 && m::log1p(0.0) == 0.0 && m::expm1(-m::inf) == -1.0 && m::expm1(m::inf) == m::inf);
static_assert(m::log1p(-1.0) == -m::inf && m::is_nan(m::log1p(-2.0)) && m::log1p(m::inf) == m::inf);
static_assert(m::is_nan(m::expm1(m::nan)) && m::is_nan(m::log1p(m::nan)));
static_assert(m::expm1(1e-30) == 1e-30 && m::log1p(1e-30) == 1e-30);   // exp(x) - 1 would be 0
static_assert(m::near(m::expm1(1.0), m::e_v - 1, 1e-15) && m::near(m::log1p(m::e_v - 1), 1.0, 1e-15));
static_assert(m::near(m::expm1(1e-5f), 1.00000500001666e-5f, 1e-12f) && m::near<long double>(m::log1p(1e-5L), 9.99995000033333e-6L, 1e-18L));

TEST(ConstexprMath, TheRunTimePathIsCmathToTheBit)
{
    for (const double x : {0.1, 1.7, 33.0, -4.2})
        if (x > 0) {
            EXPECT_EQ(m::log(x), std::log(x));
            EXPECT_EQ(m::sqrt(x), std::sqrt(x));
        }
    EXPECT_EQ(m::exp(1.7), std::exp(1.7));
    EXPECT_EQ(m::erfc(2.5), std::erfc(2.5));
}

TEST(NormalDistribution, QuantileInvertsTheCdfAcrossTheRange)
{
    for (const double p : {1e-300, 1e-100, 1e-20, 1e-10, 1e-5, 0.001, 0.02425, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97575, 0.999, 1 - 1e-10})
        EXPECT_NEAR(m::normal_cdf(m::normal_quantile(p)) / p, 1.0, 1e-12) << p;
    EXPECT_NEAR(m::normal_quantile(1e-10), -6.361340902404056, 1e-12);
    EXPECT_NEAR(m::normal_pdf(0.0), 0.3989422804014327, 1e-16);
}

TEST(GammaAndChiSquare, AgreeWithScipy)
{
    EXPECT_NEAR(m::chi_square_cdf(3.84, 1), 0.9499564787512949, 1e-13);
    EXPECT_NEAR(m::chi_square_cdf(25.0, 10), 0.9946544945128659, 1e-13);
    EXPECT_NEAR(m::chi_square_cdf(0.5, 7), 0.000553518609575035, 1e-15);
    EXPECT_NEAR(m::lgamma(0.3), 1.0957979948180756, 1e-13);
    EXPECT_NEAR(m::lgamma(171.5), 709.1431630309283, 1e-10);
    EXPECT_NEAR(m::gamma_p(4.5, 2.0), 0.08858747316832083, 1e-14);
}
static_assert(m::near(m::lgamma(5.0), 3.1780538303479458, 1e-14));   // log 4!
static_assert(m::near(m::chi_square_cdf(2.0, 2), 1 - m::exp(-1.0), 1e-14));

// ---- generic in the number type ------------------------------------------------------------------------------

// An integer computes in double, a float in float (through double at compile time), mixed in the common type.
static_assert(std::is_same_v<decltype(m::sqrt(4)), double> && m::sqrt(4) == 2.0);
static_assert(std::is_same_v<decltype(m::exp(1.0f)), float> && std::is_same_v<decltype(m::log(2.0L)), long double>);
static_assert(std::is_same_v<decltype(m::pow(2.0f, 3)), float> && m::pow(2.0f, 3) == 8.0f);
static_assert(std::is_same_v<decltype(m::atan2(1.0f, 2.0)), double>);
static_assert(m::exp(1.0f) == static_cast<float>(m::e_v));                      // rounded once from double
static_assert(m::sqrt(2.0f) == 1.41421356f && m::normal_quantile(0.5f) == 0.0f);
static_assert(m::near<float>(m::normal_cdf(1.959964f), 0.975f, 1e-6f));
static_assert(m::near<long double>(m::exp(1.0L), 2.718281828459045235360287471352662498L, 4 * std::numeric_limits<long double>::epsilon()));
static_assert(m::near<long double>(m::sin(m::math_limits<long double>::pi / 6), 0.5L, 4 * std::numeric_limits<long double>::epsilon()));
static_assert(m::is_nan(m::sqrt(-1.0f)) && m::is_inf(m::log(0.0L)) && m::exp(-1e5f) == 0.0f);

template <typename T, typename F, typename G>
void agree_in(F ours, G theirs, T from, T to, T tol, const char* name)
{
    for (int i = 0; i <= 400; ++i) {
        const T x = from + (to - from) * T(i) / T(400);
        const T want = theirs(x), got = ours(x);
        ASSERT_TRUE(m::near<T>(got, want, tol)) << name << "(" << static_cast<double>(x) << ")";
    }
}

TEST(ConstexprMathLongDouble, CompileTimeKernelsAgreeWithCmathToTheTypesPrecision)
{
    namespace d = snicholls::math::detail;
    using T = long double;
    const T tol = 64 * std::numeric_limits<T>::epsilon();
    agree_in<T>(d::exp_ct<T>, [](T x) { return std::exp(x); }, -700, 700, tol, "exp");
    agree_in<T>(d::log_ct<T>, [](T x) { return std::log(x); }, T(1e-300), T(1e300), tol, "log");
    agree_in<T>(d::sqrt_ct<T>, [](T x) { return std::sqrt(x); }, 0, T(1e6), tol, "sqrt");
    agree_in<T>(d::sin_ct<T>, [](T x) { return std::sin(x); }, -100, 100, tol, "sin");
    agree_in<T>(d::atan_ct<T>, [](T x) { return std::atan(x); }, -50, 50, tol, "atan");
    agree_in<T>(d::erf_ct<T>, [](T x) { return std::erf(x); }, -6, 6, tol, "erf");
    agree_in<T>(d::erfc_ct<T>, [](T x) { return std::erfc(x); }, -6, 26, 16 * tol, "erfc");
    agree_in<T>(d::tanh_ct<T>, [](T x) { return std::tanh(x); }, -25, 25, tol, "tanh");
    EXPECT_EQ(m::exp(T(1.7)), std::exp(T(1.7)));   // run time: <cmath>'s long double
}

TEST(ConstexprMathFloat, TheRunTimePathIsCmathsFloatToTheBit)
{
    for (const float x : {0.1f, 1.7f, 33.0f})
        EXPECT_EQ(m::log(x), std::log(x));
    EXPECT_EQ(m::exp(1.7f), std::exp(1.7f));
    EXPECT_EQ(m::erfc(2.5f), std::erfc(2.5f));
}

} // namespace

// A number type of the caller's: its own arithmetic, numeric_limits and math functions, and math_limits.
namespace money {
struct amount {
    double v;
    constexpr amount(double x = 0) : v(x) {}   // NOLINT(google-explicit-constructor)
    constexpr explicit operator double() const { return v; }
    constexpr explicit operator long long() const { return static_cast<long long>(v); }
    friend constexpr amount operator+(amount a, amount b) { return a.v + b.v; }
    friend constexpr amount operator-(amount a, amount b) { return a.v - b.v; }
    friend constexpr amount operator-(amount a) { return -a.v; }
    friend constexpr amount operator*(amount a, amount b) { return a.v * b.v; }
    friend constexpr amount operator/(amount a, amount b) { return a.v / b.v; }
    friend constexpr auto operator<=>(amount, amount) = default;
    amount& operator*=(amount b) { v *= b.v; return *this; }
    friend amount exp(amount a) { return std::exp(a.v) + 0.0; }   // found by argument-dependent lookup at run time
};
} // namespace money
template <>
struct std::numeric_limits<money::amount> : std::numeric_limits<double> {
};
template <>
struct snicholls::math::math_limits<money::amount> {
    using compute_type = double;   // the kernels run in double at compile time
};

namespace {

static_assert(m::math_number<money::amount> && !m::math_number<std::string_view>);
static_assert(m::exp(money::amount{0.0}).v == 1.0);

TEST(ConstexprMathCustomType, AtRunTimeItsOwnFunctionsAreFound)
{
    EXPECT_EQ(m::exp(money::amount{1.5}).v, std::exp(1.5));
}

} // namespace
