//
//  special_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/math/special.hpp>: Lambert W on both real branches against w e^w = z and
//  high-precision references, softplus and the logistic function in both tails, log Phi from +6 to
//  -1000, the entropy, relative entropy, Kullback-Leibler, Huber and xlogy atoms with their conventions
//  at 0 and infinity; float, double, long double and integer arguments; constant evaluation.
//
//  Every reference value below was computed independently in Python 3 with the standard library's
//  decimal module at 50 to 80 significant digits: log Phi from the erf power series (|x| / sqrt 2 < 5) and
//  Laplace's continued fraction for erfc with 4000 terms (beyond), Lambert W by Halley's iteration on
//  w e^w = z in decimal arithmetic, softplus and the logistic function from Decimal.exp and Decimal.ln.
//

#include <snicholls/core/math/special.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace {

namespace m = snicholls::math;

constexpr double inv_e = 0.36787944117144233;   // -1/e rounded to nearest is just above -1/e
constexpr double dinf = std::numeric_limits<double>::infinity();

// ---- constant evaluation -------------------------------------------------------------------------
static_assert(m::lambert_w0(0.0) == 0.0 && m::lambert_w0(-inv_e) == -1.0 && m::lambert_wm1(-inv_e) == -1.0);
static_assert(m::near(m::lambert_w0(1.0), 0.56714329040978384, 1e-15));          // the omega constant
static_assert(m::near(m::lambert_w0(m::e_v), 1.0, 1e-15));
static_assert(m::near(m::lambert_wm1(-0.1), -3.5771520639572972, 1e-15));
static_assert(m::is_nan(m::lambert_w0(-0.5)) && m::is_nan(m::lambert_wm1(0.5)) && m::lambert_wm1(0.0) == -m::inf);
static_assert(m::lambert_w0(m::inf) == m::inf && m::is_nan(m::lambert_w0(-m::inf)));
static_assert(m::near(m::softplus(0.0), 0.69314718055994531, 1e-16) && m::softplus(1000.0) == 1000.0);
static_assert(m::logistic(0.0) == 0.5 && m::logistic(1000.0) == 1.0 && m::logistic(-1000.0) == 0.0);
static_assert(m::log_logistic(-1000.0) == -1000.0);
static_assert(m::near(m::log_normcdf(-40.0), -804.60844201375380, 1e-15) && m::log_normcdf(0.0) == m::log1p(-0.5));
static_assert(m::entr(0.0) == 0.0 && m::entr(-1.0) == -m::inf && m::entr(1.0) == 0.0);
static_assert(m::rel_entr(0.0, 0.0) == 0.0 && m::rel_entr(1.0, 0.0) == m::inf && m::rel_entr(1.0, 1.0) == 0.0);
static_assert(m::kl_div(0.0, 2.0) == 2.0 && m::kl_div(3.0, 3.0) == 0.0 && m::kl_div(-1.0, 1.0) == m::inf);
static_assert(m::huber(0.5) == 0.25 && m::huber(3.0) == 5.0 && m::huber(-3.0, 2.0) == 8.0);
static_assert(m::xlogy(0.0, 0.0) == 0.0 && m::xlogy(0.0, -1.0) == 0.0 && m::xlogy(2.0, 1.0) == 0.0);
// The argument's type: float in double rounded once, long double in itself, an integer in double.
static_assert(std::is_same_v<decltype(m::lambert_w0(1.0f)), float> && std::is_same_v<decltype(m::softplus(1)), double>);
static_assert(std::is_same_v<decltype(m::rel_entr(1.0f, 2.0L)), long double> && std::is_same_v<decltype(m::huber(1.0f)), float>);
static_assert(m::lambert_w0(1.0f) == static_cast<float>(m::lambert_w0(1.0)));

double rel(double got, double want) { return std::abs(got - want) / std::abs(want); }
long double rel_l(long double got, long double want) { return std::abs(got - want) / std::abs(want); }

constexpr double eps = std::numeric_limits<double>::epsilon();
constexpr long double eps_l = std::numeric_limits<long double>::epsilon();   // equal to double's on Apple ARM and MSVC

// W's relative condition number, 1 / |1 + W|: a rounded z moves W by this many units of z's rounding
template <typename T>
T w_condition(T w) { return 1 + 1 / std::abs(1 + w); }

// ---- Lambert W ------------------------------------------------------------------------------------

TEST(SpecialLambertW0, MatchesHighPrecisionReferences)
{
    struct { long double z; long double w; } cases[] = {
        {1.0L, 0.567143290409783872999968662210L}, {10.0L, 1.745528002740699383074301264875L},
        {-0.2L, -0.25917110181907374505665195021L}, {-0.36L, -0.80608431597081777828552136162L},
        {1e-5L, 0.000009999900001499973333854155L}, {100.0L, 3.385630140290050184888244364529L},
        {1e10L, 20.02868541330495078123430607181L}, {1e100L, 224.8431064451185015393731343379L},
    };
    for (const auto& c : cases) {
        const double z = static_cast<double>(c.z);
        EXPECT_LE(rel(m::lambert_w0(z), double(c.w)), 2 * eps * w_condition(double(c.w))) << "W0(" << z << ")";
        EXPECT_LE(rel_l(m::lambert_w0(c.z), c.w), 4 * eps_l * w_condition(c.w)) << "long double W0(" << z << ")";
        if (z < 1e38) {   // a float's W0 is double's, rounded once
            EXPECT_EQ(m::lambert_w0(static_cast<float>(z)), static_cast<float>(m::lambert_w0(double(float(z)))));
        }
    }
}

TEST(SpecialLambertW0, SatisfiesItsDefiningEquationFromTheBranchPointTo1e300)
{
    for (int i = 0; i <= 3000; ++i) {
        const double z = -inv_e + std::pow(10.0, -15.0 + 315.0 * i / 3000.0);
        const double w = m::lambert_w0(z);
        ASSERT_GE(w, -1.0) << z;
        if (z < 1e300) {
            // exp(w) itself carries |w| ulps of error, so the residual is measured in those
            ASSERT_LE(std::abs(w * std::exp(w) - z), 2 * eps * (std::abs(w) + 4) * std::abs(z) + 1e-16) << "W0(" << z << ") = " << w;
        }
    }
}

TEST(SpecialLambertW0, CompileTimeAndRunTimeAgree)
{
    constexpr double ct[] = {m::lambert_w0(-0.3), m::lambert_w0(0.5), m::lambert_w0(7.0), m::lambert_w0(1e6)};
    const double z[] = {-0.3, 0.5, 7.0, 1e6};
    for (int i = 0; i < 4; ++i)
        EXPECT_LE(rel(ct[i], m::lambert_w0(z[i])), 4e-16) << z[i];
}

TEST(SpecialLambertW0, KeepsItsDomainBehaviour)
{
    EXPECT_EQ(m::lambert_w0(-inv_e), -1.0);
    EXPECT_EQ(m::lambert_w0(-inv_e - 5e-13), -1.0);   // within the fuzz of a rounded -1/e
    EXPECT_TRUE(std::isnan(m::lambert_w0(-inv_e - 2e-12)));
    EXPECT_TRUE(std::isnan(m::lambert_w0(-1.0)));
    EXPECT_TRUE(std::isnan(m::lambert_w0(std::nan(""))));
    EXPECT_EQ(m::lambert_w0(dinf), dinf);
    EXPECT_EQ(m::lambert_w0(0), 0.0);
    EXPECT_EQ(m::lambert_w0(-0.36787945f), -1.0f);   // a float's nearest -1/e lies 9e-9 below it
    // Beyond 1e300 the asymptotic series, about 1e-9 relative: log W + W = log z
    const double w = m::lambert_w0(1e305);
    EXPECT_LE(std::abs((std::log(w) + w - std::log(1e305)) / std::log(1e305)), 1e-9);
    EXPECT_TRUE(std::isfinite(m::lambert_w0(std::numeric_limits<double>::max())));
    // Close to the branch point the Puiseux series alone: W = -1 + p - p^2/3 + ..., p = sqrt(2 e (z + 1/e)).
    // z + 1/e is 1e-14 to within the 1.3e-17 by which the double nearest 1/e exceeds it.
    const double p = std::sqrt(2 * m::e_v * 1e-14);
    EXPECT_NEAR(m::lambert_w0(-inv_e + 1e-14), -1.0 + p - p * p / 3, 1e-10);
    EXPECT_NEAR(m::lambert_wm1(-inv_e + 1e-14), -1.0 - p - p * p / 3, 1e-10);
}

TEST(SpecialLambertWm1, MatchesHighPrecisionReferences)
{
    struct { long double z; long double w; } cases[] = {
        {-0.3L, -1.78133702342162761197417028151L}, {-0.1L, -3.57715206395729721840939196351L}, {-0.01L, -6.47277512439400469474105789272L},
        {-1e-5L, -14.1636008158101830091095563036L}, {-1e-100L, -235.721158875685313660460606130L}, {-0.36L, -1.22277013397850595314293807342L},
        {-0.367L, -1.07079188676805187407925872521L},
    };
    for (const auto& c : cases) {
        const double z = static_cast<double>(c.z);
        EXPECT_LE(rel(m::lambert_wm1(z), double(c.w)), 2 * eps * w_condition(double(c.w))) << "W-1(" << z << ")";
        EXPECT_LE(rel_l(m::lambert_wm1(c.z), c.w), 4 * eps_l * w_condition(c.w)) << "long double W-1(" << z << ")";
    }
}

TEST(SpecialLambertWm1, SatisfiesItsDefiningEquationOnTheWholeBranch)
{
    for (int i = 0; i <= 3000; ++i) {
        const double z = -std::pow(10.0, -320.0 + 319.5657 * i / 3000.0);   // -1e-320 (subnormal) to -1/e
        if (z < -inv_e)
            continue;
        const double w = m::lambert_wm1(z);
        ASSERT_LE(w, -1.0) << z;
        // w e^w = z, in logarithms where e^w underflows
        ASSERT_LE(std::abs(w + std::log(-w) - std::log(-z)), 4e-16 * std::abs(std::log(-z)) + 1e-15) << "W-1(" << z << ") = " << w;
        if (z < -1e-300) {
            ASSERT_LE(std::abs(w * std::exp(w) - z), 2 * eps * (std::abs(w) + 4) * std::abs(z)) << "W-1(" << z << ") = " << w;
        }
    }
}

TEST(SpecialLambertWm1, KeepsItsDomainBehaviour)
{
    EXPECT_EQ(m::lambert_wm1(-inv_e), -1.0);
    EXPECT_TRUE(std::isnan(m::lambert_wm1(-0.5)));
    EXPECT_TRUE(std::isnan(m::lambert_wm1(1e-300)));
    EXPECT_TRUE(std::isnan(m::lambert_wm1(-dinf)));
    EXPECT_TRUE(std::isnan(m::lambert_wm1(std::nan(""))));
    EXPECT_EQ(m::lambert_wm1(0.0), -dinf);
    EXPECT_EQ(m::lambert_wm1(-0.0), -dinf);
    EXPECT_TRUE(std::isfinite(m::lambert_wm1(-std::numeric_limits<double>::denorm_min())));
    // Both branches meet at -1 and straddle it just to the right of the branch point
    EXPECT_LT(m::lambert_wm1(-0.3678), -1.0);
    EXPECT_GT(m::lambert_w0(-0.3678), -1.0);
    EXPECT_LE(std::abs(m::lambert_wm1(-0.25f) - static_cast<float>(m::lambert_wm1(-0.25))), 0.0f);
}

// ---- softplus and the logistic function ------------------------------------------------------------

TEST(SpecialSoftplus, IsStableInBothTails)
{
    EXPECT_LE(rel(m::softplus(1.0), 1.31326168751822280889), 2e-16);
    EXPECT_LE(rel(m::softplus(-1.0), 0.313261687518222808890), 2e-16);
    EXPECT_LE(rel(m::softplus(20.0), 20.0000000020611530260), 2e-16);
    EXPECT_LE(rel(m::softplus(-20.0), 2.06115362031438080437e-09), 2e-16);
    EXPECT_LE(rel(m::softplus(-40.0), 4.24835425529158887095e-18), 2e-16);
    EXPECT_LE(rel(m::softplus(-700.0), 9.85967654375977077183e-305), 2e-16);   // e^-700: no 1 + e^x rounding to 1
    EXPECT_EQ(m::softplus(40.0), 40.0);
    EXPECT_EQ(m::softplus(1e308), 1e308);
    EXPECT_EQ(m::softplus(dinf), dinf);
    EXPECT_EQ(m::softplus(-dinf), 0.0);
    EXPECT_EQ(m::softplus(-800.0), 0.0);
    EXPECT_TRUE(std::isnan(m::softplus(std::nan(""))));
    // softplus(x) - softplus(-x) = x
    for (const double x : {0.1, 1.0, 5.0, 30.0})
        EXPECT_NEAR(m::softplus(x) - m::softplus(-x), x, 4e-16 * x);
    EXPECT_EQ(m::softplus(100.0f), 100.0f);
    EXPECT_GT(m::softplus(-100.0f), 0.0f);
    EXPECT_LE(rel_l(m::softplus(-30.0L), 9.3576229688397367793776974246751577e-14L), 2 * eps_l);
}

TEST(SpecialLogistic, IsStableInBothTails)
{
    EXPECT_LE(rel(m::logistic(2.0), 0.880797077977882425515), 2e-16);
    EXPECT_LE(rel(m::logistic(-2.0), 0.119202922022117560608), 2e-16);
    EXPECT_LE(rel(m::logistic(-30.0), 9.35762296883929886547e-14), 2e-16);
    EXPECT_LE(rel(m::logistic(-700.0), 9.85967654375977077183e-305), 2e-16);
    EXPECT_LE(rel(m::logistic(30.0), 0.999999999999906408199), 2e-16);
    EXPECT_EQ(m::logistic(40.0), 1.0);
    EXPECT_EQ(m::logistic(dinf), 1.0);
    EXPECT_EQ(m::logistic(-dinf), 0.0);
    EXPECT_TRUE(std::isnan(m::logistic(std::nan(""))));
    for (const double x : {0.3, 3.0, 17.0})
        EXPECT_NEAR(m::logistic(x) + m::logistic(-x), 1.0, 2 * eps);
    EXPECT_EQ(m::logistic(-1000.0f), 0.0f);
    EXPECT_EQ(m::logistic(0), 0.5);
}

TEST(SpecialLogLogistic, IsFiniteWhereTheLogisticUnderflows)
{
    EXPECT_LE(rel(m::log_logistic(-2.0), -2.12692801104297268822), 2e-16);
    EXPECT_LE(rel(m::log_logistic(2.0), -0.126928011042972493927), 2e-16);
    EXPECT_LE(rel(m::log_logistic(50.0), -1.92874984796391782056e-22), 2e-16);
    EXPECT_EQ(m::log_logistic(-50.0), -50.0);
    EXPECT_EQ(m::log_logistic(-1e300), -1e300);
    EXPECT_EQ(m::log_logistic(dinf), -0.0);
    EXPECT_EQ(m::log_logistic(-dinf), -dinf);
    EXPECT_TRUE(std::isnan(m::log_logistic(std::nan(""))));
}

// ---- log Phi ------------------------------------------------------------------------------------------

struct log_phi_case { long double x; long double want; };
// References: Python decimal (see the file comment). SciPy's log_ndtr gives the same values.
constexpr log_phi_case log_phi_cases[] = {
    {3, -0.001350809964748193798841110L},  {1, -0.172753779023449889526483173L},  {0, -0.693147180559945309417232121L},
    {-0.5, -1.175911761593618608879729093L}, {-1, -1.841021645009263505770783073L}, {-3, -6.607726221510349543276077083L},
    {-4, -10.36010148652729082786072034L}, {-5, -15.06499839398872573608370479L}, {-6, -20.73676894997470565496885371L},
    {-7, -27.38430749881107524263012388L}, {-10, -53.23128515051247057834702735L}, {-19.5, -194.0169657774974994076029694L},
    {-19.99L, -203.716707717208435366662161877L}, {-20, -203.917155371097263936804458654L}, {-20.01L, -204.117702778659901248766038550L},
    {-20.5, -214.0667289632638001665303535L}, {-25, -316.6394080080202589351957635L}, {-30, -454.3212439563431971073557713L},
    {-38, -726.5572160188201300965035175L}, {-40, -804.6084420137537881666068329L}, {-100, -5005.524208694205088626302457L},
    {-1000, -500007.8266948121843098061675L},
};

TEST(SpecialLogNormcdf, MatchesHighPrecisionReferencesInEveryType)
{
    for (const auto& c : log_phi_cases) {
        const double tol = c.x > 0 ? 2e-15 : 4e-16;   // for x > 0 the relative error of Phi(-x), conditioned by the argument
        const double x = static_cast<double>(c.x);
        EXPECT_LE(rel(m::log_normcdf(x), double(c.want)), tol * (x == c.x ? 1 : 4)) << "log Phi(" << x << ")";   // a rounded x moves log Phi by ~2 ulps
        EXPECT_LE(rel_l(m::log_normcdf(static_cast<long double>(c.x)), c.want), (c.x > 0 ? 20 : 6) * eps_l)
            << "long double log Phi(" << x << ")";
        EXPECT_LE(std::abs(m::log_normcdf(static_cast<float>(c.x)) - float(c.want)), 4e-7f * std::abs(float(c.want)))
            << "float log Phi(" << x << ")";
    }
}

TEST(SpecialLogNormcdf, IsFiniteWherePhiUnderflows)
{
    EXPECT_EQ(std::log(m::normal_cdf(-40.0)), -dinf);   // the naive form
    EXPECT_TRUE(std::isfinite(m::log_normcdf(-40.0)));
    EXPECT_TRUE(std::isfinite(m::log_normcdf(-1e150)));
    EXPECT_EQ(m::log_normcdf(-1e200), -dinf);            // -x^2/2 itself overflows
    EXPECT_EQ(m::log_normcdf(-dinf), -dinf);
    EXPECT_EQ(m::log_normcdf(dinf), 0.0);
    EXPECT_EQ(m::log_normcdf(40.0), 0.0);
    EXPECT_TRUE(std::isnan(m::log_normcdf(std::nan(""))));
    EXPECT_TRUE(std::isfinite(m::log_normcdf(-1e15f)));   // computed in double: -5e29 fits a float
}

TEST(SpecialLogNormcdf, IsIncreasingAcrossItsThreeRegions)
{
    // The regions meet at -20 (erfc and the asymptotic series, both checked against references above) and at 0
    double before = m::log_normcdf(-60.0);
    for (int i = 1; i <= 2000; ++i) {
        const double v = m::log_normcdf(-60.0 + 66.0 * i / 2000.0);
        ASSERT_GT(v, before) << i;
        before = v;
    }
}

TEST(SpecialLogNormcdf, CompileTimeAndRunTimeAgree)
{
    constexpr double ct[] = {m::log_normcdf(2.0), m::log_normcdf(-2.0), m::log_normcdf(-15.0), m::log_normcdf(-50.0)};
    const double x[] = {2.0, -2.0, -15.0, -50.0};
    for (int i = 0; i < 4; ++i)
        EXPECT_LE(rel(ct[i], m::log_normcdf(x[i])), 1e-14) << x[i];   // compile-time erfc is 1 - erf below 2, losing digits in Phi(-2)
}

// ---- the entropy atoms -----------------------------------------------------------------------------

TEST(SpecialEntr, FollowsCvxpysConventions)
{
    EXPECT_LE(rel(m::entr(0.5), 0.34657359027997265470861606072908828), 2e-16);
    EXPECT_EQ(m::entr(0.0), 0.0);
    EXPECT_EQ(m::entr(-1e-300), -dinf);
    EXPECT_EQ(m::entr(dinf), -dinf);
    EXPECT_TRUE(std::isnan(m::entr(std::nan(""))));
    EXPECT_GT(m::entr(std::numeric_limits<double>::denorm_min()), 0.0);
    EXPECT_EQ(m::entr(1), 0.0);
}

TEST(SpecialRelEntr, FollowsCvxpysConventions)
{
    EXPECT_LE(rel(m::rel_entr(2.0, 3.0), -0.81093021621632876395602623092869827), 2e-16);
    EXPECT_EQ(m::rel_entr(0.0, 5.0), 0.0);
    EXPECT_EQ(m::rel_entr(0.0, 0.0), 0.0);
    EXPECT_EQ(m::rel_entr(0.0, -1.0), dinf);
    EXPECT_EQ(m::rel_entr(1.0, 0.0), dinf);
    EXPECT_EQ(m::rel_entr(-1.0, 1.0), dinf);
    EXPECT_EQ(m::rel_entr(dinf, 1.0), dinf);
    EXPECT_EQ(m::rel_entr(1.0, dinf), -dinf);
    EXPECT_TRUE(std::isnan(m::rel_entr(std::nan(""), 1.0)));
    EXPECT_TRUE(std::isnan(m::rel_entr(0.0, std::nan(""))));
    // x / y overflows; the value does not: 1e300 * log(1e600)
    EXPECT_LE(rel(m::rel_entr(1e300, 1e-300), 1.3815510557964274104107948728106185e+303), 4e-16);
    EXPECT_TRUE(std::isfinite(m::rel_entr(1e-300, 1e300)));
    // Gibbs: the sum over a distribution is at least 0, 0 only at equality
    const double p[] = {0.2, 0.3, 0.5}, q[] = {0.4, 0.4, 0.2};
    double s = 0, s0 = 0;
    for (int i = 0; i < 3; ++i) { s += m::rel_entr(p[i], q[i]); s0 += m::rel_entr(p[i], p[i]); }
    EXPECT_GT(s, 0.0);
    EXPECT_EQ(s0, 0.0);
}

TEST(SpecialKlDiv, FollowsCvxpysConventions)
{
    EXPECT_LE(rel(m::kl_div(2.0, 3.0), 0.18906978378367123604397376907130173), 4e-16);
    EXPECT_EQ(m::kl_div(0.0, 3.0), 3.0);
    EXPECT_EQ(m::kl_div(0.0, 0.0), 0.0);
    EXPECT_EQ(m::kl_div(1.0, 0.0), dinf);
    EXPECT_EQ(m::kl_div(-1.0, 1.0), dinf);
    EXPECT_EQ(m::kl_div(dinf, 1.0), dinf);
    EXPECT_EQ(m::kl_div(1.0, dinf), dinf);
    EXPECT_TRUE(std::isnan(m::kl_div(dinf, dinf)));
    EXPECT_TRUE(std::isnan(m::kl_div(1.0, std::nan(""))));
    for (const double x : {1e-10, 0.3, 1.0, 7.0, 1e10})
        for (const double y : {1e-10, 0.3, 1.0, 7.0, 1e10})
            EXPECT_GE(m::kl_div(x, y), -1e-15 * (x + y)) << x << ", " << y;   // nonnegative
}

TEST(SpecialHuber, IsCvxpysDefinition)
{
    EXPECT_EQ(m::huber(0.0), 0.0);
    EXPECT_EQ(m::huber(1.0), 1.0);
    EXPECT_EQ(m::huber(-1.0), 1.0);
    EXPECT_EQ(m::huber(2.0), 3.0);
    EXPECT_EQ(m::huber(0.5, 2.0), 0.25);
    EXPECT_EQ(m::huber(5.0, 2.0), 16.0);
    EXPECT_EQ(m::huber(3.0, 0.0), 0.0);
    EXPECT_EQ(m::huber(dinf), dinf);
    EXPECT_EQ(m::huber(1e300), 2e300 - 1.0);   // no overflow from x^2
    EXPECT_EQ(m::huber(3.0, dinf), 9.0);
    EXPECT_TRUE(std::isnan(m::huber(1.0, -1.0)));
    EXPECT_TRUE(std::isnan(m::huber(std::nan(""))));
    // Continuous with a continuous slope at |x| = M
    const double M = 1.5, h = 1e-7;
    EXPECT_NEAR(m::huber(M - h, M), m::huber(M + h, M), 4 * M * h + 1e-15);
    EXPECT_NEAR((m::huber(M + h, M) - m::huber(M, M)) / h, 2 * M, 1e-6);
    EXPECT_EQ(m::huber(3, 1), 5.0);
    EXPECT_EQ(m::huber(0.5f), 0.25f);
}

TEST(SpecialXlogy, IsZeroWhenXIsZero)
{
    EXPECT_EQ(m::xlogy(0.0, 0.0), 0.0);
    EXPECT_EQ(m::xlogy(0.0, -3.0), 0.0);
    EXPECT_EQ(m::xlogy(0.0, dinf), 0.0);
    EXPECT_TRUE(std::isnan(m::xlogy(0.0, std::nan(""))));
    EXPECT_TRUE(std::isnan(m::xlogy(1.0, -1.0)));
    EXPECT_EQ(m::xlogy(2.0, 0.0), -dinf);
    EXPECT_EQ(m::xlogy(3.0, m::e_v), 3.0);
    EXPECT_DOUBLE_EQ(m::xlogy(2.0, 8.0), 2.0 * std::log(8.0));
    EXPECT_EQ(m::xlogy(1.0f, 1.0f), 0.0f);
}

} // namespace
