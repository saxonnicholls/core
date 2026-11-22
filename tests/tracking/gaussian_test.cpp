//
//  gaussian_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/tracking/gaussian.hpp>: the layout of gaussian<T, N>, the Mahalanobis distance by
//  hand-inverted covariances, the log-likelihood and the one-dimensional closed forms of the Hellinger distance and
//  the KL divergence, the chi-square gate against tabulated quantiles, the confidence ellipse's axes and angle in
//  every branch, each documented not_positive_definite error, and the measures in a constant expression.
//

#include <snicholls/core/tracking/gaussian.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <type_traits>
#include <utility>

namespace {

namespace tr = snicholls::tracking;
namespace la = snicholls::linalg;

using v2 = la::fixed_vector<double, 2>;
using m2 = la::fixed_matrix<double, 2, 2>;
using g1 = tr::gaussian<double, 1>;
using g2 = tr::gaussian<double, 2>;

// The layout contract: N + N^2 doubles, trivially copyable, an aggregate
static_assert(sizeof(tr::gaussian<double, 4>) == 20 * sizeof(double));
static_assert(alignof(tr::gaussian<double, 4>) == alignof(double));
static_assert(std::is_trivially_copyable_v<tr::gaussian<double, 4>>);
static_assert(std::is_aggregate_v<tr::gaussian<double, 4>>);
static_assert(std::is_trivially_copyable_v<tr::ellipse<double>>);
static_assert(tr::gaussian<double, 4>::dim == 4 && std::is_same_v<tr::gaussian<float, 3>::value_type, float>);

// noexcept for the standard floating-point types
static_assert(noexcept(tr::squared_mahalanobis(std::declval<const v2&>(), std::declval<const m2&>())));
static_assert(noexcept(tr::mahalanobis(std::declval<const v2&>(), std::declval<const m2&>())));
static_assert(noexcept(tr::whitened_squared_norm(std::declval<const v2&>(), std::declval<const m2&>())));
static_assert(noexcept(tr::log_likelihood(std::declval<const v2&>(), std::declval<const m2&>())));
static_assert(noexcept(tr::euclidean(std::declval<const v2&>(), std::declval<const v2&>())));
static_assert(noexcept(tr::euclidean_weighted(std::declval<const v2&>(), std::declval<const v2&>(), std::declval<const v2&>())));
static_assert(noexcept(tr::squared_hellinger(std::declval<const g2&>(), std::declval<const g2&>())));
static_assert(noexcept(tr::hellinger(std::declval<const g2&>(), std::declval<const g2&>())));
static_assert(noexcept(tr::kl_divergence(std::declval<const g2&>(), std::declval<const g2&>())));
static_assert(noexcept(tr::gate_threshold(0.99, 2)));
static_assert(noexcept(tr::in_gate(1.0, 2.0)));
static_assert(noexcept(tr::confidence_ellipse(std::declval<const m2&>(), 1.0)));

// r = (1, 2), S = [[4, 1], [1, 3]]: S^-1 = [[3, -1], [-1, 4]] / 11, so r^T S^-1 r = (3 - 4 + 16) / 11 = 15/11, det S = 11
constexpr v2 r{{1, 2}};
constexpr m2 s{{4, 1, 1, 3}};
constexpr m2 indefinite{{1, 2, 2, 1}};   // eigenvalues 3 and -1

// The measures in a constant expression
static_assert(*tr::squared_mahalanobis(r, s) > 15.0 / 11 - 1e-12 && *tr::squared_mahalanobis(r, s) < 15.0 / 11 + 1e-12);
static_assert(tr::squared_mahalanobis(r, indefinite).is_err());
static_assert(tr::euclidean(v2{{0, 0}}, v2{{3, 4}}) == 5);
static_assert(tr::in_gate(9.0, 9.0) && !tr::in_gate(9.0 + 1e-12, 9.0));
constexpr bool constexpr_measures()
{
    const g1 a{{{0}}, {{1}}}, b{{{1}}, {{4}}};
    const double kl = *tr::kl_divergence(a, b);                     // log 2 + (1 + 1) / 8 - 1/2
    const double h2 = *tr::squared_hellinger(a, b);                 // 1 - sqrt(4/5) e^(-1/20)
    const double ll = *tr::log_likelihood(v2{{0, 0}}, m2::identity()); // -log(2 pi)
    return kl > 0.4431 && kl < 0.4432 && h2 > 0.14919 && h2 < 0.14920 && ll > -1.8379 && ll < -1.8378;
}
static_assert(constexpr_measures());

TEST(TrackingGaussianMahalanobis, SquaredDistanceIsTheQuadraticFormOfTheInverseCovariance)
{
    EXPECT_NEAR(*tr::squared_mahalanobis(r, s), 15.0 / 11.0, 1e-14);
    EXPECT_NEAR(*tr::mahalanobis(r, s), std::sqrt(15.0 / 11.0), 1e-14);
    // A second residual: (3, -1): (27 + 6 + 4) / 11 = 37 / 11
    EXPECT_NEAR(*tr::squared_mahalanobis(v2{{3, -1}}, s), 37.0 / 11.0, 1e-14);
    EXPECT_NEAR(*tr::mahalanobis(v2{{3, -1}}, s), std::sqrt(37.0 / 11.0), 1e-14);
    // Zero residual, zero distance; identity covariance, the Euclidean norm
    EXPECT_EQ(*tr::squared_mahalanobis(v2{{0, 0}}, s), 0.0);
    EXPECT_NEAR(*tr::mahalanobis(v2{{3, 4}}, m2::identity()), 5.0, 1e-14);
}

TEST(TrackingGaussianMahalanobis, WhitenedNormWithTheCholeskyFactorIsTheSameDistance)
{
    const auto l = la::cholesky(s);
    ASSERT_TRUE(l.is_ok());
    EXPECT_NEAR(tr::whitened_squared_norm(r, *l), 15.0 / 11.0, 1e-14);
    EXPECT_NEAR(tr::whitened_squared_norm(v2{{3, -1}}, *l), 37.0 / 11.0, 1e-14);
}

TEST(TrackingGaussianMahalanobis, NotPositiveDefiniteCovarianceIsAnError)
{
    EXPECT_EQ(tr::squared_mahalanobis(r, indefinite).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::mahalanobis(r, indefinite).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::mahalanobis(r, m2{}).unwrap_err(), la::error::not_positive_definite);
}

TEST(TrackingGaussianLogLikelihood, IsTheLogOfTheNormalDensity)
{
    const double two_pi = 2 * std::numbers::pi;
    EXPECT_NEAR(*tr::log_likelihood(r, s), -(15.0 / 11.0 + std::log(11.0) + 2 * std::log(two_pi)) / 2, 1e-13);
    // One dimension: log N(2; 0, 4) = -(1 + log 4 + log 2 pi) / 2
    EXPECT_NEAR(*tr::log_likelihood(la::fixed_vector<double, 1>{{2}}, la::fixed_matrix<double, 1, 1>{{4}}), -(1.0 + std::log(4.0) + std::log(two_pi)) / 2, 1e-14);
    EXPECT_EQ(tr::log_likelihood(r, indefinite).unwrap_err(), la::error::not_positive_definite);
}

TEST(TrackingGaussianEuclidean, PlainAndWeightedDistances)
{
    EXPECT_DOUBLE_EQ(tr::euclidean(v2{{1, 1}}, v2{{4, 5}}), 5.0);
    EXPECT_EQ(tr::euclidean(v2{{1, 1}}, v2{{1, 1}}), 0.0);
    // sqrt(4 * 9 + 0.25 * 16) = sqrt(40)
    EXPECT_DOUBLE_EQ(tr::euclidean_weighted(v2{{1, 1}}, v2{{4, 5}}, v2{{4, 0.25}}), std::sqrt(40.0));
    // Unit weights are the plain distance, zero weights zero
    EXPECT_DOUBLE_EQ(tr::euclidean_weighted(v2{{1, 1}}, v2{{4, 5}}, v2{{1, 1}}), 5.0);
    EXPECT_EQ(tr::euclidean_weighted(v2{{1, 1}}, v2{{4, 5}}, v2{{0, 0}}), 0.0);
}

// One dimension: H^2 = 1 - sqrt(2 s1 s2 / (s1^2 + s2^2)) exp(-d^2 / (4 (s1^2 + s2^2)))
double hellinger_1d(double m1, double v1, double m2_, double v2_)
{
    const double s1 = std::sqrt(v1), s2 = std::sqrt(v2_);
    return 1 - std::sqrt(2 * s1 * s2 / (v1 + v2_)) * std::exp(-(m1 - m2_) * (m1 - m2_) / (4 * (v1 + v2_)));
}

TEST(TrackingGaussianHellinger, MatchesTheOneDimensionalClosedForm)
{
    const g1 a{{{0.5}}, {{1.5}}}, b{{{-1.0}}, {{0.4}}};
    EXPECT_NEAR(*tr::squared_hellinger(a, b), hellinger_1d(0.5, 1.5, -1.0, 0.4), 1e-14);
    EXPECT_NEAR(*tr::hellinger(a, b), std::sqrt(hellinger_1d(0.5, 1.5, -1.0, 0.4)), 1e-14);
    // Symmetric
    EXPECT_NEAR(*tr::squared_hellinger(a, b), *tr::squared_hellinger(b, a), 1e-15);
}

TEST(TrackingGaussianHellinger, IndependentAxesMultiplyTheirCoefficients)
{
    // With diagonal covariances the Bhattacharyya coefficient 1 - H^2 is the product of the axes'
    const g2 a{{{0.5, 2}}, {{1.5, 0, 0, 2}}}, b{{{-1, 1}}, {{0.4, 0, 0, 3}}};
    const double bc = (1 - hellinger_1d(0.5, 1.5, -1, 0.4)) * (1 - hellinger_1d(2, 2, 1, 3));
    EXPECT_NEAR(*tr::squared_hellinger(a, b), 1 - bc, 1e-14);
}

TEST(TrackingGaussianHellinger, IsZeroForEqualGaussiansAndBelowOneForDistantOnes)
{
    const g2 a{{{1, 2}}, s};
    EXPECT_GE(*tr::squared_hellinger(a, a), 0.0);   // clamped: rounding never makes it negative
    EXPECT_NEAR(*tr::squared_hellinger(a, a), 0.0, 1e-15);
    EXPECT_NEAR(*tr::hellinger(a, a), 0.0, 1e-7);
    const g2 far{{{1e3, 2e3}}, s};
    EXPECT_LE(*tr::squared_hellinger(a, far), 1.0);
    EXPECT_NEAR(*tr::squared_hellinger(a, far), 1.0, 1e-15);
}

TEST(TrackingGaussianHellinger, NotPositiveDefiniteCovarianceIsAnError)
{
    const g2 good{{{0, 0}}, s}, bad{{{0, 0}}, indefinite};
    EXPECT_EQ(tr::squared_hellinger(good, bad).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::squared_hellinger(bad, good).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::hellinger(good, bad).unwrap_err(), la::error::not_positive_definite);
    // Zero covariances on both sides
    EXPECT_EQ(tr::squared_hellinger(g2{}, g2{}).unwrap_err(), la::error::not_positive_definite);
}

TEST(TrackingGaussianKlDivergence, MatchesTheOneDimensionalClosedForm)
{
    // KL(p || q) = log(sq / sp) + (vp + d^2) / (2 vq) - 1/2
    const g1 p{{{0.5}}, {{1.5}}}, q{{{-1.0}}, {{0.4}}};
    const double expected = std::log(std::sqrt(0.4 / 1.5)) + (1.5 + 2.25) / (2 * 0.4) - 0.5;
    EXPECT_NEAR(*tr::kl_divergence(p, q), expected, 1e-14);
    // Not symmetric
    const double reverse = std::log(std::sqrt(1.5 / 0.4)) + (0.4 + 2.25) / (2 * 1.5) - 0.5;
    EXPECT_NEAR(*tr::kl_divergence(q, p), reverse, 1e-14);
}

TEST(TrackingGaussianKlDivergence, MatchesTheMatrixFormulaInTwoDimensions)
{
    // p = N((1, 2), S), q = N((0, 0), I): (tr S + |mu|^2 - 2 - log det S) / 2 = (7 + 5 - 2 - log 11) / 2
    const g2 p{{{1, 2}}, s}, q{{{0, 0}}, m2::identity()};
    EXPECT_NEAR(*tr::kl_divergence(p, q), (10.0 - std::log(11.0)) / 2, 1e-14);
    EXPECT_NEAR(*tr::kl_divergence(p, p), 0.0, 1e-14);
}

TEST(TrackingGaussianKlDivergence, NotPositiveDefiniteCovarianceIsAnError)
{
    const g2 good{{{0, 0}}, s}, bad{{{0, 0}}, indefinite};
    EXPECT_EQ(tr::kl_divergence(good, bad).unwrap_err(), la::error::not_positive_definite);
    EXPECT_EQ(tr::kl_divergence(bad, good).unwrap_err(), la::error::not_positive_definite);
}

TEST(TrackingGaussianGate, ThresholdIsTheChiSquareQuantile)
{
    // Two degrees of freedom: the quantile is -2 log(1 - p) exactly
    EXPECT_NEAR(tr::gate_threshold(0.99, 2), -2 * std::log(0.01), 1e-9);
    // Tabulated: chi2(0.95, 1) = 1.959963984540054^2, chi2(0.95, 3) = 7.814727903251178
    EXPECT_NEAR(tr::gate_threshold(0.95, 1), 1.959963984540054 * 1.959963984540054, 1e-9);
    EXPECT_NEAR(tr::gate_threshold(0.95, 3), 7.814727903251178, 1e-9);
    EXPECT_LT(tr::gate_threshold(0.9, 2), tr::gate_threshold(0.99, 2));
}

TEST(TrackingGaussianGate, InGateIncludesTheBoundary)
{
    const double g = tr::gate_threshold(0.99, 2);
    EXPECT_TRUE(tr::in_gate(0.0, g));
    EXPECT_TRUE(tr::in_gate(g, g));
    EXPECT_FALSE(tr::in_gate(g * (1 + 1e-12), g));
    EXPECT_TRUE(tr::in_gate(*tr::squared_mahalanobis(r, s), g));
}

TEST(TrackingGaussianEllipse, AxisAlignedCovariances)
{
    const auto e = tr::confidence_ellipse(m2{{4, 0, 0, 1}});
    EXPECT_DOUBLE_EQ(e.semi_major, 2.0);
    EXPECT_DOUBLE_EQ(e.semi_minor, 1.0);
    EXPECT_DOUBLE_EQ(e.angle, 0.0);
    // The longer axis along y: angle pi/2, the top of (-pi/2, pi/2]
    const auto f = tr::confidence_ellipse(m2{{1, 0, 0, 4}});
    EXPECT_DOUBLE_EQ(f.semi_major, 2.0);
    EXPECT_DOUBLE_EQ(f.semi_minor, 1.0);
    EXPECT_DOUBLE_EQ(f.angle, std::numbers::pi / 2);
}

TEST(TrackingGaussianEllipse, CorrelatedCovarianceIsRotated)
{
    // [[2, 1], [1, 2]]: eigenvalues 3 and 1, the major axis along (1, 1); with the sign of the correlation flipped,
    // along (1, -1)
    const auto e = tr::confidence_ellipse(m2{{2, 1, 1, 2}}, 2.0);
    EXPECT_NEAR(e.semi_major, 2 * std::sqrt(3.0), 1e-14);
    EXPECT_NEAR(e.semi_minor, 2.0, 1e-14);
    EXPECT_NEAR(e.angle, std::numbers::pi / 4, 1e-14);
    const auto f = tr::confidence_ellipse(m2{{2, -1, -1, 2}});
    EXPECT_NEAR(f.angle, -std::numbers::pi / 4, 1e-14);
}

TEST(TrackingGaussianEllipse, CircleHasAngleZeroAndIndefiniteAxesClampToZero)
{
    const auto c = tr::confidence_ellipse(m2{{9, 0, 0, 9}});
    EXPECT_DOUBLE_EQ(c.semi_major, 3.0);
    EXPECT_DOUBLE_EQ(c.semi_minor, 3.0);
    EXPECT_EQ(c.angle, 0.0);
    // Eigenvalues 3 and -1: the negative one counts as zero
    const auto i = tr::confidence_ellipse(indefinite);
    EXPECT_NEAR(i.semi_major, std::sqrt(3.0), 1e-14);
    EXPECT_EQ(i.semi_minor, 0.0);
    // The zero matrix: a point
    const auto z = tr::confidence_ellipse(m2{});
    EXPECT_EQ(z.semi_major, 0.0);
    EXPECT_EQ(z.semi_minor, 0.0);
}

TEST(TrackingGaussianEllipse, NinetyFivePercentScale)
{
    const double k = std::sqrt(tr::gate_threshold(0.95, 2));   // sqrt(-2 log 0.05) = 2.4477...
    const auto e = tr::confidence_ellipse(m2{{4, 0, 0, 1}}, k);
    EXPECT_NEAR(e.semi_major, 2 * std::sqrt(-2 * std::log(0.05)), 1e-8);
}

TEST(TrackingGaussianType, EqualityAndFloat)
{
    const g2 a{{{1, 2}}, s};
    g2 b = a;
    EXPECT_EQ(a, b);
    b.covariance(0, 1) = 0;
    EXPECT_NE(a, b);
    // float instantiates the same functions
    const la::fixed_vector<float, 2> rf{{1, 2}};
    const la::fixed_matrix<float, 2, 2> sf{{4, 1, 1, 3}};
    EXPECT_NEAR(*tr::squared_mahalanobis(rf, sf), 15.0f / 11.0f, 1e-6f);
}

} // namespace
