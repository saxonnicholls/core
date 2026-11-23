//
//  statistical_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/metric/statistical.hpp>: mahalanobis by hand for a diagonal covariance and against the
//  explicit (a - b)ᵀ S⁻¹ (a - b) for a full one, the not_positive_definite error, the reduced form with expand and
//  contract, the triangle inequality, a brute-force nearest search, mixed point types; hellinger, squared_hellinger
//  and kl_divergence against the tracking:: functions and the one-dimensional closed forms, NaN for an indefinite
//  covariance, the asymmetry of kl_divergence; the concepts, layout, noexcept claims and constant evaluation.
//

#include <snicholls/core/metric/statistical.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace me = snicholls::metric;
namespace st = snicholls::stats;
namespace tr = snicholls::tracking;

using v2 = la::fixed_vector<double, 2>;
using v3 = la::fixed_vector<double, 3>;
using m2 = la::fixed_matrix<double, 2, 2>;
using m3 = la::fixed_matrix<double, 3, 3>;
using g1 = tr::gaussian<double, 1>;
using g2 = tr::gaussian<double, 2>;

static_assert(me::true_metric<me::mahalanobis<double, 2>> && me::reducible<me::mahalanobis<double, 2>, v2>);
static_assert(me::distance_for<me::mahalanobis<double, 2>, std::array<double, 2>, v2>);
static_assert(std::is_trivially_copyable_v<me::mahalanobis<double, 3>> && sizeof(me::mahalanobis<double, 3>) == sizeof(m3));
static_assert(me::true_metric<me::hellinger> && !me::true_metric<me::squared_hellinger> && !me::true_metric<me::kl_divergence>);
static_assert(me::distance_for<me::hellinger, g2> && me::distance_for<me::squared_hellinger, g2> && me::distance_for<me::kl_divergence, g2>);
static_assert(noexcept(me::make_mahalanobis(m2{})));
static_assert(noexcept(std::declval<const me::mahalanobis<double, 2>&>()(v2{}, v2{})));
static_assert(noexcept(std::declval<const me::mahalanobis<double, 2>&>().reduced(v2{}, v2{})));
static_assert(noexcept(me::hellinger{}(g2{}, g2{})) && noexcept(me::squared_hellinger{}(g2{}, g2{})) && noexcept(me::kl_divergence{}(g2{}, g2{})));

constexpr bool compile_time()
{
    const auto m = me::make_mahalanobis(m2{{4, 0, 0, 9}});
    if (!m)
        return false;
    const double d = (*m)(v2{{2, 3}}, v2{{0, 0}});   // √(4/4 + 9/9) = √2
    const auto bad = me::make_mahalanobis(m2{{1, 2, 2, 1}});
    const g1 a{{{0}}, {{1}}};
    const double h = me::hellinger{}(a, a), k = me::kl_divergence{}(a, a);
    return d > 1.41421356 && d < 1.41421357 && m->reduced(v2{{2, 3}}, v2{{0, 0}}) > 1.9999999 && bad.is_err() && h == 0 && k > -1e-15 && k < 1e-15;
}
static_assert(compile_time());

// A random symmetric positive-definite matrix: A Aᵀ + I / 2
template <std::size_t N>
la::fixed_matrix<double, N, N> random_spd(st::wyrand& g)
{
    la::fixed_matrix<double, N, N> a{}, s{};
    for (auto& x : a.elements)
        x = 2 * st::uniform_real(g) - 1;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t k = 0; k < N; ++k)
                s(i, j) += a(i, k) * a(j, k);
            s(i, j) += i == j ? 0.5 : 0.0;
        }
    return s;
}

template <std::size_t N>
la::fixed_vector<double, N> random_vector(st::wyrand& g, double span)
{
    la::fixed_vector<double, N> v{};
    for (auto& x : v.elements)
        x = span * (2 * st::uniform_real(g) - 1);
    return v;
}

TEST(MetricMahalanobis, DiagonalCovarianceByHand)
{
    const auto m = me::make_mahalanobis(m2{{4, 0, 0, 25}});
    ASSERT_TRUE(m);
    // √((3 - 1)² / 4 + (10 - 0)² / 25) = √(1 + 4)
    EXPECT_NEAR((*m)(v2{{3, 10}}, v2{{1, 0}}), std::sqrt(5.0), 1e-14);
    EXPECT_NEAR(m->reduced(v2{{3, 10}}, v2{{1, 0}}), 5.0, 1e-14);
    EXPECT_EQ((*m)(v2{{3, 10}}, v2{{3, 10}}), 0.0);
    // The identity covariance is the Euclidean distance
    const auto id = me::make_mahalanobis(m3{{1, 0, 0, 0, 1, 0, 0, 0, 1}});
    ASSERT_TRUE(id);
    EXPECT_NEAR((*id)(v3{{1, 2, 3}}, v3{{4, 6, 3}}), 5.0, 1e-14);
}

TEST(MetricMahalanobis, FullCovarianceMatchesTheExplicitQuadraticForm)
{
    st::wyrand g(3);
    for (int trial = 0; trial < 200; ++trial) {
        const m3 s = random_spd<3>(g);
        const auto m = me::make_mahalanobis(s);
        ASSERT_TRUE(m) << trial;
        const auto inv = la::inverse(s);   // by LU, independently of the Cholesky factor
        ASSERT_TRUE(inv) << trial;
        const v3 a = random_vector<3>(g, 5), b = random_vector<3>(g, 5);
        double q = 0;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                q += (a[i] - b[i]) * (*inv)(i, j) * (a[j] - b[j]);
        EXPECT_NEAR((*m)(a, b), std::sqrt(q), 1e-9 * (1 + std::sqrt(q))) << trial;
        EXPECT_NEAR(m->reduced(a, b), q, 1e-9 * (1 + q)) << trial;
        EXPECT_NEAR((*m)(a, b), (*m)(b, a), 1e-12 * (1 + q)) << trial;
    }
}

TEST(MetricMahalanobis, RejectsACovarianceThatIsNotPositiveDefinite)
{
    for (const m2& s : {m2{{1, 2, 2, 1}}, m2{{0, 0, 0, 0}}, m2{{-1, 0, 0, 1}}, m2{{1, 1, 1, 1}}, m2{{std::numeric_limits<double>::quiet_NaN(), 0, 0, 1}}}) {
        const auto m = me::make_mahalanobis(s);
        ASSERT_FALSE(m);
        EXPECT_EQ(m.unwrap_err_unchecked(), la::error::not_positive_definite);
    }
    EXPECT_TRUE(me::make_mahalanobis(m2{{1, 0.99, 0.99, 1}}));
}

TEST(MetricMahalanobis, ReducedExpandAndContractAgree)
{
    st::wyrand g(5);
    const auto m = me::make_mahalanobis(random_spd<2>(g)).unwrap();
    for (int i = 0; i < 100; ++i) {
        const v2 a = random_vector<2>(g, 3), b = random_vector<2>(g, 3);
        const double d = m(a, b), r = m.reduced(a, b);
        EXPECT_NEAR(m.expand(r), d, 1e-14 * (1 + d));
        EXPECT_NEAR(m.contract(d), r, 1e-12 * (1 + r));
        EXPECT_NEAR(me::reduced_distance(m, a, b), r, 0.0);
        EXPECT_NEAR(me::expand(m, r), d, 1e-14 * (1 + d));
        EXPECT_NEAR(me::contract(m, d), r, 1e-12 * (1 + r));
    }
    EXPECT_EQ(m.expand(16.0f), 4.0f);
    EXPECT_EQ(m.contract(3.0), 9.0);
}

TEST(MetricMahalanobis, ObeysTheTriangleInequality)
{
    st::wyrand g(7);
    for (int trial = 0; trial < 100; ++trial) {
        const auto m = me::make_mahalanobis(random_spd<3>(g)).unwrap();
        for (int i = 0; i < 20; ++i) {
            const v3 a = random_vector<3>(g, 4), b = random_vector<3>(g, 4), c = random_vector<3>(g, 4);
            EXPECT_LE(m(a, b), m(a, c) + m(c, b) + 1e-12) << trial << ' ' << i;
            EXPECT_GE(m(a, b), 0.0);
        }
    }
}

TEST(MetricMahalanobis, FindsTheNearestPointByBruteForce)
{
    // A track elongated along x: (5, 0) is nearer than (0, 1) under S = diag(100, 1), farther in Euclidean terms
    const auto m = me::make_mahalanobis(m2{{100, 0, 0, 1}}).unwrap();
    const std::vector<v2> candidates{v2{{0, 1}}, v2{{5, 0}}, v2{{-3, 3}}};
    const auto nearest = [&candidates](const auto& metric, const v2& q) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < candidates.size(); ++i)
            if (me::reduced_distance(metric, q, candidates[i]) < me::reduced_distance(metric, q, candidates[best]))
                best = i;
        return best;
    };
    EXPECT_EQ(nearest(m, v2{{0, 0}}), 1u);
    EXPECT_EQ(nearest(me::euclidean{}, v2{{0, 0}}), 0u);
    // On random sets, comparing reduced distances finds the same nearest point as comparing distances
    st::wyrand g(9);
    const auto r = me::make_mahalanobis(random_spd<3>(g)).unwrap();
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<v3> pts(40);
        for (auto& p : pts)
            p = random_vector<3>(g, 10);
        const v3 q = random_vector<3>(g, 10);
        std::size_t by_reduced = 0, by_distance = 0;
        for (std::size_t i = 1; i < pts.size(); ++i) {
            by_reduced = r.reduced(q, pts[i]) < r.reduced(q, pts[by_reduced]) ? i : by_reduced;
            by_distance = r(q, pts[i]) < r(q, pts[by_distance]) ? i : by_distance;
        }
        EXPECT_EQ(by_reduced, by_distance) << trial;
    }
}

TEST(MetricMahalanobis, MixesPointTypes)
{
    const auto m = me::make_mahalanobis(m2{{4, 0, 0, 25}}).unwrap();
    const std::array<double, 2> a{3, 10};
    const std::vector<float> b{1, 0};
    const double c[2] = {1, 0};
    EXPECT_NEAR(m(a, b), std::sqrt(5.0), 1e-14);
    EXPECT_NEAR(m(a, c), std::sqrt(5.0), 1e-14);
    EXPECT_NEAR(m(v2{{3, 10}}, c), std::sqrt(5.0), 1e-14);
}

TEST(MetricGaussian, AgreesWithTheTrackingFunctions)
{
    st::wyrand g(13);
    for (int trial = 0; trial < 200; ++trial) {
        const g2 a{random_vector<2>(g, 3), random_spd<2>(g)}, b{random_vector<2>(g, 3), random_spd<2>(g)};
        EXPECT_EQ(me::hellinger{}(a, b), *tr::hellinger(a, b)) << trial;
        EXPECT_EQ(me::squared_hellinger{}(a, b), *tr::squared_hellinger(a, b)) << trial;
        EXPECT_EQ(me::kl_divergence{}(a, b), *tr::kl_divergence(a, b)) << trial;
        const double h = me::hellinger{}(a, b);
        EXPECT_GE(h, 0.0);
        EXPECT_LE(h, 1.0);
        EXPECT_NEAR(h * h, me::squared_hellinger{}(a, b), 1e-14);
        EXPECT_GE(me::kl_divergence{}(a, b), -1e-12);
    }
}

TEST(MetricGaussian, MatchesTheOneDimensionalClosedForms)
{
    st::wyrand g(17);
    for (int trial = 0; trial < 200; ++trial) {
        const double mu1 = 4 * st::uniform_real(g) - 2, mu2 = 4 * st::uniform_real(g) - 2;
        const double s1 = 0.1 + 2 * st::uniform_real(g), s2 = 0.1 + 2 * st::uniform_real(g);   // standard deviations
        const g1 a{{{mu1}}, {{s1 * s1}}}, b{{{mu2}}, {{s2 * s2}}};
        const double v = s1 * s1 + s2 * s2;
        // H² = 1 - √(2 σ₁ σ₂ / (σ₁² + σ₂²)) exp(-(μ₁ - μ₂)² / (4 (σ₁² + σ₂²)))
        const double h2 = 1 - std::sqrt(2 * s1 * s2 / v) * std::exp(-(mu1 - mu2) * (mu1 - mu2) / (4 * v));
        // KL(a ‖ b) = log(σ₂ / σ₁) + (σ₁² + (μ₁ - μ₂)²) / (2 σ₂²) - 1/2
        const double kl = std::log(s2 / s1) + (s1 * s1 + (mu1 - mu2) * (mu1 - mu2)) / (2 * s2 * s2) - 0.5;
        EXPECT_NEAR(me::squared_hellinger{}(a, b), h2, 1e-12) << trial;
        EXPECT_NEAR(me::hellinger{}(a, b), std::sqrt(h2), 1e-9) << trial;
        EXPECT_NEAR(me::kl_divergence{}(a, b), kl, 1e-10 * (1 + kl)) << trial;
    }
    const g1 same{{{1.5}}, {{0.7}}};
    EXPECT_EQ(me::hellinger{}(same, same), 0.0);
    EXPECT_NEAR(me::kl_divergence{}(same, same), 0.0, 1e-15);
}

TEST(MetricGaussian, AnIndefiniteCovarianceGivesNaN)
{
    const g2 good{v2{{0, 0}}, m2{{1, 0, 0, 1}}}, bad{v2{{0, 0}}, m2{{1, 2, 2, 1}}};
    for (const auto& [a, b] : {std::pair{good, bad}, std::pair{bad, good}, std::pair{bad, bad}}) {
        EXPECT_TRUE(std::isnan(me::hellinger{}(a, b)));
        EXPECT_TRUE(std::isnan(me::squared_hellinger{}(a, b)));
        EXPECT_TRUE(std::isnan(me::kl_divergence{}(a, b)));
        EXPECT_FALSE(me::hellinger{}(a, b) < 1.0);   // NaN never compares below a bound
    }
    const g1 negative{{{0}}, {{-1}}}, one{{{0}}, {{1}}};
    EXPECT_TRUE(std::isnan(me::kl_divergence{}(one, negative)));
    EXPECT_TRUE(std::isnan(me::hellinger{}(negative, one)));
}

TEST(MetricGaussian, KlDivergenceIsAsymmetricAndTheHellingerMeasuresAreSymmetric)
{
    const g1 narrow{{{0}}, {{1}}}, wide{{{0}}, {{4}}};
    // KL(N(0, 1) ‖ N(0, 4)) = log 2 + 1/8 - 1/2; the reverse is -log 2 + 2 - 1/2
    EXPECT_NEAR(me::kl_divergence{}(narrow, wide), std::log(2.0) + 0.125 - 0.5, 1e-14);
    EXPECT_NEAR(me::kl_divergence{}(wide, narrow), -std::log(2.0) + 1.5, 1e-14);
    EXPECT_GT(std::abs(me::kl_divergence{}(narrow, wide) - me::kl_divergence{}(wide, narrow)), 0.4);
    EXPECT_NEAR(me::hellinger{}(narrow, wide), me::hellinger{}(wide, narrow), 1e-15);
    EXPECT_NEAR(me::squared_hellinger{}(narrow, wide), me::squared_hellinger{}(wide, narrow), 1e-15);
}

} // namespace
