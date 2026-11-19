//
//  collapse_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/collapse.hpp>: embedding_moments against the two-pass formulas, full and diagonal,
//  merge equal to one pass, the effective rank of isotropic, rank-two and collapsed embeddings; VICReg's variance and
//  covariance terms against the definitions written out and their gradients by central differences; SIGReg's
//  directions (unit, deterministic per step), the statistic against the characteristic function integrated directly,
//  small for Gaussian samples and large for collapsed ones, the sums merging across workers, the gradient by central
//  differences, and gradient descent on it spreading a collapsed batch; compile-time use.
//

#include <snicholls/core/ml/collapse.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
const auto& backend = snicholls::linalg::reference;

constexpr ml::sigreg_quadrature<double> quad{};
static_assert(quad.t[16] == 3.0 && quad.phi[0] == 1.0 && quad.weight[0] == 3.0 / 16);

constexpr bool at_compile_time()
{
    std::array<double, 2> mean{}, m2{};
    ml::embedding_moments<double> acc{mean, m2};
    const std::array<double, 6> rows{1, 2, 3, 4, 5, 6};
    acc.add_rows(rows);
    std::array<double, 2> var{};
    acc.variance(var);
    return acc.count == 3 && mean[0] == 3 && mean[1] == 4 && var[0] == 4 && var[1] == 4;
}
static_assert(at_compile_time());

std::vector<double> gaussian(std::size_t n, std::uint64_t seed, double scale = 1.0)
{
    stats::wyrand g(seed);
    const stats::normal_inverse_cdf normal{};
    std::vector<double> v(n);
    for (double& x : v)
        x = scale * normal(g);
    return v;
}

TEST(MlEmbeddingMoments, AreTheTwoPassFormulasAndMerge)
{
    constexpr std::size_t d = 5, n = 40;
    const auto x = gaussian(n * d, 1, 2.0);
    std::vector<double> mean(d), m(d * d), dmean(d), dm(d);
    ml::embedding_moments<double> full{mean, m}, diag{dmean, dm};
    full.add_rows(x);
    diag.add_rows(x);
    std::vector<double> mu(d), cov(d * d);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t j = 0; j < d; ++j)
            mu[j] += x[r * d + j] / n;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j < d; ++j)
                cov[i * d + j] += (x[r * d + i] - mu[i]) * (x[r * d + j] - mu[j]) / (n - 1);
    std::vector<double> c(d * d), sd(d), sd2(d);
    full.covariance(c);
    full.standard_deviation(sd);
    diag.standard_deviation(sd2);
    for (std::size_t i = 0; i < d * d; ++i)
        EXPECT_NEAR(c[i], cov[i], 1e-12);
    for (std::size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(mean[j], mu[j], 1e-12);
        EXPECT_NEAR(sd[j], std::sqrt(cov[j * d + j]), 1e-12);
        EXPECT_NEAR(sd2[j], sd[j], 1e-12);
    }
    // Two halves merged are one pass
    std::vector<double> ma(d), a(d * d), mb(d), b(d * d);
    ml::embedding_moments<double> first{ma, a}, second{mb, b};
    first.add_rows(std::span<const double>(x).first(13 * d));
    second.add_rows(std::span<const double>(x).subspan(13 * d));
    first.merge(second);
    EXPECT_EQ(first.count, n);
    for (std::size_t i = 0; i < d * d; ++i)
        EXPECT_NEAR(a[i], m[i], 1e-10);
    ml::embedding_moments<double> empty{mb, b};
    empty.count = 0;
    first.merge(empty);   // nothing
    EXPECT_EQ(first.count, n);
}

TEST(MlEmbeddingMoments, EffectiveRankIsTheWidthWhenIsotropicAndOneWhenCollapsed)
{
    constexpr std::size_t d = 6, n = 4000;
    std::vector<double> mean(d), m(d * d), work(2 * d * d + d);
    ml::embedding_moments<double> iso{mean, m};
    iso.add_rows(gaussian(n * d, 2));
    const auto r = iso.effective_rank(backend, std::span<double>(work));
    ASSERT_TRUE(r);
    EXPECT_GT(*r, 5.9);
    EXPECT_LE(*r, 6.0 + 1e-9);
    // Every row a multiple of one vector: rank 1 (up to rounding)
    std::vector<double> mean2(d), m2(d * d), line(n * d);
    const auto z = gaussian(n, 3);
    for (std::size_t r2 = 0; r2 < n; ++r2)
        for (std::size_t j = 0; j < d; ++j)
            line[r2 * d + j] = z[r2] * double(j + 1);
    ml::embedding_moments<double> collapsed{mean2, m2};
    collapsed.add_rows(line);
    EXPECT_NEAR(*collapsed.effective_rank(backend, std::span<double>(work)), 1.0, 1e-5);
    // Two independent directions of equal spread: rank 2
    std::vector<double> mean3(d), m3(d * d), plane(n * d);
    const auto u = gaussian(n, 4), v = gaussian(n, 5);
    for (std::size_t r2 = 0; r2 < n; ++r2) {
        plane[r2 * d + 0] = u[r2];
        plane[r2 * d + 3] = v[r2];
    }
    ml::embedding_moments<double> two{mean3, m3};
    two.add_rows(plane);
    EXPECT_NEAR(*two.effective_rank(backend, std::span<double>(work)), 2.0, 0.03);   // sampling noise of 4000 rows
    // Constant rows: no spread at all
    std::vector<double> mean4(d), m4(d * d), constant(10 * d, 1.5);
    ml::embedding_moments<double> point{mean4, m4};
    point.add_rows(constant);
    EXPECT_EQ(*point.effective_rank(backend, std::span<double>(work)), 0.0);
}

// The VICReg terms written out from their definitions
double variance_term(const std::vector<double>& x, std::size_t d)
{
    const std::size_t n = x.size() / d;
    double t = 0;
    for (std::size_t j = 0; j < d; ++j) {
        double mu = 0, v = 0;
        for (std::size_t r = 0; r < n; ++r)
            mu += x[r * d + j] / n;
        for (std::size_t r = 0; r < n; ++r)
            v += (x[r * d + j] - mu) * (x[r * d + j] - mu) / (n - 1);
        t += std::max(0.0, 1.0 - std::sqrt(v + 1e-4));
    }
    return t / d;
}

double covariance_term(const std::vector<double>& x, std::size_t d)
{
    const std::size_t n = x.size() / d;
    std::vector<double> mu(d);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t j = 0; j < d; ++j)
            mu[j] += x[r * d + j] / n;
    double t = 0;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            if (i == j)
                continue;
            double c = 0;
            for (std::size_t r = 0; r < n; ++r)
                c += (x[r * d + i] - mu[i]) * (x[r * d + j] - mu[j]) / (n - 1);
            t += c * c;
        }
    return t / d;
}

template <typename F, typename G>
void expect_gradient(std::vector<double> x, F loss, const std::vector<double>& grad, G&& reset, double tol)
{
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double keep = x[i], e = 1e-6;
        x[i] = keep + e;
        const double hi = loss(x);
        x[i] = keep - e;
        const double lo = loss(x);
        x[i] = keep;
        EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), tol) << i;
    }
    reset();
}

TEST(MlVicreg, VarianceAndCovarianceTermsAreTheirDefinitionsWithTheirGradients)
{
    constexpr std::size_t d = 4, n = 7;
    auto x = gaussian(n * d, 6, 0.6);
    for (std::size_t r = 0; r < n; ++r)
        x[r * d + 2] *= 3.0;   // one dimension above the hinge: no gradient there
    std::vector<double> work(2 * d), cwork(d + d * d), dv(x.size()), dc(x.size());
    const double v = ml::vicreg_variance_rows(backend, std::span<const double>(x), d, std::span<double>(dv), std::span<double>(work), 25.0);
    const double c = ml::vicreg_covariance_rows(backend, std::span<const double>(x), d, std::span<double>(dc), std::span<double>(cwork), 2.0);
    EXPECT_NEAR(v, 25.0 * variance_term(x, d), 1e-12);
    EXPECT_NEAR(c, 2.0 * covariance_term(x, d), 1e-12);
    expect_gradient(x, [&](const std::vector<double>& y) { return 25.0 * variance_term(y, d); }, dv, [] {}, 1e-7);
    expect_gradient(x, [&](const std::vector<double>& y) { return 2.0 * covariance_term(y, d); }, dc, [] {}, 1e-7);
    // No gradient wanted: the value alone
    EXPECT_NEAR(ml::vicreg_variance_rows(backend, std::span<const double>(x), d, std::span<double>{}, std::span<double>(work)), variance_term(x, d), 1e-12);
}

TEST(MlSigreg, DirectionsAreUnitAndTheSameForAStep)
{
    std::vector<double> a(16 * 8), b(16 * 8), c(16 * 8);
    ml::sigreg_directions<double>(5, 10, 8, a);
    ml::sigreg_directions<double>(5, 10, 8, b);
    ml::sigreg_directions<double>(5, 11, 8, c);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    for (std::size_t m = 0; m < 16; ++m) {
        double s = 0;
        for (std::size_t i = 0; i < 8; ++i)
            s += a[m * 8 + i] * a[m * 8 + i];
        EXPECT_NEAR(s, 1.0, 1e-12);
    }
}

// The statistic of one direction by the integral done directly: N ∫_{-3}^{3} |φ̂(t) - e^{-t²/2}|² e^{-t²/2} dt by a
// fine midpoint rule, against the 17-knot trapezoid
TEST(MlSigreg, TheStatisticIsTheWeightedCharacteristicFunctionDistance)
{
    constexpr std::size_t n = 50, d = 3;
    const auto x = gaussian(n * d, 7, 1.3);
    const std::vector<double> dir{1.0, 0.0, 0.0};
    std::vector<double> sums(17 * 2);
    const double got = ml::sigreg_rows(backend, std::span<const double>(x), d, std::span<const double>(dir), quad, std::span<double>(sums),
                                       std::span<double>{});
    double integral = 0;
    constexpr int steps = 60000;
    for (int s = 0; s < steps; ++s) {
        const double t = -3.0 + 6.0 * (s + 0.5) / steps;
        double c = 0, sn = 0;
        for (std::size_t r = 0; r < n; ++r) {
            c += std::cos(t * x[r * d]) / n;
            sn += std::sin(t * x[r * d]) / n;
        }
        const double phi = std::exp(-t * t / 2);
        integral += ((c - phi) * (c - phi) + sn * sn) * phi * 6.0 / steps;
    }
    EXPECT_NEAR(got, n * integral, 0.02 * n * integral + 1e-3);   // the 17-knot trapezoid against the integral
}

TEST(MlSigreg, IsSmallForGaussiansLargeForCollapseAndMergesAcrossWorkers)
{
    constexpr std::size_t n = 512, d = 8, m = 32;
    std::vector<double> dirs(m * d), sums(m * 17 * 2);
    ml::sigreg_directions<double>(1, 0, d, dirs);
    const auto x = gaussian(n * d, 8);
    const double gauss = ml::sigreg_rows(backend, std::span<const double>(x), d, std::span<const double>(dirs), quad, std::span<double>(sums), {});
    std::vector<double> collapsed(n * d, 0.25);
    const double point = ml::sigreg_rows(backend, std::span<const double>(collapsed), d, std::span<const double>(dirs), quad, std::span<double>(sums), {});
    EXPECT_LT(gauss, 4.0);   // under the null the statistic is of order 1
    EXPECT_GT(point, 25 * gauss);
    // Two workers' sums added are one worker's over every row
    std::vector<double> s1(sums.size()), s2(sums.size()), all(sums.size());
    ml::sigreg_moments_rows(backend, std::span<const double>(x).first(200 * d), d, std::span<const double>(dirs), quad, std::span<double>(s1));
    ml::sigreg_moments_rows(backend, std::span<const double>(x).subspan(200 * d), d, std::span<const double>(dirs), quad, std::span<double>(s2));
    for (std::size_t i = 0; i < all.size(); ++i)
        all[i] = s1[i] + s2[i];
    EXPECT_NEAR(ml::sigreg_value(std::span<const double>(all), n, quad), gauss, 1e-9);
}

TEST(MlSigreg, GradientMatchesCentralDifferences)
{
    constexpr std::size_t n = 9, d = 4, m = 5;
    std::vector<double> dirs(m * d), sums(m * 17 * 2);
    ml::sigreg_directions<double>(2, 3, d, dirs);
    const auto x = gaussian(n * d, 9, 0.8);
    std::vector<double> dx(x.size());
    ml::sigreg_rows(backend, std::span<const double>(x), d, std::span<const double>(dirs), quad, std::span<double>(sums), std::span<double>(dx), 0.5);
    auto loss = [&](const std::vector<double>& y) {
        std::vector<double> s(sums.size());
        return ml::sigreg_rows(backend, std::span<const double>(y), d, std::span<const double>(dirs), quad, std::span<double>(s), {}, 0.5);
    };
    expect_gradient(x, loss, dx, [] {}, 1e-6);
}

TEST(MlSigreg, DescentSpreadsACollapsedBatch)
{
    constexpr std::size_t n = 128, d = 4, m = 16;
    auto x = gaussian(n * d, 10, 0.02);   // nearly one point
    std::vector<double> mean(d), var(d);
    std::vector<double> sums(m * 17 * 2), dirs(m * d), dx(x.size());
    double first = 0, last = 0;
    for (std::uint64_t step = 0; step < 300; ++step) {
        ml::sigreg_directions<double>(11, step, d, dirs);
        for (double& v : dx)
            v = 0;
        last = ml::sigreg_rows(backend, std::span<const double>(x), d, std::span<const double>(dirs), quad, std::span<double>(sums), std::span<double>(dx));
        if (step == 0)
            first = last;
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] -= 0.5 * dx[i];
    }
    EXPECT_LT(last, first / 10);
    ml::embedding_moments<double> acc{mean, var};
    acc.add_rows(x);
    std::vector<double> sd(d);
    acc.standard_deviation(sd);
    for (const double s : sd)
        EXPECT_GT(s, 0.3);
}

} // namespace
