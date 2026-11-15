//
//  randomized_svd_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/linalg/randomized_svd.hpp>: against matrices built from known singular values (an exact
//  low rank recovered to rounding, a decaying spectrum to the best rank-k error), the exact SVD when the sketch is
//  full, orthonormal factors, the same bits for the same seed, every sketch and Gaussian sampler (and one of the
//  test's own), layouts and wide matrices, float, every error, and a constant expression.
//

#include <snicholls/core/linalg/randomized_svd.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace st = snicholls::stats;
using la::layout_right;
using la::matrix_view;

template <typename T>
using rm = matrix_view<T, layout_right>;

// An m x r matrix with orthonormal columns: the thin Q of a Gaussian matrix
std::vector<double> orthonormal(std::size_t m, std::size_t r, std::uint64_t seed)
{
    std::vector<double> g(m * r), q(m * r), tau(r);
    la::gaussian_sketch<>{seed}(std::span<double>(g), m, r);
    EXPECT_TRUE(la::qr_factor(la::reference, rm<double>(g.data(), {m, r}), std::span<double>(tau)).is_ok());
    EXPECT_TRUE(la::qr_q(la::reference, rm<const double>(g.data(), {m, r}), std::span<const double>(tau), rm<double>(q.data(), {m, r})).is_ok());
    return q;
}

// A = U0 diag(sigma) V0ᵀ, m x n, with the given singular values
std::vector<double> with_spectrum(std::size_t m, std::size_t n, const std::vector<double>& sigma)
{
    const std::size_t r = sigma.size();
    const auto u0 = orthonormal(m, r, 11), v0 = orthonormal(n, r, 12);
    std::vector<double> a(m * n, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t p = 0; p < r; ++p)
                a[i * n + j] += u0[i * r + p] * sigma[p] * v0[j * r + p];
    return a;
}

struct triplets {
    std::vector<double> s, u, v;
};

template <typename S = la::gaussian_sketch<>>
triplets run(const std::vector<double>& a, std::size_t m, std::size_t n, std::size_t k, la::truncated_svd_options o = {}, S sk = S{})
{
    triplets t{std::vector<double>(k), std::vector<double>(m * k), std::vector<double>(n * k)};
    std::vector<double> work(la::truncated_svd_work_size(m, n, k, o));
    const auto done = la::truncated_svd(la::reference, rm<const double>(a.data(), {m, n}), std::span<double>(t.s),
                                        rm<double>(t.u.data(), {m, k}), rm<double>(t.v.data(), {n, k}), std::span<double>(work), o, sk);
    EXPECT_TRUE(done.is_ok());
    return t;
}

// max |A - U diag(s) Vᵀ| and the Frobenius norm of the difference
std::pair<double, double> residual(const std::vector<double>& a, const triplets& t, std::size_t m, std::size_t n)
{
    const std::size_t k = t.s.size();
    double worst = 0, f2 = 0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double x = a[i * n + j];
            for (std::size_t p = 0; p < k; ++p)
                x -= t.u[i * k + p] * t.s[p] * t.v[j * k + p];
            worst = std::max(worst, std::abs(x));
            f2 += x * x;
        }
    return {worst, std::sqrt(f2)};
}

double orthonormality(const std::vector<double>& q, std::size_t rows, std::size_t k)
{
    double worst = 0;
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t b = 0; b < k; ++b) {
            double d = 0;
            for (std::size_t i = 0; i < rows; ++i)
                d += q[i * k + a] * q[i * k + b];
            worst = std::max(worst, std::abs(d - (a == b ? 1.0 : 0.0)));
        }
    return worst;
}

TEST(LinalgTruncatedSvd, AnExactlyLowRankMatrixIsRecoveredToRounding)
{
    constexpr std::size_t m = 60, n = 40, k = 5;
    const std::vector<double> sigma{9, 4, 2, 1, 0.5};
    const auto a = with_spectrum(m, n, sigma);
    const auto t = run(a, m, n, k);
    for (std::size_t i = 0; i < k; ++i)
        EXPECT_NEAR(t.s[i], sigma[i], 1e-12 * sigma[0]) << i;
    EXPECT_LT(residual(a, t, m, n).first, 1e-12);
    EXPECT_LT(orthonormality(t.u, m, k), 1e-12);
    EXPECT_LT(orthonormality(t.v, n, k), 1e-12);
}

TEST(LinalgTruncatedSvd, ADecayingSpectrumGivesTheLeadingTripletsAndTheBestRankKError)
{
    constexpr std::size_t m = 80, n = 40, k = 8;
    std::vector<double> sigma(n);
    for (std::size_t i = 0; i < n; ++i)
        sigma[i] = std::pow(0.7, double(i));
    const auto a = with_spectrum(m, n, sigma);
    const auto t = run(a, m, n, k);
    for (std::size_t i = 0; i < k; ++i)
        EXPECT_NEAR(t.s[i], sigma[i], 1e-9) << i;
    double tail = 0;   // the best rank-k error, by Eckart-Young: the Frobenius norm of the discarded values
    for (std::size_t i = k; i < n; ++i)
        tail += sigma[i] * sigma[i];
    EXPECT_LT(residual(a, t, m, n).second, (1 + 1e-6) * std::sqrt(tail));
    EXPECT_LT(orthonormality(t.u, m, k), 1e-12);
    EXPECT_LT(orthonormality(t.v, n, k), 1e-12);
}

TEST(LinalgTruncatedSvd, AFullSketchIsTheExactSvd)
{
    constexpr std::size_t m = 30, n = 12, k = 12;
    std::vector<double> a(m * n), exact = {};
    la::gaussian_sketch<>{5}(std::span<double>(a), m, n);
    const auto t = run(a, m, n, k, {.oversample = 0, .power_iterations = 0});
    std::vector<double> u = a, s(n), v(n * n);
    ASSERT_TRUE(la::svd(la::reference, rm<double>(u.data(), {m, n}), std::span<double>(s), rm<double>(v.data(), {n, n})).is_ok());
    for (std::size_t i = 0; i < k; ++i)
        EXPECT_NEAR(t.s[i], s[i], 1e-12 * s[0]) << i;
    EXPECT_LT(residual(a, t, m, n).first, 1e-12);
}

TEST(LinalgTruncatedSvd, TheSameSeedGivesTheSameBitsAndAnotherSeedTheSameValues)
{
    constexpr std::size_t m = 50, n = 30, k = 4;
    const auto a = with_spectrum(m, n, {5, 3, 2, 1, 0.01, 0.005});
    const auto x = run(a, m, n, k, {}, la::gaussian_sketch<>{7}), y = run(a, m, n, k, {}, la::gaussian_sketch<>{7}),
               z = run(a, m, n, k, {}, la::gaussian_sketch<>{8});
    EXPECT_EQ(0, std::memcmp(x.s.data(), y.s.data(), k * sizeof(double)));
    EXPECT_EQ(0, std::memcmp(x.u.data(), y.u.data(), x.u.size() * sizeof(double)));
    for (std::size_t i = 0; i < k; ++i)
        EXPECT_NEAR(x.s[i], z.s[i], 1e-10);
}

// A sketch of the test's own: every entry from a fixed lattice, nothing random at all (the seam takes any filler)
struct lattice_sketch {
    template <la::scalar T>
    void operator()(std::span<T> omega, std::size_t rows, std::size_t cols) const
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            omega[i] = T(std::sin(0.7 * double(i) + 0.3 * double(i / cols)));
    }
};
static_assert(la::sketch<lattice_sketch, double> && la::sketch<la::rademacher_sketch, float>
              && la::sketch<la::gaussian_sketch<st::normal_box_muller>, double>);

TEST(LinalgTruncatedSvd, EverySketchAndGaussianSamplerFindsTheSameTriplets)
{
    constexpr std::size_t m = 70, n = 35, k = 6;
    std::vector<double> sigma(n);
    for (std::size_t i = 0; i < n; ++i)
        sigma[i] = std::pow(0.6, double(i));
    const auto a = with_spectrum(m, n, sigma);
    const triplets runs[] = {run(a, m, n, k, {}, la::gaussian_sketch<st::normal_inverse_cdf>{1}),
                             run(a, m, n, k, {}, la::gaussian_sketch<st::normal_box_muller>{1}), run(a, m, n, k, {}, la::rademacher_sketch{1}),
                             run(a, m, n, k, {}, lattice_sketch{})};
    for (const auto& t : runs)
        for (std::size_t i = 0; i < k; ++i)
            EXPECT_NEAR(t.s[i], sigma[i], 1e-9) << i;
    // Box-Muller and the inverse CDF draw different matrices from the same seed
    EXPECT_NE(runs[0].u, runs[1].u);
}

TEST(LinalgTruncatedSvd, AColumnMajorAndAWideMatrixWork)
{
    constexpr std::size_t m = 20, n = 45, k = 3;   // wide
    const auto at = with_spectrum(n, m, {6, 2, 1});   // n x m, row-major: its transpose is A, column-major
    triplets t{std::vector<double>(k), std::vector<double>(m * k), std::vector<double>(n * k)};
    std::vector<double> work(la::truncated_svd_work_size(m, n, k));
    ASSERT_TRUE(la::truncated_svd(la::reference, la::transposed(rm<const double>(at.data(), {n, m})), std::span<double>(t.s),
                                  rm<double>(t.u.data(), {m, k}), rm<double>(t.v.data(), {n, k}), std::span<double>(work))
                    .is_ok());
    EXPECT_NEAR(t.s[0], 6, 1e-12);
    EXPECT_NEAR(t.s[1], 2, 1e-12);
    EXPECT_NEAR(t.s[2], 1, 1e-12);
}

TEST(LinalgTruncatedSvd, FloatWorksToFloatsPrecision)
{
    constexpr std::size_t m = 40, n = 20, k = 3;
    const auto ad = with_spectrum(m, n, {4, 2, 1});
    std::vector<float> a(ad.size()), s(k), u(m * k), v(n * k), work(la::truncated_svd_work_size(m, n, k));
    std::transform(ad.begin(), ad.end(), a.begin(), [](double x) { return static_cast<float>(x); });
    ASSERT_TRUE(la::truncated_svd(la::reference, rm<const float>(a.data(), {m, n}), std::span<float>(s), rm<float>(u.data(), {m, k}),
                                  rm<float>(v.data(), {n, k}), std::span<float>(work))
                    .is_ok());
    EXPECT_NEAR(s[0], 4.0f, 1e-5f);
    EXPECT_NEAR(s[1], 2.0f, 1e-5f);
    EXPECT_NEAR(s[2], 1.0f, 1e-5f);
}

TEST(LinalgTruncatedSvd, BadShapesAndShortWorkAreReportedAndKZeroWritesNothing)
{
    constexpr std::size_t m = 6, n = 4;
    std::vector<double> a(m * n, 1.0), s(3, -1.0), u(m * 3), v(n * 3), work(la::truncated_svd_work_size(m, n, 3));
    const rm<const double> av(a.data(), {m, n});
    auto mismatch = [](const auto& r) { return r.is_err() && r.unwrap_err() == la::error::dimension_mismatch; };
    std::vector<double> s5(5), u5(m * 5), v5(n * 5), w5(la::truncated_svd_work_size(m, n, 5));
    EXPECT_TRUE(mismatch(la::truncated_svd(la::reference, av, std::span<double>(s5), rm<double>(u5.data(), {m, 5}),
                                           rm<double>(v5.data(), {n, 5}), std::span<double>(w5))));   // k > min(m, n)
    EXPECT_TRUE(mismatch(la::truncated_svd(la::reference, av, std::span<double>(s), rm<double>(u.data(), {m - 1, 3}),
                                           rm<double>(v.data(), {n, 3}), std::span<double>(work))));
    EXPECT_TRUE(mismatch(la::truncated_svd(la::reference, av, std::span<double>(s), rm<double>(u.data(), {m, 3}),
                                           rm<double>(v.data(), {n, 2}), std::span<double>(work))));
    EXPECT_TRUE(mismatch(la::truncated_svd(la::reference, av, std::span<double>(s), rm<double>(u.data(), {m, 3}),
                                           rm<double>(v.data(), {n, 3}), std::span<double>(work).first(work.size() - 1))));
    EXPECT_EQ(s, std::vector<double>(3, -1.0));   // nothing written on failure
    EXPECT_TRUE(la::truncated_svd(la::reference, av, std::span<double>(), rm<double>(u.data(), {m, 0}), rm<double>(v.data(), {n, 0}),
                                  std::span<double>())
                    .is_ok());
}

TEST(LinalgTruncatedSvd, ANanIsReported)
{
    constexpr std::size_t m = 6, n = 4, k = 2;
    std::vector<double> a(m * n, 1.0), s(k), u(m * k), v(n * k), work(la::truncated_svd_work_size(m, n, k));
    a[5] = std::numeric_limits<double>::quiet_NaN();
    const auto r = la::truncated_svd(la::reference, rm<const double>(a.data(), {m, n}), std::span<double>(s), rm<double>(u.data(), {m, k}),
                                     rm<double>(v.data(), {n, k}), std::span<double>(work));
    ASSERT_TRUE(r.is_err());
    EXPECT_TRUE(r.unwrap_err() == la::error::singular || r.unwrap_err() == la::error::did_not_converge);
}

TEST(LinalgTruncatedSvd, AZeroMatrixHasZeroSingularValues)
{
    constexpr std::size_t m = 8, n = 5, k = 2;
    const std::vector<double> a(m * n, 0.0);
    const auto t = run(a, m, n, k);
    EXPECT_EQ(t.s, std::vector<double>(k, 0.0));
}

static_assert(la::truncated_svd_work_size(100, 40, 5) == 2 * 40 * 15 + 2 * 100 * 15 + 15 * 15 + 2 * 15);
static_assert(la::truncated_svd_work_size(10, 4, 3) == 2 * 4 * 4 + 2 * 10 * 4 + 4 * 4 + 2 * 4);   // l capped at n
static_assert(la::truncated_svd_work_size(10, 4, 0) == 0);

// In a constant expression: a rank-one 4 x 3 matrix, 2 u vᵀ with unit u, v, has one singular value, 2
constexpr bool compile_time()
{
    constexpr std::size_t m = 4, n = 3;
    std::array<double, m * n> a{};
    const std::array<double, m> u{0.5, 0.5, 0.5, 0.5};
    const std::array<double, n> v{0.6, 0.8, 0.0};
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a[i * n + j] = 2 * u[i] * v[j];
    std::array<double, 1> s{};
    std::array<double, m> uo{};
    std::array<double, n> vo{};
    std::array<double, la::truncated_svd_work_size(m, n, 1)> work{};
    const auto r = la::truncated_svd(la::reference, rm<const double>(a.data(), {m, n}), std::span<double>(s), rm<double>(uo.data(), {m, 1}),
                                     rm<double>(vo.data(), {n, 1}), std::span<double>(work), {}, la::rademacher_sketch{3});
    return r.is_ok() && s[0] > 2 - 1e-12 && s[0] < 2 + 1e-12;
}
static_assert(compile_time());

} // namespace
