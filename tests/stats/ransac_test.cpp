//
//  ransac_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/ransac.hpp>: ransac_iterations against its formula (computed independently with
//  Python's math module) and at its limits; recovery of a line and of a rigid 2-D transform with a known outlier
//  fraction, the exact inlier set, the adaptive stop, the iteration cap, the tie rule, determinism for a fixed seed,
//  both errors, and the constexpr and noexcept claims.
//

#include <snicholls/core/stats/ransac.hpp>

#include <snicholls/core/geometry/alignment.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace st = snicholls::stats;
namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;

constexpr std::size_t never = std::numeric_limits<std::size_t>::max();

// ---- ransac_iterations

TEST(StatsRansacIterations, MatchesTheFormula)
{
    // ceil(log(1 - p) / log(1 - w^s)), from Python: math.ceil(math.log(1 - p) / math.log(1 - w ** s))
    EXPECT_EQ(st::ransac_iterations(0.99, 0.5, 2), 17u);    // 16.008
    EXPECT_EQ(st::ransac_iterations(0.99, 0.7, 2), 7u);     // 6.839
    EXPECT_EQ(st::ransac_iterations(0.99, 0.5, 4), 72u);    // 71.355
    EXPECT_EQ(st::ransac_iterations(0.999, 0.3, 3), 253u);  // 252.373
    EXPECT_EQ(st::ransac_iterations(0.95, 0.8, 3), 5u);     // 4.176
    EXPECT_EQ(st::ransac_iterations(0.99, 0.1, 2), 459u);   // 458.211
    for (const double p : {0.5, 0.9, 0.99, 0.9999})
        for (const double w : {0.05, 0.2, 0.5, 0.8, 0.95})
            for (const std::size_t s : {1u, 2u, 3u, 5u, 8u}) {
                const double n = std::log(1 - p) / std::log1p(-std::pow(w, static_cast<double>(s)));
                if (n < 1e15) {
                    EXPECT_EQ(st::ransac_iterations(p, w, s), static_cast<std::size_t>(std::ceil(n))) << p << " " << w << " " << s;
                }
            }
}

TEST(StatsRansacIterations, IsOneForAllInliersAndNeverForNone)
{
    EXPECT_EQ(st::ransac_iterations(0.99, 1.0, 4), 1u);
    EXPECT_EQ(st::ransac_iterations(0.99, 1.5, 4), 1u);
    EXPECT_EQ(st::ransac_iterations(0.99, 0.0, 4), never);
    EXPECT_EQ(st::ransac_iterations(0.99, -0.1, 1), never);
    EXPECT_EQ(st::ransac_iterations(0.0, 0.5, 2), 1u);    // no confidence wanted: one try
    EXPECT_EQ(st::ransac_iterations(0.0, 0.0, 2), never);
    EXPECT_EQ(st::ransac_iterations(1e-300, 0.5, 1), 1u);  // a tiny but positive count rounds up to one
}

TEST(StatsRansacIterations, ConfidenceOneIsCappedAndTinyFractionsSaturate)
{
    // Confidence 1 is read as 1 - 1e-16: finite and larger than any confidence below it
    const std::size_t one = st::ransac_iterations(1.0, 0.5, 2);
    EXPECT_LT(one, never);
    EXPECT_GE(one, st::ransac_iterations(0.999999, 0.5, 2));
    // w^s below 2^-53: log(1 - w^s) would round to 0; the count is astronomically large, so saturates
    EXPECT_EQ(st::ransac_iterations(0.99, 4e-5, 4), never);   // 1.8e18
    EXPECT_EQ(st::ransac_iterations(0.99, 1e-3, 6), never);   // 4.6e18
    // log(0.01) / log1p(-1e-15) = log(100) / 1e-15 = 4.605170185988091e15 (Python: math.log(100) / 1e-15; log1p(-1e-15)
    // is -1e-15 to 16 digits)
    EXPECT_NEAR(static_cast<double>(st::ransac_iterations(0.99, 1e-3, 5)), 4.605170185988091e15, 1e3);
}

TEST(StatsRansacIterations, FallsAsTheInlierFractionRises)
{
    std::size_t previous = never;
    for (double w = 0.05; w < 1.0; w += 0.05) {
        const std::size_t k = st::ransac_iterations(0.99, w, 3);
        EXPECT_LE(k, previous) << w;
        previous = k;
    }
}

// ---- a line model

struct line {
    double slope = 0, intercept = 0;
};

struct line_data {
    std::vector<double> x, y;
    std::vector<bool> inlier;
};

// y = 2x - 1 with noise below 0.01 on the inliers; the outliers at least 5 away from the line
line_data make_line(std::size_t n, double outlier_fraction, std::uint64_t seed)
{
    st::wyrand g(seed);
    line_data d;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = st::uniform_real(g) * 10;
        const bool in = st::uniform_real(g) >= outlier_fraction;
        const double off = in ? (st::uniform_real(g) - 0.5) * 0.02 : (st::uniform_real(g) < 0.5 ? -1 : 1) * (5 + 20 * st::uniform_real(g));
        d.x.push_back(x);
        d.y.push_back(2 * x - 1 + off);
        d.inlier.push_back(in);
    }
    return d;
}

auto line_fit(const line_data& d)
{
    return [&d](std::span<const std::size_t> s) -> std::optional<line> {
        const double dx = d.x[s[1]] - d.x[s[0]];
        if (std::abs(dx) < 1e-12)
            return std::nullopt;
        const double m = (d.y[s[1]] - d.y[s[0]]) / dx;
        return line{m, d.y[s[0]] - m * d.x[s[0]]};
    };
}

auto line_error(const line_data& d)
{
    return [&d](const line& l, std::size_t i) { return std::abs(d.y[i] - (l.slope * d.x[i] + l.intercept)); };
}

TEST(StatsRansac, RecoversALineAndExactlyItsInliersWithThirtyPercentOutliers)
{
    const auto d = make_line(200, 0.3, 1);
    std::vector<std::size_t> in(200);
    st::wyrand g(7);
    const auto r = st::ransac<2>(200, line_fit(d), line_error(d), 0.05, in, g);
    ASSERT_TRUE(r);
    EXPECT_NEAR(r->model.slope, 2.0, 0.02);
    EXPECT_NEAR(r->model.intercept, -1.0, 0.1);
    std::vector<std::size_t> expected;
    for (std::size_t i = 0; i < 200; ++i)
        if (d.inlier[i])
            expected.push_back(i);
    EXPECT_EQ(std::vector<std::size_t>(in.begin(), in.begin() + static_cast<std::ptrdiff_t>(r->inliers)), expected);
    EXPECT_GE(r->tries, 1u);
    EXPECT_LE(r->tries, 1000u);
}

TEST(StatsRansac, SucceedsAcrossSeedsAndOutlierFractions)
{
    for (const double f : {0.0, 0.1, 0.4, 0.6})
        for (std::uint64_t seed = 0; seed < 20; ++seed) {
            const auto d = make_line(120, f, 100 + seed);
            std::size_t true_inliers = 0;
            for (const bool b : d.inlier)
                true_inliers += b;
            std::vector<std::size_t> in(120);
            auto g = st::wyrand::stream(seed, 3);
            const auto r = st::ransac<2>(120, line_fit(d), line_error(d), 0.05, in, g, 5000, 0.999);
            ASSERT_TRUE(r) << f << " " << seed;
            EXPECT_EQ(r->inliers, true_inliers) << f << " " << seed;
            EXPECT_NEAR(r->model.slope, 2.0, 0.05) << f << " " << seed;
        }
}

TEST(StatsRansac, StopsAfterOneTryWhenEveryDatumIsAnInlier)
{
    line_data d;   // noise-free, so the first sample's line fits every datum
    for (int i = 0; i < 50; ++i) {
        d.x.push_back(i);
        d.y.push_back(2 * i - 1);
        d.inlier.push_back(true);
    }
    std::vector<std::size_t> in(50);
    st::wyrand g(3);
    const auto r = st::ransac<2>(50, line_fit(d), line_error(d), 0.05, in, g);
    ASSERT_TRUE(r);
    EXPECT_EQ(r->tries, 1u);   // ransac_iterations(0.99, 1, 2) is 1
    EXPECT_EQ(r->inliers, 50u);
}

TEST(StatsRansac, TheAdaptiveStopIsTheFormulaAfterTheBestModel)
{
    // After the last improvement, at try k, the loop runs to ransac_iterations(0.99, inliers / n, 2) tries, or stops
    // at once if k + 1 is already past it
    const auto d = make_line(100, 0.3, 4);
    for (std::uint64_t seed = 0; seed < 50; ++seed) {
        std::vector<std::size_t> in(100);
        st::wyrand g(seed);
        std::vector<line> models;
        const auto fit = line_fit(d);
        const auto r = st::ransac<2>(100, [&](std::span<const std::size_t> s) {
            auto m = fit(s);
            if (m)
                models.push_back(*m);
            return m;
        }, line_error(d), 0.05, in, g);
        ASSERT_TRUE(r);
        EXPECT_EQ(models.size(), r->tries);   // no degenerate samples here
        std::size_t found = 0;
        while (found < models.size() && !(models[found].slope == r->model.slope && models[found].intercept == r->model.intercept))
            ++found;
        ASSERT_LT(found, models.size());
        const std::size_t want = st::ransac_iterations(0.99, static_cast<double>(r->inliers) / 100, 2);
        EXPECT_EQ(r->tries, std::max(want, found + 1)) << seed;
    }
}

TEST(StatsRansac, MaxIterationsCapsTheTries)
{
    const auto d = make_line(100, 0.9, 5);   // so few inliers that the formula asks for hundreds of tries
    std::vector<std::size_t> in(100);
    st::wyrand g(5);
    const auto r = st::ransac<2>(100, line_fit(d), line_error(d), 0.05, in, g, 25);
    ASSERT_TRUE(r);
    EXPECT_EQ(r->tries, 25u);
}

TEST(StatsRansac, AFixedSeedReproducesTheResult)
{
    const auto d = make_line(150, 0.5, 6);
    std::vector<std::size_t> a(150), b(150);
    st::wyrand g1(99), g2(99);
    const auto r1 = st::ransac<2>(150, line_fit(d), line_error(d), 0.05, a, g1);
    const auto r2 = st::ransac<2>(150, line_fit(d), line_error(d), 0.05, b, g2);
    ASSERT_TRUE(r1 && r2);
    EXPECT_EQ(r1->model.slope, r2->model.slope);
    EXPECT_EQ(r1->model.intercept, r2->model.intercept);
    EXPECT_EQ(r1->tries, r2->tries);
    EXPECT_EQ(r1->inliers, r2->inliers);
    EXPECT_EQ(a, b);
    EXPECT_EQ(g1, g2);
}

TEST(StatsRansac, SamplesHoldDistinctIndicesInRange)
{
    st::wyrand g(8);
    std::vector<std::size_t> in(5);
    bool ok = true;
    const auto r = st::ransac<5>(5, [&](std::span<const std::size_t> s) -> std::optional<int> {   // S == n: a full permutation
        std::array<bool, 5> seen{};
        for (const std::size_t i : s) {
            ok = ok && i < 5 && !seen[i];
            seen[i] = true;
        }
        return 0;
    }, [](int, std::size_t) { return 0.0; }, 1.0, in, g);
    ASSERT_TRUE(r);
    EXPECT_TRUE(ok);
    EXPECT_EQ(r->inliers, 5u);
}

TEST(StatsRansac, TiesInTheInlierCountGoToTheLowerResidualSum)
{
    // Model k is datum k's id; every model has the same 3 inliers of 4 (datum 3 never fits) and model k's residuals
    // are res[k]. Whichever models were drawn, the one with the lowest sum must win
    constexpr std::array<double, 4> res{0.5, 0.1, 0.3, 0.2};
    for (std::uint64_t seed = 0; seed < 40; ++seed) {
        std::array<bool, 4> drawn{};
        std::vector<std::size_t> in(4);
        st::wyrand g(seed);
        const auto r = st::ransac<1>(4, [&](std::span<const std::size_t> s) -> std::optional<std::size_t> {
            drawn[s[0]] = true;
            return s[0];
        }, [&](std::size_t m, std::size_t i) { return i == 3 ? 9.0 : res[m]; }, 1.0, in, g, 1000, 0.999999);
        ASSERT_TRUE(r);
        std::size_t best = 4;
        for (std::size_t k = 0; k < 4; ++k)
            if (drawn[k] && (best == 4 || res[k] < res[best]))
                best = k;
        EXPECT_EQ(r->model, best) << seed;
        EXPECT_EQ(r->inliers, 3u);
    }
}

TEST(StatsRansac, AnInlierMayLieExactlyOnTheThreshold)
{
    std::vector<std::size_t> in(3);
    st::wyrand g(9);
    const auto r = st::ransac<1>(3, [](std::span<const std::size_t>) { return std::optional<int>(0); },
                                 [](int, std::size_t i) { return i == 1 ? 0.25 : 1.0; }, 0.25, in, g);
    ASSERT_TRUE(r);
    EXPECT_EQ(r->inliers, 1u);
    EXPECT_EQ(in[0], 1u);
}

TEST(StatsRansac, ReportsTooFewData)
{
    const auto d = make_line(10, 0.0, 10);
    std::vector<std::size_t> in(10), short_buffer(9);
    st::wyrand g(10);
    const auto a = st::ransac<2>(1, line_fit(d), line_error(d), 0.05, in, g);
    ASSERT_FALSE(a);
    EXPECT_EQ(a.unwrap_err_unchecked(), st::ransac_error::too_few_data);
    const auto b = st::ransac<2>(10, line_fit(d), line_error(d), 0.05, short_buffer, g);
    ASSERT_FALSE(b);
    EXPECT_EQ(b.unwrap_err_unchecked(), st::ransac_error::too_few_data);
    const auto c = st::ransac<2>(0, line_fit(d), line_error(d), 0.05, std::span<std::size_t>{}, g);
    ASSERT_FALSE(c);
    EXPECT_EQ(c.unwrap_err_unchecked(), st::ransac_error::too_few_data);
}

TEST(StatsRansac, ReportsNoModelWhenEverySampleIsDegenerateOrUnderSupported)
{
    std::vector<std::size_t> in(10);
    st::wyrand g(11);
    int calls = 0;
    const auto a = st::ransac<2>(10, [&](std::span<const std::size_t>) { ++calls; return std::optional<line>{}; },
                                 [](const line&, std::size_t) { return 0.0; }, 0.05, in, g, 40);
    ASSERT_FALSE(a);
    EXPECT_EQ(a.unwrap_err_unchecked(), st::ransac_error::no_model);
    EXPECT_EQ(calls, 40);   // every try was spent
    // Models exist but none has S = 2 inliers (only datum 0 ever fits)
    const auto b = st::ransac<2>(10, [](std::span<const std::size_t>) { return std::optional<line>(line{}); },
                                 [](const line&, std::size_t i) { return i == 0 ? 0.0 : 1.0; }, 0.05, in, g, 40);
    ASSERT_FALSE(b);
    EXPECT_EQ(b.unwrap_err_unchecked(), st::ransac_error::no_model);
}

TEST(StatsRansac, RobustlyAlignsTwoFramesWithWrongCorrespondences)
{
    // The documented use: align_2d on minimal samples of 2, a quarter of the pairs mismatched
    using v2 = la::fixed_vector<double, 2>;
    st::wyrand g(12);
    const double a = 0.6;
    const geo::transform_2d<double> truth{{{std::cos(a), -std::sin(a), std::sin(a), std::cos(a)}}, v2{{4, -3}}, 1.0};
    std::vector<v2> src, dst;
    for (int i = 0; i < 40; ++i) {
        src.push_back(v2{{st::uniform_real(g) * 20, st::uniform_real(g) * 20}});
        dst.push_back(i % 4 == 3 ? v2{{st::uniform_real(g) * 100, -50}} : truth.apply(src.back()));
    }
    std::vector<std::size_t> in(40);
    const auto r = st::ransac<2>(40, [&](std::span<const std::size_t> s) {
        const std::array<v2, 2> ss{src[s[0]], src[s[1]]}, dd{dst[s[0]], dst[s[1]]};
        return geo::align_2d<double>(ss, dd);   // a result: tests as a model and dereferences to one
    }, [&](const geo::transform_2d<double>& x, std::size_t i) { return la::norm(dst[i] - x.apply(src[i])); }, 1e-6, in, g);
    ASSERT_TRUE(r);
    EXPECT_EQ(r->inliers, 30u);
    EXPECT_NEAR(r->model.angle(), a, 1e-9);
    for (std::size_t k = 0; k < r->inliers; ++k)
        EXPECT_NE(in[k] % 4, 3u);
}

// ---- compile time and noexcept

constexpr bool compile_time()
{
    // A constant model from one datum: the inliers are the four values near 1
    constexpr std::array<double, 6> data{1.0, 1.1, 0.9, 1.0, 50.0, -30.0};
    std::array<std::size_t, 6> in{};
    st::wyrand g(1);
    const auto r = st::ransac<1>(6, [&](std::span<const std::size_t> s) { return std::optional<double>(data[s[0]]); },
                                 [&](double m, std::size_t i) { return data[i] > m ? data[i] - m : m - data[i]; }, 0.2, in, g);
    return r && r->inliers == 4 && in[0] == 0 && in[1] == 1 && in[2] == 2 && in[3] == 3 && st::ransac_iterations(0.99, 0.5, 2) == 17;
}
static_assert(compile_time());

struct nothrow_fit {
    std::optional<int> operator()(std::span<const std::size_t>) const noexcept { return 0; }
};
struct nothrow_error {
    double operator()(int, std::size_t) const noexcept { return 0; }
};
struct throwing_error {
    double operator()(int, std::size_t) const { throw std::runtime_error("no"); }
};
static_assert(noexcept(st::ransac<1>(1, nothrow_fit{}, nothrow_error{}, 0.0, std::span<std::size_t>{}, std::declval<st::wyrand&>())));
static_assert(!noexcept(st::ransac<1>(1, nothrow_fit{}, throwing_error{}, 0.0, std::span<std::size_t>{}, std::declval<st::wyrand&>())));
static_assert(noexcept(st::ransac_iterations(0.99, 0.5, 2)));

} // namespace
