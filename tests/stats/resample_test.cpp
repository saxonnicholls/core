//
//  resample_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/resample.hpp>: the effective sample size and the resampling test against closed
//  forms; log_sum_exp and normalise_log_weights against direct sums and at the infinities; systematic resampling
//  against an independent search over the cumulative weights and its floor-or-ceiling copy counts; stratified and
//  multinomial copy-count means and variances against their closed forms; residual resampling's exact deterministic
//  copies and its remainder's distribution under each method; ordering, zero weights never drawn, generator
//  consumption, every unusable total, and the constexpr and noexcept claims.
//

#include <snicholls/core/stats/resample.hpp>

#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace st = snicholls::stats;

constexpr double inf = std::numeric_limits<double>::infinity();

std::vector<std::size_t> counts(std::span<const std::size_t> idx, std::size_t n)
{
    std::vector<std::size_t> c(n, 0);
    for (const std::size_t i : idx) {
        EXPECT_LT(i, n);
        if (i < n)
            ++c[i];
    }
    return c;
}

// The scheme under test, as one callable
enum class scheme { systematic, stratified, multinomial, residual_systematic, residual_stratified, residual_multinomial };
constexpr std::array all_schemes{scheme::systematic, scheme::stratified, scheme::multinomial, scheme::residual_systematic, scheme::residual_stratified,
                                 scheme::residual_multinomial};

bool draw(scheme s, std::span<const double> w, std::span<std::size_t> out, st::wyrand& g)
{
    std::vector<double> residual(w.size());
    switch (s) {
    case scheme::systematic: return st::systematic_resample(w, out, g);
    case scheme::stratified: return st::stratified_resample(w, out, g);
    case scheme::multinomial: return st::multinomial_resample(w, out, g);
    case scheme::residual_systematic: return st::residual_resample(w, std::span<double>(residual), out, g, st::residual_method::systematic);
    case scheme::residual_stratified: return st::residual_resample(w, std::span<double>(residual), out, g, st::residual_method::stratified);
    case scheme::residual_multinomial: return st::residual_resample(w, std::span<double>(residual), out, g, st::residual_method::multinomial);
    }
    return false;
}

// ---- effective sample size, needs_resampling

TEST(StatsEffectiveSampleSize, IsTheClosedFormBetweenOneAndN)
{
    const std::vector<double> equal(8, 0.3), one_hot{0, 0, 5, 0}, w{1, 2, 3};
    EXPECT_NEAR(st::effective_sample_size<double>(equal), 8.0, 1e-12);
    EXPECT_EQ(st::effective_sample_size<double>(one_hot), 1.0);
    EXPECT_NEAR(st::effective_sample_size<double>(w), 36.0 / 14.0, 1e-12);   // (1 + 2 + 3)^2 / (1 + 4 + 9)
    const std::vector<double> scaled{1000, 2000, 3000};
    EXPECT_NEAR(st::effective_sample_size<double>(scaled), 36.0 / 14.0, 1e-12);   // unnormalised weights: same answer
    EXPECT_EQ(st::effective_sample_size<double>(std::span<const double>{}), 0.0);
    const std::vector<double> zeros(3, 0.0);
    EXPECT_EQ(st::effective_sample_size<double>(zeros), 0.0);
}

TEST(StatsNeedsResampling, ComparesTheEffectiveSampleSizeToAFractionOfN)
{
    const std::vector<double> w{1, 2, 3};   // ESS 2.571 of 3
    EXPECT_FALSE(st::needs_resampling<double>(w, 0.5));
    EXPECT_FALSE(st::needs_resampling<double>(w, 0.857));   // 0.857 * 3 = 2.571 just below
    EXPECT_TRUE(st::needs_resampling<double>(w, 0.858));
    const std::vector<double> one_hot{0, 1, 0, 0};
    EXPECT_TRUE(st::needs_resampling<double>(one_hot, 0.5));
    const std::vector<double> equal(4, 1.0);
    EXPECT_FALSE(st::needs_resampling<double>(equal, 1.0));   // ESS = N is not below N
}

// ---- log space

TEST(StatsLogSumExp, MatchesTheDirectSumAndAvoidsOverflow)
{
    const std::vector<double> x{std::log(1.0), std::log(2.0), std::log(3.0)};
    EXPECT_NEAR(st::log_sum_exp<double>(x), std::log(6.0), 1e-14);
    const std::vector<double> big{1000, 1000};   // exp(1000) overflows; the answer is 1000 + log 2
    EXPECT_NEAR(st::log_sum_exp<double>(big), 1000 + std::log(2.0), 1e-12);
    const std::vector<double> small{-1000, -1001};
    EXPECT_NEAR(st::log_sum_exp<double>(small), -1000 + std::log1p(std::exp(-1.0)), 1e-12);
    const std::vector<double> some_minus_inf{-inf, 0.0, -inf};
    EXPECT_NEAR(st::log_sum_exp<double>(some_minus_inf), 0.0, 1e-15);
}

TEST(StatsLogSumExp, IsMinusInfinityForNothingAndPassesPlusInfinity)
{
    EXPECT_EQ(st::log_sum_exp<double>(std::span<const double>{}), -inf);
    const std::vector<double> none{-inf, -inf};
    EXPECT_EQ(st::log_sum_exp<double>(none), -inf);
    const std::vector<double> plus{1.0, inf};
    EXPECT_EQ(st::log_sum_exp<double>(plus), inf);
}

TEST(StatsNormaliseLogWeights, MakesTheExponentialsSumToOneAndReturnsTheConstant)
{
    std::vector<double> lw{-750, -751, -752.5};   // all underflow if exponentiated directly
    const double z = st::normalise_log_weights<double>(lw);
    EXPECT_NEAR(z, -750 + std::log(1 + std::exp(-1.0) + std::exp(-2.5)), 1e-12);
    double s = 0;
    for (const double v : lw)
        s += std::exp(v);
    EXPECT_NEAR(s, 1.0, 2e-13);   // each v - z is exact only to half an ulp of 750, 5.7e-14
    EXPECT_NEAR(lw[0] - lw[1], 1.0, 1e-12);   // ratios unchanged

    std::vector<double> none{-inf, -inf};
    EXPECT_EQ(st::normalise_log_weights<double>(none), -inf);
    EXPECT_EQ(none[0], -inf);   // left as they were
    std::vector<double> plus{0.0, inf};
    EXPECT_EQ(st::normalise_log_weights<double>(plus), inf);
    EXPECT_EQ(plus[0], 0.0);
    std::vector<double> empty;
    EXPECT_EQ(st::normalise_log_weights<double>(empty), -inf);
}

// ---- systematic

TEST(StatsSystematicResample, MatchesAnIndependentSearchOfTheCumulativeWeights)
{
    st::wyrand g(11);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 1 + st::uniform_index(g, 20), m = 1 + st::uniform_index(g, 30);
        std::vector<double> w(n);
        double sum = 0;
        for (double& x : w)
            sum += (x = st::uniform_real(g) < 0.2 ? 0.0 : st::uniform_real(g));
        if (!(sum > 0))
            continue;
        std::vector<std::size_t> out(m);
        st::wyrand copy = g;
        ASSERT_TRUE(st::systematic_resample<double>(w, out, g));
        const double u0 = st::uniform_real(copy);
        EXPECT_EQ(copy, g) << "exactly one generator call";
        for (std::size_t i = 0; i < m; ++i) {
            // The first j whose cumulative normalised weight exceeds (i + U) / M, skipping zero weights
            const double u = (static_cast<double>(i) + u0) / static_cast<double>(m);
            double cum = 0;
            std::size_t j = 0;
            for (; j < n; ++j) {
                cum += w[j] / sum;
                if (u < cum)
                    break;
            }
            j = std::min(j, n - 1);
            while (w[j] == 0 && j > 0)
                --j;
            EXPECT_EQ(out[i], j) << "trial " << trial << " draw " << i;
        }
    }
}

TEST(StatsSystematicResample, EachParticleGetsTheFloorOrCeilingOfItsExpectedCopies)
{
    st::wyrand g(12);
    const std::vector<double> w{0.13, 0.31, 0.07, 0.49};
    std::vector<std::size_t> out(10);
    for (int run = 0; run < 1000; ++run) {
        ASSERT_TRUE(st::systematic_resample<double>(w, out, g));
        EXPECT_TRUE(std::is_sorted(out.begin(), out.end()));
        const auto c = counts(out, 4);
        for (std::size_t j = 0; j < 4; ++j) {
            const double e = 10 * w[j];
            EXPECT_TRUE(c[j] == static_cast<std::size_t>(std::floor(e)) || c[j] == static_cast<std::size_t>(std::ceil(e))) << j << ": " << c[j];
        }
    }
    // Integer expectations: the copies are exact, whatever the uniform
    const std::vector<double> exact{1, 1, 2};
    std::vector<std::size_t> four(4);
    for (int run = 0; run < 100; ++run) {
        ASSERT_TRUE(st::systematic_resample<double>(exact, four, g));
        EXPECT_EQ(four, (std::vector<std::size_t>{0, 1, 2, 2}));
    }
}

// ---- stratified and multinomial: statistics

struct moments {
    std::vector<double> mean, variance;
};

moments copy_moments(scheme s, const std::vector<double>& w, std::size_t m, int runs, std::uint64_t seed)
{
    st::wyrand g(seed);
    std::vector<std::size_t> out(m);
    std::vector<double> s1(w.size(), 0), s2(w.size(), 0);
    for (int r = 0; r < runs; ++r) {
        EXPECT_TRUE(draw(s, w, out, g));
        const auto c = counts(out, w.size());
        for (std::size_t j = 0; j < w.size(); ++j) {
            s1[j] += static_cast<double>(c[j]);
            s2[j] += static_cast<double>(c[j] * c[j]);
        }
    }
    moments mo;
    for (std::size_t j = 0; j < w.size(); ++j) {
        mo.mean.push_back(s1[j] / runs);
        mo.variance.push_back(s2[j] / runs - mo.mean.back() * mo.mean.back());
    }
    return mo;
}

TEST(StatsMultinomialResample, CopyCountsHaveTheMultinomialMeanAndVariance)
{
    // Count_j ~ Binomial(M, w_j): mean M w_j, variance M w_j (1 - w_j). With 40000 runs the standard error of the
    // mean is at most sqrt(2.5 / 40000) = 0.008 and of the variance about 0.7 %: tolerances are over 5 sigma
    const std::vector<double> w{1, 2, 3, 4};   // normalised: 0.1 .. 0.4
    const auto mo = copy_moments(scheme::multinomial, w, 10, 40000, 21);
    for (std::size_t j = 0; j < 4; ++j) {
        const double p = w[j] / 10;
        EXPECT_NEAR(mo.mean[j], 10 * p, 0.05) << j;
        EXPECT_NEAR(mo.variance[j], 10 * p * (1 - p), 0.06 * 10 * p * (1 - p)) << j;
    }
}

TEST(StatsStratifiedResample, CopyCountsAreUnbiasedWithLessVarianceThanMultinomial)
{
    const std::vector<double> w{1, 2, 3, 4};
    const auto mo = copy_moments(scheme::stratified, w, 10, 40000, 22);
    for (std::size_t j = 0; j < 4; ++j) {
        const double p = w[j] / 10;
        EXPECT_NEAR(mo.mean[j], 10 * p, 0.05) << j;
        EXPECT_LT(mo.variance[j], 0.8 * 10 * p * (1 - p)) << j;
    }
}

TEST(StatsStratifiedResample, EachCountIsWithinOneOfTheExpectedCopiesAndUsesOneUniformPerDraw)
{
    // Each stratum holds one draw; an interval of length L meets at most ceil(M L) + 1 strata and contains at least
    // floor(M L) - 1 whole ones
    st::wyrand g(23);
    const std::vector<double> w{0.13, 0.31, 0.07, 0.49};
    std::vector<std::size_t> out(10);
    for (int run = 0; run < 1000; ++run) {
        st::wyrand copy = g;
        ASSERT_TRUE(st::stratified_resample<double>(w, out, g));
        for (int k = 0; k < 10; ++k)
            static_cast<void>(copy());
        EXPECT_EQ(copy, g);
        EXPECT_TRUE(std::is_sorted(out.begin(), out.end()));
        const auto c = counts(out, 4);
        for (std::size_t j = 0; j < 4; ++j) {
            const double e = 10 * w[j];
            EXPECT_GE(static_cast<double>(c[j]), std::floor(e) - 1);
            EXPECT_LE(static_cast<double>(c[j]), std::ceil(e) + 1);
        }
    }
}

TEST(StatsMultinomialResample, IsSortedAndUsesMPlusOneUniforms)
{
    st::wyrand g(24);
    const std::vector<double> w{5, 1, 0.5, 3, 2};
    std::vector<std::size_t> out(17);
    for (int run = 0; run < 500; ++run) {
        st::wyrand copy = g;
        ASSERT_TRUE(st::multinomial_resample<double>(w, out, g));
        for (int k = 0; k < 18; ++k)
            static_cast<void>(copy());
        EXPECT_EQ(copy, g);
        EXPECT_TRUE(std::is_sorted(out.begin(), out.end()));
    }
}

// ---- residual

TEST(StatsResidualResample, GivesTheFloorCopiesFirstThenTheRemainderInOrder)
{
    // Weights 0.5, 0.3, 0.2 and M = 7: expected 3.5, 2.1, 1.4, so 3, 2 and 1 deterministic copies and one draw from
    // the residuals 0.5, 0.1, 0.4
    const std::vector<double> w{5, 3, 2};
    for (const scheme s : {scheme::residual_systematic, scheme::residual_stratified, scheme::residual_multinomial}) {
        st::wyrand g(31);
        std::vector<std::size_t> out(7);
        std::array<int, 3> rest{};
        constexpr int runs = 30000;
        for (int r = 0; r < runs; ++r) {
            ASSERT_TRUE(draw(s, w, out, g));
            EXPECT_EQ(std::vector<std::size_t>(out.begin(), out.begin() + 6), (std::vector<std::size_t>{0, 0, 0, 1, 1, 2}));
            ASSERT_LT(out[6], 3u);
            ++rest[out[6]];
        }
        // The remainder's distribution is the normalised residuals; standard error at most 0.003
        EXPECT_NEAR(rest[0] / double(runs), 0.5, 0.02);
        EXPECT_NEAR(rest[1] / double(runs), 0.1, 0.02);
        EXPECT_NEAR(rest[2] / double(runs), 0.4, 0.02);
    }
}

TEST(StatsResidualResample, IntegerExpectationsNeedNoRandomness)
{
    const std::vector<double> w{2, 1, 1};
    std::vector<double> residual(3);
    std::vector<std::size_t> out(8);
    st::wyrand g(32);
    const st::wyrand before = g;
    ASSERT_TRUE(st::residual_resample<double>(w, residual, out, g));
    EXPECT_EQ(out, (std::vector<std::size_t>{0, 0, 0, 0, 1, 1, 2, 2}));
    EXPECT_EQ(g, before);   // nothing left to draw
    for (const double r : residual)
        EXPECT_EQ(r, 0.0);
}

TEST(StatsResidualResample, CopyCountsAreWithinOneOfTheExpectation)
{
    st::wyrand g(33);
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = 1 + st::uniform_index(g, 12), m = 1 + st::uniform_index(g, 40);
        std::vector<double> w(n), residual(n);
        double sum = 0;
        for (double& x : w)
            sum += (x = st::uniform_real(g));
        std::vector<std::size_t> out(m);
        const auto method = static_cast<st::residual_method>(trial % 3);
        ASSERT_TRUE(st::residual_resample<double>(w, residual, out, g, method));
        const auto c = counts(out, n);
        for (std::size_t j = 0; j < n; ++j) {
            const double e = static_cast<double>(m) * w[j] / sum;
            EXPECT_GE(static_cast<double>(c[j]), std::floor(e)) << trial;
            // the remainder is at most n draws; with systematic each particle gets at most one more
            if (method == st::residual_method::systematic) {
                EXPECT_LE(static_cast<double>(c[j]), std::floor(e) + 1) << trial;
            }
        }
    }
}

TEST(StatsResidualResample, FailsWhenTheResidualBufferIsShort)
{
    const std::vector<double> w{1, 2, 3};
    std::vector<double> residual(2);
    std::vector<std::size_t> out(4);
    st::wyrand g(34);
    EXPECT_FALSE(st::residual_resample<double>(w, residual, out, g));
}

// ---- every scheme

TEST(StatsResample, NoSchemeEverDrawsAParticleOfZeroWeight)
{
    // Zeros at both ends and in the middle; the cumulative sum's rounding at the end must not land on the last zero
    const std::vector<double> w{0, 0.1, 0, 0, 0.7, 0.2, 0, 0};
    for (const scheme s : all_schemes) {
        st::wyrand g(41);
        std::vector<std::size_t> out(37);
        for (int run = 0; run < 2000; ++run) {
            ASSERT_TRUE(draw(s, w, out, g));
            for (const std::size_t i : out)
                ASSERT_GT(w[i], 0.0) << static_cast<int>(s) << " drew " << i;
        }
    }
}

TEST(StatsResample, EveryDrawIsSortedAndInRangeForRandomWeights)
{
    st::wyrand g(42);
    for (int trial = 0; trial < 600; ++trial) {
        const std::size_t n = 1 + st::uniform_index(g, 15), m = st::uniform_index(g, 25);
        std::vector<double> w(n);
        for (double& x : w)
            x = st::uniform_real(g) < 0.3 ? 0.0 : std::exp(20 * st::uniform_real(g) - 10);   // spread over 9 decades
        w[st::uniform_index(g, n)] = 1.0;   // never all zero
        const scheme s = all_schemes[static_cast<std::size_t>(trial) % all_schemes.size()];
        std::vector<std::size_t> out(m);
        ASSERT_TRUE(draw(s, w, out, g));
        const bool residual = s != scheme::systematic && s != scheme::stratified && s != scheme::multinomial;
        if (!residual) {
            EXPECT_TRUE(std::is_sorted(out.begin(), out.end())) << trial;
        }
        for (const std::size_t i : out) {
            ASSERT_LT(i, n);
            EXPECT_GT(w[i], 0.0);
        }
    }
}

TEST(StatsResample, OneParticleIsDrawnEveryTime)
{
    const std::vector<double> w{0.25};
    for (const scheme s : all_schemes) {
        st::wyrand g(43);
        std::vector<std::size_t> out(5, 99);
        ASSERT_TRUE(draw(s, w, out, g));
        EXPECT_EQ(out, (std::vector<std::size_t>(5, 0)));
    }
}

TEST(StatsResample, AnUnusableTotalReturnsFalse)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::vector<double>> bad{{}, {0, 0}, {-1, 0.5}, {nan, 1}, {inf, 1}, {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()}};
    for (const auto& w : bad)
        for (const scheme s : all_schemes) {
            st::wyrand g(44);
            std::vector<std::size_t> out(3);
            EXPECT_FALSE(draw(s, w, out, g)) << static_cast<int>(s);
        }
    // An empty index span: still the weights' verdict
    for (const scheme s : all_schemes) {
        st::wyrand g(45);
        const std::vector<double> good{1, 2};
        EXPECT_TRUE(draw(s, good, std::span<std::size_t>{}, g));
        EXPECT_FALSE(draw(s, bad[1], std::span<std::size_t>{}, g));
    }
}

TEST(StatsResample, AFixedSeedReproducesTheDraw)
{
    const std::vector<double> w{3, 1, 4, 1, 5, 9, 2, 6};
    for (const scheme s : all_schemes) {
        st::wyrand g1(2026), g2(2026);
        std::vector<std::size_t> a(20), b(20);
        ASSERT_TRUE(draw(s, w, a, g1));
        ASSERT_TRUE(draw(s, w, b, g2));
        EXPECT_EQ(a, b);
        auto g3 = st::wyrand::stream(2026, 1);
        std::vector<std::size_t> c(20);
        ASSERT_TRUE(draw(s, w, c, g3));
    }
}

TEST(StatsResample, FloatWeightsWork)
{
    const std::vector<float> w{1, 1, 2};
    std::vector<float> residual(3);
    std::vector<std::size_t> out(4);
    st::wyrand g(46);
    ASSERT_TRUE(st::systematic_resample<float>(w, out, g));
    EXPECT_EQ(out, (std::vector<std::size_t>{0, 1, 2, 2}));
    ASSERT_TRUE(st::residual_resample<float>(w, residual, out, g));
    EXPECT_EQ(out, (std::vector<std::size_t>{0, 1, 2, 2}));
    EXPECT_NEAR(st::effective_sample_size<float>(w), 16.0f / 6.0f, 1e-6f);
}

// ---- compile time

constexpr bool compile_time()
{
    const std::array<double, 4> w{0, 1, 1, 2};
    std::array<std::size_t, 4> sys{}, strat{}, multi{}, res{};
    std::array<double, 4> residual{};
    st::wyrand g(5);
    const bool ok = st::systematic_resample<double>(w, sys, g) && st::stratified_resample<double>(w, strat, g) && st::multinomial_resample<double>(w, multi, g)
                    && st::residual_resample<double>(w, residual, res, g, st::residual_method::multinomial);
    bool no_zero = true;
    for (const auto& arr : {strat, multi})
        for (const std::size_t i : arr)
            no_zero = no_zero && i != 0 && i < 4;
    std::array<double, 2> lw{0.0, 0.0};
    const double z = st::normalise_log_weights<double>(lw);
    return ok && no_zero && sys == std::array<std::size_t, 4>{1, 2, 3, 3} && res == std::array<std::size_t, 4>{1, 2, 3, 3}
           && st::effective_sample_size<double>(w) == 16.0 / 6.0 && !st::needs_resampling<double>(w, 0.5) && z > 0.693 && z < 0.694
           && st::log_sum_exp<double>(std::span<const double>{}) == -inf;
}
static_assert(compile_time());
static_assert(noexcept(st::systematic_resample<double>(std::span<const double>{}, std::span<std::size_t>{}, std::declval<st::wyrand&>())));
static_assert(noexcept(st::stratified_resample<double>(std::span<const double>{}, std::span<std::size_t>{}, std::declval<st::wyrand&>())));
static_assert(noexcept(st::multinomial_resample<double>(std::span<const double>{}, std::span<std::size_t>{}, std::declval<st::wyrand&>())));
static_assert(noexcept(st::residual_resample<double>(std::span<const double>{}, std::span<double>{}, std::span<std::size_t>{}, std::declval<st::wyrand&>())));
static_assert(noexcept(st::effective_sample_size<double>(std::span<const double>{})) && noexcept(st::log_sum_exp<double>(std::span<const double>{})));

} // namespace
