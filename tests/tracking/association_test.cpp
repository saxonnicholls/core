//
//  association_test.cpp
//  snicholls-core
//
//  Tests for <snicholls/core/tracking/association.hpp>: PDA weights against the hand formula and a numpy known
//  answer, gated detections, every false return; the PDA update against the moment-matched mixture of the
//  missed-detection prediction and each detection's Kalman update; the gate volume against the interval, ellipse
//  and ellipsoid formulas; JPDA marginals against brute-force enumeration of every joint event on random clusters
//  with forbidden entries, and its size errors; all of it in a constant expression.
//

#include <snicholls/core/tracking/association.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/kalman.hpp>
#include <snicholls/core/tracking/measurement.hpp>
#include <snicholls/core/tracking/mixture.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace st = snicholls::stats;
namespace tr = snicholls::tracking;

using vec2 = la::fixed_vector<double, 2>;
using mat2 = la::fixed_matrix<double, 2, 2>;
using vec4 = la::fixed_vector<double, 4>;
using mat4 = la::fixed_matrix<double, 4, 4>;
using sensor = tr::linear_gaussian<double, 4, std::array{0, 2}>;

constexpr double inf = std::numeric_limits<double>::infinity();

// A constant-velocity track (x, vx, y, vy) seen in position; the same numbers as the numpy script that made the
// known answers below
constexpr tr::gaussian<double, 4> prediction()
{
    return {vec4{{1, 0.5, -2, 0.3}}, mat4{{4, 0.5, 0, 0, 0.5, 1, 0, 0, 0, 0, 3, 0.4, 0, 0, 0.4, 0.8}}};
}
constexpr sensor position_sensor() { return sensor{mat2{{1, 0.2, 0.2, 0.5}}}; }
constexpr std::array<vec2, 3> detections() { return {vec2{{1.5, -1}}, vec2{{3, -3}}, vec2{{-0.5, -2.2}}}; }

// N(ν; 0, S) for a 2 x 2 S by the closed-form inverse and determinant
double density(const vec2& nu, const mat2& s)
{
    const double det = s(0, 0) * s(1, 1) - s(0, 1) * s(1, 0);
    const double q = (s(1, 1) * nu[0] * nu[0] - 2 * s(0, 1) * nu[0] * nu[1] + s(0, 0) * nu[1] * nu[1]) / det;
    return std::exp(-q / 2) / (2 * std::numbers::pi * std::sqrt(det));
}

TEST(TrackingPdaWeights, MatchTheHandFormulaNormalised)
{
    const auto mp = tr::predict_measurement(prediction(), position_sensor());
    const double pd = 0.9, pg = 0.99, lambda = 0.05;
    std::array<double, 3> ll{};
    std::array<double, 4> expect{};
    expect[0] = 1 - pd * pg;
    for (std::size_t i = 0; i < 3; ++i) {
        const vec2 nu = detections()[i] - mp.mean;
        ll[i] = *tr::log_likelihood(nu, mp.covariance);
        expect[i + 1] = pd * density(nu, mp.covariance) / lambda;
    }
    double z = 0;
    for (const double e : expect)
        z += e;
    std::array<double, 4> w{};
    ASSERT_TRUE(tr::pda_weights<double>(ll, pd, pg, lambda, w));
    double sum = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(w[i], expect[i] / z, 1e-14) << i;
        sum += w[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-15);
    // From numpy: S = H P Hᵀ + R, N(ν; 0, S) by np.linalg.solve and np.linalg.det, normalised
    const std::array<double, 4> numpy{0.06701196130149935, 0.3582756366526552, 0.23908201070584253, 0.33563039134000294};
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(w[i], numpy[i], 1e-13) << i;
}

TEST(TrackingPdaWeights, AGatedDetectionGetsWeightZero)
{
    const std::array<double, 3> ll{-2.0, -inf, -3.0};
    std::array<double, 4> w{};
    ASSERT_TRUE(tr::pda_weights<double>(ll, 0.8, 0.95, 0.1, w));
    EXPECT_EQ(w[2], 0.0);
    const double miss = 1 - 0.8 * 0.95, d1 = 0.8 * std::exp(-2.0) / 0.1, d3 = 0.8 * std::exp(-3.0) / 0.1;
    EXPECT_NEAR(w[0], miss / (miss + d1 + d3), 1e-15);
    EXPECT_NEAR(w[1], d1 / (miss + d1 + d3), 1e-15);
    EXPECT_NEAR(w[3], d3 / (miss + d1 + d3), 1e-15);
}

TEST(TrackingPdaWeights, NoDetectionsIsAllMissedAndCertainDetectionNeverMisses)
{
    std::array<double, 1> only{};
    ASSERT_TRUE(tr::pda_weights<double>(std::span<const double>{}, 0.9, 0.99, 0.05, only));
    EXPECT_EQ(only[0], 1.0);
    // P_D P_G = 1: the missed hypothesis has weight 0 (log of zero is not taken)
    const std::array<double, 2> ll{-1.0, -1.0};
    std::array<double, 3> w{};
    ASSERT_TRUE(tr::pda_weights<double>(ll, 1.0, 1.0, 0.5, w));
    EXPECT_EQ(w[0], 0.0);
    EXPECT_NEAR(w[1], 0.5, 1e-15);
    EXPECT_NEAR(w[2], 0.5, 1e-15);
}

TEST(TrackingPdaWeights, RefuseWrongLengthsNonPositiveDensityAndNoHypothesis)
{
    const std::array<double, 2> ll{-1.0, -2.0};
    std::array<double, 2> short_w{};
    std::array<double, 4> long_w{};
    std::array<double, 3> w{};
    EXPECT_FALSE(tr::pda_weights<double>(ll, 0.9, 0.99, 0.05, short_w));
    EXPECT_FALSE(tr::pda_weights<double>(ll, 0.9, 0.99, 0.05, long_w));
    EXPECT_FALSE(tr::pda_weights<double>(ll, 0.9, 0.99, 0.0, w));
    EXPECT_FALSE(tr::pda_weights<double>(ll, 0.9, 0.99, -1.0, w));
    EXPECT_FALSE(tr::pda_weights<double>(ll, 0.9, 0.99, std::numeric_limits<double>::quiet_NaN(), w));
    // Every detection gated and detection certain: no hypothesis has weight
    const std::array<double, 2> gated{-inf, -inf};
    EXPECT_FALSE(tr::pda_weights<double>(gated, 1.0, 1.0, 0.05, w));
}

// The explicit mixture: the prediction with β₀, each detection's Kalman update with βᵢ, moment matched
tr::gaussian<double, 4> explicit_mixture(const std::array<double, 4>& w)
{
    const auto p = prediction();
    const auto mp = tr::predict_measurement(p, position_sensor());
    const auto k = mp.cross_covariance * *la::inverse(mp.covariance);   // K = P Hᵀ S⁻¹
    std::array<tr::weighted_gaussian<double, 4>, 4> parts{};
    parts[0] = {w[0], p};
    for (std::size_t i = 0; i < 3; ++i) {
        const vec2 nu = detections()[i] - mp.mean;
        parts[i + 1] = {w[i + 1], {p.mean + k * nu, p.covariance - k * mp.covariance * la::transpose(k)}};
    }
    return tr::reduce_to_single<double, 4>(parts);
}

TEST(TrackingPdaUpdate, EqualsTheMomentMatchedMixtureOfEveryHypothesis)
{
    const auto p = prediction();
    const auto mp = tr::predict_measurement(p, position_sensor());
    std::array<vec2, 3> nus{};
    std::array<double, 3> ll{};
    for (std::size_t i = 0; i < 3; ++i) {
        nus[i] = detections()[i] - mp.mean;
        ll[i] = *tr::log_likelihood(nus[i], mp.covariance);
    }
    std::array<double, 4> w{};
    ASSERT_TRUE(tr::pda_weights<double>(ll, 0.9, 0.99, 0.05, w));
    const auto post = tr::pda_update<double, 4, 2>(p, mp, nus, w);
    ASSERT_TRUE(post);
    const auto want = explicit_mixture(w);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(post->mean[i], want.mean[i], 1e-13) << i;
        for (std::size_t j = 0; j < 4; ++j) {
            EXPECT_NEAR(post->covariance(i, j), want.covariance(i, j), 1e-13) << i << ',' << j;
            EXPECT_EQ(post->covariance(i, j), post->covariance(j, i));
        }
    }
    // From the same numpy script: the four hypotheses' mixture, moment matched
    EXPECT_NEAR(post->mean[0], 1.1209813001173579, 1e-13);
    EXPECT_NEAR(post->mean[2], -1.9605555862803334, 1e-13);
    EXPECT_NEAR(post->covariance(0, 0), 2.1667871167107582, 1e-13);
    EXPECT_NEAR(post->covariance(2, 3), 0.14112938285741802, 1e-13);
}

TEST(TrackingPdaUpdate, AllWeightOnOneHypothesisIsThatHypothesis)
{
    const auto p = prediction();
    const auto mp = tr::predict_measurement(p, position_sensor());
    std::array<vec2, 3> nus{};
    for (std::size_t i = 0; i < 3; ++i)
        nus[i] = detections()[i] - mp.mean;
    // Missed only: the prediction
    const auto missed = tr::pda_update<double, 4, 2>(p, mp, nus, std::array<double, 4>{1, 0, 0, 0});
    ASSERT_TRUE(missed);
    for (std::size_t i = 0; i < 16; ++i)
        EXPECT_NEAR(missed->covariance.elements[i], p.covariance.elements[i], 1e-15);
    EXPECT_EQ(missed->mean, p.mean);
    // Detection 2 only: the Kalman update with it
    const auto one = tr::pda_update<double, 4, 2>(p, mp, nus, std::array<double, 4>{0, 0, 1, 0});
    const auto k = tr::update<sensor::kinds>(p, detections()[1], mp);
    ASSERT_TRUE(one && k);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(one->mean[i], k->posterior.mean[i], 1e-14);
    for (std::size_t i = 0; i < 16; ++i)
        EXPECT_NEAR(one->covariance.elements[i], k->posterior.covariance.elements[i], 1e-14);
    // No detections at all
    const auto none = tr::pda_update<double, 4, 2>(p, mp, std::span<const vec2>{}, std::array<double, 1>{1});
    ASSERT_TRUE(none);
    EXPECT_EQ(none->mean, p.mean);
}

TEST(TrackingPdaUpdate, ReportsMismatchedLengthsAndANonPositiveDefiniteS)
{
    const auto p = prediction();
    auto mp = tr::predict_measurement(p, position_sensor());
    const std::array<vec2, 2> nus{};
    const auto bad_len = tr::pda_update<double, 4, 2>(p, mp, nus, std::array<double, 2>{0.5, 0.5});
    ASSERT_FALSE(bad_len);
    EXPECT_EQ(bad_len.unwrap_err_unchecked(), la::error::dimension_mismatch);
    mp.covariance = mat2{{1, 2, 2, 1}};
    const auto bad_s = tr::pda_update<double, 4, 2>(p, mp, nus, std::array<double, 3>{0.5, 0.25, 0.25});
    ASSERT_FALSE(bad_s);
    EXPECT_EQ(bad_s.unwrap_err_unchecked(), la::error::not_positive_definite);
}

TEST(TrackingGateVolume, MatchesTheIntervalEllipseAndEllipsoidFormulas)
{
    // 1-D: the interval [-√(γ s), √(γ s)]
    const auto v1 = tr::gate_volume(9.0, la::fixed_matrix<double, 1, 1>{{4}});
    ASSERT_TRUE(v1);
    EXPECT_NEAR(*v1, 2 * std::sqrt(9.0 * 4.0), 1e-12);
    // 2-D: an ellipse of semi-axes a = √(γ λ₁), b = √(γ λ₂), area π a b; diag(4, 9), γ = 2: π √8 √18 = 12π
    const auto v2 = tr::gate_volume(2.0, mat2{{4, 0, 0, 9}});
    ASSERT_TRUE(v2);
    EXPECT_NEAR(*v2, std::numbers::pi * std::sqrt(8.0) * std::sqrt(18.0), 1e-12);
    // Rotated: the same eigenvalues give the same area. [[5, 2], [2, 8]] has eigenvalues 4 and 9
    const auto v2r = tr::gate_volume(2.0, mat2{{5, 2, 2, 8}});
    ASSERT_TRUE(v2r);
    EXPECT_NEAR(*v2r, 12 * std::numbers::pi, 1e-12);
    // 3-D: 4/3 π a b c; diag(1, 4, 9), γ = 4: semi-axes 2, 4, 6, volume 64π
    const auto v3 = tr::gate_volume(4.0, la::fixed_matrix<double, 3, 3>{{1, 0, 0, 0, 4, 0, 0, 0, 9}});
    ASSERT_TRUE(v3);
    EXPECT_NEAR(*v3, 4.0 / 3.0 * std::numbers::pi * 2 * 4 * 6, 1e-11);
}

TEST(TrackingGateVolume, ReportsANonPositiveDefiniteCovariance)
{
    const auto v = tr::gate_volume(4.0, mat2{{1, 3, 3, 1}});
    ASSERT_FALSE(v);
    EXPECT_EQ(v.unwrap_err_unchecked(), la::error::not_positive_definite);
}

// Every joint event by recursion: each track misses or takes one unused detection, weight the product
void enumerate(const std::vector<double>& l, std::size_t tracks, std::size_t cols, std::size_t t, std::vector<std::size_t>& choice, std::vector<bool>& used, double product,
               std::vector<double>& marg, double& total)
{
    if (t == tracks) {
        total += product;
        for (std::size_t k = 0; k < tracks; ++k)
            marg[k * cols + choice[k]] += product;
        return;
    }
    for (std::size_t c = 0; c < cols; ++c) {
        if (c > 0 && used[c])
            continue;
        const double w = l[t * cols + c];
        if (!(w > 0))
            continue;
        choice[t] = c;
        if (c > 0)
            used[c] = true;
        enumerate(l, tracks, cols, t + 1, choice, used, product * w, marg, total);
        if (c > 0)
            used[c] = false;
    }
}

TEST(TrackingJpdaMarginals, MatchBruteForceEnumerationOnRandomClusters)
{
    st::wyrand g(20261123);
    int feasible = 0, infeasible = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const std::size_t tracks = 1 + st::uniform_index(g, 4), detections = st::uniform_index(g, 5), cols = detections + 1;
        std::vector<double> l(tracks * cols);
        for (double& x : l)
            x = st::uniform_index(g, 4) == 0 ? 0.0 : 0.01 + st::uniform_real(g);   // a quarter forbidden
        std::vector<double> marg(tracks * cols, 0.0), got(tracks * cols, -1.0);
        std::vector<std::size_t> choice(tracks);
        std::vector<bool> used(cols, false);
        double total = 0;
        enumerate(l, tracks, cols, 0, choice, used, 1.0, marg, total);
        const bool ok = tr::jpda_marginals<4, 4, double>(l, tracks, detections, got);
        ASSERT_EQ(ok, total > 0) << trial;
        if (!ok) {
            ++infeasible;
            continue;
        }
        ++feasible;
        for (std::size_t t = 0; t < tracks; ++t) {
            double row = 0;
            for (std::size_t c = 0; c < cols; ++c) {
                EXPECT_NEAR(got[t * cols + c], marg[t * cols + c] / total, 1e-13) << trial << ' ' << t << ' ' << c;
                if (l[t * cols + c] == 0) {
                    EXPECT_EQ(got[t * cols + c], 0.0);
                }
                row += got[t * cols + c];
            }
            EXPECT_NEAR(row, 1.0, 1e-13);
        }
    }
    EXPECT_GT(feasible, 400);
    EXPECT_GT(infeasible, 0);
}

TEST(TrackingJpdaMarginals, TwoTracksCompetingForOneDetectionByHand)
{
    // Rows (miss, d1): track 0 (1, 3), track 1 (1, 1). Events: both miss 1, t0 takes d1 3, t1 takes d1 1; total 5
    const std::array<double, 4> l{1, 3, 1, 1};
    std::array<double, 4> m{};
    ASSERT_TRUE((tr::jpda_marginals<2, 1, double>(l, 2, 1, m)));
    EXPECT_NEAR(m[0], 2.0 / 5, 1e-15);
    EXPECT_NEAR(m[1], 3.0 / 5, 1e-15);
    EXPECT_NEAR(m[2], 4.0 / 5, 1e-15);
    EXPECT_NEAR(m[3], 1.0 / 5, 1e-15);
}

TEST(TrackingJpdaMarginals, NoTracksIsTrivialAndNoFeasibleEventIsRefused)
{
    std::array<double, 1> none{};
    EXPECT_TRUE((tr::jpda_marginals<2, 2, double>(std::span<const double>{}, 0, 2, none)));
    // Both tracks must take detection 1 (missing is forbidden): infeasible
    const std::array<double, 4> clash{0, 1, 0, 1};
    std::array<double, 4> m{};
    EXPECT_FALSE((tr::jpda_marginals<2, 1, double>(clash, 2, 1, m)));
    const std::array<double, 4> zeros{};
    EXPECT_FALSE((tr::jpda_marginals<2, 1, double>(zeros, 2, 1, m)));
}

TEST(TrackingJpdaMarginals, RefusesSizesBeyondItsMaximaAndShortSpans)
{
    const std::array<double, 12> l{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::array<double, 12> m{};
    EXPECT_TRUE((tr::jpda_marginals<3, 3, double>(l, 3, 3, m)));
    EXPECT_FALSE((tr::jpda_marginals<2, 3, double>(l, 3, 3, m)));   // too many tracks
    EXPECT_FALSE((tr::jpda_marginals<3, 2, double>(l, 3, 3, m)));   // too many detections
    EXPECT_FALSE((tr::jpda_marginals<3, 3, double>(std::span<const double>(l).first(11), 3, 3, m)));
    EXPECT_FALSE((tr::jpda_marginals<3, 3, double>(l, 3, 3, std::span<double>(m).first(11))));
}

// Compile time: weights, an update, a gate volume and a JPDA cluster
constexpr bool compile_time()
{
    const std::array<double, 2> ll{-1.0, -2.0};
    std::array<double, 3> w{};
    if (!tr::pda_weights<double>(ll, 0.9, 0.9, 0.1, w))
        return false;
    const auto p = prediction();
    const auto mp = tr::predict_measurement(p, position_sensor());
    const std::array<vec2, 2> nus{vec2{{0.5, 1}}, vec2{{-1, 0}}};
    const auto post = tr::pda_update<double, 4, 2>(p, mp, nus, w);
    const auto v = tr::gate_volume(2.0, mat2{{4, 0, 0, 9}});
    const std::array<double, 4> l{1, 3, 1, 1};
    std::array<double, 4> m{};
    const bool j = tr::jpda_marginals<2, 1, double>(l, 2, 1, m);
    const double sum = w[0] + w[1] + w[2];
    return sum > 0.999999 && sum < 1.000001 && post.is_ok() && post->covariance(0, 0) < p.covariance(0, 0) && v.is_ok() && *v > 37.69 && *v < 37.70 && j && m[1] > 0.5999 && m[1] < 0.6001;
}
static_assert(compile_time());

static_assert(noexcept(tr::pda_weights<double>(std::span<const double>{}, 0.9, 0.9, 0.1, std::span<double>{})));
static_assert(noexcept(tr::pda_update<double, 4, 2>(std::declval<const tr::gaussian<double, 4>&>(), std::declval<const tr::measurement_prediction<double, 4, 2>&>(),
                                                     std::span<const vec2>{}, std::span<const double>{})));
static_assert(noexcept(tr::gate_volume(1.0, std::declval<const mat2&>())));
static_assert(noexcept(tr::jpda_marginals<2, 2, double>(std::span<const double>{}, 0, 0, std::span<double>{})));

} // namespace
