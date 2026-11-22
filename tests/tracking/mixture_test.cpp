//
//  mixture_test.cpp
//  snicholls-core
//
//  Tests for <snicholls/core/tracking/mixture.hpp>: the moment-matched single Gaussian by hand and against explicit
//  moments of random mixtures, the zero Gaussian for no weight; merge_pair's moments and its weight cap; Runnalls'
//  cost by hand, non-negative, zero for identical components, and its error; reduce_mixture's pruning (the pruned
//  weight shared equally), merging (measured with the candidate's covariance, lightest candidate first, moments
//  preserved), heaviest-first order, max_components and its refusal to merge a component with no Cholesky
//  factor; the layout of the value types, and all of it in a constant expression.
//

#include <snicholls/core/tracking/mixture.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace {

namespace la = snicholls::linalg;
namespace st = snicholls::stats;
namespace tr = snicholls::tracking;

using wg1 = tr::weighted_gaussian<double, 1>;
using wg2 = tr::weighted_gaussian<double, 2>;
using vec1 = la::fixed_vector<double, 1>;
using mat1 = la::fixed_matrix<double, 1, 1>;
using vec2 = la::fixed_vector<double, 2>;
using mat2 = la::fixed_matrix<double, 2, 2>;

constexpr wg1 c1(double w, double mean, double var) { return {w, {vec1{{mean}}, mat1{{var}}}}; }

// Layout: a weight and a Gaussian, trivially copyable, nothing more
static_assert(std::is_trivially_copyable_v<wg2> && std::is_standard_layout_v<wg2>);
static_assert(sizeof(wg2) == sizeof(double) + sizeof(tr::gaussian<double, 2>));
static_assert(std::is_trivially_copyable_v<tr::mixture_reduction<double>>);
static_assert(tr::mixture_reduction<double>{}.prune_threshold == 1e-9 && tr::mixture_reduction<double>{}.merge_threshold == 16
              && tr::mixture_reduction<double>{}.max_components == std::numeric_limits<std::size_t>::max());

TEST(TrackingReduceToSingle, MatchesTheMixtureMomentsByHand)
{
    // Weights 1 and 3 (unnormalised): μ = 3, P = ¼ (1 + 3²) + ¾ (2 + 1²) = 4.75
    const std::array<wg1, 2> m{c1(1, 0, 1), c1(3, 4, 2)};
    const auto g = tr::reduce_to_single<double, 1>(m);
    EXPECT_NEAR(g.mean[0], 3.0, 1e-15);
    EXPECT_NEAR(g.covariance(0, 0), 4.75, 1e-15);
    // One component is itself
    const std::array<wg1, 1> one{c1(0.2, -1.5, 0.7)};
    const auto s = tr::reduce_to_single<double, 1>(one);
    EXPECT_NEAR(s.mean[0], -1.5, 1e-15);
    EXPECT_NEAR(s.covariance(0, 0), 0.7, 1e-15);
}

TEST(TrackingReduceToSingle, MatchesExplicitMomentsOfRandomMixtures)
{
    st::wyrand g(7);
    for (int trial = 0; trial < 200; ++trial) {
        std::array<wg2, 5> m{};
        for (auto& c : m) {
            c.weight = 0.05 + st::uniform_real(g);
            c.state.mean = vec2{{10 * st::uniform_real(g) - 5, 10 * st::uniform_real(g) - 5}};
            const double a = 0.1 + st::uniform_real(g), b = 0.1 + st::uniform_real(g), r = 0.9 * (2 * st::uniform_real(g) - 1);
            c.state.covariance = mat2{{a, r * std::sqrt(a * b), r * std::sqrt(a * b), b}};
        }
        // E[x] and E[x xᵀ] - E[x] E[x]ᵀ, the second moment taken about the origin: a different order of arithmetic
        double total = 0, mx = 0, my = 0, sxx = 0, sxy = 0, syy = 0;
        for (const auto& c : m) {
            total += c.weight;
            mx += c.weight * c.state.mean[0];
            my += c.weight * c.state.mean[1];
            sxx += c.weight * (c.state.covariance(0, 0) + c.state.mean[0] * c.state.mean[0]);
            sxy += c.weight * (c.state.covariance(0, 1) + c.state.mean[0] * c.state.mean[1]);
            syy += c.weight * (c.state.covariance(1, 1) + c.state.mean[1] * c.state.mean[1]);
        }
        mx /= total, my /= total;
        const auto r = tr::reduce_to_single<double, 2>(m);
        EXPECT_NEAR(r.mean[0], mx, 1e-12);
        EXPECT_NEAR(r.mean[1], my, 1e-12);
        EXPECT_NEAR(r.covariance(0, 0), sxx / total - mx * mx, 1e-11);
        EXPECT_NEAR(r.covariance(0, 1), sxy / total - mx * my, 1e-11);
        EXPECT_NEAR(r.covariance(1, 1), syy / total - my * my, 1e-11);
        EXPECT_EQ(r.covariance(0, 1), r.covariance(1, 0));
    }
}

TEST(TrackingReduceToSingle, NoComponentsOrNoWeightIsTheZeroGaussian)
{
    EXPECT_EQ((tr::reduce_to_single<double, 2>(std::span<const wg2>{})), (tr::gaussian<double, 2>{}));
    const std::array<wg1, 2> zero{c1(0, 3, 1), c1(0, -2, 5)};
    EXPECT_EQ((tr::reduce_to_single<double, 1>(zero)), (tr::gaussian<double, 1>{}));
}

TEST(TrackingMergePair, PreservesMomentsAndCapsTheWeightAtOne)
{
    const wg1 a = c1(0.2, 0, 1), b = c1(0.3, 2, 3);
    const auto m = tr::merge_pair(a, b);
    // μ = 0.4·0 + 0.6·2 = 1.2; P = 0.4·1 + 0.6·3 + 0.4·0.6·2² = 3.16
    EXPECT_NEAR(m.weight, 0.5, 1e-15);
    EXPECT_NEAR(m.state.mean[0], 1.2, 1e-15);
    EXPECT_NEAR(m.state.covariance(0, 0), 3.16, 1e-14);
    const std::array<wg1, 2> pair{a, b};
    const auto r = tr::reduce_to_single<double, 1>(pair);
    EXPECT_NEAR(r.mean[0], m.state.mean[0], 1e-15);
    EXPECT_NEAR(r.covariance(0, 0), m.state.covariance(0, 0), 1e-14);
    // Weights that sum past 1 are capped, as Stone Soup caps them; the moments still use the true proportions
    const auto heavy = tr::merge_pair(c1(0.7, 0, 1), c1(0.6, 1, 1));
    EXPECT_EQ(heavy.weight, 1.0);
    EXPECT_NEAR(heavy.state.mean[0], 0.6 / 1.3, 1e-15);
}

TEST(TrackingRunnallsCost, MatchesTheHandValueAndIsZeroForIdenticalComponents)
{
    // Unit variances at 0 and 2, weights 1: merged variance 2, cost (2 log 2 - 0 - 0) / 2 = log 2
    const auto c = tr::runnalls_cost(c1(1, 0, 1), c1(1, 2, 1));
    ASSERT_TRUE(c);
    EXPECT_NEAR(*c, std::log(2.0), 1e-15);
    const wg2 a{0.4, {vec2{{1, 2}}, mat2{{2, 0.5, 0.5, 1}}}};
    const auto z = tr::runnalls_cost(a, a);
    ASSERT_TRUE(z);
    EXPECT_NEAR(*z, 0.0, 1e-14);
}

TEST(TrackingRunnallsCost, IsNonNegativeOnRandomPairs)
{
    st::wyrand g(11);
    for (int trial = 0; trial < 500; ++trial) {
        std::array<wg2, 2> p{};
        for (auto& c : p) {
            c.weight = 0.01 + st::uniform_real(g);
            c.state.mean = vec2{{4 * st::uniform_real(g), 4 * st::uniform_real(g)}};
            const double a = 0.1 + st::uniform_real(g), b = 0.1 + st::uniform_real(g), r = 0.95 * (2 * st::uniform_real(g) - 1);
            c.state.covariance = mat2{{a, r * std::sqrt(a * b), r * std::sqrt(a * b), b}};
        }
        const auto c = tr::runnalls_cost(p[0], p[1]);
        ASSERT_TRUE(c);
        EXPECT_GE(*c, -1e-12) << trial;
    }
}

TEST(TrackingRunnallsCost, ReportsACovarianceWithoutACholeskyFactor)
{
    const auto c = tr::runnalls_cost(c1(1, 0, 0), c1(1, 0, 0));
    ASSERT_FALSE(c);
    EXPECT_EQ(c.unwrap_err_unchecked(), la::error::not_positive_definite);
    const auto d = tr::runnalls_cost(c1(1, 0, 1), c1(1, 0, -1));
    ASSERT_FALSE(d);
}

TEST(TrackingReduceMixture, NoComponentsIsNone)
{
    EXPECT_EQ((tr::reduce_mixture<double, 1>(std::span<wg1>{})), 0u);
}

TEST(TrackingReduceMixture, PrunesAndSharesThePrunedWeightEqually)
{
    std::array<wg1, 5> m{c1(0.5, 0, 1), c1(0.01, 100, 1), c1(0.3, 200, 1), c1(0.03, 300, 1), c1(0.16, 400, 1)};
    const std::size_t n = tr::reduce_mixture<double, 1>(m, {0.05, 16, std::numeric_limits<std::size_t>::max()});
    ASSERT_EQ(n, 3u);
    // 0.04 pruned, 0.04 / 3 to each survivor, heaviest first
    EXPECT_NEAR(m[0].weight, 0.5 + 0.04 / 3, 1e-15);
    EXPECT_NEAR(m[1].weight, 0.3 + 0.04 / 3, 1e-15);
    EXPECT_NEAR(m[2].weight, 0.16 + 0.04 / 3, 1e-15);
    EXPECT_EQ(m[0].state.mean[0], 0.0);
    EXPECT_EQ(m[1].state.mean[0], 200.0);
    EXPECT_EQ(m[2].state.mean[0], 400.0);
    // A zero threshold keeps every component
    std::array<wg1, 3> keep{c1(1e-12, 0, 1), c1(0.5, 100, 1), c1(0.5, 200, 1)};
    EXPECT_EQ((tr::reduce_mixture<double, 1>(keep, {0, 0, std::numeric_limits<std::size_t>::max()})), 3u);
}

TEST(TrackingReduceMixture, SortsHeaviestFirstStablyWithoutMerging)
{
    std::array<wg1, 4> m{c1(0.1, 1, 1), c1(0.4, 2, 1), c1(0.1, 3, 1), c1(0.4, 4, 1)};
    ASSERT_EQ((tr::reduce_mixture<double, 1>(m, {0, 0, std::numeric_limits<std::size_t>::max()})), 4u);
    EXPECT_EQ(m[0].state.mean[0], 2.0);
    EXPECT_EQ(m[1].state.mean[0], 4.0);
    EXPECT_EQ(m[2].state.mean[0], 1.0);
    EXPECT_EQ(m[3].state.mean[0], 3.0);
}

TEST(TrackingReduceMixture, MeasuresTheMergeWithTheCandidatesCovariance)
{
    // Heavy, wide component at 0 (variance 100); a narrow candidate at 10 (variance 1): d² = 100 with the candidate's
    // covariance, so no merge (it would be 1 with the heavy one's)
    std::array<wg1, 2> wide{c1(0.6, 0, 100), c1(0.4, 10, 1)};
    EXPECT_EQ((tr::reduce_mixture<double, 1>(wide)), 2u);
    // The reverse: narrow heavy component, wide candidate: d² = 1, merged (it would be 100 the other way)
    std::array<wg1, 2> narrow{c1(0.6, 0, 1), c1(0.4, 10, 100)};
    ASSERT_EQ((tr::reduce_mixture<double, 1>(narrow)), 1u);
    const auto m = tr::merge_pair(c1(0.6, 0, 1), c1(0.4, 10, 100));
    EXPECT_NEAR(narrow[0].weight, 1.0, 1e-15);
    EXPECT_NEAR(narrow[0].state.mean[0], m.state.mean[0], 1e-15);
    EXPECT_NEAR(narrow[0].state.covariance(0, 0), m.state.covariance(0, 0), 1e-13);
}

TEST(TrackingReduceMixture, TestsTheLightestCandidateFirst)
{
    // From the heaviest at 0: C at 4.5 (lightest) is tested first, d² = 20.25, kept; then B at 3, d² = 9, merged
    // (the merged mean moves to 1.125, from which C would have been within 16, but C was already tested)
    std::array<wg1, 3> m{c1(0.5, 0, 1), c1(0.3, 3, 1), c1(0.2, 4.5, 1)};
    ASSERT_EQ((tr::reduce_mixture<double, 1>(m)), 2u);
    EXPECT_NEAR(m[0].weight, 0.8, 1e-15);
    EXPECT_NEAR(m[0].state.mean[0], 1.125, 1e-15);
    EXPECT_NEAR(m[1].weight, 0.2, 1e-15);
    EXPECT_EQ(m[1].state.mean[0], 4.5);
}

TEST(TrackingReduceMixture, MergingPreservesTheMixturesMoments)
{
    st::wyrand g(3);
    for (int trial = 0; trial < 100; ++trial) {
        std::array<wg2, 8> m{};
        double total = 0;
        for (auto& c : m) {
            c.weight = 0.01 + st::uniform_real(g);
            total += c.weight;
            c.state.mean = vec2{{6 * st::uniform_real(g), 6 * st::uniform_real(g)}};
            const double a = 0.5 + st::uniform_real(g), b = 0.5 + st::uniform_real(g);
            c.state.covariance = mat2{{a, 0.2, 0.2, b}};
        }
        for (auto& c : m)
            c.weight /= total * 1.0001;   // under 1 in all, so no merge is capped
        const auto before = tr::reduce_to_single<double, 2>(m);
        const std::size_t n = tr::reduce_mixture<double, 2>(m, {0, 4, std::numeric_limits<std::size_t>::max()});
        ASSERT_GE(n, 1u);
        ASSERT_LE(n, 8u);
        const auto after = tr::reduce_to_single<double, 2>(std::span<const wg2>(m).first(n));
        for (std::size_t i = 0; i < 2; ++i)
            EXPECT_NEAR(after.mean[i], before.mean[i], 1e-12);
        for (std::size_t i = 0; i < 4; ++i)
            EXPECT_NEAR(after.covariance.elements[i], before.covariance.elements[i], 1e-11);
        double w = 0;
        for (std::size_t i = 0; i < n; ++i) {
            w += m[i].weight;
            if (i > 0) {
                EXPECT_GE(m[i - 1].weight, m[i].weight);
            }
        }
        EXPECT_NEAR(w, 1 / 1.0001, 1e-12);
    }
}

TEST(TrackingReduceMixture, TruncatesToTheHeaviestMaxComponents)
{
    std::array<wg1, 3> m{c1(0.2, 0, 1), c1(0.5, 100, 1), c1(0.3, 200, 1)};
    ASSERT_EQ((tr::reduce_mixture<double, 1>(m, {0, 16, 2})), 2u);
    EXPECT_EQ(m[0].state.mean[0], 100.0);
    EXPECT_EQ(m[1].state.mean[0], 200.0);
    std::array<wg1, 2> few{c1(0.5, 0, 1), c1(0.5, 100, 1)};
    EXPECT_EQ((tr::reduce_mixture<double, 1>(few, {0, 16, 5})), 2u);
}

TEST(TrackingReduceMixture, NeverMergesAComponentWithoutACholeskyFactor)
{
    // The candidate at the heavy one's mean has zero variance: its Mahalanobis distance has no answer, so it stays
    std::array<wg1, 2> m{c1(0.6, 0, 1), c1(0.4, 0, 0)};
    ASSERT_EQ((tr::reduce_mixture<double, 1>(m)), 2u);
    EXPECT_EQ(m[1].state.covariance(0, 0), 0.0);
}

constexpr bool compile_time()
{
    std::array<wg1, 4> m{c1(0.4, 0, 1), c1(0.3, 1, 1), c1(0.2, 50, 1), c1(1e-12, 0, 1)};
    const std::size_t n = tr::reduce_mixture<double, 1>(m);
    const auto g = tr::reduce_to_single<double, 1>(std::span<const wg1>(m).first(n));
    const auto c = tr::runnalls_cost(c1(1, 0, 1), c1(1, 2, 1));
    const auto p = tr::merge_pair(c1(0.2, 0, 1), c1(0.3, 2, 3));
    return n == 2 && m[0].weight > 0.6999 && m[0].weight < 0.7001 && g.mean[0] > 0 && c.is_ok() && *c > 0.6931 && *c < 0.6932 && p.state.mean[0] > 1.1999 && p.state.mean[0] < 1.2001;
}
static_assert(compile_time());

static_assert(noexcept(tr::reduce_to_single<double, 2>(std::span<const wg2>{})));
static_assert(noexcept(tr::merge_pair(std::declval<const wg2&>(), std::declval<const wg2&>())));
static_assert(noexcept(tr::runnalls_cost(std::declval<const wg2&>(), std::declval<const wg2&>())));
static_assert(noexcept(tr::reduce_mixture<double, 2>(std::span<wg2>{}, {})));

} // namespace
