//
//  fusion_test.cpp
//  snicholls-core
//
//  Tests for <snicholls/core/tracking/fusion.hpp>: covariance intersection against its information-form definition
//  for two and for n estimates, ω at its bounds, every error; optimal_omega against a fine grid for the trace and for
//  the determinant, its bounds and its error; fast_ci_weights proportional to 1 / tr P, clamped to its bounds and
//  renormalised, and each false return; the criterion's layout, and all of it in a constant expression.
//

#include <snicholls/core/tracking/fusion.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

namespace {

namespace la = snicholls::linalg;
namespace st = snicholls::stats;
namespace tr = snicholls::tracking;

using vec2 = la::fixed_vector<double, 2>;
using mat2 = la::fixed_matrix<double, 2, 2>;
using g2 = tr::gaussian<double, 2>;

static_assert(std::is_same_v<std::underlying_type_t<tr::ci_criterion>, std::uint8_t>);
static_assert(static_cast<int>(tr::ci_criterion::trace) == 0 && static_cast<int>(tr::ci_criterion::determinant) == 1);

// Two estimates elongated in different directions: neither dominates
constexpr g2 a_est() { return {vec2{{1, 0}}, mat2{{4, 0.5, 0.5, 1}}}; }
constexpr g2 b_est() { return {vec2{{0, 1}}, mat2{{1, -0.3, -0.3, 3}}}; }

// The fused information form: Y = Σ ωᵢ Pᵢ⁻¹, y = Σ ωᵢ Pᵢ⁻¹ xᵢ, by the general inverse (not the Cholesky the header uses)
g2 by_definition(std::span<const g2> e, std::span<const double> w)
{
    mat2 info{};
    vec2 vec{};
    for (std::size_t i = 0; i < e.size(); ++i) {
        const auto y = *la::inverse(e[i].covariance);
        info += y * w[i];
        vec += y * e[i].mean * w[i];
    }
    const auto p = *la::inverse(info);
    return {p * vec, p};
}

void expect_near(const g2& x, const g2& y, double tol)
{
    for (std::size_t i = 0; i < 2; ++i)
        EXPECT_NEAR(x.mean[i], y.mean[i], tol) << i;
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(x.covariance.elements[i], y.covariance.elements[i], tol) << i;
}

TEST(TrackingCovarianceIntersection, TwoEstimatesMatchTheInformationFormDefinition)
{
    for (const double omega : {0.1, 0.35, 0.5, 0.8}) {
        const auto f = tr::covariance_intersection(a_est(), b_est(), omega);
        ASSERT_TRUE(f) << omega;
        const std::array<g2, 2> e{a_est(), b_est()};
        const std::array<double, 2> w{omega, 1 - omega};
        expect_near(*f, by_definition(e, w), 1e-13);
        // Consistency: P_f⁻¹ = ω P_a⁻¹ + (1 - ω) P_b⁻¹
        const auto lhs = *la::inverse(f->covariance);
        const auto rhs = *la::inverse(a_est().covariance) * omega + *la::inverse(b_est().covariance) * (1 - omega);
        for (std::size_t i = 0; i < 4; ++i)
            EXPECT_NEAR(lhs.elements[i], rhs.elements[i], 1e-13);
        EXPECT_EQ(f->covariance(0, 1), f->covariance(1, 0));
    }
}

TEST(TrackingCovarianceIntersection, OmegaAtItsBoundsReturnsOneEstimate)
{
    const auto one = tr::covariance_intersection(a_est(), b_est(), 1.0);
    ASSERT_TRUE(one);
    expect_near(*one, a_est(), 1e-14);
    const auto zero = tr::covariance_intersection(a_est(), b_est(), 0.0);
    ASSERT_TRUE(zero);
    expect_near(*zero, b_est(), 1e-14);
    // An estimate with weight 0 is not read, even when its covariance is singular
    const g2 singular{vec2{{9, 9}}, mat2{}};
    const auto skip = tr::covariance_intersection(a_est(), singular, 1.0);
    ASSERT_TRUE(skip);
    expect_near(*skip, a_est(), 1e-14);
}

TEST(TrackingCovarianceIntersection, NEstimatesMatchTheDefinition)
{
    const std::array<g2, 3> e{a_est(), b_est(), g2{vec2{{-1, -1}}, mat2{{2, 0, 0, 2}}}};
    const std::array<double, 3> w{0.2, 0.5, 0.3};
    const auto f = tr::covariance_intersection<double, 2>(e, w);
    ASSERT_TRUE(f);
    expect_near(*f, by_definition(e, w), 1e-13);
}

TEST(TrackingCovarianceIntersection, ReportsEmptyMismatchedAndNonPositiveDefiniteInputs)
{
    const std::array<g2, 2> e{a_est(), b_est()};
    const auto empty = tr::covariance_intersection<double, 2>(std::span<const g2>{}, std::span<const double>{});
    ASSERT_FALSE(empty);
    EXPECT_EQ(empty.unwrap_err_unchecked(), la::error::dimension_mismatch);
    const auto mismatch = tr::covariance_intersection<double, 2>(e, std::array<double, 1>{1.0});
    ASSERT_FALSE(mismatch);
    EXPECT_EQ(mismatch.unwrap_err_unchecked(), la::error::dimension_mismatch);
    const g2 bad{vec2{}, mat2{{1, 2, 2, 1}}};
    const auto indefinite = tr::covariance_intersection(a_est(), bad, 0.5);
    ASSERT_FALSE(indefinite);
    EXPECT_EQ(indefinite.unwrap_err_unchecked(), la::error::not_positive_definite);
    // Every weight zero: no information, so the fused information has no Cholesky factor
    const auto none = tr::covariance_intersection<double, 2>(e, std::array<double, 2>{0.0, 0.0});
    ASSERT_FALSE(none);
    EXPECT_EQ(none.unwrap_err_unchecked(), la::error::not_positive_definite);
}

double fused_trace(double w) { return la::trace(tr::covariance_intersection(a_est(), b_est(), w)->covariance); }
double fused_det(double w) { return la::determinant(tr::covariance_intersection(a_est(), b_est(), w)->covariance); }

// The best of 100001 grid points in [lo, hi]
template <typename F>
double grid_min(F f, double lo, double hi)
{
    double best = lo, fb = f(lo);
    for (int i = 1; i <= 100000; ++i) {
        const double w = lo + (hi - lo) * i / 100000.0;
        if (f(w) < fb)
            best = w, fb = f(w);
    }
    return best;
}

TEST(TrackingOptimalOmega, MinimisesTheTraceAgainstAFineGrid)
{
    const auto w = tr::optimal_omega(a_est(), b_est());
    ASSERT_TRUE(w);
    const double g = grid_min(fused_trace, 0, 1);
    EXPECT_GT(*w, 0.0);
    EXPECT_LT(*w, 1.0);
    EXPECT_NEAR(*w, g, 1e-4);
    EXPECT_LE(fused_trace(*w), fused_trace(g) + 1e-12);
}

TEST(TrackingOptimalOmega, MinimisesTheDeterminantAgainstAFineGrid)
{
    const auto w = tr::optimal_omega(a_est(), b_est(), 0.0, 1.0, tr::ci_criterion::determinant);
    ASSERT_TRUE(w);
    const double g = grid_min(fused_det, 0, 1);
    EXPECT_NEAR(*w, g, 1e-4);
    EXPECT_LE(fused_det(*w), fused_det(g) + 1e-12);
}

TEST(TrackingOptimalOmega, StaysInsideItsBounds)
{
    const double free = *tr::optimal_omega(a_est(), b_est());
    // A bound below the free optimum holds ω at the bound; one around it changes nothing
    const auto held = tr::optimal_omega(a_est(), b_est(), 0.0, free / 2);
    ASSERT_TRUE(held);
    EXPECT_NEAR(*held, free / 2, 1e-9);
    EXPECT_LE(*held, free / 2);
    const auto raised = tr::optimal_omega(a_est(), b_est(), (1 + free) / 2, 1.0);
    ASSERT_TRUE(raised);
    EXPECT_NEAR(*raised, (1 + free) / 2, 1e-9);
    EXPECT_GE(*raised, (1 + free) / 2);
    const auto around = tr::optimal_omega(a_est(), b_est(), free / 2, (1 + free) / 2);
    ASSERT_TRUE(around);
    EXPECT_NEAR(*around, free, 1e-7);
}

TEST(TrackingOptimalOmega, ADominatingEstimateGetsAllTheWeight)
{
    // a's covariance is inside b's in every direction: the trace is least at ω = 1 (to rounding, at the bound)
    const g2 tight{vec2{}, mat2{{0.5, 0, 0, 0.5}}};
    const g2 loose{vec2{{1, 1}}, mat2{{5, 1, 1, 4}}};
    const auto w = tr::optimal_omega(tight, loose);
    ASSERT_TRUE(w);
    EXPECT_NEAR(*w, 1.0, 1e-12);
    EXPECT_LE(*w, 1.0);
    const auto v = tr::optimal_omega(loose, tight);
    ASSERT_TRUE(v);
    EXPECT_NEAR(*v, 0.0, 1e-12);
    EXPECT_GE(*v, 0.0);
}

TEST(TrackingOptimalOmega, MatchesTheGridOnRandomPairs)
{
    st::wyrand g(5);
    auto random_estimate = [&] {
        const double a = 0.2 + 5 * st::uniform_real(g), b = 0.2 + 5 * st::uniform_real(g), r = 0.9 * (2 * st::uniform_real(g) - 1);
        return g2{vec2{{st::uniform_real(g), st::uniform_real(g)}}, mat2{{a, r * std::sqrt(a * b), r * std::sqrt(a * b), b}}};
    };
    for (int trial = 0; trial < 30; ++trial) {
        const g2 a = random_estimate(), b = random_estimate();
        const auto w = tr::optimal_omega(a, b);
        ASSERT_TRUE(w);
        auto t = [&](double x) { return la::trace(tr::covariance_intersection(a, b, x)->covariance); };
        EXPECT_LE(t(*w), t(grid_min(t, 0, 1)) + 1e-10) << trial;
    }
}

TEST(TrackingOptimalOmega, ReportsANonPositiveDefiniteCovariance)
{
    const g2 bad{vec2{}, mat2{{1, 2, 2, 1}}};
    const auto w = tr::optimal_omega(a_est(), bad);
    ASSERT_FALSE(w);
    EXPECT_EQ(w.unwrap_err_unchecked(), la::error::not_positive_definite);
    EXPECT_FALSE(tr::optimal_omega(bad, a_est()));
}

std::array<g2, 3> by_trace()
{
    // Traces 1, 2 and 4
    return {g2{vec2{}, mat2{{0.5, 0, 0, 0.5}}}, g2{vec2{}, mat2{{1.5, 0.2, 0.2, 0.5}}}, g2{vec2{}, mat2{{2, 0, 0, 2}}}};
}

TEST(TrackingFastCiWeights, AreProportionalToTheInverseTrace)
{
    std::array<double, 3> w{};
    ASSERT_TRUE((tr::fast_ci_weights<double, 2>(by_trace(), w)));
    EXPECT_NEAR(w[0], 4.0 / 7, 1e-15);
    EXPECT_NEAR(w[1], 2.0 / 7, 1e-15);
    EXPECT_NEAR(w[2], 1.0 / 7, 1e-15);
    const std::array<g2, 1> one{a_est()};
    std::array<double, 1> w1{};
    ASSERT_TRUE((tr::fast_ci_weights<double, 2>(one, w1)));
    EXPECT_EQ(w1[0], 1.0);
}

TEST(TrackingFastCiWeights, RespectTheirBoundsAndStillSumToOne)
{
    // At most 0.5 for the first: it takes 0.5 and the other two share 0.5 in the ratio 2 : 1
    const std::array<double, 3> hi{0.5, 1, 1};
    std::array<double, 3> w{};
    ASSERT_TRUE((tr::fast_ci_weights<double, 2>(by_trace(), w, {}, hi)));
    EXPECT_NEAR(w[0], 0.5, 1e-15);
    EXPECT_NEAR(w[1], 1.0 / 3, 1e-15);
    EXPECT_NEAR(w[2], 1.0 / 6, 1e-15);
    // At least 0.3 for the third: the other two share 0.7 in the ratio 2 : 1
    const std::array<double, 3> lo{0, 0, 0.3};
    ASSERT_TRUE((tr::fast_ci_weights<double, 2>(by_trace(), w, lo)));
    EXPECT_NEAR(w[0], 0.7 * 2 / 3, 1e-15);
    EXPECT_NEAR(w[1], 0.7 / 3, 1e-15);
    EXPECT_NEAR(w[2], 0.3, 1e-15);
    // Random bounds that admit a solution: within them, and summing to 1
    st::wyrand g(9);
    for (int trial = 0; trial < 200; ++trial) {
        std::array<double, 3> l{}, h{};
        for (std::size_t i = 0; i < 3; ++i) {
            l[i] = 0.3 * st::uniform_real(g);
            h[i] = l[i] + 0.1 + 0.6 * st::uniform_real(g);
        }
        if (l[0] + l[1] + l[2] > 1 || h[0] + h[1] + h[2] < 1)
            continue;
        ASSERT_TRUE((tr::fast_ci_weights<double, 2>(by_trace(), w, l, h)));
        double s = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_GE(w[i], l[i] - 1e-15) << trial;
            EXPECT_LE(w[i], h[i] + 1e-15) << trial;
            s += w[i];
        }
        EXPECT_NEAR(s, 1.0, 1e-12) << trial;
    }
}

TEST(TrackingFastCiWeights, ReleaseALowerBoundWhenUpperBoundsTakeWeight)
{
    // Proportional (4, 2, 1) / 7 with the first held to 0.5, the second to 0.3 and the third at least 0.15: the
    // clamped weights are clamp(λ rᵢ) for the λ that sums them to 1, so the third rises off its lower bound to 0.2
    // (clamping the third to 0.15 first and never releasing it would leave a sum of 0.95)
    const std::array<double, 3> lo{0, 0, 0.15}, hi{0.5, 0.3, 1};
    std::array<double, 3> w{};
    ASSERT_TRUE((tr::fast_ci_weights<double, 2>(by_trace(), w, lo, hi)));
    EXPECT_NEAR(w[0], 0.5, 1e-15);
    EXPECT_NEAR(w[1], 0.3, 1e-15);
    EXPECT_NEAR(w[2], 0.2, 1e-15);
}

TEST(TrackingFastCiWeights, ZeroTracesGiveEqualWeights)
{
    const std::array<g2, 2> flat{g2{}, g2{}};
    std::array<double, 2> w{};
    ASSERT_TRUE((tr::fast_ci_weights<double, 2>(flat, w)));
    EXPECT_EQ(w[0], 0.5);
    EXPECT_EQ(w[1], 0.5);
}

TEST(TrackingFastCiWeights, RefuseBadSpansAndInfeasibleBounds)
{
    std::array<double, 3> w{};
    std::array<double, 2> w2{};
    EXPECT_FALSE((tr::fast_ci_weights<double, 2>(std::span<const g2>{}, std::span<double>{})));
    EXPECT_FALSE((tr::fast_ci_weights<double, 2>(by_trace(), w2)));
    EXPECT_FALSE((tr::fast_ci_weights<double, 2>(by_trace(), w, std::array<double, 2>{0, 0})));
    EXPECT_FALSE((tr::fast_ci_weights<double, 2>(by_trace(), w, {}, std::array<double, 2>{1, 1})));
    EXPECT_FALSE((tr::fast_ci_weights<double, 2>(by_trace(), w, std::array<double, 3>{0.5, 0.5, 0.5})));   // lower bounds sum past 1
    EXPECT_FALSE((tr::fast_ci_weights<double, 2>(by_trace(), w, {}, std::array<double, 3>{0.3, 0.3, 0.3})));  // upper bounds short of 1
}

constexpr bool compile_time()
{
    const auto f = tr::covariance_intersection(a_est(), b_est(), 0.4);
    const auto w = tr::optimal_omega(a_est(), b_est());
    const std::array<g2, 3> e{g2{vec2{}, mat2{{0.5, 0, 0, 0.5}}}, g2{vec2{}, mat2{{1, 0, 0, 1}}}, g2{vec2{}, mat2{{2, 0, 0, 2}}}};
    std::array<double, 3> fw{};
    const bool fast = tr::fast_ci_weights<double, 2>(e, fw);
    return f.is_ok() && f->covariance(0, 0) > 0 && w.is_ok() && *w > 0 && *w < 1 && fast && fw[0] > 0.5713 && fw[0] < 0.5715;
}
static_assert(compile_time());

static_assert(noexcept(tr::covariance_intersection(std::declval<const g2&>(), std::declval<const g2&>(), 0.5)));
static_assert(noexcept(tr::covariance_intersection<double, 2>(std::span<const g2>{}, std::span<const double>{})));
static_assert(noexcept(tr::optimal_omega(std::declval<const g2&>(), std::declval<const g2&>())));
static_assert(noexcept(tr::fast_ci_weights<double, 2>(std::span<const g2>{}, std::span<double>{})));

} // namespace
