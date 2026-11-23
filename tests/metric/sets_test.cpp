//
//  sets_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/metric/sets.hpp>: OSPA and GOSPA against hand-computed small cases and against brute
//  force over every assignment for random sets of different sizes; OSPA in [0, c]; empty sets; the GOSPA
//  decomposition for α = 2 (distance^p = localisation + c^p / 2 · (missed + false)) and its distance against the
//  partial-assignment definition; the metric axioms; a large solver on the heap; invalid parameters and too_large;
//  the base metric plugged in (manhattan, chebyshev, angular, haversine) against brute force under that metric;
//  mixed point and range types; hausdorff by hand, against brute force and on empty sets; the value types' layout,
//  and all of it in a constant expression.
//

#include <snicholls/core/metric/sets.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/angular.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/geodesic.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/metric/statistical.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace st = snicholls::stats;
namespace me = snicholls::metric;

using pt = la::fixed_vector<double, 2>;
using solver = me::set_metric_solver<double, 6, 6>;

static_assert(std::is_same_v<std::underlying_type_t<me::set_metric_error>, std::uint8_t>);
static_assert(static_cast<int>(me::set_metric_error::too_large) == 1 && static_cast<int>(me::set_metric_error::invalid_parameters) == 2);
static_assert(std::is_trivially_copyable_v<me::gospa_result<double>> && std::is_standard_layout_v<me::gospa_result<double>>);
static_assert(me::gospa_result<double>{}.distance == 0 && me::gospa_result<double>{}.localisation == 0 && me::gospa_result<double>{}.missed == 0
              && me::gospa_result<double>{}.false_tracks == 0);

double dist(const pt& a, const pt& b) { return std::hypot(a[0] - b[0], a[1] - b[1]); }

// The Euclidean distance written out, the default base metric of the brute-force references
struct hand_euclidean {
    double operator()(const pt& a, const pt& b) const { return dist(a, b); }
};

// min over injections of the smaller set into the larger of Σ min(d, c)^p, by recursion
template <typename P, typename M>
double best_injection(const std::vector<P>& small, const std::vector<P>& large, double c, double p, std::size_t i, std::vector<bool>& used, const M& m)
{
    if (i == small.size())
        return 0;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < large.size(); ++j) {
        if (used[j])
            continue;
        used[j] = true;
        best = std::min(best, std::pow(std::min(static_cast<double>(m(small[i], large[j])), c), p) + best_injection(small, large, c, p, i + 1, used, m));
        used[j] = false;
    }
    return best;
}

template <typename P, typename M = hand_euclidean>
double assigned(const std::vector<P>& x, const std::vector<P>& y, double c, double p, const M& m = {})
{
    const auto& small = x.size() <= y.size() ? x : y;
    const auto& large = x.size() <= y.size() ? y : x;
    std::vector<bool> used(large.size(), false);
    return best_injection(small, large, c, p, 0, used, m);
}

template <typename P, typename M = hand_euclidean>
double brute_ospa(const std::vector<P>& x, const std::vector<P>& y, double c, double p, const M& metric = {})
{
    const std::size_t n = std::max(x.size(), y.size()), m = std::min(x.size(), y.size());
    if (n == 0)
        return 0;
    return std::pow((assigned(x, y, c, p, metric) + std::pow(c, p) * static_cast<double>(n - m)) / static_cast<double>(n), 1 / p);
}

template <typename P, typename M = hand_euclidean>
double brute_gospa(const std::vector<P>& x, const std::vector<P>& y, double c, double p, double alpha, const M& metric = {})
{
    const std::size_t n = std::max(x.size(), y.size()), m = std::min(x.size(), y.size());
    return std::pow(assigned(x, y, c, p, metric) + std::pow(c, p) / alpha * static_cast<double>(n - m), 1 / p);
}

// GOSPA with α = 2 by its own definition: the least, over partial assignments γ, of Σ_γ d^p + c^p / 2 · (unassigned)
double partial(const std::vector<pt>& x, const std::vector<pt>& y, double c, double p, std::size_t i, std::vector<bool>& used, std::size_t pairs)
{
    if (i == x.size())
        return std::pow(c, p) / 2 * static_cast<double>(x.size() + y.size() - 2 * pairs);
    double best = partial(x, y, c, p, i + 1, used, pairs);   // truth i unassigned
    for (std::size_t j = 0; j < y.size(); ++j) {
        const double d = dist(x[i], y[j]);
        if (used[j] || !(d < c))
            continue;
        used[j] = true;
        best = std::min(best, std::pow(d, p) + partial(x, y, c, p, i + 1, used, pairs + 1));
        used[j] = false;
    }
    return best;
}

std::vector<pt> random_set(st::wyrand& g, std::size_t n, double span)
{
    std::vector<pt> s(n);
    for (auto& q : s)
        q = pt{{span * st::uniform_real(g), span * st::uniform_real(g)}};
    return s;
}

TEST(MetricOspa, MatchesHandComputedCases)
{
    solver s;
    const std::array<pt, 2> truths{pt{{0, 0}}, pt{{10, 0}}};
    const std::array<pt, 1> estimate{pt{{1, 0}}};
    // m = 1, n = 2: the estimate pairs with (0, 0) at 1, plus one cardinality error of c: ((1 + 5) / 2) = 3
    const auto p1 = s.ospa(truths, estimate, 5.0);
    ASSERT_TRUE(p1);
    EXPECT_NEAR(*p1, 3.0, 1e-14);
    // p = 2: √((1 + 25) / 2) = √13
    const auto p2 = s.ospa(truths, estimate, 5.0, 2.0);
    ASSERT_TRUE(p2);
    EXPECT_NEAR(*p2, std::sqrt(13.0), 1e-14);
    // A pair beyond the cut-off costs c: one truth at 0, one estimate at 7, c = 5
    const std::array<pt, 1> far{pt{{7, 0}}};
    EXPECT_NEAR(*s.ospa(std::span<const pt>(truths).first(1), far, 5.0), 5.0, 1e-14);
    // Equal sets in a different order: 0
    const std::array<pt, 2> swapped{pt{{10, 0}}, pt{{0, 0}}};
    EXPECT_EQ(*s.ospa(truths, swapped, 5.0), 0.0);
}

TEST(MetricOspa, EmptySetsAreZeroOrTheCutOff)
{
    solver s;
    const std::array<pt, 2> some{pt{{0, 0}}, pt{{1, 1}}};
    EXPECT_EQ(*s.ospa(std::span<const pt>{}, std::span<const pt>{}, 3.0), 0.0);
    EXPECT_EQ(*s.ospa(some, std::span<const pt>{}, 3.0), 3.0);
    EXPECT_EQ(*s.ospa(std::span<const pt>{}, some, 3.0, 2.0), 3.0);
}

TEST(MetricOspa, MatchesBruteForceOnRandomSetsAndLiesInZeroToC)
{
    st::wyrand g(31);
    solver s;
    for (int trial = 0; trial < 400; ++trial) {
        const auto x = random_set(g, st::uniform_index(g, 7), 10), y = random_set(g, st::uniform_index(g, 7), 10);
        const double c = 1 + 4 * st::uniform_real(g), p = 1 + 2 * st::uniform_real(g);
        const auto o = s.ospa(x, y, c, p);
        ASSERT_TRUE(o) << trial;
        EXPECT_NEAR(*o, brute_ospa(x, y, c, p), 1e-12) << trial << ' ' << x.size() << ' ' << y.size();
        EXPECT_GE(*o, 0.0);
        EXPECT_LE(*o, c + 1e-12);
    }
}

TEST(MetricOspa, IsAMetric)
{
    st::wyrand g(37);
    solver s;
    for (int trial = 0; trial < 200; ++trial) {
        const auto x = random_set(g, st::uniform_index(g, 6), 8), y = random_set(g, st::uniform_index(g, 6), 8), z = random_set(g, st::uniform_index(g, 6), 8);
        const double xy = *s.ospa(x, y, 3.0, 2.0), yx = *s.ospa(y, x, 3.0, 2.0), xz = *s.ospa(x, z, 3.0, 2.0), zy = *s.ospa(z, y, 3.0, 2.0);
        EXPECT_NEAR(xy, yx, 1e-12) << trial;
        EXPECT_LE(xy, xz + zy + 1e-12) << trial;
        auto shuffled = x;
        st::shuffle(std::span<pt>(shuffled), g);
        EXPECT_NEAR(*s.ospa(x, shuffled, 3.0, 2.0), 0.0, 1e-12);
    }
}

TEST(MetricOspa, ReportsInvalidParametersAndTooManyPoints)
{
    solver s;
    const std::array<pt, 1> one{pt{{0, 0}}};
    for (const auto& [c, p] : {std::pair{0.0, 1.0}, std::pair{-1.0, 1.0}, std::pair{1.0, 0.5}, std::pair{std::numeric_limits<double>::quiet_NaN(), 1.0}, std::pair{1.0, std::numeric_limits<double>::quiet_NaN()}}) {
        const auto o = s.ospa(one, one, c, p);
        ASSERT_FALSE(o);
        EXPECT_EQ(o.unwrap_err_unchecked(), me::set_metric_error::invalid_parameters);
    }
    me::set_metric_solver<double, 3, 3> small;
    const std::array<pt, 4> four{};
    const auto rows = small.ospa(four, one, 1.0);
    ASSERT_FALSE(rows);
    EXPECT_EQ(rows.unwrap_err_unchecked(), me::set_metric_error::too_large);
    const auto cols = small.ospa(one, four, 1.0);
    ASSERT_FALSE(cols);
    EXPECT_EQ(cols.unwrap_err_unchecked(), me::set_metric_error::too_large);
}

TEST(MetricGospa, MatchesHandComputedCases)
{
    solver s;
    const std::array<pt, 2> truths{pt{{0, 0}}, pt{{10, 0}}};
    const std::array<pt, 1> estimate{pt{{1, 0}}};
    // One pair at 1 and one missed truth at c / 2: 1 + 5 / 2 = 3.5
    const auto g1 = s.gospa(truths, estimate, 5.0);
    ASSERT_TRUE(g1);
    EXPECT_NEAR(g1->distance, 3.5, 1e-14);
    EXPECT_NEAR(g1->localisation, 1.0, 1e-14);
    EXPECT_EQ(g1->missed, 1u);
    EXPECT_EQ(g1->false_tracks, 0u);
    // p = 2: √(1 + 25 / 2)
    const auto g2 = s.gospa(truths, estimate, 5.0, 2.0);
    ASSERT_TRUE(g2);
    EXPECT_NEAR(g2->distance, std::sqrt(13.5), 1e-14);
    // α = 1: the cardinality error costs c^p: 1 + 5
    EXPECT_NEAR(s.gospa(truths, estimate, 5.0, 1.0, 1.0)->distance, 6.0, 1e-14);
    // A pair beyond the cut-off is one missed truth and one false track: 5 / 2 · 2 = 5
    const std::array<pt, 1> far{pt{{7, 0}}};
    const auto g3 = s.gospa(std::span<const pt>(truths).first(1), far, 5.0);
    ASSERT_TRUE(g3);
    EXPECT_NEAR(g3->distance, 5.0, 1e-14);
    EXPECT_EQ(g3->localisation, 0.0);
    EXPECT_EQ(g3->missed, 1u);
    EXPECT_EQ(g3->false_tracks, 1u);
}

TEST(MetricGospa, EmptySetsCountEveryPointAsMissedOrFalse)
{
    solver s;
    const std::array<pt, 3> three{pt{{0, 0}}, pt{{1, 1}}, pt{{2, 2}}};
    const auto none = s.gospa(std::span<const pt>{}, std::span<const pt>{}, 2.0);
    ASSERT_TRUE(none);
    EXPECT_EQ(none->distance, 0.0);
    const auto missed = s.gospa(three, std::span<const pt>{}, 2.0, 2.0);
    ASSERT_TRUE(missed);
    EXPECT_NEAR(missed->distance, std::sqrt(4.0 / 2 * 3), 1e-14);
    EXPECT_EQ(missed->missed, 3u);
    EXPECT_EQ(missed->false_tracks, 0u);
    const auto false_tracks = s.gospa(std::span<const pt>{}, three, 2.0);
    ASSERT_TRUE(false_tracks);
    EXPECT_NEAR(false_tracks->distance, 3.0, 1e-14);
    EXPECT_EQ(false_tracks->missed, 0u);
    EXPECT_EQ(false_tracks->false_tracks, 3u);
}

TEST(MetricGospa, MatchesBruteForceAndDecomposesForAlphaTwo)
{
    st::wyrand g(41);
    solver s;
    for (int trial = 0; trial < 400; ++trial) {
        const auto x = random_set(g, st::uniform_index(g, 7), 10), y = random_set(g, st::uniform_index(g, 7), 10);
        const double c = 1 + 4 * st::uniform_real(g), p = 1 + 2 * st::uniform_real(g), alpha = 0.25 + 1.75 * st::uniform_real(g);
        const auto ga = s.gospa(x, y, c, p, alpha);
        ASSERT_TRUE(ga) << trial;
        EXPECT_NEAR(ga->distance, brute_gospa(x, y, c, p, alpha), 1e-12) << trial;
        const auto g2 = s.gospa(x, y, c, p);
        ASSERT_TRUE(g2);
        std::vector<bool> used(y.size(), false);
        const double def = partial(x, y, c, p, 0, used, 0);
        EXPECT_NEAR(std::pow(g2->distance, p), def, 1e-10 * (1 + def)) << trial;
        // distance^p = localisation + c^p / 2 · (missed + false), and the counts are consistent with the sizes
        const double cp = std::pow(c, p);
        EXPECT_NEAR(std::pow(g2->distance, p), g2->localisation + cp / 2 * static_cast<double>(g2->missed + g2->false_tracks), 1e-10 * (1 + def)) << trial;
        EXPECT_EQ(x.size() - g2->missed, y.size() - g2->false_tracks);
        EXPECT_LE(g2->missed, x.size());
        EXPECT_LE(g2->false_tracks, y.size());
    }
}

TEST(MetricGospa, ReportsInvalidParametersAndTooManyPoints)
{
    solver s;
    const std::array<pt, 1> one{pt{{0, 0}}};
    for (const auto& [c, p, a] : {std::array{0.0, 1.0, 2.0}, std::array{1.0, 0.9, 2.0}, std::array{1.0, 1.0, 0.0}, std::array{1.0, 1.0, -1.0}, std::array{1.0, 1.0, 2.5},
                                  std::array{1.0, 1.0, std::numeric_limits<double>::quiet_NaN()}}) {
        const auto r = s.gospa(one, one, c, p, a);
        ASSERT_FALSE(r);
        EXPECT_EQ(r.unwrap_err_unchecked(), me::set_metric_error::invalid_parameters);
    }
    EXPECT_TRUE(s.gospa(one, one, 1.0, 1.0, 2.0));   // α = 2 is inside (0, 2]
    me::set_metric_solver<double, 3, 3> small;
    const std::array<pt, 4> four{};
    const auto rows = small.gospa(four, one, 1.0);
    ASSERT_FALSE(rows);
    EXPECT_EQ(rows.unwrap_err_unchecked(), me::set_metric_error::too_large);
    const auto cols = small.gospa(one, four, 1.0);
    ASSERT_FALSE(cols);
    EXPECT_EQ(cols.unwrap_err_unchecked(), me::set_metric_error::too_large);
}

TEST(MetricSetMetricSolver, ALargeSolverOnTheHeapScoresAGrid)
{
    // 50 truths on a grid of spacing 10, the estimates each within 0.5 of one, in reverse order, plus 5 false
    // tracks far away. With c = 3 the identity pairing is optimal: OSPA^p = (Σ dᵢ^p + 5 c^p) / 55
    auto s = std::make_unique<me::set_metric_solver<double, 64, 64>>();
    st::wyrand g(43);
    std::vector<pt> truths, estimates;
    double sum = 0;
    for (int i = 0; i < 50; ++i) {
        truths.push_back(pt{{10.0 * (i % 10), 10.0 * (i / 10)}});
        const pt e{{truths.back()[0] + 0.5 * st::uniform_real(g) - 0.25, truths.back()[1] + 0.5 * st::uniform_real(g) - 0.25}};
        estimates.push_back(e);
        sum += std::pow(dist(truths.back(), e), 2);
    }
    std::reverse(estimates.begin(), estimates.end());
    for (int i = 0; i < 5; ++i)
        estimates.push_back(pt{{500.0 + 10 * i, 500.0}});
    const auto o = s->ospa(truths, estimates, 3.0, 2.0);
    ASSERT_TRUE(o);
    EXPECT_NEAR(*o, std::sqrt((sum + 5 * 9.0) / 55), 1e-12);
    const auto gs = s->gospa(truths, estimates, 3.0, 2.0);
    ASSERT_TRUE(gs);
    EXPECT_NEAR(gs->localisation, sum, 1e-12);
    EXPECT_EQ(gs->missed, 0u);
    EXPECT_EQ(gs->false_tracks, 5u);
}

constexpr bool compile_time()
{
    me::set_metric_solver<double, 3, 3> s;
    const std::array<pt, 2> truths{pt{{0, 0}}, pt{{10, 0}}};
    const std::array<pt, 1> estimate{pt{{1, 0}}};
    const auto o = s.ospa(truths, estimate, 5.0);
    const auto g = s.gospa(truths, estimate, 5.0);
    const auto bad = s.ospa(truths, estimate, 0.0);
    return o.is_ok() && *o > 2.9999 && *o < 3.0001 && g.is_ok() && g->distance > 3.4999 && g->distance < 3.5001 && g->missed == 1 && g->false_tracks == 0 && bad.is_err();
}
static_assert(compile_time());

constexpr bool hausdorff_compile_time()
{
    const std::array<pt, 2> x{pt{{0, 0}}, pt{{1, 0}}};
    const std::array<pt, 2> y{pt{{0, 0}}, pt{{5, 0}}};
    const double h = me::hausdorff(x, y), m = me::hausdorff(x, y, me::manhattan{});
    return h > 3.9999 && h < 4.0001 && m > 3.9999 && m < 4.0001 && me::hausdorff(std::span<const pt>{}, std::span<const pt>{}) == 0
           && me::hausdorff(x, std::span<const pt>{}) == std::numeric_limits<double>::infinity();
}
static_assert(hausdorff_compile_time());

static_assert(me::point_range<std::vector<std::array<double, 2>>> && me::point_range<std::array<pt, 3>> && me::point_range<std::span<const pt>>);
static_assert(!me::point_range<std::vector<char>> && !me::point_range<int>);

using ll = std::array<double, 2>;   // (latitude, longitude), radians
constexpr ll degrees(double lat, double lon) { return {lat * std::numbers::pi / 180, lon * std::numbers::pi / 180}; }

// Hausdorff by its definition, over every pair
template <typename P, typename M = hand_euclidean>
double brute_hausdorff(const std::vector<P>& x, const std::vector<P>& y, const M& m = {})
{
    double worst = 0;
    for (const auto& a : x) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& b : y)
            best = std::min(best, static_cast<double>(m(a, b)));
        worst = std::max(worst, best);
    }
    for (const auto& b : y) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& a : x)
            best = std::min(best, static_cast<double>(m(a, b)));
        worst = std::max(worst, best);
    }
    return worst;
}

TEST(MetricSetBaseMetric, ManhattanAndChebyshevPlugIn)
{
    solver s;
    const std::array<pt, 1> truth{pt{{0, 0}}};
    const std::array<pt, 1> estimate{pt{{3, 4}}};
    EXPECT_NEAR(*s.ospa(truth, estimate, 10.0), 5.0, 1e-14);                        // euclidean by default
    EXPECT_NEAR(*s.ospa(truth, estimate, 10.0, 1.0, me::manhattan{}), 7.0, 1e-14);
    EXPECT_NEAR(*s.ospa(truth, estimate, 10.0, 1.0, me::chebyshev{}), 4.0, 1e-14);
    EXPECT_NEAR(*s.ospa(truth, estimate, 6.0, 1.0, me::manhattan{}), 6.0, 1e-14);   // over the cut-off under manhattan only
    const auto g = s.gospa(truth, estimate, 6.0, 1.0, 2.0, me::manhattan{});
    ASSERT_TRUE(g);
    EXPECT_EQ(g->missed, 1u);
    EXPECT_EQ(g->false_tracks, 1u);
    EXPECT_NEAR(s.gospa(truth, estimate, 6.0, 1.0, 2.0, me::chebyshev{})->localisation, 4.0, 1e-14);
    st::wyrand r(47);
    for (int trial = 0; trial < 200; ++trial) {
        const auto x = random_set(r, st::uniform_index(r, 6), 10), y = random_set(r, st::uniform_index(r, 6), 10);
        const double c = 1 + 4 * st::uniform_real(r), p = 1 + 2 * st::uniform_real(r);
        EXPECT_NEAR(*s.ospa(x, y, c, p, me::manhattan{}), brute_ospa(x, y, c, p, me::manhattan{}), 1e-12) << trial;
        EXPECT_NEAR(*s.ospa(x, y, c, p, me::chebyshev{}), brute_ospa(x, y, c, p, me::chebyshev{}), 1e-12) << trial;
        EXPECT_NEAR(s.gospa(x, y, c, p, 1.0, me::manhattan{})->distance, brute_gospa(x, y, c, p, 1.0, me::manhattan{}), 1e-12) << trial;
        EXPECT_NEAR(s.gospa(x, y, c, p, 2.0, me::chebyshev{})->distance, brute_gospa(x, y, c, p, 2.0, me::chebyshev{}), 1e-12) << trial;
    }
}

TEST(MetricSetBaseMetric, AngularComparesDirectionsOnly)
{
    solver s;
    const std::array<pt, 2> truths{pt{{1, 0}}, pt{{0, 1}}};
    const std::array<pt, 2> scaled{pt{{0, 7}}, pt{{3, 0}}};    // the same directions, other lengths, other order
    const std::array<pt, 1> diagonal{pt{{2, 2}}};
    EXPECT_NEAR(*s.ospa(truths, scaled, 1.0, 1.0, me::angular{}), 0.0, 1e-7);
    EXPECT_GT(*s.ospa(truths, scaled, 1.0), 0.9);   // but far apart in Euclidean terms
    // (1, 0) and (1, 1): angular distance √(2 - √2); the other truth is a cardinality error of c = 2
    const double chord = std::sqrt(2 - std::sqrt(2.0));
    EXPECT_NEAR(*s.ospa(truths, diagonal, 2.0, 1.0, me::angular{}), (chord + 2) / 2, 1e-12);
    EXPECT_NEAR(me::hausdorff(truths, diagonal, me::angular{}), chord, 1e-12);
}

TEST(MetricSetBaseMetric, HaversineScoresPositionsOnTheEarth)
{
    solver s;
    // 343556.535 m London-Paris and 5570229.874 m London-New York, from Python 3 / numpy (see geodesic_test.cpp)
    const std::vector<ll> truths{degrees(51.5074, -0.1278), degrees(40.7128, -74.0060)};
    const std::vector<ll> estimates{degrees(48.8566, 2.3522)};
    const double c = 1e6;
    const auto o = s.ospa(truths, estimates, c, 1.0, me::haversine{});
    ASSERT_TRUE(o);
    EXPECT_NEAR(*o, (343556.53488088306 + c) / 2, 1e-3);
    const auto g = s.gospa(truths, estimates, c, 1.0, 2.0, me::haversine{});
    ASSERT_TRUE(g);
    EXPECT_NEAR(g->localisation, 343556.53488088306, 1e-3);
    EXPECT_EQ(g->missed, 1u);
    EXPECT_EQ(g->false_tracks, 0u);
    EXPECT_NEAR(me::hausdorff(truths, estimates, me::haversine{}), 5837248.966566375, 1e-3);   // New York to Paris (numpy), the worst miss
    EXPECT_NEAR(me::hausdorff(truths, estimates, me::haversine{}), me::haversine{}(truths[1], estimates[0]), 1e-9);
    // On a unit sphere the same scene is in radians
    EXPECT_NEAR(*s.ospa(truths, estimates, 1.0, 1.0, me::haversine{1.0}), (343556.53488088306 / 6371008.8 + 1) / 2, 1e-9);
}

TEST(MetricSetBaseMetric, MahalanobisPlugsIn)
{
    solver s;
    const auto m = me::make_mahalanobis(la::fixed_matrix<double, 2, 2>{{100, 0, 0, 1}}).unwrap();
    const std::array<pt, 1> truth{pt{{0, 0}}};
    const std::array<pt, 1> estimate{pt{{20, 0}}};
    EXPECT_NEAR(*s.ospa(truth, estimate, 3.0, 1.0, m), 2.0, 1e-12);   // 2 sigma along x, inside the cut-off
    EXPECT_NEAR(*s.ospa(truth, estimate, 3.0), 3.0, 1e-12);            // 20 m, beyond it
}

TEST(MetricSetRanges, MixesPointAndRangeTypes)
{
    solver s;
    st::wyrand g(53);
    for (int trial = 0; trial < 100; ++trial) {
        const auto x = random_set(g, 1 + st::uniform_index(g, 5), 10);
        std::vector<std::array<double, 2>> xa;
        for (const auto& q : x)
            xa.push_back({q[0], q[1]});
        const std::array<pt, 3> y{pt{{2 * st::uniform_real(g), 3.0}}, pt{{5.0, 5 * st::uniform_real(g)}}, pt{{9.0, 1.0}}};
        const std::vector<pt> yv(y.begin(), y.end());
        const double c = 4, p = 2;
        EXPECT_NEAR(*s.ospa(xa, y, c, p), *s.ospa(x, yv, c, p), 1e-14) << trial;
        EXPECT_NEAR(*s.ospa(y, xa, c, p), *s.ospa(x, yv, c, p), 1e-12) << trial;
        const auto ga = s.gospa(xa, y, c, p), gp = s.gospa(x, yv, c, p);
        ASSERT_TRUE(ga && gp);
        EXPECT_NEAR(ga->distance, gp->distance, 1e-14);
        EXPECT_EQ(ga->missed, gp->missed);
        EXPECT_EQ(ga->false_tracks, gp->false_tracks);
        EXPECT_EQ(me::hausdorff(xa, y), me::hausdorff(x, yv));
        EXPECT_EQ(me::hausdorff(xa, std::span<const pt>(y)), me::hausdorff(x, yv));
    }
}

TEST(MetricHausdorff, MatchesHandComputedCases)
{
    const std::array<pt, 2> x{pt{{0, 0}}, pt{{1, 0}}};
    const std::array<pt, 2> y{pt{{0, 0}}, pt{{5, 0}}};
    // sup over x of the nearest in y is 1; sup over y of the nearest in x is 4 (from (5, 0) to (1, 0))
    EXPECT_NEAR(me::hausdorff(x, y), 4.0, 1e-14);
    EXPECT_NEAR(me::hausdorff(y, x), 4.0, 1e-14);
    const std::array<pt, 1> one{pt{{3, 4}}};
    const std::array<pt, 1> origin{pt{{0, 0}}};
    EXPECT_NEAR(me::hausdorff(one, origin), 5.0, 1e-14);
    EXPECT_NEAR(me::hausdorff(one, origin, me::manhattan{}), 7.0, 1e-14);
    EXPECT_NEAR(me::hausdorff(one, origin, me::chebyshev{}), 4.0, 1e-14);
    const std::array<pt, 2> swapped{pt{{1, 0}}, pt{{0, 0}}};
    EXPECT_EQ(me::hausdorff(x, swapped), 0.0);
}

TEST(MetricHausdorff, EmptySetsAreZeroOrInfinity)
{
    const std::array<pt, 2> some{pt{{0, 0}}, pt{{1, 1}}};
    EXPECT_EQ(me::hausdorff(std::span<const pt>{}, std::span<const pt>{}), 0.0);
    EXPECT_EQ(me::hausdorff(some, std::span<const pt>{}), std::numeric_limits<double>::infinity());
    EXPECT_EQ(me::hausdorff(std::vector<std::array<double, 2>>{}, some), std::numeric_limits<double>::infinity());
}

TEST(MetricHausdorff, MatchesBruteForceAndIsAMetric)
{
    st::wyrand g(59);
    for (int trial = 0; trial < 300; ++trial) {
        const auto x = random_set(g, 1 + st::uniform_index(g, 8), 10), y = random_set(g, 1 + st::uniform_index(g, 8), 10),
                   z = random_set(g, 1 + st::uniform_index(g, 8), 10);
        const double xy = me::hausdorff(x, y);
        EXPECT_NEAR(xy, brute_hausdorff(x, y), 1e-14) << trial;
        EXPECT_NEAR(me::hausdorff(x, y, me::manhattan{}), brute_hausdorff(x, y, me::manhattan{}), 1e-14) << trial;
        EXPECT_EQ(xy, me::hausdorff(y, x)) << trial;
        EXPECT_LE(xy, me::hausdorff(x, z) + me::hausdorff(z, y) + 1e-12) << trial;
        auto shuffled = x;
        st::shuffle(std::span<pt>(shuffled), g);
        EXPECT_EQ(me::hausdorff(x, shuffled), 0.0);
    }
}

} // namespace

// ospa and gospa are noexcept for a noexcept metric over double, as the header documents
static_assert(noexcept(std::declval<snicholls::metric::set_metric_solver<double, 4, 4>&>().ospa(std::declval<const std::array<std::array<double, 2>, 2>&>(), std::declval<const std::array<std::array<double, 2>, 2>&>(), 1.0)));
static_assert(noexcept(std::declval<snicholls::metric::set_metric_solver<double, 4, 4>&>().gospa(std::declval<const std::array<std::array<double, 2>, 2>&>(), std::declval<const std::array<std::array<double, 2>, 2>&>(), 1.0)));
