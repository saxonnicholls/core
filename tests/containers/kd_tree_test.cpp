//
//  kd_tree_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/kd_tree.hpp>: every query (nearest, k-nearest, radius, box) checked against a
//  linear scan over many random point sets in 1 to 5 dimensions, on an integer grid so that duplicates, ties and
//  points exactly on the radius or the box are common, under every decomposable metric (euclidean, manhattan,
//  chebyshev, minkowski<3>, weighted_minkowski, squared_euclidean); float points with double queries, a point type of
//  the test's own through point_traits, the early exit at the current bound, empty and one-point trees, all points
//  identical, a prefix index span, k beyond the size, a large tree, and the constexpr and noexcept claims.
//

#include <snicholls/core/containers/kd_tree.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace kd_tree_test {

// A target of the test's own, read through point_traits; reads count, to observe the early exit
struct target {
    float east = 0, north = 0;
    int id = 0;
};

// A point of eight coordinates whose reads are counted per axis
struct counted {
    std::array<double, 8> c{};
};
inline std::array<std::size_t, 8> counted_reads{};

} // namespace kd_tree_test

template <>
struct snicholls::metric::point_traits<kd_tree_test::target> {
    using value_type = float;
    static constexpr std::size_t extent = 2;
    static constexpr std::size_t dimension(const kd_tree_test::target&) noexcept { return 2; }
    static constexpr float at(const kd_tree_test::target& t, std::size_t axis) noexcept { return axis == 0 ? t.east : t.north; }
};

template <>
struct snicholls::metric::point_traits<kd_tree_test::counted> {
    using value_type = double;
    static constexpr std::size_t extent = 8;
    static std::size_t dimension(const kd_tree_test::counted&) noexcept { return 8; }
    static double at(const kd_tree_test::counted& p, std::size_t axis) noexcept
    {
        ++kd_tree_test::counted_reads[axis];
        return p.c[axis];
    }
};

namespace {

namespace ct = snicholls::containers;
namespace la = snicholls::linalg;
namespace mt = snicholls::metric;
namespace st = snicholls::stats;
using kd_tree_test::counted;
using kd_tree_test::target;

template <std::size_t K>
using point = std::array<double, K>;

// Coordinates on a small integer grid: duplicates and exact ties are frequent
template <std::size_t K>
point<K> grid_point(st::wyrand& g, std::uint64_t side)
{
    point<K> p{};
    for (auto& c : p)
        c = static_cast<double>(st::uniform_index(g, side)) - static_cast<double>(side / 2);
    return p;
}

// Every query against a linear scan under the metric m. Distances are compared exactly: the tree and the scan
// compute the same terms in the same order. Radius membership is decided in the reduced form (radius contracted), as
// the radius is documented; the radius is an integer, so contracted it is exact and points on the sphere are common.
template <std::size_t K, typename Metric = mt::euclidean>
void check_against_linear_scan(std::uint64_t seed, const Metric& m = Metric{})
{
    st::wyrand g(seed);
    for (int trial = 0; trial < 60; ++trial) {
        const std::size_t n = st::uniform_index(g, 120);
        const std::uint64_t side = 2 + st::uniform_index(g, 12);
        std::vector<point<K>> pts(n);
        for (auto& p : pts)
            p = grid_point<K>(g, side);
        std::vector<std::uint32_t> idx(n);
        const ct::kd_tree<point<K>, K, Metric> tree(pts, idx, m);
        ASSERT_EQ(tree.size(), n);
        ASSERT_EQ(tree.empty(), n == 0);
        // The indices are a permutation
        std::vector<std::uint32_t> sorted(idx);
        std::sort(sorted.begin(), sorted.end());
        for (std::size_t i = 0; i < n; ++i)
            ASSERT_EQ(sorted[i], i);

        for (int q = 0; q < 25; ++q) {
            const point<K> query = grid_point<K>(g, side + 2);
            std::vector<double> all(n), reduced(n);
            for (std::size_t i = 0; i < n; ++i) {
                all[i] = m(pts[i], query);
                reduced[i] = m.reduced(pts[i], query);
            }

            // nearest
            const auto near = tree.nearest(query);
            if (n == 0) {
                EXPECT_FALSE(near);
            } else {
                ASSERT_TRUE(near);
                EXPECT_EQ(near->distance, *std::min_element(all.begin(), all.end()));
                EXPECT_EQ(near->distance, all[near->index]);
            }

            // k nearest, k from 0 to beyond n
            const std::size_t k = st::uniform_index(g, n + 4);
            std::vector<ct::kd_neighbour> out(k);
            const std::size_t got = tree.k_nearest(query, out);
            ASSERT_EQ(got, std::min(k, n));
            std::vector<double> expected(all);
            std::sort(expected.begin(), expected.end());
            std::vector<int> seen(n, 0);
            for (std::size_t i = 0; i < got; ++i) {
                EXPECT_EQ(out[i].distance, expected[i]) << "neighbour " << i;
                EXPECT_EQ(out[i].distance, all[out[i].index]);
                EXPECT_EQ(seen[out[i].index]++, 0) << "index repeated";
            }

            // within a radius
            const double radius = static_cast<double>(st::uniform_index(g, side + 1));

            // k nearest strictly within the radius (SciPy's distance_upper_bound): the brute force's first k below it
            {
                const double upper = m.contract(radius);
                std::vector<double> below;
                for (std::size_t i = 0; i < n; ++i)
                    if (reduced[i] < upper)
                        below.push_back(all[i]);
                std::sort(below.begin(), below.end());
                std::vector<ct::kd_neighbour> within_k(k);
                const std::size_t w = tree.k_nearest_within(query, radius, within_k);
                ASSERT_EQ(w, std::min(k, below.size()));
                for (std::size_t i = 0; i < w; ++i) {
                    EXPECT_EQ(within_k[i].distance, below[i]) << "bounded neighbour " << i;
                    EXPECT_EQ(within_k[i].distance, all[within_k[i].index]);
                    EXPECT_LT(m.contract(within_k[i].distance), upper);
                }
            }
            const double bound = m.contract(radius);
            std::vector<std::uint32_t> within, brute;
            tree.for_each_within(query, radius, [&](std::uint32_t i, double dd) {
                EXPECT_EQ(dd, all[i]);
                within.push_back(i);
            });
            for (std::uint32_t i = 0; i < n; ++i)
                if (reduced[i] <= bound)
                    brute.push_back(i);
            std::sort(within.begin(), within.end());
            EXPECT_EQ(within, brute);

            // in a box, its faces on the grid
            point<K> lo{}, hi{};
            for (std::size_t a = 0; a < K; ++a) {
                const double u = grid_point<1>(g, side)[0], v = grid_point<1>(g, side)[0];
                lo[a] = std::min(u, v);
                hi[a] = std::max(u, v);
            }
            std::vector<std::uint32_t> boxed, brute_box;
            tree.for_each_in_box(lo, hi, [&](std::uint32_t i) { boxed.push_back(i); });
            for (std::uint32_t i = 0; i < n; ++i) {
                bool in = true;
                for (std::size_t a = 0; a < K; ++a)
                    in = in && pts[i][a] >= lo[a] && pts[i][a] <= hi[a];
                if (in)
                    brute_box.push_back(i);
            }
            std::sort(boxed.begin(), boxed.end());
            EXPECT_EQ(boxed, brute_box);
        }
    }
}

TEST(ContainersKdTreeQueries, MatchALinearScanInOneDimension) { check_against_linear_scan<1>(1); }
TEST(ContainersKdTreeQueries, MatchALinearScanInTwoDimensions) { check_against_linear_scan<2>(2); }
TEST(ContainersKdTreeQueries, MatchALinearScanInThreeDimensions) { check_against_linear_scan<3>(3); }
TEST(ContainersKdTreeQueries, MatchALinearScanInFiveDimensions) { check_against_linear_scan<5>(5); }

TEST(ContainersKdTreeMetrics, ManhattanMatchesALinearScan)
{
    check_against_linear_scan<2, mt::manhattan>(11);
    check_against_linear_scan<4, mt::manhattan>(12);
}

TEST(ContainersKdTreeMetrics, ChebyshevMatchesALinearScan)
{
    check_against_linear_scan<2, mt::chebyshev>(13);
    check_against_linear_scan<4, mt::chebyshev>(14);
}

TEST(ContainersKdTreeMetrics, MinkowskiOfOrderThreeMatchesALinearScan)
{
    check_against_linear_scan<2, mt::minkowski<3>>(15);
    check_against_linear_scan<3, mt::minkowski<3>>(16);
}

TEST(ContainersKdTreeMetrics, WeightedMinkowskiMatchesALinearScan)
{
    check_against_linear_scan<2>(17, mt::weighted<2>(point<2>{0.5, 4}));
    check_against_linear_scan<3>(18, mt::weighted<2>(point<3>{2, 0.25, 1}));
    check_against_linear_scan<3>(19, mt::weighted<1>(point<3>{3, 1, 0.5}));
}

TEST(ContainersKdTreeMetrics, SquaredEuclideanMatchesALinearScan)
{
    check_against_linear_scan<2, mt::squared_euclidean>(20);
    check_against_linear_scan<5, mt::squared_euclidean>(21);
}

TEST(ContainersKdTreeMetrics, DistancesAreInTheMetricsOwnUnits)
{
    const std::vector<point<2>> pts{{0, 0}, {3, 4}};
    std::vector<std::uint32_t> a(2), b(2), c(2), d(2);
    const ct::kd_tree<point<2>, 2> euclid(pts, a);
    const ct::kd_tree<point<2>, 2, mt::squared_euclidean> squared(pts, b);
    const ct::kd_tree<point<2>, 2, mt::manhattan> taxi(pts, c);
    const ct::kd_tree<point<2>, 2, mt::chebyshev> king(pts, d);
    const point<2> far{6, 8};
    EXPECT_EQ(euclid.nearest(far)->distance, 5.0);
    EXPECT_EQ(squared.nearest(far)->distance, 25.0);
    EXPECT_EQ(taxi.nearest(far)->distance, 7.0);
    EXPECT_EQ(king.nearest(far)->distance, 4.0);
    // The radius is in the same units: (3, 4) is 5 from the origin, so a radius of 5 includes it and 4.99 does not
    int inside = 0;
    euclid.for_each_within(point<2>{0, 0}, 5.0, [&](std::uint32_t, double) { ++inside; });
    EXPECT_EQ(inside, 2);
    inside = 0;
    euclid.for_each_within(point<2>{0, 0}, 4.99, [&](std::uint32_t, double) { ++inside; });
    EXPECT_EQ(inside, 1);
    inside = 0;
    squared.for_each_within(point<2>{0, 0}, 25.0, [&](std::uint32_t, double) { ++inside; });
    EXPECT_EQ(inside, 2);
}

TEST(ContainersKdTreeMetrics, TheMetricIsKeptAndReturned)
{
    const std::vector<point<2>> pts{{0, 0}, {1, 0}, {0, 1}};
    std::vector<std::uint32_t> idx(pts.size());
    const auto w = mt::weighted<2>(point<2>{1, 100});
    static_assert(std::is_same_v<decltype(w), const mt::weighted_minkowski<2, point<2>>>);
    const ct::kd_tree<point<2>, 2, mt::weighted_minkowski<2, point<2>>> tree(pts, idx, w);
    EXPECT_EQ(tree.distance_metric().weights[1], 100.0);
    // The weight on the second axis makes (0, 1), 0.4 off on it, nearer to (0.6, 0.6) than (1, 0), 0.6 off on it
    EXPECT_EQ(tree.nearest(point<2>{0.6, 0.6})->index, 2u);
    EXPECT_EQ(tree.nearest(point<2>{0.6, 0.6})->distance, w(pts[2], point<2>{0.6, 0.6}));
}

TEST(ContainersKdTreeQueries, MatchALinearScanForRealCoordinates)
{
    st::wyrand g(6);
    const mt::euclidean m;
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<point<2>> pts(300);
        for (auto& p : pts)
            p = {st::uniform_real(g), st::uniform_real(g) * 3};
        std::vector<std::uint32_t> idx(pts.size());
        const ct::kd_tree<point<2>, 2> tree(pts, idx);
        for (int q = 0; q < 50; ++q) {
            const point<2> query{st::uniform_real(g) * 1.2 - 0.1, st::uniform_real(g) * 3};
            std::vector<std::pair<double, std::uint32_t>> all;
            for (std::uint32_t i = 0; i < pts.size(); ++i)
                all.push_back({m(pts[i], query), i});
            std::sort(all.begin(), all.end());
            std::array<ct::kd_neighbour, 7> out{};
            ASSERT_EQ(tree.k_nearest(query, out), 7u);
            for (std::size_t i = 0; i < 7; ++i) {   // distinct real distances: the indices agree too
                EXPECT_EQ(out[i].index, all[i].second);
                EXPECT_EQ(out[i].distance, all[i].first);
            }
            EXPECT_EQ(tree.nearest(query)->index, all[0].second);
        }
    }
}

TEST(ContainersKdTreeQueries, FloatPointsAnswerDoubleQueries)
{
    st::wyrand g(8);
    std::vector<std::array<float, 3>> pts(500);
    for (auto& p : pts)
        p = {static_cast<float>(st::uniform_real(g)), static_cast<float>(st::uniform_real(g)), static_cast<float>(st::uniform_real(g))};
    std::vector<std::uint32_t> idx(pts.size());
    const ct::kd_tree<std::array<float, 3>, 3, mt::manhattan> tree(pts, idx);
    const mt::manhattan m;
    for (int q = 0; q < 100; ++q) {
        const point<3> query{st::uniform_real(g), st::uniform_real(g), st::uniform_real(g)};   // double, not float
        std::vector<std::pair<double, std::uint32_t>> all;
        for (std::uint32_t i = 0; i < pts.size(); ++i)
            all.push_back({m(pts[i], query), i});   // in double: the common floating type
        std::sort(all.begin(), all.end());
        std::array<ct::kd_neighbour, 4> out{};
        ASSERT_EQ(tree.k_nearest(query, out), 4u);
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(out[i].index, all[i].second);
            EXPECT_EQ(out[i].distance, all[i].first);
        }
        std::size_t count = 0, brute = 0;
        tree.for_each_within(query, 0.2, [&](std::uint32_t i, double d) {
            EXPECT_EQ(d, m(pts[i], query));
            ++count;
        });
        for (const auto& p : pts)
            brute += m(p, query) <= 0.2;
        EXPECT_EQ(count, brute);
    }
}

TEST(ContainersKdTree, AnEmptyTreeAnswersNothing)
{
    const std::vector<point<2>> none;
    std::vector<std::uint32_t> idx;
    const ct::kd_tree<point<2>, 2> tree(none, idx);
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_FALSE(tree.nearest(point<2>{0, 0}));
    std::array<ct::kd_neighbour, 3> out{};
    EXPECT_EQ(tree.k_nearest(point<2>{0, 0}, out), 0u);
    int calls = 0;
    tree.for_each_within(point<2>{0, 0}, 1e300, [&](std::uint32_t, double) { ++calls; });
    tree.for_each_in_box(point<2>{-1e300, -1e300}, point<2>{1e300, 1e300}, [&](std::uint32_t) { ++calls; });
    EXPECT_EQ(calls, 0);
}

TEST(ContainersKdTree, OnePointIsAlwaysTheNearest)
{
    const std::vector<point<3>> one{{1, 2, 3}};
    std::vector<std::uint32_t> idx(1);
    const ct::kd_tree<point<3>, 3> tree(one, idx);
    const auto n = tree.nearest(point<3>{1, 2, 5});
    ASSERT_TRUE(n);
    EXPECT_EQ(n->index, 0u);
    EXPECT_EQ(n->distance, 2.0);
    std::array<ct::kd_neighbour, 4> out{};
    EXPECT_EQ(tree.k_nearest(point<3>{0, 0, 0}, out), 1u);   // k beyond the size: min(k, size)
    EXPECT_EQ(out[0].distance, mt::euclidean{}(one[0], point<3>{0, 0, 0}));   // √14
}

TEST(ContainersKdTree, IdenticalPointsAreAllFound)
{
    const std::vector<point<2>> same(500, point<2>{3, -1});
    std::vector<std::uint32_t> idx(same.size());
    const ct::kd_tree<point<2>, 2> tree(same, idx);
    std::vector<std::uint32_t> got;
    tree.for_each_within(point<2>{3, -1}, 0.0, [&](std::uint32_t i, double) { got.push_back(i); });   // radius 0 includes them
    EXPECT_EQ(got.size(), 500u);
    got.clear();
    tree.for_each_in_box(point<2>{3, -1}, point<2>{3, -1}, [&](std::uint32_t i) { got.push_back(i); });   // a degenerate box
    EXPECT_EQ(got.size(), 500u);
    std::array<ct::kd_neighbour, 10> out{};
    ASSERT_EQ(tree.k_nearest(point<2>{0, 3}, out), 10u);
    for (const auto& o : out)
        EXPECT_EQ(o.distance, 5.0);
}

TEST(ContainersKdTree, AShortIndexSpanIndexesAPrefix)
{
    const std::vector<point<1>> pts{{0}, {10}, {20}, {30}, {1}};
    std::vector<std::uint32_t> idx(3);
    const ct::kd_tree<point<1>, 1> tree(pts, idx);
    EXPECT_EQ(tree.size(), 3u);
    EXPECT_EQ(tree.nearest(point<1>{2})->index, 0u);   // point 4 at 1 is not indexed
    std::vector<std::uint32_t> got;
    tree.for_each_in_box(point<1>{-100}, point<1>{100}, [&](std::uint32_t i) { got.push_back(i); });
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, (std::vector<std::uint32_t>{0, 1, 2}));
    // A longer span uses only as many entries as there are points
    std::vector<std::uint32_t> long_idx(8, 77);
    const ct::kd_tree<point<1>, 1> whole(pts, long_idx);
    EXPECT_EQ(whole.size(), 5u);
    EXPECT_EQ(long_idx[5], 77u);
}

TEST(ContainersKdTreePoints, APointTypeOfTheCallersOwnThroughPointTraits)
{
    static_assert(mt::point<target>);
    const std::vector<target> ts{{0, 0, 10}, {5, 5, 11}, {-3, 4, 12}, {6, -1, 13}};
    std::vector<std::uint32_t> idx(ts.size());
    const ct::kd_tree<target, 2> tree(ts, idx);
    EXPECT_EQ(ts[tree.nearest(target{5.5f, 4.0f, 0})->index].id, 11);
    EXPECT_EQ(ts[tree.nearest(point<2>{-2.5, 3.5})->index].id, 12);   // a query of another type
    std::vector<int> ids;
    tree.for_each_in_box(target{-4, -2, 0}, target{1, 5, 0}, [&](std::uint32_t i) { ids.push_back(ts[i].id); });
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (std::vector<int>{10, 12}));
    ids.clear();
    tree.for_each_within(target{0, 0, 0}, 5.0, [&](std::uint32_t i, double) { ids.push_back(ts[i].id); });   // (-3, 4) is exactly 5 away
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (std::vector<int>{10, 12}));
    static_assert(noexcept(tree.nearest(std::declval<const target&>())));
}

TEST(ContainersKdTreePoints, FixedVectorsArePointsAndTheQueryTypeMayDiffer)
{
    using v2 = la::fixed_vector<double, 2>;
    const std::vector<v2> pts{v2{{0, 0}}, v2{{1, 0}}, v2{{0, 1}}, v2{{4, 4}}};
    std::vector<std::uint32_t> idx(pts.size());
    const ct::kd_tree<v2, 2> tree(pts, idx);
    EXPECT_EQ(tree.nearest(std::array<int, 2>{3, 3})->index, 3u);   // any type with operator[]
    EXPECT_EQ(tree.nearest(v2{{0.9, 0.2}})->index, 1u);
}

TEST(ContainersKdTreeSearch, ACandidateStopsBeingReadOnceItExceedsTheBound)
{
    // Points spread on the first axis only: once a near point is found, every farther candidate exceeds the bound
    // after its first coordinate, and its other seven are never read (a candidate within the bound is read to the
    // end). The splits on axes 1 to 7 cannot prune (every point is 0 there), so many candidates are visited and only
    // the early exit saves the reads: axis 1 is read for far fewer of them than axis 0.
    st::wyrand g(9);
    std::vector<counted> pts(4096);
    for (auto& p : pts)
        p.c[0] = st::uniform_real(g) * 1000;
    std::vector<std::uint32_t> idx(pts.size());
    const ct::kd_tree<counted, 8> tree(pts, idx);
    kd_tree_test::counted_reads = {};
    std::size_t correct = 0;
    for (int q = 0; q < 50; ++q) {
        point<8> query{};
        query[0] = st::uniform_real(g) * 1000;
        const auto n = tree.nearest(query);
        double best = 1e300;
        std::uint32_t at = 0;
        for (std::uint32_t i = 0; i < pts.size(); ++i) {
            const double d = query[0] > pts[i].c[0] ? query[0] - pts[i].c[0] : pts[i].c[0] - query[0];
            if (d < best) {
                best = d;
                at = i;
            }
        }
        correct += n && n->index == at && n->distance == best;
    }
    const auto reads = kd_tree_test::counted_reads;
    EXPECT_EQ(correct, 50u);
    // The brute-force scan above reads pts[i].c directly, so these are the tree's reads alone
    EXPECT_LT(reads[1] * 4, reads[0]) << "axis 0: " << reads[0] << ", axis 1: " << reads[1];
    EXPECT_GT(reads[7], 0u);
}

TEST(ContainersKdTreeSearch, KNearestStopsAtTheKthBestSoFar)
{
    // The same in k_nearest: only the candidates still able to enter the k best are read to the end
    st::wyrand g(10);
    std::vector<counted> pts(4096);
    for (auto& p : pts)
        p.c[0] = st::uniform_real(g) * 1000;
    std::vector<std::uint32_t> idx(pts.size());
    const ct::kd_tree<counted, 8> tree(pts, idx);
    kd_tree_test::counted_reads = {};
    std::array<ct::kd_neighbour, 5> out{};
    point<8> query{};
    query[0] = 500;
    ASSERT_EQ(tree.k_nearest(query, out), 5u);
    for (std::size_t i = 1; i < 5; ++i)
        EXPECT_LE(out[i - 1].distance, out[i].distance);
    const auto reads = kd_tree_test::counted_reads;
    EXPECT_LT(reads[1] * 4, reads[0]) << "axis 0: " << reads[0] << ", axis 1: " << reads[1];
    std::vector<double> all;
    for (const auto& p : pts)
        all.push_back(p.c[0] > 500 ? p.c[0] - 500 : 500 - p.c[0]);
    std::sort(all.begin(), all.end());
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_EQ(out[i].distance, all[i]);
}

TEST(ContainersKdTree, ALargeTreeStaysWithinItsStackAndMatchesALinearScan)
{
    st::wyrand g(7);
    const mt::euclidean m;
    std::vector<point<3>> pts(100000);
    for (auto& p : pts)
        p = {st::uniform_real(g), st::uniform_real(g), st::uniform_real(g)};
    std::vector<std::uint32_t> idx(pts.size());
    const ct::kd_tree<point<3>, 3> tree(pts, idx);
    for (int q = 0; q < 20; ++q) {
        const point<3> query{st::uniform_real(g), st::uniform_real(g), st::uniform_real(g)};
        double best = 1e300;
        for (const auto& p : pts)
            best = std::min(best, m(p, query));
        EXPECT_EQ(tree.nearest(query)->distance, best);
        std::size_t count = 0, brute = 0;
        tree.for_each_within(query, 0.05, [&](std::uint32_t, double) { ++count; });
        for (const auto& p : pts)
            brute += m.reduced(p, query) <= m.contract(0.05);
        EXPECT_EQ(count, brute);
    }
}

// ---- compile time and noexcept

constexpr bool compile_time()
{
    constexpr std::array<point<2>, 6> pts{{{0, 0}, {4, 1}, {-2, 3}, {5, 5}, {1, -1}, {3, 3}}};
    std::array<std::uint32_t, 6> idx{};
    const ct::kd_tree<point<2>, 2> tree(pts, idx);
    const auto n = tree.nearest(point<2>{3.2, 2.9});
    std::array<ct::kd_neighbour, 2> two{};
    const std::size_t got = tree.k_nearest(point<2>{0, 0}, two);
    int within = 0, boxed = 0;
    tree.for_each_within(point<2>{0, 0}, 1.5, [&](std::uint32_t, double) { ++within; });   // (0, 0) and (1, -1)
    tree.for_each_in_box(point<2>{0, 0}, point<2>{5, 5}, [&](std::uint32_t) { ++boxed; });   // (0, 0), (4, 1), (5, 5), (3, 3)
    std::array<std::uint32_t, 6> idx1{};
    const ct::kd_tree<point<2>, 2, mt::manhattan> taxi(pts, idx1);
    const auto t = taxi.nearest(point<2>{2, 2});   // (3, 3) at 2
    return n && n->index == 5 && got == 2 && two[0].index == 0 && two[1].index == 4 && two[1].distance == snicholls::math::sqrt(2.0) && within == 2 &&
           boxed == 4 && t && t->index == 5 && t->distance == 2.0;
}
static_assert(compile_time());

using tree2 = ct::kd_tree<point<2>, 2>;
static_assert(std::is_same_v<tree2, ct::kd_tree<point<2>, 2, mt::euclidean>>);
static_assert(std::is_nothrow_constructible_v<tree2, std::span<const point<2>>, std::span<std::uint32_t>>);
static_assert(noexcept(std::declval<const tree2&>().nearest(std::declval<const point<2>&>())));
static_assert(noexcept(std::declval<const tree2&>().k_nearest(std::declval<const point<2>&>(), std::span<ct::kd_neighbour>{})));
inline constexpr auto quiet = [](std::uint32_t, double) noexcept {};
inline constexpr auto loud = [](std::uint32_t) { throw std::runtime_error("no"); };
static_assert(noexcept(std::declval<const tree2&>().for_each_within(std::declval<const point<2>&>(), 1.0, quiet)));
static_assert(!noexcept(std::declval<const tree2&>().for_each_in_box(std::declval<const point<2>&>(), std::declval<const point<2>&>(), loud)));
static_assert(std::is_trivially_copyable_v<ct::kd_neighbour>);

} // namespace
