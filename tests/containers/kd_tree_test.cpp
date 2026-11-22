//
//  kd_tree_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/kd_tree.hpp>: every query (nearest, k-nearest, radius, box) checked against a
//  linear scan over many random point sets in 1 to 5 dimensions, on an integer grid so that duplicates, ties and
//  points exactly on the radius or the box are common; empty and one-point trees, all points identical, a prefix
//  index span, k beyond the size, a custom coordinate reader, a large tree, and the constexpr and noexcept claims.
//

#include <snicholls/core/containers/kd_tree.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace ct = snicholls::containers;
namespace la = snicholls::linalg;
namespace st = snicholls::stats;

template <std::size_t K>
using point = std::array<double, K>;

template <std::size_t K>
double d2(const point<K>& a, const point<K>& b)
{
    double s = 0;
    for (std::size_t i = 0; i < K; ++i)
        s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

// Coordinates on a small integer grid: duplicates and exact ties are frequent
template <std::size_t K>
point<K> grid_point(st::wyrand& g, std::uint64_t side)
{
    point<K> p{};
    for (auto& c : p)
        c = static_cast<double>(st::uniform_index(g, side)) - static_cast<double>(side / 2);
    return p;
}

template <std::size_t K>
void check_against_linear_scan(std::uint64_t seed)
{
    st::wyrand g(seed);
    for (int trial = 0; trial < 60; ++trial) {
        const std::size_t n = st::uniform_index(g, 120);
        const std::uint64_t side = 2 + st::uniform_index(g, 12);
        std::vector<point<K>> pts(n);
        for (auto& p : pts)
            p = grid_point<K>(g, side);
        std::vector<std::uint32_t> idx(n);
        const ct::kd_tree<point<K>, K> tree(pts, idx);
        ASSERT_EQ(tree.size(), n);
        ASSERT_EQ(tree.empty(), n == 0);
        // The indices are a permutation
        std::vector<std::uint32_t> sorted(idx);
        std::sort(sorted.begin(), sorted.end());
        for (std::size_t i = 0; i < n; ++i)
            ASSERT_EQ(sorted[i], i);

        for (int q = 0; q < 25; ++q) {
            const point<K> query = grid_point<K>(g, side + 2);
            std::vector<double> all(n);
            for (std::size_t i = 0; i < n; ++i)
                all[i] = d2(pts[i], query);

            // nearest
            const auto near = tree.nearest(query);
            if (n == 0) {
                EXPECT_FALSE(near);
            } else {
                ASSERT_TRUE(near);
                EXPECT_EQ(near->distance2, *std::min_element(all.begin(), all.end()));
                EXPECT_EQ(near->distance2, all[near->index]);
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
                EXPECT_EQ(out[i].distance2, expected[i]) << "neighbour " << i;
                EXPECT_EQ(out[i].distance2, all[out[i].index]);
                EXPECT_EQ(seen[out[i].index]++, 0) << "index repeated";
            }

            // within a radius (squared radii are integers, so points on the circle are common)
            const double r2 = static_cast<double>(st::uniform_index(g, side * side));
            std::vector<std::uint32_t> within, brute;
            tree.for_each_within(query, r2, [&](std::uint32_t i, double dd) {
                EXPECT_EQ(dd, all[i]);
                within.push_back(i);
            });
            for (std::uint32_t i = 0; i < n; ++i)
                if (all[i] <= r2)
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

TEST(ContainersKdTreeQueries, MatchALinearScanForRealCoordinates)
{
    st::wyrand g(6);
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
                all.push_back({d2(pts[i], query), i});
            std::sort(all.begin(), all.end());
            std::array<ct::kd_neighbour, 7> out{};
            ASSERT_EQ(tree.k_nearest(query, out), 7u);
            for (std::size_t i = 0; i < 7; ++i) {   // distinct real distances: the indices agree too
                EXPECT_EQ(out[i].index, all[i].second);
                EXPECT_EQ(out[i].distance2, all[i].first);
            }
            EXPECT_EQ(tree.nearest(query)->index, all[0].second);
        }
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
    EXPECT_EQ(n->distance2, 4.0);
    std::array<ct::kd_neighbour, 4> out{};
    EXPECT_EQ(tree.k_nearest(point<3>{0, 0, 0}, out), 1u);   // k beyond the size: min(k, size)
    EXPECT_EQ(out[0].distance2, 14.0);
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
        EXPECT_EQ(o.distance2, 25.0);
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

TEST(ContainersKdTree, ACustomTraitsReadsAnyPointType)
{
    struct target {
        float east, north;
        int id;
    };
    struct target_traits {
        static constexpr double coordinate(const target& t, std::size_t axis) noexcept { return axis == 0 ? t.east : t.north; }
    };
    const std::vector<target> ts{{0, 0, 10}, {5, 5, 11}, {-3, 4, 12}, {6, -1, 13}};
    std::vector<std::uint32_t> idx(ts.size());
    const ct::kd_tree<target, 2, target_traits> tree(ts, idx);
    EXPECT_EQ(ts[tree.nearest(target{5.5f, 4.0f, 0})->index].id, 11);
    std::vector<int> ids;
    tree.for_each_in_box(target{-4, -2, 0}, target{1, 5, 0}, [&](std::uint32_t i) { ids.push_back(ts[i].id); });
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (std::vector<int>{10, 12}));
    static_assert(noexcept(tree.nearest(std::declval<const target&>())));
}

TEST(ContainersKdTree, FixedVectorsArePointsAndTheQueryTypeMayDiffer)
{
    using v2 = la::fixed_vector<double, 2>;
    const std::vector<v2> pts{v2{{0, 0}}, v2{{1, 0}}, v2{{0, 1}}, v2{{4, 4}}};
    std::vector<std::uint32_t> idx(pts.size());
    const ct::kd_tree<v2, 2> tree(pts, idx);
    EXPECT_EQ(tree.nearest(std::array<int, 2>{3, 3})->index, 3u);   // any type with operator[]
    EXPECT_EQ(tree.nearest(v2{{0.9, 0.2}})->index, 1u);
}

TEST(ContainersKdTree, ALargeTreeStaysWithinItsStackAndMatchesALinearScan)
{
    st::wyrand g(7);
    std::vector<point<3>> pts(100000);
    for (auto& p : pts)
        p = {st::uniform_real(g), st::uniform_real(g), st::uniform_real(g)};
    std::vector<std::uint32_t> idx(pts.size());
    const ct::kd_tree<point<3>, 3> tree(pts, idx);
    for (int q = 0; q < 20; ++q) {
        const point<3> query{st::uniform_real(g), st::uniform_real(g), st::uniform_real(g)};
        double best = 1e300;
        for (const auto& p : pts)
            best = std::min(best, d2(p, query));
        EXPECT_EQ(tree.nearest(query)->distance2, best);
        std::size_t count = 0, brute = 0;
        tree.for_each_within(query, 0.0025, [&](std::uint32_t, double) { ++count; });
        for (const auto& p : pts)
            brute += d2(p, query) <= 0.0025;
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
    tree.for_each_within(point<2>{0, 0}, 2.0, [&](std::uint32_t, double) { ++within; });   // (0, 0) and (1, -1)
    tree.for_each_in_box(point<2>{0, 0}, point<2>{5, 5}, [&](std::uint32_t) { ++boxed; });   // (0, 0), (4, 1), (5, 5), (3, 3)
    return n && n->index == 5 && got == 2 && two[0].index == 0 && two[1].index == 4 && two[1].distance2 == 2 && within == 2 && boxed == 4;
}
static_assert(compile_time());

using tree2 = ct::kd_tree<point<2>, 2>;
static_assert(std::is_nothrow_constructible_v<tree2, std::span<const point<2>>, std::span<std::uint32_t>>);
static_assert(noexcept(std::declval<const tree2&>().nearest(std::declval<const point<2>&>())));
static_assert(noexcept(std::declval<const tree2&>().k_nearest(std::declval<const point<2>&>(), std::span<ct::kd_neighbour>{})));
inline constexpr auto quiet = [](std::uint32_t, double) noexcept {};
inline constexpr auto loud = [](std::uint32_t) { throw std::runtime_error("no"); };
static_assert(noexcept(std::declval<const tree2&>().for_each_within(std::declval<const point<2>&>(), 1.0, quiet)));
static_assert(!noexcept(std::declval<const tree2&>().for_each_in_box(std::declval<const point<2>&>(), std::declval<const point<2>&>(), loud)));

} // namespace
