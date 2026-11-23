//
//  ann_forest_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/ann_forest.hpp>: recall against brute force above a stated floor, with a fixed
//  seed, under euclidean, manhattan, angular, cosine_distance, inner_product (normalised vectors) and hamming (recall@1
//  on planted near-duplicates); every returned distance exact and nearest first; the same seed giving the same index,
//  element by element; an index attached to copies of its arrays answering identically; the structure (every point
//  once per tree, leaves within leaf_size, the layout's sizes); the edge cases (one point, fewer points than a leaf,
//  identical points, one tree, a budget smaller than a leaf, a full frontier, k beyond n, empty spans); the errors
//  (invalid_parameters and out_of_storage); a point type and a metric of the test's own; suggested_storage, the
//  layout contract of the flat records, and a build and query in a constant expression.
//

#include <snicholls/core/containers/ann_forest.hpp>

#include <snicholls/core/metric/angular.hpp>
#include <snicholls/core/metric/binary.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ann_forest_test {

// A point of the test's own: three named coordinates, read through point_traits
struct sample {
    float x = 0, y = 0, z = 0;
};

// A metric of the test's own that orders by direction (the cosine distance, scaled); it declares its split below
struct scaled_cosine {
    template <typename A, typename B>
    constexpr double operator()(const A& a, const B& b) const
    {
        return 10.0 * snicholls::metric::cosine_distance{}(a, b);
    }
};

} // namespace ann_forest_test

template <>
struct snicholls::metric::point_traits<ann_forest_test::sample> {
    using value_type = float;
    static constexpr std::size_t extent = 3;
    static constexpr std::size_t dimension(const ann_forest_test::sample&) noexcept { return 3; }
    static constexpr float at(const ann_forest_test::sample& s, std::size_t i) noexcept { return i == 0 ? s.x : i == 1 ? s.y : s.z; }
};

template <>
struct snicholls::containers::ann_default_split<ann_forest_test::scaled_cosine> {
    using type = snicholls::containers::ann_angular_split;
};

namespace {

namespace ct = snicholls::containers;
namespace mt = snicholls::metric;
namespace st = snicholls::stats;
using ann_forest_test::sample;
using ann_forest_test::scaled_cosine;

template <std::size_t D>
using vec = std::array<float, D>;

// Clustered data: `clusters` centres uniform in [-1, 1]^D, points normal around them with standard deviation `sd`
template <std::size_t D>
std::vector<vec<D>> clustered(st::wyrand& g, std::size_t n, std::size_t clusters, double sd, bool normalise = false)
{
    std::vector<vec<D>> centres(clusters);
    for (auto& c : centres)
        for (auto& x : c)
            x = static_cast<float>(st::uniform_real(g) * 2 - 1);
    st::normal_box_muller z;
    std::vector<vec<D>> pts(n);
    for (auto& p : pts) {
        const auto& c = centres[st::uniform_index(g, clusters)];
        double s = 0;
        for (std::size_t i = 0; i < D; ++i) {
            p[i] = static_cast<float>(c[i] + sd * z(g));
            s += static_cast<double>(p[i]) * p[i];
        }
        if (normalise)
            for (auto& x : p)
                x = static_cast<float>(x / std::sqrt(s));
    }
    return pts;
}

// The storage of an index and the index over it
template <typename Point, std::size_t D, typename Metric = mt::euclidean>
struct built {
    using forest = ct::ann_forest<Point, D, Metric>;
    std::vector<ct::ann_node> nodes;
    std::vector<std::uint32_t> items;
    std::vector<float> normals;
    std::optional<forest> f;
    snicholls::result<forest, ct::ann_error> outcome = snicholls::err(ct::ann_error::invalid_parameters);

    built(std::span<const Point> pts, std::size_t trees, std::size_t leaf, std::uint64_t seed, const Metric& m = Metric{})
    {
        const auto s = forest::suggested_storage(pts.size(), trees, leaf);
        nodes.resize(s.nodes);
        items.resize(s.items);
        normals.resize(s.normals);
        st::wyrand g(seed);
        outcome = forest::build(pts, trees, leaf, nodes, items, normals, g, m);
        if (outcome)
            f = outcome.unwrap();
    }
};

// Brute force: the k smallest of m(points[i], q), nearest first
template <typename Point, typename Q, typename Metric>
std::vector<std::pair<double, std::uint32_t>> brute(const std::vector<Point>& pts, const Q& q, std::size_t k, const Metric& m)
{
    std::vector<std::pair<double, std::uint32_t>> all(pts.size());
    for (std::uint32_t i = 0; i < pts.size(); ++i)
        all[i] = {static_cast<double>(m(pts[i], q)), i};
    std::sort(all.begin(), all.end());
    all.resize(std::min(k, all.size()));
    return all;
}

// Queries an index and checks what is exact regardless of recall: distances equal the metric's, nearest first, no
// index twice. Returns the neighbours found.
template <typename Forest, typename Point, typename Q, typename Metric>
std::vector<ct::ann_neighbour> query_checked(const Forest& f, const std::vector<Point>& pts, const Q& q, std::size_t k, std::size_t budget,
                                             const Metric& m, std::size_t frontier = 256)
{
    std::vector<ct::ann_neighbour> out(k);
    std::vector<std::uint32_t> cand(budget);
    std::vector<ct::ann_frontier> front(frontier);
    out.resize(f.query(q, out, cand, front));
    std::vector<std::uint32_t> seen;
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_EQ(out[i].distance, static_cast<double>(m(pts[out[i].index], q))) << "neighbour " << i;
        if (i > 0) {
            EXPECT_LE(out[i - 1].distance, out[i].distance) << "neighbour " << i;
        }
        seen.push_back(out[i].index);
    }
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end()) << "an index returned twice";
    return out;
}

// Recall@k over `queries`: the fraction of the true k nearest found
template <typename Forest, typename Point, typename Metric>
double recall(const Forest& f, const std::vector<Point>& pts, const std::vector<Point>& queries, std::size_t k, std::size_t budget, const Metric& m)
{
    std::size_t hit = 0;
    for (const auto& q : queries) {
        const auto truth = brute(pts, q, k, m);
        const auto got = query_checked(f, pts, q, k, budget, m);
        for (const auto& t : truth)
            hit += std::any_of(got.begin(), got.end(), [&](const ct::ann_neighbour& o) { return o.index == t.second; });
    }
    return static_cast<double>(hit) / static_cast<double>(queries.size() * k);
}

template <typename Metric>
double recall_on_clusters(std::uint64_t seed, bool normalise = false)
{
    constexpr std::size_t D = 16;
    st::wyrand g(seed);
    const auto pts = clustered<D>(g, 3000, 30, 0.15, normalise);
    const auto queries = clustered<D>(g, 100, 30, 0.15, normalise);
    const built<vec<D>, D, Metric> ix(pts, 10, 16, seed);
    EXPECT_TRUE(ix.f);
    return ix.f ? recall(*ix.f, pts, queries, 10, 1000, Metric{}) : 0.0;
}

// ---- recall against brute force (3,000 clustered points in 16 dimensions, 10 trees, a budget of 1,000)

TEST(ContainersAnnForestRecall, EuclideanFindsMostTrueNeighbours)
{
    const double r = recall_on_clusters<mt::euclidean>(1);
    RecordProperty("recall_at_10", std::to_string(r));
    EXPECT_GE(r, 0.90) << "recall@10 " << r;
}

TEST(ContainersAnnForestRecall, ManhattanFindsMostTrueNeighbours)
{
    const double r = recall_on_clusters<mt::manhattan>(2);
    RecordProperty("recall_at_10", std::to_string(r));
    EXPECT_GE(r, 0.85) << "recall@10 " << r;
}

TEST(ContainersAnnForestRecall, AngularFindsMostTrueNeighbours)
{
    const double r = recall_on_clusters<mt::angular>(3);
    RecordProperty("recall_at_10", std::to_string(r));
    EXPECT_GE(r, 0.90) << "recall@10 " << r;
}

TEST(ContainersAnnForestRecall, CosineDistanceFindsMostTrueNeighbours)
{
    const double r = recall_on_clusters<mt::cosine_distance>(4);
    RecordProperty("recall_at_10", std::to_string(r));
    EXPECT_GE(r, 0.90) << "recall@10 " << r;
}

TEST(ContainersAnnForestRecall, InnerProductOnNormalisedVectorsFindsMostTrueNeighbours)
{
    const double r = recall_on_clusters<mt::inner_product>(5, true);
    RecordProperty("recall_at_10", std::to_string(r));
    EXPECT_GE(r, 0.90) << "recall@10 " << r;
}

TEST(ContainersAnnForestRecall, HammingFindsPlantedNearDuplicates)
{
    // Random 256-bit codes have no structure but their planted near-duplicates: each query is a code with 8 of its
    // bits flipped, about 120 bits nearer than any other code
    using code = std::array<std::uint64_t, 4>;
    st::wyrand g(6);
    std::vector<code> pts(2000);
    for (auto& c : pts)
        for (auto& w : c)
            w = g();
    const built<code, 4, mt::hamming> ix(pts, 10, 16, 6);
    ASSERT_TRUE(ix.f);
    std::size_t hit = 0;
    constexpr std::size_t queries = 200;
    for (std::size_t q = 0; q < queries; ++q) {
        const std::uint32_t target = static_cast<std::uint32_t>(st::uniform_index(g, pts.size()));
        code c = pts[target];
        for (int b = 0; b < 8; ++b) {
            const auto bit = st::uniform_index(g, 256);
            c[bit / 64] ^= std::uint64_t{1} << (bit % 64);
        }
        const auto got = query_checked(*ix.f, pts, c, 1, 400, mt::hamming{});
        hit += !got.empty() && got[0].index == target;
    }
    const double r = static_cast<double>(hit) / queries;
    RecordProperty("recall_at_1", std::to_string(r));
    EXPECT_GE(r, 0.90) << "recall@1 " << r;
}

TEST(ContainersAnnForestRecall, MoreTreesAndALargerBudgetDoNotLowerRecall)
{
    constexpr std::size_t D = 16;
    st::wyrand g(7);
    const auto pts = clustered<D>(g, 3000, 30, 0.15);
    const auto queries = clustered<D>(g, 60, 30, 0.15);
    const built<vec<D>, D> few(pts, 2, 16, 7), many(pts, 16, 16, 7);
    ASSERT_TRUE(few.f && many.f);
    const double small = recall(*few.f, pts, queries, 10, 40, mt::euclidean{});
    const double large = recall(*many.f, pts, queries, 10, 3000, mt::euclidean{});
    EXPECT_LT(small, large);
    EXPECT_GE(large, 0.97);
}

// ---- exactness

TEST(ContainersAnnForestExactness, ABudgetOfEveryPointIsBruteForce)
{
    // With every item of every tree as candidates, the re-ranking sees every point: the answer is exact
    constexpr std::size_t D = 8;
    st::wyrand g(8);
    const auto pts = clustered<D>(g, 500, 5, 0.3);
    const built<vec<D>, D> ix(pts, 3, 10, 8);
    ASSERT_TRUE(ix.f);
    for (int q = 0; q < 30; ++q) {
        const auto query = clustered<D>(g, 1, 5, 0.3)[0];
        const auto truth = brute(pts, query, 7, mt::euclidean{});
        const auto got = query_checked(*ix.f, pts, query, 7, pts.size() * 3, mt::euclidean{}, 4096);
        ASSERT_EQ(got.size(), 7u);
        for (std::size_t i = 0; i < 7; ++i) {
            EXPECT_EQ(got[i].index, truth[i].second);
            EXPECT_EQ(got[i].distance, truth[i].first);
        }
    }
}

TEST(ContainersAnnForestExactness, DoubleQueriesAgainstFloatPointsAreExactInDouble)
{
    constexpr std::size_t D = 8;
    st::wyrand g(9);
    const auto pts = clustered<D>(g, 800, 8, 0.2);
    const built<vec<D>, D, mt::angular> ix(pts, 5, 12, 9);
    ASSERT_TRUE(ix.f);
    for (int q = 0; q < 20; ++q) {
        std::array<double, D> query{};
        for (auto& x : query)
            x = st::uniform_real(g) * 2 - 1;
        const auto got = query_checked(*ix.f, pts, query, 5, 200, mt::angular{});
        EXPECT_EQ(got.size(), 5u);
    }
}

// ---- determinism and attaching

template <typename Point, std::size_t D, typename Metric>
void expect_identical(const ct::ann_forest<Point, D, Metric>& a, const ct::ann_forest<Point, D, Metric>& b)
{
    EXPECT_EQ(a.layout().points, b.layout().points);
    EXPECT_EQ(a.layout().trees, b.layout().trees);
    EXPECT_EQ(a.layout().leaf_size, b.layout().leaf_size);
    EXPECT_EQ(a.layout().nodes, b.layout().nodes);
    ASSERT_EQ(a.nodes().size(), b.nodes().size());
    for (std::size_t i = 0; i < a.nodes().size(); ++i) {
        const auto &x = a.nodes()[i], &y = b.nodes()[i];
        EXPECT_TRUE(x.left == y.left && x.right == y.right && x.first == y.first && x.count == y.count && x.split == y.split && x.offset == y.offset)
            << "node " << i;
    }
    ASSERT_EQ(a.items().size(), b.items().size());
    EXPECT_TRUE(std::equal(a.items().begin(), a.items().end(), b.items().begin()));
    ASSERT_EQ(a.normals().size(), b.normals().size());
    EXPECT_TRUE(std::equal(a.normals().begin(), a.normals().end(), b.normals().begin()));
}

TEST(ContainersAnnForestDeterminism, TheSameSeedBuildsTheSameIndex)
{
    constexpr std::size_t D = 12;
    st::wyrand g(10);
    const auto pts = clustered<D>(g, 2000, 20, 0.2);
    const built<vec<D>, D> a(pts, 6, 16, 99), b(pts, 6, 16, 99), c(pts, 6, 16, 100);
    ASSERT_TRUE(a.f && b.f && c.f);
    expect_identical(*a.f, *b.f);
    EXPECT_FALSE(std::equal(a.f->items().begin(), a.f->items().end(), c.f->items().begin())) << "another seed, another index";
    const built<vec<D>, D, mt::angular> d(pts, 4, 16, 5), e(pts, 4, 16, 5);
    ASSERT_TRUE(d.f && e.f);
    expect_identical(*d.f, *e.f);
}

TEST(ContainersAnnForestAttach, AnIndexAttachedToCopiesOfItsArraysAnswersIdentically)
{
    constexpr std::size_t D = 12;
    st::wyrand g(11);
    const auto pts = clustered<D>(g, 2000, 20, 0.2);
    const built<vec<D>, D> ix(pts, 6, 16, 11);
    ASSERT_TRUE(ix.f);
    // As if written to a file and mapped back: the layout and three flat arrays
    const ct::ann_layout layout = ix.f->layout();
    const std::vector<ct::ann_node> nodes(ix.f->nodes().begin(), ix.f->nodes().end());
    const std::vector<std::uint32_t> items(ix.f->items().begin(), ix.f->items().end());
    const std::vector<float> normals(ix.f->normals().begin(), ix.f->normals().end());
    const ct::ann_forest<vec<D>, D> attached(layout, pts, nodes, items, normals);
    EXPECT_EQ(attached.size(), pts.size());
    expect_identical(*ix.f, attached);
    EXPECT_NE(attached.nodes().data(), ix.f->nodes().data());
    for (int q = 0; q < 50; ++q) {
        const auto query = clustered<D>(g, 1, 20, 0.2)[0];
        const auto a = query_checked(*ix.f, pts, query, 10, 300, mt::euclidean{});
        const auto b = query_checked(attached, pts, query, 10, 300, mt::euclidean{});
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i].index, b[i].index);
            EXPECT_EQ(a[i].distance, b[i].distance);
        }
    }
}

// ---- structure

template <typename Forest>
void expect_well_formed(const Forest& f, std::size_t n, std::size_t trees, std::size_t leaf, std::size_t D, bool bits)
{
    const auto& l = f.layout();
    EXPECT_EQ(l.points, n);
    EXPECT_EQ(l.trees, trees);
    EXPECT_EQ(l.leaf_size, leaf);
    EXPECT_EQ(l.nodes, f.nodes().size());
    EXPECT_GE(l.nodes, trees);
    EXPECT_EQ(f.items().size(), n * trees);
    EXPECT_EQ(f.normals().size(), bits ? 0u : l.nodes * D);
    EXPECT_EQ(f.size(), n);
    std::vector<int> reached(f.nodes().size(), 0);
    for (std::size_t t = 0; t < trees; ++t) {
        std::vector<std::uint32_t> got;
        std::vector<std::int32_t> stack{static_cast<std::int32_t>(t)};
        while (!stack.empty()) {
            const auto k = static_cast<std::size_t>(stack.back());
            stack.pop_back();
            ASSERT_LT(k, f.nodes().size());
            ASSERT_EQ(reached[k]++, 0) << "node " << k << " reached twice";
            const auto& node = f.nodes()[k];
            if (node.count > 0) {
                EXPECT_LE(node.count, leaf) << "leaf " << k;
                EXPECT_GE(node.first, t * n);
                EXPECT_LE(node.first + node.count, (t + 1) * n);
                for (std::uint32_t i = 0; i < node.count; ++i)
                    got.push_back(f.items()[node.first + i]);
            } else {
                EXPECT_GE(node.left, static_cast<std::int32_t>(trees));
                EXPECT_GE(node.right, static_cast<std::int32_t>(trees));
                stack.push_back(node.left);
                stack.push_back(node.right);
            }
        }
        std::sort(got.begin(), got.end());
        std::vector<std::uint32_t> all(n);
        std::iota(all.begin(), all.end(), 0u);
        EXPECT_EQ(got, all) << "tree " << t << ": every point exactly once";
    }
    EXPECT_TRUE(std::all_of(reached.begin(), reached.end(), [](int r) { return r == 1; })) << "every node belongs to one tree";
}

TEST(ContainersAnnForestStructure, EveryPointOncePerTreeAndLeavesWithinTheirSize)
{
    constexpr std::size_t D = 10;
    st::wyrand g(12);
    const auto pts = clustered<D>(g, 1500, 12, 0.25);
    for (const std::size_t leaf : {1u, 7u, 32u}) {
        const built<vec<D>, D> ix(pts, 4, leaf, leaf);
        ASSERT_TRUE(ix.f);
        expect_well_formed(*ix.f, pts.size(), 4, leaf, D, false);
    }
    const built<vec<D>, D, mt::angular> ang(pts, 3, 16, 3);
    ASSERT_TRUE(ang.f);
    expect_well_formed(*ang.f, pts.size(), 3, 16, D, false);
}

TEST(ContainersAnnForestStructure, AngularSplitsPassThroughTheOriginWithUnitNormals)
{
    constexpr std::size_t D = 10;
    st::wyrand g(13);
    const auto pts = clustered<D>(g, 1500, 12, 0.25);
    const built<vec<D>, D, mt::angular> ix(pts, 3, 16, 13);
    ASSERT_TRUE(ix.f);
    for (std::size_t k = 0; k < ix.f->nodes().size(); ++k) {
        const auto& node = ix.f->nodes()[k];
        if (node.count > 0)
            continue;
        EXPECT_EQ(node.offset, 0.0f);
        double s = 0;
        for (std::size_t c = 0; c < D; ++c)
            s += static_cast<double>(ix.f->normals()[k * D + c]) * ix.f->normals()[k * D + c];
        EXPECT_NEAR(s, 1.0, 1e-5) << "node " << k;
    }
}

TEST(ContainersAnnForestStructure, BitSplitsStoreNoNormals)
{
    using code = std::array<std::uint64_t, 2>;
    st::wyrand g(14);
    std::vector<code> pts(700);
    for (auto& c : pts)
        c = {g(), g()};
    const built<code, 2, mt::hamming> ix(pts, 5, 8, 14);
    ASSERT_TRUE(ix.f);
    expect_well_formed(*ix.f, pts.size(), 5, 8, 2, true);
    for (const auto& node : ix.f->nodes())
        if (node.count == 0) {
            EXPECT_LT(node.split, 128u);
        }
}

TEST(ContainersAnnForestStructure, SuggestedStorageIsAsDocumented)
{
    using f = ct::ann_forest<vec<4>, 4>;
    constexpr auto s = f::suggested_storage(100, 3, 10);
    static_assert(s.nodes == 3 * (4 * 10 + 3) && s.items == 300 && s.normals == s.nodes * 4);
    constexpr auto ragged = f::suggested_storage(101, 2, 10);   // ceil(101 / 10) = 11 leaves
    static_assert(ragged.nodes == 2 * (4 * 11 + 3) && ragged.items == 202);
    constexpr auto zero_leaf = f::suggested_storage(5, 1, 0);   // a zero leaf size counts as 1
    static_assert(zero_leaf.nodes == 4 * 5 + 3);
    constexpr auto bits = ct::ann_forest<std::array<std::uint64_t, 2>, 2, mt::hamming>::suggested_storage(100, 3, 10);
    static_assert(bits.normals == 0 && bits.nodes == s.nodes);
    EXPECT_EQ(s.items, 300u);
}

// ---- edge cases

TEST(ContainersAnnForestEdges, OnePointIsItsOwnLeafAndTheOnlyAnswer)
{
    const std::vector<vec<3>> one{{1, 2, 3}};
    const built<vec<3>, 3> ix(one, 4, 8, 1);
    ASSERT_TRUE(ix.f);
    expect_well_formed(*ix.f, 1, 4, 8, 3, false);
    EXPECT_EQ(ix.f->layout().nodes, 4u);   // one leaf per tree, no splits
    const auto got = query_checked(*ix.f, one, vec<3>{0, 0, 0}, 5, 10, mt::euclidean{});
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0].index, 0u);
}

TEST(ContainersAnnForestEdges, FewerPointsThanALeafAreFoundExactly)
{
    const std::vector<vec<2>> pts{{0, 0}, {3, 4}, {1, 1}, {-2, 0}, {5, 5}};
    const built<vec<2>, 2> ix(pts, 3, 10, 2);
    ASSERT_TRUE(ix.f);
    EXPECT_EQ(ix.f->layout().nodes, 3u);
    for (const auto& node : ix.f->nodes())
        EXPECT_EQ(node.count, 5u);
    const auto truth = brute(pts, vec<2>{1, 0}, 5, mt::euclidean{});
    const auto got = query_checked(*ix.f, pts, vec<2>{1, 0}, 5, 15, mt::euclidean{});
    ASSERT_EQ(got.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_EQ(got[i].distance, truth[i].first);
}

TEST(ContainersAnnForestEdges, IdenticalPointsTakeTheDegenerateSplit)
{
    const std::vector<vec<4>> same(300, vec<4>{1, -2, 3, 0.5f});
    const built<vec<4>, 4> ix(same, 3, 8, 3);
    ASSERT_TRUE(ix.f);
    expect_well_formed(*ix.f, same.size(), 3, 8, 4, false);
    // No hyperplane separates identical points: every split is neutral (a zero normal), so both sides are explored
    for (std::size_t k = 0; k < ix.f->nodes().size(); ++k) {
        if (ix.f->nodes()[k].count > 0)
            continue;
        EXPECT_EQ(ix.f->nodes()[k].offset, 0.0f);
        for (std::size_t c = 0; c < 4; ++c)
            EXPECT_EQ(ix.f->normals()[k * 4 + c], 0.0f);
    }
    const auto got = query_checked(*ix.f, same, vec<4>{1, -2, 3, 0.5f}, 20, 100, mt::euclidean{});
    ASSERT_EQ(got.size(), 20u);
    for (const auto& o : got)
        EXPECT_EQ(o.distance, 0.0);
    // The same under bit splits: no bit varies, so the points are shuffled and halved
    using code = std::array<std::uint64_t, 1>;
    const std::vector<code> codes(200, code{0xf0f0u});
    const built<code, 1, mt::hamming> bits(codes, 2, 8, 4);
    ASSERT_TRUE(bits.f);
    expect_well_formed(*bits.f, codes.size(), 2, 8, 1, true);
    EXPECT_EQ(query_checked(*bits.f, codes, code{0xf0f1u}, 10, 50, mt::hamming{}).size(), 10u);
}

TEST(ContainersAnnForestEdges, OneTreeIsAnIndex)
{
    constexpr std::size_t D = 8;
    st::wyrand g(15);
    const auto pts = clustered<D>(g, 1000, 10, 0.2);
    const auto queries = clustered<D>(g, 50, 10, 0.2);
    const built<vec<D>, D> ix(pts, 1, 16, 15);
    ASSERT_TRUE(ix.f);
    expect_well_formed(*ix.f, pts.size(), 1, 16, D, false);
    EXPECT_GE(recall(*ix.f, pts, queries, 5, 300, mt::euclidean{}), 0.7);
}

TEST(ContainersAnnForestEdges, ABudgetSmallerThanALeafStopsWithinIt)
{
    constexpr std::size_t D = 4;
    st::wyrand g(16);
    const auto pts = clustered<D>(g, 400, 4, 0.3);
    const built<vec<D>, D> ix(pts, 4, 32, 16);
    ASSERT_TRUE(ix.f);
    const auto got = query_checked(*ix.f, pts, pts[17], 10, 3, mt::euclidean{});
    EXPECT_GE(got.size(), 1u);
    EXPECT_LE(got.size(), 3u);
}

TEST(ContainersAnnForestEdges, AFullFrontierLowersRecallButNotExactness)
{
    constexpr std::size_t D = 8;
    st::wyrand g(17);
    const auto pts = clustered<D>(g, 2000, 10, 0.2);
    const built<vec<D>, D> ix(pts, 8, 16, 17);
    ASSERT_TRUE(ix.f);
    for (int q = 0; q < 30; ++q) {
        const auto query = clustered<D>(g, 1, 10, 0.2)[0];
        // A frontier of one: only the first root enters, and each split keeps only its positive side
        const auto got = query_checked(*ix.f, pts, query, 10, 1000, mt::euclidean{}, 1);
        EXPECT_GE(got.size(), 1u);
        EXPECT_LE(got.size(), 16u);
    }
}

TEST(ContainersAnnForestEdges, KBeyondTheSizeReturnsEveryPoint)
{
    constexpr std::size_t D = 3;
    st::wyrand g(18);
    const auto pts = clustered<D>(g, 20, 2, 0.5);
    const built<vec<D>, D> ix(pts, 3, 4, 18);
    ASSERT_TRUE(ix.f);
    const auto got = query_checked(*ix.f, pts, vec<D>{0, 0, 0}, 50, 60, mt::euclidean{}, 512);
    ASSERT_EQ(got.size(), 20u);
    const auto truth = brute(pts, vec<D>{0, 0, 0}, 50, mt::euclidean{});
    for (std::size_t i = 0; i < 20; ++i)
        EXPECT_EQ(got[i].index, truth[i].second);
}

TEST(ContainersAnnForestEdges, EmptyOutputCandidateOrFrontierSpansAnswerNothing)
{
    const std::vector<vec<2>> pts{{0, 0}, {1, 1}, {2, 2}};
    const built<vec<2>, 2> ix(pts, 2, 1, 1);
    ASSERT_TRUE(ix.f);
    std::array<ct::ann_neighbour, 2> out{};
    std::array<std::uint32_t, 8> cand{};
    std::array<ct::ann_frontier, 8> front{};
    const vec<2> q{0, 0};
    EXPECT_EQ(ix.f->query(q, std::span<ct::ann_neighbour>{}, cand, front), 0u);
    EXPECT_EQ(ix.f->query(q, out, std::span<std::uint32_t>{}, front), 0u);
    EXPECT_EQ(ix.f->query(q, out, cand, std::span<ct::ann_frontier>{}), 0u);
    EXPECT_EQ(ix.f->query(q, out, cand, front), 2u);
}

// ---- errors

TEST(ContainersAnnForestErrors, InvalidParametersAreRefused)
{
    using f = ct::ann_forest<vec<2>, 2>;
    const std::vector<vec<2>> pts{{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    std::vector<ct::ann_node> nodes(64);
    std::vector<std::uint32_t> items(16);
    std::vector<float> normals(128);
    st::wyrand g(1);
    auto refused = [](const auto& r) { return r.is_err() && r.unwrap_err() == ct::ann_error::invalid_parameters; };
    EXPECT_TRUE(refused(f::build(std::span<const vec<2>>{}, 2, 2, nodes, items, normals, g)));    // no points
    EXPECT_TRUE(refused(f::build(pts, 0, 2, nodes, items, normals, g)));                         // no trees
    EXPECT_TRUE(refused(f::build(pts, 2, 0, nodes, items, normals, g)));                         // a zero leaf size
    EXPECT_TRUE(refused(f::build(pts, 2, 2, nodes, std::span(items).first(7), normals, g)));   // items < points · trees
    EXPECT_TRUE(f::build(pts, 2, 2, nodes, std::span(items).first(8), normals, g).is_ok());     // exactly enough

    // A std::vector point carries its dimension at run time: one of the wrong length is refused
    using dyn = ct::ann_forest<std::vector<float>, 3>;
    std::vector<std::vector<float>> vs{{0, 0, 0}, {1, 1, 1}, {2, 2}};
    EXPECT_TRUE(refused(dyn::build(vs, 1, 1, nodes, items, normals, g)));
    vs[2].push_back(2);
    const auto fine = dyn::build(vs, 1, 1, nodes, items, normals, g);
    ASSERT_TRUE(fine.is_ok());
    std::array<ct::ann_neighbour, 1> out{};
    std::array<std::uint32_t, 8> cand{};
    std::array<ct::ann_frontier, 8> front{};
    ASSERT_EQ(fine.unwrap().query(std::vector<float>{1.9f, 2.1f, 2}, out, cand, front), 1u);
    EXPECT_EQ(out[0].index, 2u);
}

TEST(ContainersAnnForestErrors, TooLittleNodeOrNormalStorageIsOutOfStorage)
{
    constexpr std::size_t D = 4;
    using f = ct::ann_forest<vec<D>, D>;
    st::wyrand g(19);
    const auto pts = clustered<D>(g, 500, 5, 0.3);
    std::vector<std::uint32_t> items(pts.size() * 3);
    std::vector<ct::ann_node> nodes(f::suggested_storage(pts.size(), 3, 8).nodes);
    std::vector<float> normals(f::suggested_storage(pts.size(), 3, 8).normals);
    auto short_of_storage = [](const auto& r) { return r.is_err() && r.unwrap_err() == ct::ann_error::out_of_storage; };
    EXPECT_TRUE(short_of_storage(f::build(pts, 3, 8, std::span(nodes).first(2), items, normals, g)));    // fewer nodes than trees
    EXPECT_TRUE(short_of_storage(f::build(pts, 3, 8, std::span(nodes).first(20), items, normals, g)));   // too few to split
    EXPECT_TRUE(short_of_storage(f::build(pts, 3, 8, nodes, items, std::span<float>{}, g)));            // no normals
    EXPECT_TRUE(short_of_storage(f::build(pts, 3, 8, nodes, items, std::span(normals).first(3 * D), g)));
    EXPECT_TRUE(f::build(pts, 3, 8, nodes, items, normals, g).is_ok());
    // Bit splits need no normals at all
    using code = std::array<std::uint64_t, 1>;
    std::vector<code> codes(100);
    for (auto& c : codes)
        c = {g()};
    EXPECT_TRUE((ct::ann_forest<code, 1, mt::hamming>::build(codes, 3, 8, nodes, items, std::span<float>{}, g).is_ok()));
}

// ---- extensions of the caller's own

TEST(ContainersAnnForestExtensions, APointTypeOfTheCallersOwnThroughPointTraits)
{
    static_assert(mt::point<sample>);
    st::wyrand g(20);
    std::vector<sample> pts(1000);
    for (auto& p : pts)
        p = {static_cast<float>(st::uniform_real(g)), static_cast<float>(st::uniform_real(g)), static_cast<float>(st::uniform_real(g))};
    const built<sample, 3> ix(pts, 6, 10, 20);
    ASSERT_TRUE(ix.f);
    expect_well_formed(*ix.f, pts.size(), 6, 10, 3, false);
    std::vector<sample> queries(50);
    for (auto& p : queries)
        p = {static_cast<float>(st::uniform_real(g)), static_cast<float>(st::uniform_real(g)), static_cast<float>(st::uniform_real(g))};
    EXPECT_GE(recall(*ix.f, pts, queries, 5, 200, mt::euclidean{}), 0.9);
}

TEST(ContainersAnnForestExtensions, AMetricOfTheCallersOwnWithItsSplitDeclared)
{
    static_assert(std::is_same_v<ct::ann_default_split<scaled_cosine>::type, ct::ann_angular_split>);
    static_assert(std::is_same_v<ct::ann_forest<vec<8>, 8, scaled_cosine>, ct::ann_forest<vec<8>, 8, scaled_cosine, ct::ann_angular_split>>);
    static_assert(std::is_same_v<ct::ann_default_split<mt::chebyshev>::type, ct::ann_euclidean_split>);   // the primary template
    constexpr std::size_t D = 8;
    st::wyrand g(21);
    const auto pts = clustered<D>(g, 1500, 15, 0.2);
    const auto queries = clustered<D>(g, 50, 15, 0.2);
    const built<vec<D>, D, scaled_cosine> ix(pts, 8, 16, 21);
    ASSERT_TRUE(ix.f);
    for (const auto& node : ix.f->nodes())
        if (node.count == 0) {
            EXPECT_EQ(node.offset, 0.0f);   // the angular split: through the origin
        }
    EXPECT_GE(recall(*ix.f, pts, queries, 5, 400, scaled_cosine{}), 0.9);
}

// ---- layout contract and compile time

static_assert(std::is_trivially_copyable_v<ct::ann_node> && std::is_standard_layout_v<ct::ann_node>);
static_assert(sizeof(ct::ann_node) == 24 && alignof(ct::ann_node) == 4);
static_assert(std::is_trivially_copyable_v<ct::ann_layout> && sizeof(ct::ann_layout) == 16);
static_assert(std::is_trivially_copyable_v<ct::ann_frontier> && sizeof(ct::ann_frontier) == 8);
static_assert(std::is_trivially_copyable_v<ct::ann_neighbour>);
static_assert(std::is_trivially_copyable_v<ct::ann_forest<vec<4>, 4>>);
static_assert(std::is_same_v<ct::ann_default_split<mt::euclidean>::type, ct::ann_euclidean_split>);
static_assert(std::is_same_v<ct::ann_default_split<mt::manhattan>::type, ct::ann_euclidean_split>);
static_assert(std::is_same_v<ct::ann_default_split<mt::angular>::type, ct::ann_angular_split>);
static_assert(std::is_same_v<ct::ann_default_split<mt::cosine_distance>::type, ct::ann_angular_split>);
static_assert(std::is_same_v<ct::ann_default_split<mt::angle>::type, ct::ann_angular_split>);
static_assert(std::is_same_v<ct::ann_default_split<mt::inner_product>::type, ct::ann_angular_split>);
static_assert(std::is_same_v<ct::ann_default_split<mt::hamming>::type, ct::ann_bit_split>);

constexpr bool compile_time()
{
    constexpr std::array<vec<2>, 8> pts{{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {10, 10}, {11, 10}, {10, 11}, {11, 11}}};
    using f = ct::ann_forest<vec<2>, 2>;
    constexpr auto s = f::suggested_storage(8, 2, 2);
    std::array<ct::ann_node, s.nodes> nodes{};
    std::array<std::uint32_t, s.items> items{};
    std::array<float, s.normals> normals{};
    st::wyrand g(3);
    const auto r = f::build(pts, 2, 2, nodes, items, normals, g);
    if (!r)
        return false;
    std::array<ct::ann_neighbour, 1> out{};
    std::array<std::uint32_t, 16> cand{};
    std::array<ct::ann_frontier, 16> front{};
    return r.unwrap().query(vec<2>{10.2f, 10.9f}, out, cand, front) == 1 && out[0].index == 6;
}
static_assert(compile_time());

} // namespace
