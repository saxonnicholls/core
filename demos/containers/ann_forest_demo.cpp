//
//  ann_forest_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/ann_forest.hpp>: 5,000 clustered 24-dimensional embeddings indexed by a forest of
//  random-projection trees in storage the caller owns, queried for their ten nearest under the Euclidean and the
//  angular metric and compared with brute force (recall and exact distances); the index rebuilt from the same seed,
//  attached again from copies of its three arrays, a hamming index over packed bits, and both errors. Exits non-zero
//  if a result is not what it documents.
//

#include <snicholls/core/containers/ann_forest.hpp>

#include <snicholls/core/metric/angular.hpp>
#include <snicholls/core/metric/binary.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace ct = snicholls::containers;
namespace mt = snicholls::metric;
namespace st = snicholls::stats;

int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

constexpr std::size_t D = 24;
using embedding = std::array<float, D>;

// Embeddings in 40 clusters: centres uniform in [-1, 1]^D, members normal around them
std::vector<embedding> clustered(st::wyrand& g, const std::vector<embedding>& centres, std::size_t n)
{
    st::normal_box_muller z;
    std::vector<embedding> out(n);
    for (auto& p : out) {
        const auto& c = centres[st::uniform_index(g, centres.size())];
        for (std::size_t i = 0; i < D; ++i)
            p[i] = static_cast<float>(c[i] + 0.15 * z(g));
    }
    return out;
}

// The true k nearest by brute force
template <typename Point, typename Metric>
std::vector<std::uint32_t> brute(const std::vector<Point>& pts, const Point& q, std::size_t k, const Metric& m)
{
    std::vector<std::pair<double, std::uint32_t>> all(pts.size());
    for (std::uint32_t i = 0; i < pts.size(); ++i)
        all[i] = {static_cast<double>(m(pts[i], q)), i};
    std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k), all.end());
    std::vector<std::uint32_t> out(k);
    for (std::size_t i = 0; i < k; ++i)
        out[i] = all[i].second;
    return out;
}

// Recall@10 of a forest over the queries, and whether every distance it returned was exact and nearest first
template <typename Forest, typename Metric>
std::pair<double, bool> evaluate(const Forest& forest, const std::vector<embedding>& pts, const std::vector<embedding>& queries, const Metric& m)
{
    std::array<ct::ann_neighbour, 10> ten{};
    std::vector<std::uint32_t> candidates(10 * 10 * 10);   // trees · k · 10
    std::array<ct::ann_frontier, 512> frontier{};
    std::size_t hit = 0;
    bool exact = true;
    for (const auto& q : queries) {
        const std::size_t got = forest.query(q, ten, candidates, frontier);
        exact = exact && got == 10;
        for (std::size_t i = 0; i < got; ++i)
            exact = exact && ten[i].distance == static_cast<double>(m(pts[ten[i].index], q)) && (i == 0 || ten[i - 1].distance <= ten[i].distance);
        for (const std::uint32_t t : brute(pts, q, 10, m))
            hit += std::any_of(ten.begin(), ten.begin() + static_cast<std::ptrdiff_t>(got), [&](const ct::ann_neighbour& o) { return o.index == t; });
    }
    return {static_cast<double>(hit) / static_cast<double>(queries.size() * 10), exact};
}

} // namespace

int main()
{
    std::printf("ann_forest: approximate nearest neighbours by a forest of random-projection trees\n");
    st::wyrand data(2026);
    std::vector<embedding> centres(40);
    for (auto& c : centres)
        for (auto& x : c)
            x = static_cast<float>(st::uniform_real(data) * 2 - 1);
    const std::vector<embedding> points = clustered(data, centres, 5000);
    const std::vector<embedding> queries = clustered(data, centres, 100);

    //! [ann_forest]
    // Storage the caller owns, sized for 10 trees with leaves of at most 16 points
    using forest = ct::ann_forest<embedding, D>;   // euclidean by default; the split follows the metric
    const ct::ann_storage size = forest::suggested_storage(points.size(), 10, 16);
    std::vector<ct::ann_node> nodes(size.nodes);
    std::vector<std::uint32_t> items(size.items);
    std::vector<float> normals(size.normals);

    st::wyrand g(7);   // the caller's generator: the same seed builds the same index
    const auto built = forest::build(points, 10, 16, nodes, items, normals, g);
    if (!built)
        return 1;
    const forest index = built.unwrap();

    // The ten nearest of one query, examining at most 1,000 candidates; every distance returned is exact
    std::array<ct::ann_neighbour, 10> ten{};
    std::vector<std::uint32_t> candidates(1000);
    std::array<ct::ann_frontier, 512> frontier{};
    const std::size_t found = index.query(queries[0], ten, candidates, frontier);

    // The index is three flat arrays and a layout: write them out, map them back, attach without rebuilding
    const ct::ann_layout layout = index.layout();
    const std::vector<ct::ann_node> saved_nodes(index.nodes().begin(), index.nodes().end());
    const std::vector<std::uint32_t> saved_items(index.items().begin(), index.items().end());
    const std::vector<float> saved_normals(index.normals().begin(), index.normals().end());
    const forest attached(layout, points, saved_nodes, saved_items, saved_normals);
    //! [ann_forest]

    std::printf("    %zu points, %u trees, %u nodes; the nearest of query 0 is #%u at %.4f\n", index.size(), layout.trees, layout.nodes,
                found > 0 ? ten[0].index : 0u, found > 0 ? ten[0].distance : 0.0);
    expect(index.size() == 5000 && layout.points == 5000 && layout.trees == 10 && layout.leaf_size == 16, "layout(): the points, trees and leaf size built");
    expect(index.nodes().size() == layout.nodes && index.items().size() == 5000 * 10 && index.normals().size() == layout.nodes * D,
           "nodes(), items(), normals(): the three arrays, as sized");
    expect(size.items == 50000 && size.normals == size.nodes * D, "suggested_storage: items = points x trees, normals = nodes x D");

    const auto [r_euclid, exact_euclid] = evaluate(index, points, queries, mt::euclidean{});
    std::printf("    euclidean recall@10 over 100 queries: %.3f\n", r_euclid);
    expect(found == 10, "query: ten neighbours written");
    expect(exact_euclid, "query: every distance exact, nearest first");
    expect(r_euclid >= 0.9, "euclidean: recall@10 of at least 0.9 against brute force");

    // The same seed builds the same index; the attached copy answers identically
    std::vector<ct::ann_node> nodes2(size.nodes);
    std::vector<std::uint32_t> items2(size.items);
    std::vector<float> normals2(size.normals);
    st::wyrand g2(7);
    const auto again = forest::build(points, 10, 16, nodes2, items2, normals2, g2);
    expect(again && std::equal(items2.begin(), items2.end(), index.items().begin()) && std::equal(normals2.begin(), normals2.begin() + static_cast<std::ptrdiff_t>(layout.nodes * D), index.normals().begin()),
           "build: the same seed, the same index");
    std::array<ct::ann_neighbour, 10> ten2{};
    const std::size_t found2 = attached.query(queries[0], ten2, candidates, frontier);
    bool same = found2 == found;
    for (std::size_t i = 0; i < found && same; ++i)
        same = ten2[i].index == ten[i].index && ten2[i].distance == ten[i].distance;
    expect(same, "the constructor from a layout and spans: attached, the same answers");

    // Angular: the split passes through the origin (Annoy's "angular")
    std::vector<ct::ann_node> anodes(size.nodes);
    std::vector<std::uint32_t> aitems(size.items);
    std::vector<float> anormals(size.normals);
    using angular_forest = ct::ann_forest<embedding, D, mt::angular>;
    const auto angular = angular_forest::build(points, 10, 16, anodes, aitems, anormals, g);
    const auto [r_angular, exact_angular] = angular ? evaluate(angular.unwrap(), points, queries, mt::angular{}) : std::pair{0.0, false};
    std::printf("    angular recall@10 over 100 queries: %.3f\n", r_angular);
    expect(angular && exact_angular && r_angular >= 0.9, "angular: recall@10 of at least 0.9, every distance exact");

    // Hamming over 128-bit codes: a planted near-duplicate (5 bits flipped) is found
    using code = std::array<std::uint64_t, 2>;
    std::vector<code> codes(2000);
    for (auto& c : codes)
        c = {data(), data()};
    using bit_forest = ct::ann_forest<code, 2, mt::hamming>;
    const auto bsize = bit_forest::suggested_storage(codes.size(), 8, 16);
    std::vector<ct::ann_node> bnodes(bsize.nodes);
    std::vector<std::uint32_t> bitems(bsize.items);
    const auto bits = bit_forest::build(codes, 8, 16, bnodes, bitems, std::span<float>{}, g);
    code probe = codes[123];
    for (const unsigned b : {3u, 40u, 64u, 77u, 120u})
        probe[b / 64] ^= std::uint64_t{1} << (b % 64);
    std::array<ct::ann_neighbour, 1> one{};
    const bool near_dup = bits && bits.unwrap().query(probe, one, candidates, frontier) == 1 && one[0].index == 123 && one[0].distance == 5;
    expect(bsize.normals == 0 && near_dup, "hamming: bit splits, no normals, the planted code found at 5 bits");

    // Errors are values
    std::vector<ct::ann_node> tiny(12);
    const auto short_nodes = forest::build(points, 10, 16, tiny, items, normals, g);
    const auto no_trees = forest::build(points, 0, 16, nodes, items, normals, g);
    expect(short_nodes.is_err() && short_nodes.unwrap_err() == ct::ann_error::out_of_storage, "build: too little node storage is out_of_storage");
    expect(no_trees.is_err() && no_trees.unwrap_err() == ct::ann_error::invalid_parameters, "build: no trees is invalid_parameters");

    std::printf(failures == 0 ? "ann_forest demo: ok\n" : "ann_forest demo: FAILED\n");
    return failures == 0 ? 0 : 1;
}
