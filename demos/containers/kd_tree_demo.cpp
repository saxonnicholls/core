//
//  kd_tree_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/kd_tree.hpp>: a tracker's scan of detections indexed once, then for each
//  track the nearest detection, the k nearest, every detection inside a gate radius and inside a gate box, each
//  checked against a linear scan; the same scan under the Manhattan metric, a detection type of the caller's own
//  through point_traits, and the empty tree. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/containers/kd_tree.hpp>

#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ct = snicholls::containers;
namespace mt = snicholls::metric;
namespace st = snicholls::stats;

int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

using xy = std::array<double, 2>;

// A detection type of the caller's own
struct detection {
    float east = 0, north = 0;
    int sensor = 0;
};

} // namespace

// ... made a point by specialising point_traits, as std::hash is specialised
template <>
struct snicholls::metric::point_traits<detection> {
    using value_type = float;
    static constexpr std::size_t extent = 2;
    static constexpr std::size_t dimension(const detection&) noexcept { return 2; }
    static constexpr float at(const detection& d, std::size_t axis) noexcept { return axis == 0 ? d.east : d.north; }
};

int main()
{
    std::printf("kd_tree: neighbour queries over the caller's points, built in place\n");
    st::wyrand g(2026);
    std::vector<xy> detections(400);
    for (auto& d : detections)
        d = {st::uniform_real(g) * 1000, st::uniform_real(g) * 1000};

    //! [kd_tree]
    // One scan: index the detections (the tree keeps only a permutation, in storage the caller gives it)
    std::vector<std::uint32_t> order(detections.size());
    const ct::kd_tree<xy, 2> tree(detections, order);

    const xy track{500, 500};   // a track's predicted position
    const auto closest = tree.nearest(track);                  // index and distance, in metres

    std::array<ct::kd_neighbour, 5> five{};
    const std::size_t found = tree.k_nearest(track, five);     // nearest first

    std::array<ct::kd_neighbour, 5> near_five{};              // the five nearest, but only within 40 m (kriging)
    const std::size_t near_found = tree.k_nearest_within(track, 40.0, near_five);

    std::vector<std::uint32_t> gated;                         // every detection within 40 m
    tree.for_each_within(track, 40.0, [&](std::uint32_t i, double) { gated.push_back(i); });

    std::vector<std::uint32_t> boxed;                         // every detection in the gate's bounding box
    tree.for_each_in_box(xy{460, 460}, xy{540, 540}, [&](std::uint32_t i) { boxed.push_back(i); });

    // The metric is a template parameter: the same scan, gated by city-block distance
    std::vector<std::uint32_t> order_l1(detections.size());
    const ct::kd_tree<xy, 2, mt::manhattan> by_blocks(detections, order_l1);
    const auto closest_l1 = by_blocks.nearest(track);
    //! [kd_tree]

    std::printf("    %zu detections; nearest #%u at %.2f m, %zu within 40 m, %zu in the box; by blocks #%u at %.2f m\n", tree.size(),
                closest ? closest->index : 0u, closest ? closest->distance : 0.0, gated.size(), boxed.size(), closest_l1 ? closest_l1->index : 0u,
                closest_l1 ? closest_l1->distance : 0.0);

    // The same answers by a linear scan
    const mt::euclidean euclid = tree.distance_metric();
    const mt::manhattan blocks = by_blocks.distance_metric();
    std::vector<double> all, all_l1;
    for (const auto& d : detections) {
        all.push_back(euclid(d, track));
        all_l1.push_back(blocks(d, track));
    }
    std::vector<double> sorted(all);
    std::sort(sorted.begin(), sorted.end());
    expect(tree.size() == 400 && !tree.empty(), "size() and empty()");
    expect(closest && closest->distance == sorted[0] && all[closest->index] == sorted[0], "nearest: the linear scan's minimum");
    bool k_ok = found == 5;
    for (std::size_t i = 0; i < found; ++i)
        k_ok = k_ok && five[i].distance == sorted[i] && all[five[i].index] == sorted[i];
    expect(k_ok, "k_nearest: the five smallest distances, nearest first");
    std::size_t in_radius = 0, in_box = 0;
    for (const auto& d : detections) {
        in_radius += euclid.reduced(d, track) <= 40.0 * 40.0;
        in_box += d[0] >= 460 && d[0] <= 540 && d[1] >= 460 && d[1] <= 540;
    }
    expect(gated.size() == in_radius, "for_each_within: every point inside the radius");
    expect(boxed.size() == in_box && boxed.size() >= gated.size(), "for_each_in_box: every point in the box, a superset of the disc");
    expect(closest_l1 && closest_l1->distance == *std::min_element(all_l1.begin(), all_l1.end()) && all_l1[closest_l1->index] == closest_l1->distance,
           "manhattan: the nearest by city-block distance");

    // A point type of the caller's own, through point_traits; the query may be another point type
    const std::array<detection, 3> mine{{{10, 10, 1}, {200, 50, 2}, {15, 12, 3}}};
    std::array<std::uint32_t, 3> mine_order{};
    const ct::kd_tree<detection, 2> by_traits(mine, mine_order);
    const auto m = by_traits.nearest(detection{14, 14, 0});
    const auto m2 = by_traits.nearest(xy{190, 60});
    expect(m && mine[m->index].sensor == 3 && m2 && mine[m2->index].sensor == 2, "point_traits: the nearest of the caller's own type");

    // An empty scan answers nothing
    const ct::kd_tree<xy, 2> empty(std::span<const xy>{}, std::span<std::uint32_t>{});
    expect(empty.empty() && !empty.nearest(track) && empty.k_nearest(track, five) == 0, "an empty tree: no nearest, no neighbours");
    expect(near_found <= found && near_found <= gated.size(), "k nearest within 40 m is no more than k nearest, nor than all within 40 m");
    for (std::size_t i = 0; i < near_found; ++i)
        expect(near_five[i].distance < 40.0, "every bounded neighbour is strictly within the radius");

    std::printf(failures == 0 ? "kd_tree demo: ok\n" : "kd_tree demo: FAILED\n");
    return failures == 0 ? 0 : 1;
}
