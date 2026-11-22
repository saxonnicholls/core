//
//  kd_tree_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/kd_tree.hpp>: a tracker's scan of detections indexed once, then for each
//  track the nearest detection, the k nearest, every detection inside a gate radius and inside a gate box, each
//  checked against a linear scan; a custom coordinate reader and the empty tree. Exits non-zero if a result is not
//  what it documents.
//

#include <snicholls/core/containers/kd_tree.hpp>

#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ct = snicholls::containers;
namespace st = snicholls::stats;

int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

using xy = std::array<double, 2>;

double d2(const xy& a, const xy& b) { return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]); }

// A detection type of the caller's own, read through a traits class
struct detection {
    float east = 0, north = 0;
    int sensor = 0;
};
struct detection_traits {
    static constexpr double coordinate(const detection& d, std::size_t axis) noexcept { return axis == 0 ? d.east : d.north; }
};

} // namespace

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
    const auto closest = tree.nearest(track);                  // index and squared distance

    std::array<ct::kd_neighbour, 5> five{};
    const std::size_t found = tree.k_nearest(track, five);     // nearest first

    std::vector<std::uint32_t> gated;                         // every detection within 40 m
    tree.for_each_within(track, 40.0 * 40.0, [&](std::uint32_t i, double) { gated.push_back(i); });

    std::vector<std::uint32_t> boxed;                         // every detection in the gate's bounding box
    tree.for_each_in_box(xy{460, 460}, xy{540, 540}, [&](std::uint32_t i) { boxed.push_back(i); });
    //! [kd_tree]

    std::printf("    %zu detections; nearest #%u at %.2f m, %zu within 40 m, %zu in the box\n", tree.size(), closest ? closest->index : 0u,
                closest ? std::sqrt(closest->distance2) : 0.0, gated.size(), boxed.size());

    // The same answers by a linear scan
    std::vector<double> all;
    for (const auto& d : detections)
        all.push_back(d2(d, track));
    std::vector<double> sorted(all);
    std::sort(sorted.begin(), sorted.end());
    expect(tree.size() == 400 && !tree.empty(), "size() and empty()");
    expect(closest && closest->distance2 == sorted[0] && all[closest->index] == sorted[0], "nearest: the linear scan's minimum");
    bool k_ok = found == 5;
    for (std::size_t i = 0; i < found; ++i)
        k_ok = k_ok && five[i].distance2 == sorted[i];
    expect(k_ok, "k_nearest: the five smallest distances, nearest first");
    std::size_t in_radius = 0, in_box = 0;
    for (const auto& d : detections) {
        in_radius += d2(d, track) <= 1600;
        in_box += d[0] >= 460 && d[0] <= 540 && d[1] >= 460 && d[1] <= 540;
    }
    expect(gated.size() == in_radius, "for_each_within: every point inside the radius");
    expect(boxed.size() == in_box && boxed.size() >= gated.size(), "for_each_in_box: every point in the box, a superset of the disc");

    // A point type of the caller's own, through Traits
    const std::array<detection, 3> mine{{{10, 10, 1}, {200, 50, 2}, {15, 12, 3}}};
    std::array<std::uint32_t, 3> mine_order{};
    const ct::kd_tree<detection, 2, detection_traits> by_traits(mine, mine_order);
    const auto m = by_traits.nearest(detection{14, 14, 0});
    expect(m && mine[m->index].sensor == 3, "a custom Traits: the nearest of the caller's own type");

    // An empty scan answers nothing
    const ct::kd_tree<xy, 2> empty(std::span<const xy>{}, std::span<std::uint32_t>{});
    expect(empty.empty() && !empty.nearest(track) && empty.k_nearest(track, five) == 0, "an empty tree: no nearest, no neighbours");

    std::printf(failures == 0 ? "kd_tree demo: ok\n" : "kd_tree demo: FAILED\n");
    return failures == 0 ? 0 : 1;
}
