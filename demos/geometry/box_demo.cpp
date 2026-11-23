//
//  box_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/geometry/box.hpp>: a detector's YOLO centre-and-size boxes are converted to pixel
//  corners, suppressed by non-maximum suppression and matched to ground truth by IoU and GIoU; the measures, set
//  operations, bounding and squared_distance on the kept boxes; and a geo_box covering a coastline that spans the
//  antimeridian, extended and sampled. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/geometry/box.hpp>

#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <span>

namespace {

namespace geo = snicholls::geometry;
namespace st = snicholls::stats;
using b2d = geo::box<double, 2>;
using a2d = std::array<double, 2>;
constexpr double pi = std::numbers::pi;
constexpr double deg = pi / 180;

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

bool near(double a, double b, double tol = 1e-12)
{
    return std::fabs(a - b) <= tol;
}

} // namespace

int main()
{
    std::printf("snicholls-core geometry box demo\n");
    bool ok = true;

    //! [box]
    // A detector's output on a 640 x 480 frame, YOLO style: centre and size normalised to [0, 1], and a score
    struct detection {
        double cx, cy, w, h, score;
    };
    constexpr std::array<detection, 5> raw{{
        {0.25, 0.50, 0.20, 0.40, 0.92},   // a person
        {0.26, 0.51, 0.20, 0.40, 0.85},   // the same person, a second anchor
        {0.70, 0.40, 0.30, 0.30, 0.78},   // a car
        {0.72, 0.42, 0.30, 0.30, 0.40},   // the car again, weaker
        {0.10, 0.10, 0.05, 0.05, 0.30},   // a lone weak box
    }};
    // To pixel corners: scale the centre-and-size box, then read its corners
    std::array<b2d, raw.size()> boxes{};
    std::array<double, raw.size()> scores{};
    for (std::size_t i = 0; i < raw.size(); ++i) {
        boxes[i] = geo::from_cxcywh(raw[i].cx * 640, raw[i].cy * 480, raw[i].w * 640, raw[i].h * 480);
        scores[i] = raw[i].score;
    }
    // Greedy NMS at IoU 0.5: duplicates of a better box go, best first
    std::array<std::uint32_t, raw.size()> kept{};
    const std::size_t k = geo::non_max_suppression(std::span<const b2d>(boxes), std::span<const double>(scores), 0.5, std::span<std::uint32_t>(kept));
    // Match the kept boxes to the ground truth by IoU; GIoU still ranks a miss by how far off it is
    const b2d truth_person = geo::from_xywh(96.0, 144.0, 128.0, 192.0);
    const b2d truth_car = geo::from_xyxy(544.0, 264.0, 352.0, 120.0);  // corners in any order
    const double person_iou = geo::iou(boxes[kept[0]], truth_person);
    const double car_iou = geo::iou(boxes[kept[1]], truth_car);
    const double lone_giou = geo::giou(boxes[kept[2]], truth_person);

    // A coastline across the antimeridian: Fiji, 21° S to 12° S, 177° E to 178° W
    auto fiji = geo::geo_box<double>::from_degrees(-21, 177, -12, -178);
    fiji = fiji.extend(-19.0 * deg, -178.5 * deg);   // inside already: unchanged
    fiji = fiji.extend(-22.3 * deg, -176.2 * deg);   // 'Ata, Tonga: 1.8° east is shorter than 353.2° west
    const auto centre = fiji.centre();
    st::wyrand g(2026);
    const auto survey_point = fiji.sample(g);        // a uniform point by area, for a survey
    //! [box]

    std::printf("  kept %zu of %zu detections:", k, raw.size());
    for (std::size_t i = 0; i < k; ++i)
        std::printf(" %u", static_cast<unsigned>(kept[i]));
    std::printf("\n  person IoU %.3f, car IoU %.3f, lone box GIoU vs the person %.3f\n", person_iou, car_iou, lone_giou);
    std::printf("  Fiji spans %.2f deg of longitude, centre (%.2f, %.2f) deg, survey point (%.3f, %.3f) deg\n", fiji.longitude_span() / deg,
                centre[0] / deg, centre[1] / deg, survey_point[0] / deg, survey_point[1] / deg);

    std::printf("conversions\n");
    // The person: centre (160, 240), size 128 x 192, corners [96, 224] x [144, 336]
    ok &= check(near(boxes[0].lo[0], 96) && near(boxes[0].hi[0], 224) && near(boxes[0].lo[1], 144) && near(boxes[0].hi[1], 336), "from_cxcywh: the person at [96, 224] x [144, 336]");
    const auto xyxy = geo::to_xyxy(truth_car);
    ok &= check(xyxy == std::array{352.0, 120.0, 544.0, 264.0}, "from_xyxy orders the corners; to_xyxy reads them");
    ok &= check(geo::to_xywh(truth_person) == std::array{96.0, 144.0, 128.0, 192.0}, "to_xywh inverts from_xywh");
    const auto c = geo::to_cxcywh(boxes[0]);
    ok &= check(near(c[0], 160) && near(c[1], 240) && near(c[2], 128) && near(c[3], 192), "to_cxcywh inverts from_cxcywh");

    std::printf("non-maximum suppression and matching\n");
    // IoU(0, 1) and IoU(2, 3) are above 0.5; 4 overlaps nothing
    ok &= check(geo::iou(boxes[0], boxes[1]) > 0.5 && geo::iou(boxes[2], boxes[3]) > 0.5, "the duplicates overlap by more than 0.5");
    ok &= check(k == 3 && kept[0] == 0 && kept[1] == 2 && kept[2] == 4, "NMS keeps 0, 2 and 4, best first");
    ok &= check(near(person_iou, 1.0), "the person matches its ground truth exactly (IoU 1)");
    // The car is [352, 544] x [120, 264], exactly the ground truth
    ok &= check(near(car_iou, 1.0) && near(geo::giou(boxes[kept[1]], truth_car), 1.0), "the car matches too (IoU and GIoU 1)");
    ok &= check(geo::iou(boxes[4], truth_person) == 0 && lone_giou < 0 && lone_giou > -1, "the lone box misses: IoU 0, GIoU in (-1, 0)");
    std::array<std::uint32_t, 2> too_short{};
    ok &= check(geo::non_max_suppression(std::span<const b2d>(boxes), std::span<const double>(scores), 0.5, std::span<std::uint32_t>(too_short)) == 0, "an output span too short keeps nothing");

    std::printf("measures and set operations\n");
    const b2d& person = boxes[0];
    const b2d& person2 = boxes[1];
    ok &= check(!geo::empty(person) && near(geo::extent(person, 0), 128) && near(geo::area(person), 128.0 * 192.0), "extent and area of the person");
    ok &= check(near(geo::volume(person), geo::area(person)) && near(geo::centre(person)[0], 160) && near(geo::centre(person)[1], 240), "volume is area in the plane; the centre");
    const b2d both = geo::intersection(person, person2);
    const b2d either = geo::hull(person, person2);
    ok &= check(geo::contains(person, both) && geo::contains(either, person) && geo::contains(either, person2), "intersection inside each; the hull holds both");
    ok &= check(geo::intersects(person, person2) && !geo::intersects(person, boxes[2]), "the duplicates intersect; the person and the car do not");
    ok &= check(geo::empty(geo::intersection(person, boxes[2])) && geo::iou(person, boxes[2]) == 0, "disjoint: an empty intersection and IoU 0");
    const b2d frame{{0, 0}, {640, 480}};
    const b2d margin = geo::expand(frame, -16.0);
    ok &= check(margin == b2d{{16, 16}, {624, 464}} && geo::contains(margin, person), "expand by -16 leaves a safe area that holds the person");
    const b2d moved = geo::translate(person, a2d{10, -5});
    ok &= check(near(moved.lo[0], 106) && near(moved.lo[1], 139) && near(geo::area(moved), geo::area(person)), "translate moves a box and keeps its area");
    ok &= check(geo::contains(person, a2d{96, 144}) && !geo::contains(person, a2d{95, 144}), "contains a point, the boundary included");

    std::printf("bounding and distance\n");
    // A tracker's last few centres of the person
    constexpr std::array<a2d, 4> track{a2d{150, 230}, a2d{155, 236}, a2d{162, 241}, a2d{171, 239}};
    const b2d gate = geo::bounding<double, 2>(std::span<const a2d>(track));
    ok &= check(gate == b2d{{150, 230}, {171, 241}}, "bounding box of a track");
    ok &= check(geo::empty(geo::bounding<double, 2>(std::span<const a2d>{})), "no points: an empty box");
    ok &= check(geo::squared_distance(gate, a2d{160, 235}) == 0 && geo::squared_distance(gate, a2d{174, 245}) == 9 + 16, "squared distance: 0 inside, 3^2 + 4^2 past a corner");

    std::printf("geo_box across the antimeridian\n");
    ok &= check(fiji.crosses_antimeridian() && near(fiji.east, -176.2 * deg), "extend went east across the antimeridian");
    ok &= check(near(fiji.south, -22.3 * deg) && near(fiji.north, -12 * deg) && near(fiji.west, 177 * deg), "the south edge widened; the west edge stayed");
    // 177° E to 176.2° W: 3° + 3.8° = 6.8°
    ok &= check(near(fiji.longitude_span(), 6.8 * deg), "the span is 6.8 deg, not 353.2");
    // Centre: 177° + 3.4° = 180.4° E = 179.6° W; latitude (-22.3 - 12) / 2 = -17.15°
    ok &= check(near(centre[1], -179.6 * deg) && near(centre[0], -17.15 * deg), "the centre lies on the box's own arc, past 180");
    ok &= check(fiji.contains(-18.14 * deg, 178.44 * deg) && fiji.contains(-15 * deg, pi) && !fiji.contains(-15 * deg, 0), "contains Suva and the antimeridian, not Greenwich");
    ok &= check(fiji.contains(fiji.south, fiji.west) && fiji.contains(fiji.north, fiji.east), "contains its own corners");
    ok &= check(fiji.contains(survey_point[0], survey_point[1]), "the sample lies in the box");

    std::printf("%s\n", ok ? "all ok" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
