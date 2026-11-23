//
//  box_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/geometry/box.hpp>: box<T, D> for int, float and double in one, two and three dimensions;
//  the xyxy, xywh and cxcywh conversions and their round trips; the measures, intersection, hull, containment,
//  expand, translate, bounding (std::array, fixed_vector, a point type of the test's own, no points) and
//  squared_distance; IoU and GIoU against hand values, their symmetry and ranges; non_max_suppression by hand, at its
//  threshold, with ties and bad spans, and against a brute-force reference on random boxes; geo_box across the
//  antimeridian, its span, containment at the edges, centre, extend the shorter way, and sample's uniformity by area;
//  constant evaluation, noexcept and layout.
//

#include <snicholls/core/geometry/box.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

// A point type of the test's own: two named coordinates
struct pixel {
    int x, y;
};

// A point whose coordinate read may throw
struct checked {
    std::array<double, 2> v;
};

} // namespace

template <>
struct snicholls::metric::point_traits<pixel> {
    using value_type = int;
    static constexpr std::size_t extent = 2;
    static constexpr std::size_t dimension(const pixel&) noexcept { return 2; }
    static constexpr int at(const pixel& p, std::size_t i) noexcept { return i == 0 ? p.x : p.y; }
};

template <>
struct snicholls::metric::point_traits<checked> {
    using value_type = double;
    static constexpr std::size_t extent = 2;
    static constexpr std::size_t dimension(const checked&) noexcept { return 2; }
    static double at(const checked& p, std::size_t i)
    {
        if (i >= 2)
            throw std::out_of_range("checked");
        return p.v[i];
    }
};

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
namespace st = snicholls::stats;
using b1d = geo::box<double, 1>;
using b2i = geo::box<int, 2>;
using b2f = geo::box<float, 2>;
using b2d = geo::box<double, 2>;
using b3d = geo::box<double, 3>;
using gbd = geo::geo_box<double>;
using a2d = std::array<double, 2>;
using a3d = std::array<double, 3>;
constexpr double pi = std::numbers::pi;
constexpr double deg = pi / 180;

// Layout: two corner arrays and nothing else; aggregates, trivially copyable
static_assert(std::is_aggregate_v<b2d> && std::is_trivially_copyable_v<b2d> && std::is_trivially_copyable_v<b3d>);
static_assert(sizeof(b2i) == 4 * sizeof(int) && sizeof(b2f) == 4 * sizeof(float) && sizeof(b3d) == 6 * sizeof(double) && sizeof(b1d) == 2 * sizeof(double));
static_assert(alignof(b2d) == alignof(double) && b3d::dim == 3 && std::is_same_v<b2i::value_type, int>);
static_assert(std::is_aggregate_v<gbd> && std::is_trivially_copyable_v<gbd> && sizeof(gbd) == 4 * sizeof(double) && sizeof(geo::geo_box<float>) == 4 * sizeof(float));

// noexcept for arithmetic coordinates and noexcept points; a throwing coordinate read is reported
constexpr b2d unit{{0, 0}, {1, 1}};
static_assert(noexcept(geo::from_xyxy(0.0, 0.0, 1.0, 1.0)) && noexcept(geo::from_xywh(0, 0, 1, 1)) && noexcept(geo::from_cxcywh(0.f, 0.f, 1.f, 1.f)));
static_assert(noexcept(geo::to_xyxy(unit)) && noexcept(geo::to_xywh(unit)) && noexcept(geo::to_cxcywh(unit)));
static_assert(noexcept(geo::empty(unit)) && noexcept(geo::extent(unit, 0)) && noexcept(geo::centre(unit)) && noexcept(geo::volume(unit)) && noexcept(geo::area(unit)));
static_assert(noexcept(geo::intersection(unit, unit)) && noexcept(geo::hull(unit, unit)) && noexcept(geo::intersects(unit, unit)) && noexcept(geo::contains(unit, unit)));
static_assert(noexcept(geo::contains(unit, a2d{})) && noexcept(geo::squared_distance(unit, a2d{})) && noexcept(geo::contains(unit, pixel{})));
static_assert(!noexcept(geo::contains(unit, checked{})) && !noexcept(geo::squared_distance(unit, checked{})) && !noexcept(geo::bounding<double, 2>(std::span<const checked>{})));
static_assert(noexcept(geo::expand(unit, 1.0)) && noexcept(geo::translate(unit, a2d{})) && noexcept(geo::bounding<double, 2>(std::span<const a2d>{})));
static_assert(noexcept(geo::iou(unit, unit)) && noexcept(geo::giou(unit, unit)));
static_assert(noexcept(geo::non_max_suppression(std::span<const b2d>{}, std::span<const float>{}, 0.5, std::span<std::uint32_t>{})));
static_assert(noexcept(gbd::from_degrees(0, 0, 1, 1)) && noexcept(gbd{}.crosses_antimeridian()) && noexcept(gbd{}.longitude_span()) && noexcept(gbd{}.contains(0, 0)));
static_assert(noexcept(gbd{}.centre()) && noexcept(gbd{}.extend(0, 0)) && noexcept(gbd{}.sample(std::declval<st::wyrand&>())));

// Constant evaluation
static_assert(geo::from_xyxy(3, 4, 1, 2) == b2i{{1, 2}, {3, 4}});
static_assert(geo::to_xywh(geo::from_xywh(1, 2, 3, 4)) == std::array{1, 2, 3, 4});
static_assert(geo::to_cxcywh(geo::from_cxcywh(4.0, 6.0, 2.0, 4.0)) == std::array{4.0, 6.0, 2.0, 4.0});
static_assert(geo::area(geo::from_xywh(1, 2, 3, 4)) == 12.0 && geo::volume(b3d{{0, 0, 0}, {1, 2, 3}}) == 6.0);
static_assert(geo::iou(b2d{{0, 0}, {2, 2}}, b2d{{1, 1}, {3, 3}}) == 1.0 / 7.0);
static_assert(geo::giou(b2d{{0, 0}, {1, 1}}, b2d{{2, 0}, {3, 1}}) == -1.0 / 3.0);
static_assert(geo::squared_distance(b2d{{0, 0}, {2, 1}}, a2d{3, 3}) == 5.0);
static_assert(geo::contains(unit, a2d{1, 0}) && !geo::contains(unit, a2d{1.5, 0}) && geo::intersects(unit, b2d{{1, 0}, {2, 1}}));
static_assert([] {
    constexpr std::array pts{a2d{3, -1}, a2d{-2, 4}, a2d{0, 0}};
    return geo::bounding<double, 2>(std::span<const a2d>(pts)) == b2d{{-2, -1}, {3, 4}};
}());
static_assert([] {
    constexpr std::array boxes{b2d{{0, 0}, {10, 10}}, b2d{{1, 1}, {11, 11}}, b2d{{20, 20}, {30, 30}}};
    constexpr std::array scores{0.9, 0.8, 0.7};
    std::array<std::uint32_t, 3> kept{};
    const auto k = geo::non_max_suppression(std::span<const b2d>(boxes), std::span<const double>(scores), 0.5, std::span<std::uint32_t>(kept));
    return k == 2 && kept[0] == 0 && kept[1] == 2;
}());
static_assert(gbd::from_degrees(-21, 177, -12, -178).crosses_antimeridian() && !gbd::from_degrees(0, 10, 10, 20).crosses_antimeridian());
static_assert(gbd{0, -1, 1, 1}.contains(0.5, 1) && !gbd{0, -1, 1, 1}.contains(0.5, 1.5) && gbd{0, 3, 1, -3}.contains(0.5, 3.1));
static_assert(gbd{0, 0, 1, 1}.extend(2, 1.5) == gbd{0, 0, 2, 1.5});
static_assert(gbd{0, 0, 1, 1}.centre() == std::array{0.5, 0.5});
static_assert([] {
    st::wyrand g(7);
    const gbd b{-0.5, 3.0, 0.5, -3.0};
    const auto p = b.sample(g);
    return b.contains(p[0], p[1]);
}());

// ---- box: construction, conversions -----------------------------------------------------------------------------

TEST(GeometryBoxConversion, FromXyxyOrdersTheCorners)
{
    EXPECT_EQ(geo::from_xyxy(3, 4, 1, 2), (b2i{{1, 2}, {3, 4}}));
    EXPECT_EQ(geo::from_xyxy(1, 4, 3, 2), (b2i{{1, 2}, {3, 4}}));
    EXPECT_EQ(geo::from_xyxy(3, 2, 1, 4), (b2i{{1, 2}, {3, 4}}));
    EXPECT_EQ(geo::from_xyxy(1.0, 2.0, 3.0, 4.0), (b2d{{1, 2}, {3, 4}}));
    EXPECT_EQ(geo::from_xyxy(2.f, 2.f, 2.f, 2.f), (b2f{{2, 2}, {2, 2}}));  // a point is a box
}

TEST(GeometryBoxConversion, XyxyRoundTrips)
{
    EXPECT_EQ(geo::to_xyxy(geo::from_xyxy(1, 2, 3, 4)), (std::array{1, 2, 3, 4}));
    EXPECT_EQ(geo::to_xyxy(geo::from_xyxy(-1.5, 2.25, 3.5, 4.75)), (std::array{-1.5, 2.25, 3.5, 4.75}));
}

TEST(GeometryBoxConversion, XywhRoundTripsForEveryType)
{
    EXPECT_EQ(geo::from_xywh(10, 20, 30, 40), (b2i{{10, 20}, {40, 60}}));
    EXPECT_EQ(geo::to_xywh(geo::from_xywh(10, 20, 30, 40)), (std::array{10, 20, 30, 40}));
    EXPECT_EQ(geo::to_xywh(geo::from_xywh(0.5f, 1.5f, 2.f, 4.f)), (std::array{0.5f, 1.5f, 2.f, 4.f}));
    EXPECT_EQ(geo::to_xywh(geo::from_xywh(-3.0, 7.0, 0.0, 1.0)), (std::array{-3.0, 7.0, 0.0, 1.0}));
}

TEST(GeometryBoxConversion, CxcywhRoundTripsForEveryType)
{
    EXPECT_EQ(geo::from_cxcywh(0.5, 0.5, 0.2, 0.4), (b2d{{0.4, 0.3}, {0.6, 0.7}}));
    const auto d = geo::to_cxcywh(geo::from_cxcywh(0.5, 0.5, 0.2, 0.4));
    EXPECT_DOUBLE_EQ(d[0], 0.5);
    EXPECT_DOUBLE_EQ(d[1], 0.5);
    EXPECT_DOUBLE_EQ(d[2], 0.2);
    EXPECT_DOUBLE_EQ(d[3], 0.4);
    const auto f = geo::to_cxcywh(geo::from_cxcywh(320.f, 240.f, 64.f, 48.f));
    EXPECT_EQ(f, (std::array{320.f, 240.f, 64.f, 48.f}));
    // Integers: even sizes are exact
    EXPECT_EQ(geo::from_cxcywh(10, 20, 4, 6), (b2i{{8, 17}, {12, 23}}));
    EXPECT_EQ(geo::to_cxcywh(geo::from_cxcywh(10, 20, 4, 6)), (std::array{10, 20, 4, 6}));
}

TEST(GeometryBoxConversion, FormsAgreeOnOneBox)
{
    // The box [100, 300] x [50, 150]: xywh (100, 50, 200, 100), cxcywh (200, 100, 200, 100)
    const b2d b{{100, 50}, {300, 150}};
    EXPECT_EQ(geo::from_xywh(100.0, 50.0, 200.0, 100.0), b);
    EXPECT_EQ(geo::from_cxcywh(200.0, 100.0, 200.0, 100.0), b);
    EXPECT_EQ(geo::to_xywh(b), (std::array{100.0, 50.0, 200.0, 100.0}));
    EXPECT_EQ(geo::to_cxcywh(b), (std::array{200.0, 100.0, 200.0, 100.0}));
}

// ---- box: measures ----------------------------------------------------------------------------------------------

TEST(GeometryBoxMeasure, EmptyWhenAnyAxisIsInvertedOrNaN)
{
    EXPECT_FALSE(geo::empty(unit));
    EXPECT_FALSE(geo::empty(b2d{{1, 1}, {1, 1}}));  // a point
    EXPECT_TRUE(geo::empty(b2d{{1, 0}, {0, 1}}));
    EXPECT_TRUE(geo::empty(b3d{{0, 0, 1}, {1, 1, 0}}));
    EXPECT_TRUE(geo::empty(b1d{{2}, {1}}));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(geo::empty(b2d{{0, nan}, {1, 1}}));
    EXPECT_TRUE(geo::empty(b2i{{5, 0}, {4, 3}}));
}

TEST(GeometryBoxMeasure, ExtentCentreVolumeAndArea)
{
    const b3d b{{0, 1, 2}, {2, 4, 7}};
    EXPECT_EQ(geo::extent(b, 0), 2.0);
    EXPECT_EQ(geo::extent(b, 1), 3.0);
    EXPECT_EQ(geo::extent(b, 2), 5.0);
    EXPECT_EQ(geo::centre(b), (a3d{1, 2.5, 4.5}));
    EXPECT_EQ(geo::volume(b), 30.0);
    // Integers: centre and volume in double
    const b2i p{{1, 2}, {4, 4}};
    static_assert(std::is_same_v<decltype(geo::centre(p)), std::array<double, 2>> && std::is_same_v<decltype(geo::volume(p)), double>);
    EXPECT_EQ(geo::centre(p), (a2d{2.5, 3}));
    EXPECT_EQ(geo::area(p), 6.0);
    EXPECT_EQ(geo::extent(p, 0), 3);
    // Float stays float
    const b2f f{{0, 0}, {0.5f, 4}};
    static_assert(std::is_same_v<decltype(geo::area(f)), float>);
    EXPECT_EQ(geo::area(f), 2.f);
    // One dimension: length
    EXPECT_EQ(geo::volume(b1d{{-1}, {2.5}}), 3.5);
    EXPECT_EQ(geo::centre(b1d{{-1}, {2}}), (std::array{0.5}));
}

TEST(GeometryBoxMeasure, EmptyBoxHasVolumeAndExtentZero)
{
    const b2d e{{1, 1}, {0, 3}};
    EXPECT_EQ(geo::volume(e), 0.0);
    EXPECT_EQ(geo::area(e), 0.0);
    EXPECT_EQ(geo::extent(e, 0), 0.0);
    EXPECT_EQ(geo::extent(e, 1), 0.0);  // even on the axis that is not inverted
    EXPECT_EQ(geo::volume(b2i{{0, 0}, {-1, -1}}), 0.0);
}

// ---- box: set operations ----------------------------------------------------------------------------------------

TEST(GeometryBoxSet, IntersectionAndHull)
{
    const b2d a{{0, 0}, {2, 2}}, b{{1, 1}, {3, 3}};
    EXPECT_EQ(geo::intersection(a, b), (b2d{{1, 1}, {2, 2}}));
    EXPECT_EQ(geo::hull(a, b), (b2d{{0, 0}, {3, 3}}));
    EXPECT_EQ(geo::intersection(a, b), geo::intersection(b, a));
    EXPECT_EQ(geo::hull(a, b), geo::hull(b, a));
    // Disjoint: the intersection is empty
    EXPECT_TRUE(geo::empty(geo::intersection(a, b2d{{5, 5}, {6, 6}})));
    // An empty box contributes nothing to a hull
    const b2d e{{9, 9}, {-9, -9}};
    EXPECT_EQ(geo::hull(a, e), a);
    EXPECT_EQ(geo::hull(e, a), a);
    // Three dimensions and integers
    EXPECT_EQ(geo::intersection(geo::box<int, 3>{{0, 0, 0}, {4, 4, 4}}, geo::box<int, 3>{{2, -1, 3}, {5, 2, 9}}), (geo::box<int, 3>{{2, 0, 3}, {4, 2, 4}}));
    EXPECT_EQ(geo::hull(b1d{{0}, {1}}, b1d{{3}, {4}}), (b1d{{0}, {4}}));
}

TEST(GeometryBoxSet, TouchingBoxesIntersectWithIouZero)
{
    const b2d a{{0, 0}, {1, 1}};
    const b2d side{{1, 0}, {2, 1}};
    const b2d corner{{1, 1}, {2, 2}};
    EXPECT_TRUE(geo::intersects(a, side));
    EXPECT_TRUE(geo::intersects(a, corner));
    EXPECT_EQ(geo::iou(a, side), 0.0);
    EXPECT_EQ(geo::iou(a, corner), 0.0);
    // GIoU of side-touching boxes: hull area 2 = union, so 0 - 0/2 = 0
    EXPECT_EQ(geo::giou(a, side), 0.0);
    // Corner-touching: union 2, hull 4: 0 - 2/4
    EXPECT_EQ(geo::giou(a, corner), -0.5);
    EXPECT_FALSE(geo::intersects(a, b2d{{1.5, 0}, {2, 1}}));
    EXPECT_TRUE(geo::intersects(geo::box<int, 3>{{0, 0, 0}, {1, 1, 1}}, geo::box<int, 3>{{1, 1, 1}, {2, 2, 2}}));
}

TEST(GeometryBoxSet, EmptyBoxContainsAndIntersectsNothing)
{
    const b2d e{{1, 1}, {0, 0}};
    EXPECT_FALSE(geo::contains(e, a2d{0.5, 0.5}));
    EXPECT_FALSE(geo::contains(e, a2d{1, 1}));
    EXPECT_FALSE(geo::contains(e, a2d{0, 0}));
    EXPECT_FALSE(geo::intersects(e, unit));
    EXPECT_FALSE(geo::intersects(unit, e));
    EXPECT_FALSE(geo::intersects(e, e));
    // An empty box is inside every box, and contains no non-empty box
    EXPECT_TRUE(geo::contains(unit, e));
    EXPECT_TRUE(geo::contains(e, e));
    EXPECT_FALSE(geo::contains(e, unit));
}

TEST(GeometryBoxSet, ContainsPointsOnTheBoundary)
{
    const b3d b{{0, 0, 0}, {1, 2, 3}};
    EXPECT_TRUE(geo::contains(b, a3d{0, 0, 0}));
    EXPECT_TRUE(geo::contains(b, a3d{1, 2, 3}));
    EXPECT_TRUE(geo::contains(b, a3d{0.5, 1, 1.5}));
    EXPECT_FALSE(geo::contains(b, a3d{1.0000001, 1, 1}));
    EXPECT_FALSE(geo::contains(b, a3d{0.5, 1, -0.0001}));
    EXPECT_FALSE(geo::contains(b, a3d{0.5, 1, std::numeric_limits<double>::quiet_NaN()}));
    // Other point types: a fixed_vector, a point of the caller's own, an int box
    EXPECT_TRUE(geo::contains(b, la::fixed_vector<double, 3>{{1, 1, 1}}));
    EXPECT_TRUE(geo::contains(b2i{{0, 0}, {640, 480}}, pixel{640, 0}));
    EXPECT_FALSE(geo::contains(b2i{{0, 0}, {640, 480}}, pixel{641, 0}));
    EXPECT_TRUE(geo::contains(b1d{{-1}, {1}}, std::array{1.0}));
}

TEST(GeometryBoxSet, ContainsBoxes)
{
    const b2d outer{{0, 0}, {10, 10}};
    EXPECT_TRUE(geo::contains(outer, outer));
    EXPECT_TRUE(geo::contains(outer, b2d{{0, 5}, {10, 6}}));
    EXPECT_FALSE(geo::contains(outer, b2d{{-1, 5}, {10, 6}}));
    EXPECT_FALSE(geo::contains(outer, b2d{{5, 5}, {11, 6}}));
    EXPECT_FALSE(geo::contains(b2d{{0, 5}, {10, 6}}, outer));
}

TEST(GeometryBoxSet, ExpandAndTranslate)
{
    EXPECT_EQ(geo::expand(unit, 0.5), (b2d{{-0.5, -0.5}, {1.5, 1.5}}));
    EXPECT_EQ(geo::expand(b2i{{0, 0}, {10, 4}}, -1), (b2i{{1, 1}, {9, 3}}));
    // Shrinking past zero leaves an empty box
    EXPECT_TRUE(geo::empty(geo::expand(b2i{{0, 0}, {10, 4}}, -3)));
    EXPECT_EQ(geo::translate(b3d{{0, 0, 0}, {1, 2, 3}}, a3d{1, -2, 0.5}), (b3d{{1, -2, 0.5}, {2, 0, 3.5}}));
    EXPECT_EQ(geo::translate(b2f{{0, 0}, {1, 1}}, std::array{2.f, 3.f}), (b2f{{2, 3}, {3, 4}}));
    // Translation keeps the volume
    EXPECT_EQ(geo::volume(geo::translate(b3d{{0, 0, 0}, {1, 2, 3}}, a3d{7, 8, 9})), 6.0);
}

// ---- box: bounding and squared_distance -------------------------------------------------------------------------

TEST(GeometryBoxBounding, OverArraysFixedVectorsAndAPointOfOurOwn)
{
    const std::vector<a3d> pts{{1, 2, 3}, {-1, 5, 0}, {4, -2, 1}};
    EXPECT_EQ((geo::bounding<double, 3>(std::span<const a3d>(pts))), (b3d{{-1, -2, 0}, {4, 5, 3}}));
    using v2 = la::fixed_vector<float, 2>;
    const std::array<v2, 2> fv{v2{{0.5f, -1.f}}, v2{{-0.5f, 2.f}}};
    EXPECT_EQ((geo::bounding<float, 2>(std::span<const v2>(fv))), (b2f{{-0.5f, -1.f}, {0.5f, 2.f}}));
    const std::array<pixel, 3> px{pixel{10, 20}, pixel{5, 40}, pixel{30, 25}};
    EXPECT_EQ((geo::bounding<int, 2>(std::span<const pixel>(px))), (b2i{{5, 20}, {30, 40}}));
    // A single point is a degenerate, non-empty box that contains it
    const std::array<a2d, 1> one{a2d{3, 4}};
    const auto b = geo::bounding<double, 2>(std::span<const a2d>(one));
    EXPECT_EQ(b, (b2d{{3, 4}, {3, 4}}));
    EXPECT_FALSE(geo::empty(b));
    EXPECT_TRUE(geo::contains(b, one[0]));
    // Every point is inside its bounding box
    for (const auto& p : pts)
        EXPECT_TRUE(geo::contains(geo::bounding<double, 3>(std::span<const a3d>(pts)), p));
}

TEST(GeometryBoxBounding, NoPointsGiveAnEmptyBox)
{
    const auto b = geo::bounding<double, 2>(std::span<const a2d>{});
    EXPECT_TRUE(geo::empty(b));
    EXPECT_EQ(geo::volume(b), 0.0);
    EXPECT_TRUE(geo::empty(geo::bounding<int, 3>(std::span<const std::array<int, 3>>{})));
    // It contributes nothing to a hull
    EXPECT_EQ(geo::hull(b, unit), unit);
}

TEST(GeometryBoxDistance, ZeroInsideAndOnTheEdgeSquaredOutside)
{
    const b2d b{{0, 0}, {2, 1}};
    EXPECT_EQ(geo::squared_distance(b, a2d{1, 0.5}), 0.0);   // inside
    EXPECT_EQ(geo::squared_distance(b, a2d{2, 0.5}), 0.0);   // on an edge
    EXPECT_EQ(geo::squared_distance(b, a2d{0, 1}), 0.0);     // on a corner
    EXPECT_EQ(geo::squared_distance(b, a2d{3, 0.5}), 1.0);   // beside an edge: 1 away in x
    EXPECT_EQ(geo::squared_distance(b, a2d{1, -3}), 9.0);    // below: 3 away in y
    EXPECT_EQ(geo::squared_distance(b, a2d{3, 3}), 5.0);     // past the (2, 1) corner: 1² + 2²
    EXPECT_EQ(geo::squared_distance(b, a2d{-1, -2}), 5.0);   // past the (0, 0) corner: 1² + 2²
    EXPECT_EQ(geo::squared_distance(b3d{{0, 0, 0}, {1, 1, 1}}, a3d{2, 3, 4}), 1.0 + 4.0 + 9.0);
    EXPECT_EQ(geo::squared_distance(b2i{{0, 0}, {10, 10}}, pixel{13, 14}), 25.0);
    EXPECT_EQ(geo::squared_distance(b1d{{0}, {1}}, std::array{-0.5}), 0.25);
}

// ---- iou and giou -----------------------------------------------------------------------------------------------

TEST(GeometryBoxIou, HandValuesInOneTwoAndThreeDimensions)
{
    // [0, 2] vs [1, 3]: intersection 1, union 3
    EXPECT_DOUBLE_EQ(geo::iou(b1d{{0}, {2}}, b1d{{1}, {3}}), 1.0 / 3.0);
    // [0, 2]² vs [1, 3]²: intersection 1, union 4 + 4 - 1 = 7
    EXPECT_DOUBLE_EQ(geo::iou(b2d{{0, 0}, {2, 2}}, b2d{{1, 1}, {3, 3}}), 1.0 / 7.0);
    EXPECT_DOUBLE_EQ(geo::iou(b2i{{0, 0}, {2, 2}}, b2i{{1, 1}, {3, 3}}), 1.0 / 7.0);
    EXPECT_NEAR(geo::iou(b2f{{0, 0}, {2, 2}}, b2f{{1, 1}, {3, 3}}), 1.0 / 7.0, 1e-7);
    // [0, 2]³ vs [1, 3]³: intersection 1, union 8 + 8 - 1 = 15
    EXPECT_DOUBLE_EQ(geo::iou(b3d{{0, 0, 0}, {2, 2, 2}}, b3d{{1, 1, 1}, {3, 3, 3}}), 1.0 / 15.0);
    // One inside another: the area ratio
    EXPECT_DOUBLE_EQ(geo::iou(b2d{{0, 0}, {4, 4}}, b2d{{1, 1}, {3, 3}}), 0.25);
}

TEST(GeometryBoxIou, EqualBoxesGiveOneAndEmptyBoxesZero)
{
    const b2d a{{1, 2}, {5, 9}};
    EXPECT_EQ(geo::iou(a, a), 1.0);
    EXPECT_EQ(geo::giou(a, a), 1.0);
    EXPECT_EQ(geo::iou(b3d{{0, 0, 0}, {1, 1, 1}}, b3d{{0, 0, 0}, {1, 1, 1}}), 1.0);
    EXPECT_EQ(geo::giou(b2i{{0, 0}, {3, 3}}, b2i{{0, 0}, {3, 3}}), 1.0);
    // Equal degenerate (zero-area) boxes: IoU 0, GIoU 1 as documented
    const b2d point{{1, 1}, {1, 1}};
    EXPECT_EQ(geo::iou(point, point), 0.0);
    EXPECT_EQ(geo::giou(point, point), 1.0);
    const b2d e{{1, 1}, {0, 0}};
    EXPECT_EQ(geo::iou(e, a), 0.0);
    EXPECT_EQ(geo::iou(a, e), 0.0);
    EXPECT_EQ(geo::iou(e, e), 0.0);
    EXPECT_EQ(geo::giou(e, a), 0.0);
    EXPECT_EQ(geo::giou(a, e), 0.0);
    EXPECT_EQ(geo::giou(e, e), 0.0);
    // An empty box overlapping a box's corners still has IoU 0
    EXPECT_EQ(geo::iou(b2d{{0, 0}, {3, -1}}, b2d{{0, -1}, {3, 0}}), 0.0);
}

TEST(GeometryBoxGiou, DisjointAndOverlappingBoxesMatchHandValues)
{
    // [0, 1]² and [2, 3] x [0, 1]: IoU 0, union 2, hull [0, 3] x [0, 1] of area 3: 0 - (3 - 2) / 3
    EXPECT_DOUBLE_EQ(geo::giou(b2d{{0, 0}, {1, 1}}, b2d{{2, 0}, {3, 1}}), -1.0 / 3.0);
    // [0, 1]² and [9, 10]²: union 2, hull 100: -(98 / 100); further apart is more negative
    EXPECT_DOUBLE_EQ(geo::giou(b2d{{0, 0}, {1, 1}}, b2d{{9, 9}, {10, 10}}), -0.98);
    // [0, 2]² and [1, 3]²: 1/7 - (9 - 7)/9 = -5/63
    EXPECT_DOUBLE_EQ(geo::giou(b2d{{0, 0}, {2, 2}}, b2d{{1, 1}, {3, 3}}), -5.0 / 63.0);
    // [0, 2]³ and [1, 3]³: 1/15 - (27 - 15)/27 = -17/45
    EXPECT_DOUBLE_EQ(geo::giou(b3d{{0, 0, 0}, {2, 2, 2}}, b3d{{1, 1, 1}, {3, 3, 3}}), -17.0 / 45.0);
    // Intervals [0, 1] and [3, 4]: union 2, hull 4: -1/2
    EXPECT_DOUBLE_EQ(geo::giou(b1d{{0}, {1}}, b1d{{3}, {4}}), -0.5);
    // One inside another: the hull is the outer box, so GIoU is IoU
    EXPECT_DOUBLE_EQ(geo::giou(b2d{{0, 0}, {4, 4}}, b2d{{1, 1}, {3, 3}}), 0.25);
}

TEST(GeometryBoxIou, SymmetricAndInRangeOnRandomBoxes)
{
    st::wyrand g(20261123);
    auto random_box = [&] {
        const double x = st::uniform_real(g) * 10, y = st::uniform_real(g) * 10;
        return geo::from_xywh(x, y, st::uniform_real(g) * 5, st::uniform_real(g) * 5);
    };
    for (int t = 0; t < 2000; ++t) {
        const b2d a = random_box(), b = random_box();
        const double i = geo::iou(a, b), j = geo::giou(a, b);
        EXPECT_EQ(i, geo::iou(b, a));
        EXPECT_EQ(j, geo::giou(b, a));
        EXPECT_GE(i, 0.0);
        EXPECT_LE(i, 1.0);
        EXPECT_GT(j, -1.0);
        EXPECT_LE(j, 1.0);
        EXPECT_LE(j, i);  // GIoU never exceeds IoU
        EXPECT_EQ(i == 0.0, !geo::intersects(a, b) || geo::area(geo::intersection(a, b)) == 0.0);
    }
}

// ---- non_max_suppression ----------------------------------------------------------------------------------------

template <typename T, std::size_t D, typename S>
std::vector<std::uint32_t> nms(const std::vector<geo::box<T, D>>& boxes, const std::vector<S>& scores, double threshold)
{
    std::vector<std::uint32_t> kept(boxes.size());
    const auto k = geo::non_max_suppression(std::span<const geo::box<T, D>>(boxes), std::span<const S>(scores), threshold, std::span<std::uint32_t>(kept));
    kept.resize(k);
    return kept;
}

TEST(GeometryBoxNms, HandCase)
{
    // Two pairs of overlapping boxes; within each pair the IoU is 81 / 119 ≈ 0.68
    const std::vector<b2d> boxes{{{0, 0}, {10, 10}}, {{1, 1}, {11, 11}}, {{20, 20}, {30, 30}}, {{21, 21}, {31, 31}}};
    const std::vector<double> scores{0.9, 0.8, 0.7, 0.95};
    EXPECT_EQ(nms(boxes, scores, 0.5), (std::vector<std::uint32_t>{3, 0}));
    // At 0.7 nothing overlaps enough: all kept, best first
    EXPECT_EQ(nms(boxes, scores, 0.7), (std::vector<std::uint32_t>{3, 0, 1, 2}));
    // At -1 every box suppresses every other, even disjoint ones (IoU 0 > -1): only the best survives
    EXPECT_EQ(nms(boxes, scores, -1.0), (std::vector<std::uint32_t>{3}));
    // A box suppressed does not itself suppress: 0 suppresses 1, and 2 overlaps only 1
    const std::vector<b2i> chain{{{0, 0}, {10, 10}}, {{2, 0}, {12, 10}}, {{8, 0}, {18, 10}}};
    // IoU(0, 1) = 80 / 120, IoU(1, 2) = 60 / 140, IoU(0, 2) = 20 / 180
    EXPECT_EQ(nms(chain, std::vector<int>{3, 2, 1}, 0.5), (std::vector<std::uint32_t>{0, 2}));
    // Three dimensions and float scores
    const std::vector<b3d> cubes{{{0, 0, 0}, {1, 1, 1}}, {{0, 0, 0}, {1, 1, 1.1}}, {{5, 5, 5}, {6, 6, 6}}};
    EXPECT_EQ(nms(cubes, std::vector<float>{0.1f, 0.2f, 0.3f}, 0.5), (std::vector<std::uint32_t>{2, 1}));
}

TEST(GeometryBoxNms, TiesAreBrokenByIndex)
{
    const std::vector<b2d> disjoint{{{0, 0}, {1, 1}}, {{2, 0}, {3, 1}}, {{4, 0}, {5, 1}}, {{6, 0}, {7, 1}}};
    EXPECT_EQ(nms(disjoint, std::vector<double>{0.5, 0.5, 0.5, 0.5}, 0.5), (std::vector<std::uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(nms(disjoint, std::vector<double>{0.1, 0.5, 0.1, 0.5}, 0.5), (std::vector<std::uint32_t>{1, 3, 0, 2}));
    // Equal boxes with equal scores: the lowest index wins
    const std::vector<b2d> same(5, b2d{{0, 0}, {1, 1}});
    EXPECT_EQ(nms(same, std::vector<double>(5, 0.7), 0.5), (std::vector<std::uint32_t>{0}));
    EXPECT_EQ(nms(same, std::vector<double>{0.1, 0.7, 0.3, 0.7, 0.2}, 0.5), (std::vector<std::uint32_t>{1}));
}

TEST(GeometryBoxNms, IouExactlyAtTheThresholdIsKept)
{
    // [0, 1]² inside [0, 2] x [0, 1]: intersection 1, union 2, IoU exactly 0.5 in double
    const std::vector<b2d> boxes{{{0, 0}, {2, 1}}, {{0, 0}, {1, 1}}};
    ASSERT_EQ(geo::iou(boxes[0], boxes[1]), 0.5);
    const std::vector<double> scores{0.9, 0.8};
    EXPECT_EQ(nms(boxes, scores, 0.5), (std::vector<std::uint32_t>{0, 1}));
    EXPECT_EQ(nms(boxes, scores, std::nextafter(0.5, 0.0)), (std::vector<std::uint32_t>{0}));
    // Threshold 1: only exact duplicates could exceed it, and none can, so everything is kept
    const std::vector<b2d> same(3, b2d{{0, 0}, {1, 1}});
    EXPECT_EQ(nms(same, std::vector<double>{1, 2, 3}, 1.0), (std::vector<std::uint32_t>{2, 1, 0}));
    // Threshold 0: touching boxes (IoU 0) are kept, overlapping ones are not
    const std::vector<b2d> row{{{0, 0}, {1, 1}}, {{1, 0}, {2, 1}}, {{1.5, 0}, {2.5, 1}}};
    EXPECT_EQ(nms(row, std::vector<double>{3, 2, 1}, 0.0), (std::vector<std::uint32_t>{0, 1}));
}

TEST(GeometryBoxNms, MismatchedSpansReturnZero)
{
    const std::array<b2d, 3> boxes{unit, unit, unit};
    const std::array<double, 3> scores{1, 2, 3};
    const std::array<double, 2> short_scores{1, 2};
    std::array<std::uint32_t, 3> kept{9, 9, 9};
    std::array<std::uint32_t, 2> short_kept{9, 9};
    EXPECT_EQ(geo::non_max_suppression(std::span<const b2d>(boxes), std::span<const double>(short_scores), 0.5, std::span<std::uint32_t>(kept)), 0u);
    EXPECT_EQ(kept, (std::array<std::uint32_t, 3>{9, 9, 9}));  // untouched
    EXPECT_EQ(geo::non_max_suppression(std::span<const b2d>(boxes), std::span<const double>(scores), 0.5, std::span<std::uint32_t>(short_kept)), 0u);
    EXPECT_EQ(short_kept, (std::array<std::uint32_t, 2>{9, 9}));
    // A longer output span is fine
    std::array<std::uint32_t, 5> long_kept{};
    EXPECT_EQ(geo::non_max_suppression(std::span<const b2d>(boxes), std::span<const double>(scores), 0.5, std::span<std::uint32_t>(long_kept)), 1u);
    EXPECT_EQ(long_kept[0], 2u);
    // No boxes: nothing kept
    EXPECT_EQ(geo::non_max_suppression(std::span<const b2d>{}, std::span<const double>{}, 0.5, std::span<std::uint32_t>{}), 0u);
}

// An independent reference: visit in score order (a stable sort keeps ties in index order), keep a box unless some
// already kept one overlaps it by more than the threshold
std::vector<std::uint32_t> reference_nms(const std::vector<b2d>& boxes, const std::vector<double>& scores, double threshold)
{
    std::vector<std::uint32_t> order(boxes.size());
    for (std::uint32_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return scores[a] > scores[b]; });
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<std::uint32_t> kept;
    for (std::size_t a = 0; a < order.size(); ++a) {
        if (suppressed[order[a]])
            continue;
        kept.push_back(order[a]);
        for (std::size_t b = a + 1; b < order.size(); ++b) {
            const b2d& p = boxes[order[a]];
            const b2d& q = boxes[order[b]];
            // IoU written out here, not by the header
            const double w = std::min(p.hi[0], q.hi[0]) - std::max(p.lo[0], q.lo[0]);
            const double h = std::min(p.hi[1], q.hi[1]) - std::max(p.lo[1], q.lo[1]);
            const double inter = w > 0 && h > 0 ? w * h : 0.0;
            const double uni = (p.hi[0] - p.lo[0]) * (p.hi[1] - p.lo[1]) + (q.hi[0] - q.lo[0]) * (q.hi[1] - q.lo[1]) - inter;
            if (uni > 0 && inter / uni > threshold)
                suppressed[order[b]] = true;
        }
    }
    return kept;
}

TEST(GeometryBoxNms, AgreesWithABruteForceReferenceOnRandomBoxes)
{
    st::wyrand g(42);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 1 + st::uniform_index(g, 60);
        std::vector<b2d> boxes(n);
        std::vector<double> scores(n);
        for (std::size_t i = 0; i < n; ++i) {
            boxes[i] = geo::from_cxcywh(st::uniform_real(g) * 100, st::uniform_real(g) * 100, 5 + st::uniform_real(g) * 30, 5 + st::uniform_real(g) * 30);
            // Scores on a coarse grid, so ties happen
            scores[i] = static_cast<double>(st::uniform_index(g, 10)) / 10;
        }
        const double threshold = 0.1 + 0.8 * st::uniform_real(g);
        EXPECT_EQ(nms(boxes, scores, threshold), reference_nms(boxes, scores, threshold)) << "trial " << trial;
    }
}

// ---- geo_box ----------------------------------------------------------------------------------------------------

// Fiji: 21° S to 12° S, 177° E across the antimeridian to 178° W
const gbd fiji = gbd::from_degrees(-21, 177, -12, -178);

TEST(GeometryGeoBox, FromDegreesConvertsToRadians)
{
    EXPECT_DOUBLE_EQ(fiji.south, -21 * deg);
    EXPECT_DOUBLE_EQ(fiji.west, 177 * deg);
    EXPECT_DOUBLE_EQ(fiji.north, -12 * deg);
    EXPECT_DOUBLE_EQ(fiji.east, -178 * deg);
    const auto f = geo::geo_box<float>::from_degrees(-90, -180, 90, 180);
    EXPECT_FLOAT_EQ(f.south, -std::numbers::pi_v<float> / 2);
    EXPECT_FLOAT_EQ(f.east, std::numbers::pi_v<float>);
}

TEST(GeometryGeoBox, FijiCrossesTheAntimeridian)
{
    EXPECT_TRUE(fiji.crosses_antimeridian());
    EXPECT_FALSE(gbd::from_degrees(0, 10, 10, 20).crosses_antimeridian());
    EXPECT_FALSE(gbd::from_degrees(-90, -180, 90, 180).crosses_antimeridian());
    // Suva (18.14° S, 178.44° E) and Lakeba (18.2° S, 178.8° W) are in; Nadi's longitude in the Atlantic is not
    EXPECT_TRUE(fiji.contains(-18.14 * deg, 178.44 * deg));
    EXPECT_TRUE(fiji.contains(-18.2 * deg, -178.8 * deg));
    EXPECT_TRUE(fiji.contains(-15 * deg, pi));
    EXPECT_TRUE(fiji.contains(-15 * deg, -pi));
    EXPECT_FALSE(fiji.contains(-18 * deg, -30 * deg));
    EXPECT_FALSE(fiji.contains(-18 * deg, 176 * deg));
    EXPECT_FALSE(fiji.contains(-18 * deg, -177 * deg));
    EXPECT_FALSE(fiji.contains(-25 * deg, 179 * deg));
}

TEST(GeometryGeoBox, LongitudeSpanBothWays)
{
    // Not crossing: 20° - 10° = 10°
    EXPECT_DOUBLE_EQ(gbd::from_degrees(0, 10, 10, 20).longitude_span(), 10 * deg);
    // Crossing: 177° E to 178° W is 3° + 2° = 5°
    EXPECT_NEAR(fiji.longitude_span(), 5 * deg, 1e-15);
    // The same edges the other way round cover the rest of the circle: 355°
    EXPECT_NEAR(gbd::from_degrees(-21, -178, -12, 177).longitude_span(), 355 * deg, 1e-14);
    EXPECT_DOUBLE_EQ(gbd::from_degrees(-90, -180, 90, 180).longitude_span(), 2 * pi);
    EXPECT_EQ(gbd::from_degrees(0, 30, 10, 30).longitude_span(), 0.0);
    EXPECT_NEAR(geo::geo_box<float>::from_degrees(0, 170, 1, -170).longitude_span(), 20 * std::numbers::pi_v<float> / 180, 1e-6f);
}

TEST(GeometryGeoBox, ContainsAtTheBoundaries)
{
    for (const gbd& b : {fiji, gbd::from_degrees(10, -40, 20, -30)}) {
        EXPECT_TRUE(b.contains(b.south, b.west));
        EXPECT_TRUE(b.contains(b.south, b.east));
        EXPECT_TRUE(b.contains(b.north, b.west));
        EXPECT_TRUE(b.contains(b.north, b.east));
        EXPECT_FALSE(b.contains(std::nextafter(b.south, -10.0), b.west));
        EXPECT_FALSE(b.contains(std::nextafter(b.north, 10.0), b.east));
        EXPECT_FALSE(b.contains(b.south, std::nextafter(b.west, -10.0)));
        EXPECT_FALSE(b.contains(b.north, std::nextafter(b.east, 10.0)));
    }
    // The whole circle contains both ends of the longitude range; a pole cap contains its pole
    const gbd cap = gbd::from_degrees(80, -180, 90, 180);
    EXPECT_TRUE(cap.contains(pi / 2, 0));
    EXPECT_TRUE(cap.contains(85 * deg, -pi));
    EXPECT_TRUE(cap.contains(85 * deg, pi));
    EXPECT_FALSE(cap.contains(79 * deg, 0));
}

TEST(GeometryGeoBox, CentreLiesOnTheBoxsOwnArc)
{
    const auto c = gbd::from_degrees(0, 10, 10, 20).centre();
    EXPECT_DOUBLE_EQ(c[0], 5 * deg);
    EXPECT_DOUBLE_EQ(c[1], 15 * deg);
    // Fiji: 177° + 5°/2 = 179.5° E, not the 0.5° W of the naive (west + east) / 2
    const auto f = fiji.centre();
    EXPECT_DOUBLE_EQ(f[0], -16.5 * deg);
    EXPECT_NEAR(f[1], 179.5 * deg, 1e-14);
    EXPECT_TRUE(fiji.contains(f[0], f[1]));
    // A crossing box whose centre is past the antimeridian: 178° E to 176° W, 6° wide, centre 181° E = 179° W
    const auto w = gbd::from_degrees(-20, 178, -10, -176).centre();
    EXPECT_NEAR(w[1], -179 * deg, 1e-14);
    // The complement of Fiji's longitudes: 178° W to 177° E the long way, centre 0.5° W
    EXPECT_NEAR(gbd::from_degrees(-21, -178, -12, 177).centre()[1], -0.5 * deg, 1e-14);
}

TEST(GeometryGeoBox, ExtendTakesTheShorterWay)
{
    const gbd b = gbd::from_degrees(0, 10, 10, 20);
    // A point inside changes nothing
    EXPECT_EQ(b.extend(5 * deg, 15 * deg), b);
    // 25° E: 5° east against 345° west
    EXPECT_EQ(b.extend(5 * deg, 25 * deg), (gbd{0, 10 * deg, 10 * deg, 25 * deg}));
    // 0°: 340° east against 10° west
    EXPECT_EQ(b.extend(5 * deg, 0), (gbd{0, 0, 10 * deg, 20 * deg}));
    // Latitude widens both ways, with the longitude already inside
    EXPECT_EQ(b.extend(-5 * deg, 12 * deg), (gbd{-5 * deg, 10 * deg, 10 * deg, 20 * deg}));
    EXPECT_EQ(b.extend(30 * deg, 12 * deg), (gbd{0, 10 * deg, 30 * deg, 20 * deg}));
}

TEST(GeometryGeoBox, ExtendCrossesTheAntimeridianWhenShorter)
{
    // [170° E, 175° E] and 178° W: 7° east (across), 348° west
    const gbd b = gbd::from_degrees(-20, 170, -10, 175);
    const gbd e = b.extend(-15 * deg, -178 * deg);
    EXPECT_EQ(e.west, b.west);
    EXPECT_EQ(e.east, -178 * deg);
    EXPECT_TRUE(e.crosses_antimeridian());
    EXPECT_NEAR(e.longitude_span(), 12 * deg, 1e-14);
    EXPECT_TRUE(e.contains(-15 * deg, pi));
    // Fiji and 175° W: 3° east; Fiji and 170° E: 7° west (348° east)
    EXPECT_EQ(fiji.extend(-15 * deg, -175 * deg).east, -175 * deg);
    EXPECT_EQ(fiji.extend(-15 * deg, -175 * deg).west, fiji.west);
    EXPECT_EQ(fiji.extend(-15 * deg, 170 * deg).west, 170 * deg);
    EXPECT_EQ(fiji.extend(-15 * deg, 170 * deg).east, fiji.east);
    EXPECT_NEAR(fiji.extend(-15 * deg, 170 * deg).longitude_span(), 12 * deg, 1e-14);
    // The extended box holds the point, and the old box
    const gbd x = fiji.extend(-25 * deg, -170 * deg);
    EXPECT_TRUE(x.contains(-25 * deg, -170 * deg));
    EXPECT_TRUE(x.contains(fiji.north, fiji.west));
}

TEST(GeometryGeoBox, SampleStaysInTheBox)
{
    st::wyrand g(1);
    for (const gbd& b : {fiji, gbd::from_degrees(-30, 170, 60, -170), gbd::from_degrees(10, -40, 20, -30), gbd::from_degrees(-90, -180, 90, 180)}) {
        for (int i = 0; i < 5000; ++i) {
            const auto p = b.sample(g);
            ASSERT_TRUE(b.contains(p[0], p[1])) << p[0] << ", " << p[1];
            ASSERT_GE(p[1], -pi);
            ASSERT_LE(p[1], pi);
        }
    }
    st::wyrand gf(2);
    const auto f = geo::geo_box<float>::from_degrees(-21, 177, -12, -178);
    for (int i = 0; i < 2000; ++i) {
        const auto p = f.sample(gf);
        // float rounding of asin can land a hair past an edge
        EXPECT_GE(p[0], f.south - 1e-6f);
        EXPECT_LE(p[0], f.north + 1e-6f);
        EXPECT_TRUE(p[1] >= f.west - 1e-6f || p[1] <= f.east + 1e-6f);
    }
}

TEST(GeometryGeoBox, SampleIsUniformByArea)
{
    // 30° S to 60° N, 170° E to 170° W. Uniform by area means sin(latitude) is uniform on [sin -30°, sin 60°] and
    // the longitude offset from the west edge is uniform on [0, 20°]. Chi-square over 10 equal bins of each, with
    // 40000 samples (4000 expected per bin) and seed 12345: the 99.9% point of chi-square with 9 degrees of freedom is
    // 27.877 (scipy.stats.chi2.ppf(0.999, 9)).
    const gbd b = gbd::from_degrees(-30, 170, 60, -170);
    constexpr int n = 40000, bins = 10;
    const double s0 = std::sin(b.south), s1 = std::sin(b.north);
    std::array<int, bins> lat_bins{}, lon_bins{};
    int above_30 = 0;
    st::wyrand g(12345);
    for (int i = 0; i < n; ++i) {
        const auto p = b.sample(g);
        const double u = (std::sin(p[0]) - s0) / (s1 - s0);
        ++lat_bins[std::min(bins - 1, static_cast<int>(u * bins))];
        double off = p[1] - b.west;
        if (off < 0)
            off += 2 * pi;
        ++lon_bins[std::min(bins - 1, static_cast<int>(off / b.longitude_span() * bins))];
        above_30 += p[0] > 30 * deg;
    }
    const double expected = static_cast<double>(n) / bins;
    double chi_lat = 0, chi_lon = 0;
    for (int k = 0; k < bins; ++k) {
        chi_lat += (lat_bins[k] - expected) * (lat_bins[k] - expected) / expected;
        chi_lon += (lon_bins[k] - expected) * (lon_bins[k] - expected) / expected;
    }
    EXPECT_LT(chi_lat, 27.877);
    EXPECT_LT(chi_lon, 27.877);
    // The fraction north of 30° N is the area fraction (sin 60° - sin 30°) / (sin 60° + sin 30°) = 0.26795, not the
    // 30/90 = 0.333 of equal angle; 5 standard errors (sqrt(0.268 * 0.732 / 40000) = 0.0022) is 0.011
    EXPECT_NEAR(static_cast<double>(above_30) / n, 0.26795, 0.011);
}

TEST(GeometryGeoBox, SampleIsReproducibleFromTheSeed)
{
    st::wyrand a(99), b(99);
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(fiji.sample(a), fiji.sample(b));
}

} // namespace
