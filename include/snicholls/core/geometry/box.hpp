//
//  box.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::geometry::box<T, D>`, `iou`, `giou`, `non_max_suppression` and `geo_box`: axis-aligned boxes in any dimension, the overlap measures of object detection, and latitude-longitude boxes that cross the antimeridian.
/// \ingroup core
///
/// **Overview.** Boxes are everywhere a thing occupies space: a detector's bounding boxes, a tracker's gates, a
/// spatial index's nodes, a map's viewport.
///
/// - `box<T, D>`: corners `lo` and `hi` (`lo ≤ hi` on every axis for a non-empty box), any arithmetic `T` (`int` for
///   pixels, `double` for metres), any dimension. An aggregate, trivially copyable.
/// - Conversions in the plane:
///   - from corners (`from_xyxy`), from a corner and a size (`from_xywh`), and from YOLO's centre and size
///     (`from_cxcywh`);
///   - `to_xyxy`, `to_xywh`, `to_cxcywh`.
/// - Measures: `empty`, `extent`, `centre`, `volume` (`area` in the plane), `intersection`, `hull` (the smallest box
///   holding both), `contains` (a point or a box), `intersects`, `expand`, `translate`, `bounding` (of a set of
///   points), and `squared_distance` (from a box to a point; zero inside). That last is what a spatial index's
///   nearest search prunes with.
/// - Overlap:
///   - `iou` (intersection over union);
///   - `giou` (generalised IoU, Rezatofighi et al. 2019: `IoU - |C \ (A ∪ B)| / |C|`, with `C` the hull);
///   - `non_max_suppression`: greedy, by descending score, keeping a box unless it overlaps a kept one by more than a
///     threshold (a detector's post-processing step).
/// - `geo_box<T>`: south, west, north and east in radians, crossing the antimeridian when `west > east`.
///   - `contains`, `longitude_span`, `centre` (on the shorter arc), `extend` (with a point);
///   - `from_degrees`;
///   - `sample`: a uniform point on the sphere within the box (latitude by `asin` of a uniform sine), with the
///     caller's generator.
///
/// Boxes are closed and continuous: area is `Π (hi - lo)`, so a pixel box `[x1, x2]` has width `x2 - x1`; add one
/// yourself for inclusive pixel counts.
///
/// \snippet geometry/box_demo.cpp box
///
/// **Relationships.**
/// - `metric/concepts.hpp`: `bounding` reads any point type through `point_traits`.
/// - `geometry/geodetic.hpp`: radians and the ellipsoid; `metric/geodesic.hpp`: distances on the sphere.
/// - `stats/random.hpp`: `geo_box::sample`'s generator.
/// - The R-tree to come indexes `box`es.
///
/// Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for arithmetic `T`, no allocation (`non_max_suppression` uses the
/// caller's span). IoU and GIoU are symmetric; IoU is in `[0, 1]`, GIoU in `(-1, 1]`, both 1 for equal non-empty
/// boxes. Complexity: `non_max_suppression` O(n log n + n k) for k kept.
///
/// **Errors.** None. An empty box has volume 0, contains nothing, and its IoU with anything is 0; two empty boxes
/// have IoU 0 and GIoU 0.
///
/// **Performance.** Not benchmarked: a few comparisons per box, per axis.
///
/// **Tests and demos.** `tests/geometry/box_test.cpp`, `demos/geometry/box_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<numbers>`, `<span>`,
/// `<type_traits>`, `math/constexpr_math.hpp`, `metric/concepts.hpp`, `stats/random.hpp`.
///
/// **Known issues.** `geo_box` treats a box whose longitude span is the whole circle as `west = -π, east = π`; a box
/// over a pole is a latitude cap (south or north at ±π/2).
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_geometry_box_hpp
#define snicholls_core_geometry_box_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <type_traits>

namespace snicholls::geometry {

    /// \brief An axis-aligned box: corners `lo` and `hi`, non-empty when `lo[i] ≤ hi[i]` on every axis.
    /// \ingroup core
    /// \tparam T The coordinate type (any arithmetic type).
    /// \tparam D The dimension, at least 1.
    template <typename T, std::size_t D>
        requires std::is_arithmetic_v<T>
    struct box {
        static_assert(D > 0, "snicholls::geometry::box: the dimension is at least 1");
        using value_type = T;                   ///< The coordinate type.
        static constexpr std::size_t dim = D;   ///< The dimension.
        std::array<T, D> lo{};                  ///< The lower corner.
        std::array<T, D> hi{};                  ///< The upper corner.

        /// \brief Corner-wise equality. \return Whether both corners are equal.
        friend constexpr bool operator==(const box&, const box&) = default;
    };

    /// \brief A box in the plane from its corners. \tparam T The coordinate type. \param x1 The left.
    /// \ingroup core
    /// \param y1 The bottom (or top, in image coordinates). \param x2 The right. \param y2 The other edge.
    /// \return The box, its corners ordered.
    template <typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] constexpr box<T, 2> from_xyxy(T x1, T y1, T x2, T y2) noexcept
    {
        return {{x1 < x2 ? x1 : x2, y1 < y2 ? y1 : y2}, {x1 < x2 ? x2 : x1, y1 < y2 ? y2 : y1}};
    }

    /// \brief A box in the plane from a corner and a size. \tparam T The coordinate type.
    /// \ingroup core
    /// \param x The corner's x. \param y The corner's y. \param w The width, at least 0. \param h The height, at least 0.
    /// \return `[x, x + w] × [y, y + h]`.
    template <typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] constexpr box<T, 2> from_xywh(T x, T y, T w, T h) noexcept
    {
        return {{x, y}, {static_cast<T>(x + w), static_cast<T>(y + h)}};
    }

    /// \brief A box in the plane from its centre and size (YOLO's form). \tparam T The coordinate type.
    /// \ingroup core
    /// \param cx The centre's x. \param cy The centre's y. \param w The width. \param h The height.
    /// \return `[cx - w/2, cx + w/2] × [cy - h/2, cy + h/2]`.
    template <typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] constexpr box<T, 2> from_cxcywh(T cx, T cy, T w, T h) noexcept
    {
        return {{static_cast<T>(cx - w / T(2)), static_cast<T>(cy - h / T(2))}, {static_cast<T>(cx + w / T(2)), static_cast<T>(cy + h / T(2))}};
    }

    /// \brief `(x1, y1, x2, y2)`. \tparam T The coordinate type. \param b The box. \return The corners.
    /// \ingroup core
    template <typename T>
    [[nodiscard]] constexpr std::array<T, 4> to_xyxy(const box<T, 2>& b) noexcept
    {
        return {b.lo[0], b.lo[1], b.hi[0], b.hi[1]};
    }

    /// \brief `(x, y, w, h)`. \tparam T The coordinate type. \param b The box. \return Corner and size.
    /// \ingroup core
    template <typename T>
    [[nodiscard]] constexpr std::array<T, 4> to_xywh(const box<T, 2>& b) noexcept
    {
        return {b.lo[0], b.lo[1], static_cast<T>(b.hi[0] - b.lo[0]), static_cast<T>(b.hi[1] - b.lo[1])};
    }

    /// \brief `(cx, cy, w, h)`. \tparam T The coordinate type. \param b The box. \return Centre and size.
    /// \ingroup core
    template <typename T>
    [[nodiscard]] constexpr std::array<T, 4> to_cxcywh(const box<T, 2>& b) noexcept
    {
        return {static_cast<T>((b.lo[0] + b.hi[0]) / T(2)), static_cast<T>((b.lo[1] + b.hi[1]) / T(2)), static_cast<T>(b.hi[0] - b.lo[0]), static_cast<T>(b.hi[1] - b.lo[1])};
    }

    /// \brief Whether a box is empty (some `hi[i] < lo[i]`, or NaN). \tparam T The coordinate type.
    /// \ingroup core
    /// \tparam D The dimension. \param b The box. \return True if it holds no point.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr bool empty(const box<T, D>& b) noexcept
    {
        for (std::size_t i = 0; i < D; ++i)
            if (!(b.lo[i] <= b.hi[i]))
                return true;
        return false;
    }

    /// \brief The box's length along an axis. \tparam T The coordinate type. \tparam D The dimension.
    /// \ingroup core
    /// \param b The box. \param axis The axis. \return `hi - lo`, 0 for an empty box.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr T extent(const box<T, D>& b, std::size_t axis) noexcept
    {
        return empty(b) ? T(0) : static_cast<T>(b.hi[axis] - b.lo[axis]);
    }

    /// \brief The box's centre, in `double` for integer coordinates. \tparam T The coordinate type.
    /// \ingroup core
    /// \tparam D The dimension. \param b The box. \return `(lo + hi) / 2`.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr auto centre(const box<T, D>& b) noexcept
    {
        using R = std::conditional_t<std::is_floating_point_v<T>, T, double>;
        std::array<R, D> c{};
        for (std::size_t i = 0; i < D; ++i)
            c[i] = (static_cast<R>(b.lo[i]) + static_cast<R>(b.hi[i])) / R(2);
        return c;
    }

    /// \brief The box's volume (its area in the plane), in `double` for integer coordinates. \ingroup core
    /// \tparam T The coordinate type. \tparam D The dimension. \param b The box. \return `Π (hi - lo)`, 0 when empty.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr auto volume(const box<T, D>& b) noexcept
    {
        using R = std::conditional_t<std::is_floating_point_v<T>, T, double>;
        if (empty(b))
            return R(0);
        R v(1);
        for (std::size_t i = 0; i < D; ++i)
            v *= static_cast<R>(b.hi[i]) - static_cast<R>(b.lo[i]);
        return v;
    }

    /// \brief The box's area: `volume` in the plane. \tparam T The coordinate type. \param b The box.
    /// \ingroup core
    /// \return Width times height, 0 when empty.
    template <typename T>
    [[nodiscard]] constexpr auto area(const box<T, 2>& b) noexcept
    {
        return volume(b);
    }

    /// \brief The intersection of two boxes (empty when they do not meet). \tparam T The coordinate type.
    /// \ingroup core
    /// \tparam D The dimension. \param a One box. \param b The other. \return `[max lo, min hi]` per axis.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr box<T, D> intersection(const box<T, D>& a, const box<T, D>& b) noexcept
    {
        box<T, D> r{};
        for (std::size_t i = 0; i < D; ++i) {
            r.lo[i] = a.lo[i] > b.lo[i] ? a.lo[i] : b.lo[i];
            r.hi[i] = a.hi[i] < b.hi[i] ? a.hi[i] : b.hi[i];
        }
        return r;
    }

    /// \brief The smallest box holding both (an empty box contributes nothing). \tparam T The coordinate type.
    /// \ingroup core
    /// \tparam D The dimension. \param a One box. \param b The other. \return `[min lo, max hi]` per axis.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr box<T, D> hull(const box<T, D>& a, const box<T, D>& b) noexcept
    {
        if (empty(a))
            return b;
        if (empty(b))
            return a;
        box<T, D> r{};
        for (std::size_t i = 0; i < D; ++i) {
            r.lo[i] = a.lo[i] < b.lo[i] ? a.lo[i] : b.lo[i];
            r.hi[i] = a.hi[i] > b.hi[i] ? a.hi[i] : b.hi[i];
        }
        return r;
    }

    /// \brief Whether two boxes share a point (touching counts). \tparam T The coordinate type.
    /// \ingroup core
    /// \tparam D The dimension. \param a One box. \param b The other. \return True if they intersect.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr bool intersects(const box<T, D>& a, const box<T, D>& b) noexcept
    {
        return !empty(a) && !empty(b) && !empty(intersection(a, b));
    }

    /// \brief Whether a box contains a point (the boundary included). \tparam T The coordinate type.
    /// \ingroup core
    /// \tparam D The dimension. \tparam P A `metric::point` of dimension D. \param b The box. \param p The point.
    /// \return True if `lo ≤ p ≤ hi` on every axis.
    template <typename T, std::size_t D, metric::point P>
    [[nodiscard]] constexpr bool contains(const box<T, D>& b, const P& p) noexcept(noexcept(metric::coordinate(p, 0)))
    {
        for (std::size_t i = 0; i < D; ++i) {
            const auto x = metric::coordinate(p, i);
            if (!(x >= b.lo[i] && x <= b.hi[i]))
                return false;
        }
        return true;
    }

    /// \brief Whether a box contains another (an empty box is contained in every box). \ingroup core
    /// \tparam T The coordinate type. \tparam D The dimension. \param outer The containing box. \param inner The other.
    /// \return True if `inner ⊆ outer`.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr bool contains(const box<T, D>& outer, const box<T, D>& inner) noexcept
    {
        if (empty(inner))
            return true;
        for (std::size_t i = 0; i < D; ++i)
            if (!(inner.lo[i] >= outer.lo[i] && inner.hi[i] <= outer.hi[i]))
                return false;
        return true;
    }

    /// \brief The box grown by `margin` on every side (shrunk for a negative margin). \ingroup core
    /// \tparam T The coordinate type. \tparam D The dimension. \param b The box. \param margin The margin.
    /// \return `[lo - margin, hi + margin]`.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr box<T, D> expand(const box<T, D>& b, T margin) noexcept
    {
        box<T, D> r = b;
        for (std::size_t i = 0; i < D; ++i) {
            r.lo[i] = static_cast<T>(r.lo[i] - margin);
            r.hi[i] = static_cast<T>(r.hi[i] + margin);
        }
        return r;
    }

    /// \brief The box moved by an offset. \tparam T The coordinate type. \tparam D The dimension.
    /// \ingroup core
    /// \param b The box. \param offset The offset per axis. \return The moved box.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr box<T, D> translate(const box<T, D>& b, const std::array<T, D>& offset) noexcept
    {
        box<T, D> r = b;
        for (std::size_t i = 0; i < D; ++i) {
            r.lo[i] = static_cast<T>(r.lo[i] + offset[i]);
            r.hi[i] = static_cast<T>(r.hi[i] + offset[i]);
        }
        return r;
    }

    /// \brief The smallest box holding every point.
    /// \ingroup core
    /// \tparam T The coordinate type of the result.
    /// \tparam D The dimension.
    /// \tparam P A `metric::point` of dimension D.
    /// \param points The points.
    /// \return Their bounding box; an empty box (`lo` above `hi`) for no points.
    template <typename T, std::size_t D, metric::point P>
    [[nodiscard]] constexpr box<T, D> bounding(std::span<const P> points) noexcept(noexcept(metric::coordinate(std::declval<const P&>(), 0)))
    {
        box<T, D> r{};
        if (points.empty()) {
            for (std::size_t i = 0; i < D; ++i) {
                r.lo[i] = T(1);
                r.hi[i] = T(0);
            }
            return r;
        }
        for (std::size_t i = 0; i < D; ++i)
            r.lo[i] = r.hi[i] = static_cast<T>(metric::coordinate(points[0], i));
        for (const auto& p : points)
            for (std::size_t i = 0; i < D; ++i) {
                const T x = static_cast<T>(metric::coordinate(p, i));
                r.lo[i] = x < r.lo[i] ? x : r.lo[i];
                r.hi[i] = x > r.hi[i] ? x : r.hi[i];
            }
        return r;
    }

    /// \brief The squared Euclidean distance from a box to a point: zero inside (a spatial index's lower bound).
    /// \ingroup core
    /// \tparam T The coordinate type.
    /// \tparam D The dimension.
    /// \tparam P A `metric::point` of dimension D.
    /// \param b The box.
    /// \param p The point.
    /// \return `Σ max(lo - p, 0, p - hi)²`, in `double`.
    template <typename T, std::size_t D, metric::point P>
    [[nodiscard]] constexpr double squared_distance(const box<T, D>& b, const P& p) noexcept(noexcept(metric::coordinate(p, 0)))
    {
        double s = 0;
        for (std::size_t i = 0; i < D; ++i) {
            const double x = static_cast<double>(metric::coordinate(p, i));
            const double d = x < static_cast<double>(b.lo[i]) ? static_cast<double>(b.lo[i]) - x : x > static_cast<double>(b.hi[i]) ? x - static_cast<double>(b.hi[i]) : 0.0;
            s += d * d;
        }
        return s;
    }

    /// \brief Intersection over union, in `[0, 1]`: 1 for equal non-empty boxes, 0 for disjoint ones (or any empty).
    /// \tparam T The coordinate type. \tparam D The dimension. \param a One box. \param b The other.
    /// \ingroup core
    /// \return `|A ∩ B| / |A ∪ B|`, in `double`.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr double iou(const box<T, D>& a, const box<T, D>& b) noexcept
    {
        const double i = static_cast<double>(volume(intersection(a, b)));
        const double u = static_cast<double>(volume(a)) + static_cast<double>(volume(b)) - i;
        return u > 0 ? i / u : 0.0;
    }

    /// \brief Generalised IoU (Rezatofighi et al. 2019), in `(-1, 1]`: `IoU - |C \ (A ∪ B)| / |C|`, with `C` the hull.
    /// Unlike IoU it still measures how far apart disjoint boxes are, so it is a loss that gives a gradient.
    /// \tparam T The coordinate type. \tparam D The dimension. \param a One box. \param b The other.
    /// \ingroup core
    /// \return GIoU; 0 when either box is empty.
    template <typename T, std::size_t D>
    [[nodiscard]] constexpr double giou(const box<T, D>& a, const box<T, D>& b) noexcept
    {
        if (empty(a) || empty(b))
            return 0.0;
        const double i = static_cast<double>(volume(intersection(a, b)));
        const double u = static_cast<double>(volume(a)) + static_cast<double>(volume(b)) - i;
        const double c = static_cast<double>(volume(hull(a, b)));
        if (!(c > 0))
            return u > 0 ? i / u : 1.0;  // degenerate (zero-volume) boxes: equal points are a perfect match
        return (u > 0 ? i / u : 0.0) - (c - u) / c;
    }

    /// \brief Greedy non-maximum suppression: the boxes in descending score, each kept unless its IoU with a kept box
    /// exceeds `threshold`.
    /// \ingroup core
    /// \tparam T The coordinate type.
    /// \tparam D The dimension.
    /// \tparam S The score type.
    /// \param boxes The candidate boxes.
    /// \param scores One score per box.
    /// \param threshold The largest IoU a kept box may have with a better kept one.
    /// \param kept Receives the kept boxes' indices, best first; at least `boxes.size()` long (it is also the working
    /// order).
    /// \return How many were kept; 0 if the spans do not fit.
    template <typename T, std::size_t D, typename S>
    [[nodiscard]] constexpr std::size_t non_max_suppression(std::span<const box<T, D>> boxes, std::span<const S> scores, double threshold, std::span<std::uint32_t> kept) noexcept
    {
        const std::size_t n = boxes.size();
        if (scores.size() != n || kept.size() < n)
            return 0;
        for (std::size_t i = 0; i < n; ++i)
            kept[i] = static_cast<std::uint32_t>(i);
        // Descending score, ties by index so the result is deterministic
        std::sort(kept.begin(), kept.begin() + static_cast<std::ptrdiff_t>(n), [&](std::uint32_t x, std::uint32_t y) {
            return scores[x] > scores[y] || (!(scores[y] > scores[x]) && x < y);
        });
        std::size_t k = 0;
        for (std::size_t c = 0; c < n; ++c) {
            const std::uint32_t candidate = kept[c];
            bool keep = true;
            for (std::size_t j = 0; j < k && keep; ++j)
                keep = !(iou(boxes[candidate], boxes[kept[j]]) > threshold);
            if (keep)
                kept[k++] = candidate;
        }
        return k;
    }

    /// \brief A latitude-longitude box on the sphere, in radians: south ≤ north; it crosses the antimeridian when
    /// `west > east`.
    /// \ingroup core
    /// \tparam T The number type.
    template <typename T>
        requires std::is_floating_point_v<T>
    struct geo_box {
        T south{};  ///< The southern latitude, radians.
        T west{};   ///< The western longitude, radians, in `[-π, π]`.
        T north{};  ///< The northern latitude, radians.
        T east{};   ///< The eastern longitude, radians, in `[-π, π]`.

        /// \brief A box from degrees. \param south The southern latitude. \param west The western longitude.
        /// \param north The northern latitude. \param east The eastern longitude. \return The box in radians.
        [[nodiscard]] static constexpr geo_box from_degrees(T south, T west, T north, T east) noexcept
        {
            const T k = static_cast<T>(std::numbers::pi_v<long double> / 180.0L);
            return {south * k, west * k, north * k, east * k};
        }
        /// \brief Whether the box crosses the antimeridian. \return `west > east`.
        [[nodiscard]] constexpr bool crosses_antimeridian() const noexcept { return west > east; }
        /// \brief The longitude span, radians, in `[0, 2π]`. \return `east - west`, wrapped.
        [[nodiscard]] constexpr T longitude_span() const noexcept
        {
            const T two_pi = T(2) * static_cast<T>(std::numbers::pi_v<long double>);
            return crosses_antimeridian() ? east - west + two_pi : east - west;
        }
        /// \brief Whether a point lies in the box (the boundary included). \param lat The latitude, radians.
        /// \param lon The longitude, radians, in `[-π, π]`. \return True if inside.
        [[nodiscard]] constexpr bool contains(T lat, T lon) const noexcept
        {
            if (!(lat >= south && lat <= north))
                return false;
            return crosses_antimeridian() ? (lon >= west || lon <= east) : (lon >= west && lon <= east);
        }
        /// \brief The centre, on the box's own longitude arc. \return `(latitude, longitude)`, radians.
        [[nodiscard]] constexpr std::array<T, 2> centre() const noexcept
        {
            const T pi = static_cast<T>(std::numbers::pi_v<long double>);
            T lon = west + longitude_span() / T(2);
            if (lon > pi)
                lon -= T(2) * pi;
            return {(south + north) / T(2), lon};
        }
        /// \brief The smallest box that also holds a point: latitude widened, longitude extended the shorter way.
        /// \param lat The latitude, radians. \param lon The longitude, radians, in `[-π, π]`. \return The box.
        [[nodiscard]] constexpr geo_box extend(T lat, T lon) const noexcept
        {
            geo_box r = *this;
            r.south = lat < south ? lat : south;
            r.north = lat > north ? lat : north;
            const bool lon_inside = crosses_antimeridian() ? (lon >= west || lon <= east) : (lon >= west && lon <= east);
            if (lon_inside)
                return r;
            const T two_pi = T(2) * static_cast<T>(std::numbers::pi_v<long double>);
            // Growing east to lon, or west to lon: whichever adds less span
            T grow_east = lon - east;
            if (grow_east < T(0))
                grow_east += two_pi;
            T grow_west = west - lon;
            if (grow_west < T(0))
                grow_west += two_pi;
            if (grow_east <= grow_west)
                r.east = lon;
            else
                r.west = lon;
            return r;
        }
        /// \brief A uniformly distributed point of the sphere within the box (equal area, not equal angle).
        /// \tparam G A `stats::full_range_generator`. \param g The generator.
        /// \return `(latitude, longitude)`, radians.
        template <stats::full_range_generator G>
        [[nodiscard]] constexpr std::array<T, 2> sample(G& g) const noexcept(noexcept(g()))
        {
            const T pi = static_cast<T>(std::numbers::pi_v<long double>);
            const T s0 = static_cast<T>(math::sin(south)), s1 = static_cast<T>(math::sin(north));
            const T lat = static_cast<T>(math::asin(s0 + (s1 - s0) * static_cast<T>(stats::uniform_real(g))));
            T lon = west + longitude_span() * static_cast<T>(stats::uniform_real(g));
            if (lon > pi)
                lon -= T(2) * pi;
            return {lat, lon};
        }

        /// \brief Member-wise equality. \return Whether the four edges are equal.
        friend constexpr bool operator==(const geo_box&, const geo_box&) = default;
    };

} // namespace snicholls::geometry

#endif /* snicholls_core_geometry_box_hpp */
