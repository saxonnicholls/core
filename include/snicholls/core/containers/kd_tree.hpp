//
//  kd_tree.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::containers::kd_tree<Point, K, Metric>`: a balanced k-d tree built in place over the caller's points, under any coordinate-wise metric, with nearest, k-nearest, radius and box queries, iterative and allocation-free.
/// \ingroup core
///
/// **Overview.** A scan brings hundreds of detections and a tracker asks, for each track, which detections lie
/// near its prediction. A k-d tree answers that in about `log n` per query instead of `n`. This one is built once
/// over a span of points the caller owns, and stores nothing but a permutation of their indices (the caller's
/// span, too). The tree is implicit: the median of each index range is its node, split on axis
/// `depth mod K`. So there are no node objects and no pointers, and a rebuild per scan costs O(n log n).
///
/// - `Point` is any `metric::point` (`fixed_vector`, `std::array`, a struct of the caller's with `point_traits`).
///   Queries may be any compatible point type.
/// - `Metric` is any `metric::decomposable` metric (`euclidean` by default; `manhattan`, `chebyshev`,
///   `minkowski<P>`, `weighted_minkowski`, a `mahalanobis` made coordinate-wise by whitening the points first, or the
///   caller's own): a single coordinate's term bounds the whole, which is what lets a subtree be skipped.
/// - `nearest(q)`: the closest point and its distance, or nothing for an empty tree.
/// - `k_nearest(q, out)`: the `out.size()` closest, nearest first; equal distances in ascending index order, and a tie
///   at the k-th place cut by index (so the result is one well-defined list, as a reference that sorts by
///   (distance, index) gives).
/// - `k_nearest_within(q, r, out)`: the same, strictly within distance `r` (SciPy's `distance_upper_bound`), for kriging's
///   neighbourhood search.
/// - `for_each_within(q, r, f)`: `f(index, distance)` for every point within distance `r`.
/// - `for_each_in_box(lo, hi, f)`: `f(index)` for every point with `lo ≤ p ≤ hi` on every axis.
///
/// Distances are reported in the metric's own units (`expand` of the reduced form the search compares in).
///
/// \snippet containers/kd_tree_demo.cpp kd_tree
///
/// **Relationships.** `metric/concepts.hpp` and `metric/minkowski.hpp`, `intrusive_avl_tree.hpp` (the same policy
/// style), `containers/ann_forest.hpp` (approximate search for high dimensions, where k-d trees lose their edge).
/// Stone Soup's `DetectionKDTreeNN` and `KDTreePDA` associators are built on it in the framework above core. Group:
/// \ref core.
///
/// **Implementations.** Build: an explicit stack of index ranges, each partitioned at its median by
/// `std::nth_element` (constexpr since C++20). Queries: an explicit stack of at most 64 frames, near child first,
/// each carrying a lower bound on its points' reduced distance (the larger of its parent's and the splitting
/// plane's term), skipped when the bound exceeds the best so far. Candidate distances are accumulated with an early
/// exit at the current bound.
///
/// **Guarantees.** `constexpr`, no allocation, no recursion. Build O(n log n) expected; queries O(log n) expected
/// for well-spread points in a few dimensions. The tree reads the points through the span: they must not change or
/// move while the tree is in use. `noexcept` when the points' coordinate reads and the callbacks are.
///
/// **Errors.** None. An index span shorter than the points builds a tree over the first `indices.size()` points.
///
/// **Performance.** Not yet benchmarked.
///
/// **Tests and demos.** `tests/containers/kd_tree_test.cpp`, `demos/containers/kd_tree_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<optional>`, `<span>`,
/// `<utility>`, `metric/concepts.hpp`, `metric/minkowski.hpp`.
///
/// **Known issues.** Up to 2³² points (the indices are `std::uint32_t`), and depth at most 64, which a balanced
/// tree of fewer than 2⁶³ points never reaches. Changed in 0.20.0: the metric is a template parameter, points are read
/// through `metric::point_traits` (the `kd_point_traits` parameter is gone), `kd_neighbour::distance` is the
/// metric's distance (it was `distance2`, the squared Euclidean distance), and `for_each_within` takes a radius
/// (it took its square).
///
/// **Status.** Experimental, new in 0.19.0; generic over the metric in 0.20.0.

#ifndef snicholls_core_containers_kd_tree_hpp
#define snicholls_core_containers_kd_tree_hpp

#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>

namespace snicholls::containers {

    /// \brief One query result: a point's index and its distance.
    /// \ingroup core
    struct kd_neighbour {
        std::uint32_t index = 0;  ///< The point's index in the span the tree was built over.
        double distance = 0;      ///< The distance under the tree's metric.
    };

    /// \brief A balanced k-d tree over the caller's points, stored as a permutation of their indices.
    /// \ingroup core
    /// \tparam Point A `metric::point`.
    /// \tparam K The dimension, at least 1.
    /// \tparam Metric A `metric::decomposable` metric.
    template <metric::point Point, std::size_t K, typename Metric = metric::euclidean>
        requires metric::decomposable<Metric, double>
    class kd_tree {
        static_assert(K > 0, "snicholls::containers::kd_tree: the dimension is at least 1");
        static constexpr std::size_t max_depth = 64;
        template <typename Q>
        static constexpr bool nothrow_read = noexcept(metric::coordinate(std::declval<const Point&>(), std::size_t{})) &&
                                             noexcept(metric::coordinate(std::declval<const Q&>(), std::size_t{}));

        struct frame {
            std::size_t lo, hi, depth;
            double bound;  // a lower bound on the reduced distance of every point in the range
        };

        static constexpr double at(const auto& p, std::size_t axis) noexcept(noexcept(metric::coordinate(p, axis)))
        {
            return static_cast<double>(metric::coordinate(p, axis));
        }

    public:
        /// \brief Builds the tree.
        /// \param points The points, which must outlive the tree and not change while it is used.
        /// \param indices Storage for the permutation, `points.size()` long (a shorter one indexes a prefix).
        /// \param m The metric.
        constexpr kd_tree(std::span<const Point> points, std::span<std::uint32_t> indices, const Metric& m = Metric{}) noexcept(nothrow_read<Point>)
            : points_(points), index_(indices.first(indices.size() < points.size() ? indices.size() : points.size())), metric_(m)
        {
            for (std::size_t i = 0; i < index_.size(); ++i)
                index_[i] = static_cast<std::uint32_t>(i);
            std::array<frame, max_depth> stack{};
            std::size_t top = 0;
            if (index_.size() > 1)
                stack[top++] = {0, index_.size(), 0, 0};
            while (top > 0) {
                const frame f = stack[--top];
                const std::size_t mid = f.lo + (f.hi - f.lo) / 2;
                const std::size_t axis = f.depth % K;
                std::nth_element(index_.begin() + static_cast<std::ptrdiff_t>(f.lo), index_.begin() + static_cast<std::ptrdiff_t>(mid),
                                 index_.begin() + static_cast<std::ptrdiff_t>(f.hi),
                                 [&](std::uint32_t a, std::uint32_t b) { return at(points_[a], axis) < at(points_[b], axis); });
                if (mid - f.lo > 1)
                    stack[top++] = {f.lo, mid, f.depth + 1, 0};
                if (f.hi - (mid + 1) > 1)
                    stack[top++] = {mid + 1, f.hi, f.depth + 1, 0};
            }
        }

        /// \brief The points indexed. \return Their number.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return index_.size(); }
        /// \brief Whether the tree is empty. \return True for no points.
        [[nodiscard]] constexpr bool empty() const noexcept { return index_.empty(); }
        /// \brief The metric. \return It.
        [[nodiscard]] constexpr const Metric& distance_metric() const noexcept { return metric_; }

        /// \brief The point closest to `q`.
        /// \tparam Q A `metric::point` compatible with `Point`.
        /// \param q The query.
        /// \return The nearest point's index and distance; nothing for an empty tree.
        template <metric::point Q>
        [[nodiscard]] constexpr std::optional<kd_neighbour> nearest(const Q& q) const noexcept(nothrow_read<Q>)
        {
            if (empty())
                return std::nullopt;
            kd_neighbour best{0, std::numeric_limits<double>::infinity()};
            search(q, [&](std::uint32_t i, double r) {
                if (before(r, i, best))   // at an equal distance, the lower index
                    best = {i, r};
            }, [&] { return best.distance; });
            best.distance = metric::expand(metric_, best.distance);
            return best;
        }

        /// \brief The `out.size()` points closest to `q`, nearest first.
        /// \tparam Q A `metric::point` compatible with `Point`.
        /// \param q The query.
        /// \param out Receives the neighbours.
        /// \return How many were written: `min(out.size(), size())`.
        template <metric::point Q>
        constexpr std::size_t k_nearest(const Q& q, std::span<kd_neighbour> out) const noexcept(nothrow_read<Q>)
        {
            const std::size_t k = out.size() < size() ? out.size() : size();
            if (k == 0)
                return 0;
            std::size_t have = 0;
            search(q, [&](std::uint32_t i, double r) {
                if (have == k && !before(r, i, out[k - 1]))
                    return;
                std::size_t j = have < k ? have++ : k - 1;
                for (; j > 0 && before(r, i, out[j - 1]); --j)
                    out[j] = out[j - 1];
                out[j] = {i, r};
            }, [&] { return have < k ? std::numeric_limits<double>::infinity() : out[k - 1].distance; });
            for (std::size_t j = 0; j < k; ++j)
                out[j].distance = metric::expand(metric_, out[j].distance);
            return k;
        }

        /// \brief The up to `out.size()` points closest to `q` at a distance strictly below `radius`, nearest first: the
        /// k-nearest search with an upper bound, as SciPy's `cKDTree.query(k, distance_upper_bound=radius)` and as
        /// kriging's neighbourhood search use it. Points at `radius` or beyond are never returned, so the count can be
        /// below `out.size()`.
        /// \tparam Q A `metric::point` compatible with `Point`.
        /// \param q The query.
        /// \param radius The bound, in the metric's units (exclusive).
        /// \param out Receives the neighbours.
        /// \return How many were written.
        template <metric::point Q>
        constexpr std::size_t k_nearest_within(const Q& q, double radius, std::span<kd_neighbour> out) const noexcept(nothrow_read<Q>)
        {
            const std::size_t k = out.size() < size() ? out.size() : size();
            if (k == 0)
                return 0;
            const double bound = metric::contract(metric_, radius);
            std::size_t have = 0;
            search(q, [&](std::uint32_t i, double r) {
                if (!(r < bound) || (have == k && !before(r, i, out[k - 1])))
                    return;
                std::size_t j = have < k ? have++ : k - 1;
                for (; j > 0 && before(r, i, out[j - 1]); --j)
                    out[j] = out[j - 1];
                out[j] = {i, r};
            }, [&] { return have < k ? bound : (out[k - 1].distance < bound ? out[k - 1].distance : bound); });
            for (std::size_t j = 0; j < have; ++j)
                out[j].distance = metric::expand(metric_, out[j].distance);
            return have;
        }

        /// \brief Calls `f(index, distance)` for every point within distance `radius` of `q`, in no particular order.
        /// \tparam Q A `metric::point` compatible with `Point`.
        /// \tparam F The callback.
        /// \param q The query.
        /// \param radius The radius, in the metric's units.
        /// \param f The callback.
        template <metric::point Q, typename F>
        constexpr void for_each_within(const Q& q, double radius, F&& f) const noexcept(nothrow_read<Q> && noexcept(f(std::uint32_t{}, double{})))
        {
            const double r = metric::contract(metric_, radius);
            search(q, [&](std::uint32_t i, double d) {
                if (d <= r)
                    f(i, metric::expand(metric_, d));
            }, [&] { return r; });
        }

        /// \brief Calls `f(index)` for every point inside the box `[lo, hi]`, in no particular order.
        /// \tparam Q A `metric::point` compatible with `Point`.
        /// \tparam F The callback.
        /// \param lo The box's lower corner.
        /// \param hi The box's upper corner.
        /// \param f The callback.
        template <metric::point Q, typename F>
        constexpr void for_each_in_box(const Q& lo, const Q& hi, F&& f) const noexcept(nothrow_read<Q> && noexcept(f(std::uint32_t{})))
        {
            std::array<frame, max_depth> stack{};
            std::size_t top = 0;
            if (!empty())
                stack[top++] = {0, index_.size(), 0, 0};
            while (top > 0) {
                const frame fr = stack[--top];
                const std::size_t mid = fr.lo + (fr.hi - fr.lo) / 2;
                const std::size_t axis = fr.depth % K;
                const auto& p = points_[index_[mid]];
                bool inside = true;
                for (std::size_t a = 0; a < K && inside; ++a) {
                    const double c = at(p, a);
                    inside = c >= at(lo, a) && c <= at(hi, a);
                }
                if (inside)
                    f(index_[mid]);
                const double c = at(p, axis);
                if (mid > fr.lo && at(lo, axis) <= c)
                    stack[top++] = {fr.lo, mid, fr.depth + 1, 0};
                if (mid + 1 < fr.hi && at(hi, axis) >= c)
                    stack[top++] = {mid + 1, fr.hi, fr.depth + 1, 0};
            }
        }

    private:
        // The order of neighbours: nearer first, and at an equal distance the lower point index first, so a query's
        // result is one well-defined list (it matches a reference that breaks ties by index, bit for bit)
        static constexpr bool before(double r, std::uint32_t i, const kd_neighbour& n) noexcept
        {
            return r < n.distance || (r == n.distance && i < n.index);
        }

        // The reduced distance, stopping early once it exceeds bound
        template <typename Q>
        constexpr double reduced(const Q& q, std::uint32_t i, double bound) const noexcept(nothrow_read<Q>)
        {
            double acc = 0;
            for (std::size_t a = 0; a < K && !(acc > bound); ++a)
                acc = metric_.combine(acc, metric_.term(a, at(points_[i], a) - at(q, a)));
            return acc;
        }

        // Visits candidates nearest-first along the tree, pruning ranges whose bound is beyond bound()
        template <typename Q, typename Visit, typename Bound>
        constexpr void search(const Q& q, Visit&& visit, Bound&& bound) const
        {
            std::array<frame, max_depth> stack{};
            std::size_t top = 0;
            if (!empty())
                stack[top++] = {0, index_.size(), 0, 0};
            while (top > 0) {
                const frame fr = stack[--top];
                if (fr.bound > bound())
                    continue;
                const std::size_t mid = fr.lo + (fr.hi - fr.lo) / 2;
                const std::size_t axis = fr.depth % K;
                const std::uint32_t node = index_[mid];
                visit(node, reduced(q, node, bound()));
                const double diff = at(q, axis) - at(points_[node], axis);
                const double t = metric_.term(axis, diff);
                const double far_bound = fr.bound > t ? fr.bound : t;
                const bool left_near = diff < 0;
                const frame lo_child{fr.lo, mid, fr.depth + 1, left_near ? fr.bound : far_bound};
                const frame hi_child{mid + 1, fr.hi, fr.depth + 1, left_near ? far_bound : fr.bound};
                // Push the far child first so the near one is searched first
                const frame& near = left_near ? lo_child : hi_child;
                const frame& far = left_near ? hi_child : lo_child;
                if (far.hi > far.lo)
                    stack[top++] = far;
                if (near.hi > near.lo)
                    stack[top++] = near;
            }
        }

        std::span<const Point> points_;
        std::span<std::uint32_t> index_;
        Metric metric_;
    };

} // namespace snicholls::containers

#endif /* snicholls_core_containers_kd_tree_hpp */
