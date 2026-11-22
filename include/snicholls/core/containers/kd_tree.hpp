//
//  kd_tree.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::containers::kd_tree<Point, K, Traits>`: a balanced k-d tree built in place over the caller's points, with nearest, k-nearest, radius and box queries, iterative and allocation-free.
/// \ingroup core
///
/// **Overview.** A scan brings hundreds of detections and a tracker asks, for each track, which detections lie
/// near its prediction. A k-d tree answers that in about `log n` per query instead of `n`. This one is built once
/// over a span of points the caller owns, and stores nothing but a permutation of their indices (the caller's
/// span, too). The tree is implicit: the median of each index range is its node, split on axis
/// `depth mod K`. So there are no node objects and no pointers, and a rebuild per scan costs O(n log n).
///
/// - `Point` is any type; `Traits::coordinate(p, axis)` reads one coordinate. `kd_point_traits` (the default) reads
///   `p[axis]`, for `fixed_vector`, `std::array` and anything else with `operator[]`.
/// - `nearest(q)`: the closest point and its squared distance, or nothing for an empty tree.
/// - `k_nearest(q, out)`: the `out.size()` closest, nearest first.
/// - `for_each_within(q, r², f)`: `f(index, d²)` for every point within radius `r`.
/// - `for_each_in_box(lo, hi, f)`: `f(index)` for every point with `lo ≤ p ≤ hi` on every axis.
///
/// Distances are Euclidean. For Mahalanobis gating, query in whitened coordinates (points mapped through `L⁻¹` of
/// the innovation covariance), or use the box query for the gate's bounding box and test each candidate.
///
/// \snippet containers/kd_tree_demo.cpp kd_tree
///
/// **Relationships.** `intrusive_avl_tree.hpp` (the same policy style), `linalg/fixed.hpp` (points). Stone Soup's
/// `DetectionKDTreeNN` and `KDTreePDA` associators are built on it in the framework above core. Group: \ref core.
///
/// **Implementations.** Build: an explicit stack of index ranges, each partitioned at its median by
/// `std::nth_element` (constexpr since C++20). Queries: an explicit stack of at most 64 frames, near child first,
/// a far child skipped when its splitting plane is farther than the best distance so far.
///
/// **Guarantees.** `constexpr`, no allocation, no recursion. Build O(n log n) expected; queries O(log n) expected
/// for well-spread points. The tree reads the points through the span: they must not change or move while the tree
/// is in use. `noexcept` when `Traits::coordinate` and the callbacks are.
///
/// **Errors.** None. An index span shorter than the points builds a tree over the first `indices.size()` points.
///
/// **Performance.** Not yet benchmarked.
///
/// **Tests and demos.** `tests/containers/kd_tree_test.cpp`, `demos/containers/kd_tree_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<optional>`, `<span>`, `<utility>`.
///
/// **Known issues.** Up to 2³² points (the indices are `std::uint32_t`), and depth at most 64, which a balanced
/// tree of fewer than 2⁶³ points never reaches.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_containers_kd_tree_hpp
#define snicholls_core_containers_kd_tree_hpp

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>

namespace snicholls::containers {

    /// \brief The default coordinate reader: `p[axis]`.
    /// \ingroup core
    struct kd_point_traits {
        /// \brief One coordinate. \tparam P The point type. \param p The point. \param axis The axis.
        /// \return `p[axis]` as `double`.
        template <typename P>
        [[nodiscard]] static constexpr double coordinate(const P& p, std::size_t axis) noexcept(noexcept(p[axis]))
        {
            return static_cast<double>(p[axis]);
        }
    };

    /// \brief One query result: a point's index and its squared distance.
    /// \ingroup core
    struct kd_neighbour {
        std::uint32_t index = 0;   ///< The point's index in the span the tree was built over.
        double distance2 = 0;      ///< The squared Euclidean distance.
    };

    /// \brief A balanced k-d tree over the caller's points, stored as a permutation of their indices.
    /// \ingroup core
    /// \tparam Point The point type.
    /// \tparam K The dimension, at least 1.
    /// \tparam Traits The coordinate reader: `Traits::coordinate(p, axis)`.
    template <typename Point, std::size_t K, typename Traits = kd_point_traits>
    class kd_tree {
        static_assert(K > 0, "snicholls::containers::kd_tree: the dimension is at least 1");
        static constexpr std::size_t max_depth = 64;
        template <typename Q>
        static constexpr bool nothrow_read = noexcept(Traits::coordinate(std::declval<const Point&>(), std::size_t{}))
                                             && noexcept(Traits::coordinate(std::declval<const Q&>(), std::size_t{}));

        struct frame {
            std::size_t lo, hi, depth;
            double plane2;  // the squared distance to the plane that bounds this range, a lower bound on its points
        };

    public:
        /// \brief Builds the tree.
        /// \param points The points, which must outlive the tree and not change while it is used.
        /// \param indices Storage for the permutation, `points.size()` long (a shorter one indexes a prefix).
        constexpr kd_tree(std::span<const Point> points, std::span<std::uint32_t> indices) noexcept(nothrow_read<Point>)
            : points_(points), index_(indices.first(indices.size() < points.size() ? indices.size() : points.size()))
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
                                 [&](std::uint32_t a, std::uint32_t b) { return Traits::coordinate(points_[a], axis) < Traits::coordinate(points_[b], axis); });
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

        /// \brief The point closest to `q`.
        /// \tparam Q A type `Traits` reads.
        /// \param q The query.
        /// \return The nearest point's index and squared distance; nothing for an empty tree.
        template <typename Q>
        [[nodiscard]] constexpr std::optional<kd_neighbour> nearest(const Q& q) const noexcept(nothrow_read<Q>)
        {
            if (empty())
                return std::nullopt;
            kd_neighbour best{0, std::numeric_limits<double>::infinity()};
            search(q, [&](std::uint32_t i, double d2) {
                if (d2 < best.distance2)
                    best = {i, d2};
            }, [&] { return best.distance2; });
            return best;
        }

        /// \brief The `out.size()` points closest to `q`, nearest first.
        /// \tparam Q A type `Traits` reads.
        /// \param q The query.
        /// \param out Receives the neighbours.
        /// \return How many were written: `min(out.size(), size())`.
        template <typename Q>
        constexpr std::size_t k_nearest(const Q& q, std::span<kd_neighbour> out) const noexcept(nothrow_read<Q>)
        {
            const std::size_t k = out.size() < size() ? out.size() : size();
            if (k == 0)
                return 0;
            std::size_t have = 0;
            search(q, [&](std::uint32_t i, double d2) {
                if (have == k && !(d2 < out[k - 1].distance2))
                    return;
                std::size_t j = have < k ? have++ : k - 1;
                for (; j > 0 && out[j - 1].distance2 > d2; --j)
                    out[j] = out[j - 1];
                out[j] = {i, d2};
            }, [&] { return have < k ? std::numeric_limits<double>::infinity() : out[k - 1].distance2; });
            return k;
        }

        /// \brief Calls `f(index, d²)` for every point within distance `√radius2` of `q`, in no particular order.
        /// \tparam Q A type `Traits` reads.
        /// \tparam F The callback.
        /// \param q The query.
        /// \param radius2 The squared radius.
        /// \param f The callback.
        template <typename Q, typename F>
        constexpr void for_each_within(const Q& q, double radius2, F&& f) const noexcept(nothrow_read<Q> && noexcept(f(std::uint32_t{}, double{})))
        {
            search(q, [&](std::uint32_t i, double d2) {
                if (d2 <= radius2)
                    f(i, d2);
            }, [&] { return radius2; });
        }

        /// \brief Calls `f(index)` for every point inside the box `[lo, hi]`, in no particular order.
        /// \tparam Q A type `Traits` reads.
        /// \tparam F The callback.
        /// \param lo The box's lower corner.
        /// \param hi The box's upper corner.
        /// \param f The callback.
        template <typename Q, typename F>
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
                    const double c = Traits::coordinate(p, a);
                    inside = c >= Traits::coordinate(lo, a) && c <= Traits::coordinate(hi, a);
                }
                if (inside)
                    f(index_[mid]);
                const double c = Traits::coordinate(p, axis);
                if (mid > fr.lo && Traits::coordinate(lo, axis) <= c)
                    stack[top++] = {fr.lo, mid, fr.depth + 1, 0};
                if (mid + 1 < fr.hi && Traits::coordinate(hi, axis) >= c)
                    stack[top++] = {mid + 1, fr.hi, fr.depth + 1, 0};
            }
        }

    private:
        template <typename Q>
        constexpr double distance2(const Q& q, std::uint32_t i) const
        {
            double s = 0;
            for (std::size_t a = 0; a < K; ++a) {
                const double d = Traits::coordinate(points_[i], a) - Traits::coordinate(q, a);
                s += d * d;
            }
            return s;
        }

        // Visits candidates nearest-first along the tree, pruning ranges whose plane is beyond bound()
        template <typename Q, typename Visit, typename Bound>
        constexpr void search(const Q& q, Visit&& visit, Bound&& bound) const
        {
            std::array<frame, max_depth> stack{};
            std::size_t top = 0;
            if (!empty())
                stack[top++] = {0, index_.size(), 0, 0};
            while (top > 0) {
                const frame fr = stack[--top];
                if (fr.plane2 > bound())
                    continue;
                const std::size_t mid = fr.lo + (fr.hi - fr.lo) / 2;
                const std::size_t axis = fr.depth % K;
                const std::uint32_t node = index_[mid];
                visit(node, distance2(q, node));
                const double diff = Traits::coordinate(q, axis) - Traits::coordinate(points_[node], axis);
                const bool left_near = diff < 0;
                const frame lo_child{fr.lo, mid, fr.depth + 1, left_near ? fr.plane2 : (fr.plane2 > diff * diff ? fr.plane2 : diff * diff)};
                const frame hi_child{mid + 1, fr.hi, fr.depth + 1, left_near ? (fr.plane2 > diff * diff ? fr.plane2 : diff * diff) : fr.plane2};
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
    };

} // namespace snicholls::containers

#endif /* snicholls_core_containers_kd_tree_hpp */
