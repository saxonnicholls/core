//
//  ann_forest.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::containers::ann_forest<Point, D, Metric>`: approximate nearest neighbours by a forest of random-projection trees (the algorithm of Spotify's Annoy), under any metric, in flat storage the caller owns.
/// \ingroup core
///
/// **Overview.** In tens or hundreds of dimensions (embeddings), exact trees degrade to a linear scan. A forest of
/// random-projection trees trades a little recall for orders of magnitude in speed. Each tree splits its points
/// recursively by a hyperplane between two centroids found by a short two-means on a sample. A query walks all the
/// trees best-first by how near it is to each split, gathers a budget of candidates from the leaves it reaches, and
/// re-ranks them exactly with the metric. More trees and a larger candidate budget buy recall.
///
/// - `Point` is any `metric::point` of dimension `D`; `Metric` is any metric (`distance_for`). The split is chosen
///   by the metric (`ann_default_split`): hyperplanes between centroids for the Minkowski metrics
///   (`ann_euclidean_split`), hyperplanes through the origin between unit-length centroids for `angular`,
///   `cosine_distance`, `angle` and `inner_product` (`ann_angular_split`), and random bits for `hamming`
///   (`ann_bit_split`). A metric of the caller's specialises `ann_default_split`, or names the split explicitly.
/// - `build(points, trees, leaf_size, nodes, items, normals, g)`: builds into the caller's storage (sized by
///   `suggested_storage`) with the caller's generator, so a seed reproduces the index exactly.
/// - The index is three flat arrays of trivially copyable values (`ann_node`s, item indices and split normals)
///   linked by indices, not pointers, plus an `ann_layout`. A consumer can therefore write them to a file and map
///   them back, sharing one index between processes as Annoy does. The constructor from a layout and spans attaches
///   to storage without rebuilding.
/// - `query(q, out, candidates, frontier)`: the `out.size()` nearest found, nearest first, with their exact
///   distances. `candidates` is the budget of points examined (Annoy's `search_k`: `trees · k` is a reasonable
///   start); `frontier` is the best-first queue's storage.
///
/// \snippet containers/ann_forest_demo.cpp ann_forest
///
/// **Relationships.** `metric/` (every metric plugs in), `containers/kd_tree.hpp` (exact, for a few dimensions),
/// `stats/random.hpp` (the generator), `result.hpp`. After Bernhardsson's Annoy (github.com/spotify/annoy). Group:
/// \ref core.
///
/// **Implementations.** Splits: two-means of 200 steps over random members of the node's points (Annoy's
/// `two_means`), centroids weighted by their counts. When a split leaves one side empty, the points are shuffled and
/// halved and the split's normal is zero, so a query explores both sides. Queries: a binary max-heap on
/// `min(parent priority, ±margin)`, every root starting at +∞; candidates are sorted, de-duplicated and re-ranked.
/// Build is breadth-first over the node array (the array is the queue); query is iterative.
///
/// **Guarantees.** `constexpr`, no allocation, no recursion. Build O(trees · n log n) expected; query
/// O(candidates · D). Deterministic for a given generator state. `noexcept` is not claimed (the caller's metric and
/// generator decide). The results are approximate: every returned distance is exact, but a closer point may be
/// missed.
///
/// **Errors.** `ann_error::out_of_storage`: the node or normal storage was too small for the trees built (rebuild
/// with more; `suggested_storage` is generous for typical data). `ann_error::invalid_parameters`: no points, no
/// trees, a leaf size of zero, more than 2³¹ points, an item span shorter than `points · trees`, or points of the
/// wrong dimension.
///
/// **Performance.** Not yet benchmarked: a benchmark against brute force (recall and speed) and against Annoy itself
/// on the same data is planned.
///
/// **Tests and demos.** `tests/containers/ann_forest_test.cpp`, `demos/containers/ann_forest_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`,
/// `math/constexpr_math.hpp`, `metric/angular.hpp`, `metric/binary.hpp`, `metric/concepts.hpp`,
/// `metric/minkowski.hpp`, `result.hpp`, `stats/random.hpp`.
///
/// **Known issues.** `inner_product` (maximum inner product search) uses the angular split without Annoy's extra
/// norm dimension, so its recall is lower for vectors of very different lengths; normalise them, or append
/// `√(max |x|² - |x|²)` as a last coordinate yourself. A full frontier drops further entries (lower recall, never a
/// wrong distance).
///
/// **Status.** Experimental, new in 0.20.0.

#ifndef snicholls_core_containers_ann_forest_hpp
#define snicholls_core_containers_ann_forest_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/metric/angular.hpp>
#include <snicholls/core/metric/binary.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::containers {

    /// \brief Why an index was not built.
    /// \ingroup core
    enum class ann_error : std::uint8_t {
        out_of_storage = 1,      ///< The node or normal storage was too small.
        invalid_parameters = 2,  ///< No points or trees, a zero leaf size, too many points, short storage, wrong dimension.
    };

    /// \brief One node of a tree: a leaf (`count > 0`, items `[first, first + count)`) or a split (`count == 0`).
    /// \ingroup core
    struct ann_node {
        std::int32_t left = -1;    ///< A split's child on the negative side.
        std::int32_t right = -1;   ///< A split's child on the positive side.
        std::uint32_t first = 0;   ///< A leaf's first item.
        std::uint32_t count = 0;   ///< A leaf's items; 0 for a split.
        std::uint32_t split = 0;   ///< A split's bit (bit splits); its normal is at `index · D` (hyperplane splits).
        float offset = 0;          ///< A hyperplane split's offset.
    };

    /// \brief The shape of a built index: what to keep beside the three arrays to attach to them again.
    /// \ingroup core
    struct ann_layout {
        std::uint32_t points = 0;     ///< The points indexed.
        std::uint32_t trees = 0;      ///< The trees; their roots are nodes `0 .. trees - 1`.
        std::uint32_t leaf_size = 0;  ///< The most points in a leaf.
        std::uint32_t nodes = 0;      ///< The nodes used.
    };

    /// \brief One result: a point's index and its exact distance.
    /// \ingroup core
    struct ann_neighbour {
        std::uint32_t index = 0;  ///< The point's index.
        double distance = 0;      ///< The distance under the forest's metric.
    };

    /// \brief An entry of the best-first queue.
    /// \ingroup core
    struct ann_frontier {
        float priority = 0;      ///< How near the query is to this node's region (larger is nearer).
        std::int32_t node = 0;   ///< The node.
    };

    /// \brief Hyperplanes between two centroids, offset to their midpoint: for the Minkowski metrics. \ingroup core
    struct ann_euclidean_split {};
    /// \brief Hyperplanes through the origin between two unit-length centroids: for the angular metrics. \ingroup core
    struct ann_angular_split {};
    /// \brief One random bit per split: for `hamming` over packed words. \ingroup core
    struct ann_bit_split {};

    /// \brief The split a metric uses by default: hyperplanes between centroids. Specialise it for a metric of your own.
    /// \ingroup core
    /// \tparam M The metric.
    template <typename M>
    struct ann_default_split {
        using type = ann_euclidean_split;  ///< The split.
    };
    /// \brief `angular` splits through the origin. \ingroup core
    template <>
    struct ann_default_split<metric::angular> {
        using type = ann_angular_split;  ///< The split.
    };
    /// \brief `cosine_distance` splits through the origin. \ingroup core
    template <>
    struct ann_default_split<metric::cosine_distance> {
        using type = ann_angular_split;  ///< The split.
    };
    /// \brief `angle` splits through the origin. \ingroup core
    template <>
    struct ann_default_split<metric::angle> {
        using type = ann_angular_split;  ///< The split.
    };
    /// \brief `inner_product` splits through the origin. \ingroup core
    template <>
    struct ann_default_split<metric::inner_product> {
        using type = ann_angular_split;  ///< The split.
    };
    /// \brief `hamming` splits on bits. \ingroup core
    template <>
    struct ann_default_split<metric::hamming> {
        using type = ann_bit_split;  ///< The split.
    };

    /// \brief Storage sizes for an index.
    /// \ingroup core
    struct ann_storage {
        std::size_t nodes = 0;    ///< `ann_node`s.
        std::size_t items = 0;    ///< Item indices: `points · trees`.
        std::size_t normals = 0;  ///< Split normal coordinates: `nodes · D` for hyperplane splits, 0 for bit splits.
    };

    /// \brief An approximate-nearest-neighbour index over points of dimension `D`.
    /// \ingroup core
    /// \tparam Point A `metric::point`.
    /// \tparam D The dimension (the number of words, for a bit split).
    /// \tparam Metric The metric.
    /// \tparam Split The split: `ann_euclidean_split`, `ann_angular_split` or `ann_bit_split`.
    /// \tparam Scalar The type of the stored normals.
    template <metric::point Point, std::size_t D, typename Metric = metric::euclidean, typename Split = typename ann_default_split<Metric>::type, typename Scalar = float>
        requires metric::distance_for<Metric, Point>
    class ann_forest {
        static_assert(D > 0, "snicholls::containers::ann_forest: the dimension is at least 1");
        static constexpr bool bits = std::is_same_v<Split, ann_bit_split>;
        static constexpr bool through_origin = std::is_same_v<Split, ann_angular_split>;
        using vec = std::array<Scalar, D>;

    public:
        /// \brief Generous storage for an index of `n` points, `trees` trees and leaves of at most `leaf_size` points.
        /// \param n The points. \param trees The trees. \param leaf_size The most points in a leaf.
        /// \return The sizes; `build` reports `out_of_storage` in the rare case they are not enough.
        [[nodiscard]] static constexpr ann_storage suggested_storage(std::size_t n, std::size_t trees, std::size_t leaf_size) noexcept
        {
            const std::size_t leaf = leaf_size == 0 ? 1 : leaf_size;
            const std::size_t per_tree = 4 * ((n + leaf - 1) / leaf) + 3;
            const std::size_t nodes = trees * per_tree;
            return {nodes, n * trees, bits ? 0 : nodes * D};
        }

        /// \brief Builds an index into the caller's storage.
        /// \tparam G A `stats::full_range_generator`.
        /// \param points The points, of dimension `D`, which must outlive the index and not change.
        /// \param trees The trees (more trees, better recall, larger index).
        /// \param leaf_size The most points in a leaf.
        /// \param nodes Node storage.
        /// \param items Item storage, at least `points.size() · trees`.
        /// \param normals Normal storage (hyperplane splits).
        /// \param g The generator.
        /// \param m The metric.
        /// \return The index, or an `ann_error`.
        template <stats::full_range_generator G>
        [[nodiscard]] static constexpr result<ann_forest, ann_error> build(std::span<const Point> points, std::size_t trees, std::size_t leaf_size, std::span<ann_node> nodes,
                                                                          std::span<std::uint32_t> items, std::span<Scalar> normals, G& g, const Metric& m = Metric{})
        {
            const std::size_t n = points.size();
            if (n == 0 || trees == 0 || leaf_size == 0 || n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) || items.size() < n * trees)
                return err(ann_error::invalid_parameters);
            for (const auto& p : points)
                if (metric::dimension(p) != D)
                    return err(ann_error::invalid_parameters);
            if (nodes.size() < trees)
                return err(ann_error::out_of_storage);
            // Breadth-first over the node array itself: a pending node holds its item range in `first` and `count`
            // and is marked by `left == -2`; its children are allocated at the end, so the array is the queue. No
            // stack, no depth limit.
            std::size_t used = trees;
            for (std::size_t t = 0; t < trees; ++t) {
                const std::size_t base = t * n;
                for (std::size_t i = 0; i < n; ++i)
                    items[base + i] = static_cast<std::uint32_t>(i);
                nodes[t] = {-2, -1, static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(n), 0, 0};
            }
            for (std::size_t k = 0; k < used; ++k) {
                ann_node& node = nodes[k];
                if (node.left != -2)
                    continue;
                const std::size_t lo = node.first, hi = lo + node.count;
                if (node.count <= leaf_size) {
                    node.left = -1;
                    continue;
                }
                if (used + 2 > nodes.size())
                    return err(ann_error::out_of_storage);
                if constexpr (!bits) {
                    if ((k + 1) * D > normals.size())
                        return err(ann_error::out_of_storage);
                }
                const std::size_t mid = split_range(points, items, lo, hi, k, node, normals, g, m);
                const std::size_t left = used, right = used + 1;
                used += 2;
                nodes[left] = {-2, -1, static_cast<std::uint32_t>(lo), static_cast<std::uint32_t>(mid - lo), 0, 0};
                nodes[right] = {-2, -1, static_cast<std::uint32_t>(mid), static_cast<std::uint32_t>(hi - mid), 0, 0};
                ann_node& split = nodes[k];
                split.left = static_cast<std::int32_t>(left);
                split.right = static_cast<std::int32_t>(right);
                split.first = 0;
                split.count = 0;
            }
            const ann_layout layout{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(trees), static_cast<std::uint32_t>(leaf_size), static_cast<std::uint32_t>(used)};
            return ok(ann_forest(layout, points, nodes.first(used), items.first(n * trees), normals.first(bits ? 0 : used * D), m));
        }

        /// \brief Attaches to an index already built (or mapped from a file).
        /// \param layout The index's shape.
        /// \param points The points it was built over.
        /// \param nodes Its nodes.
        /// \param items Its item indices.
        /// \param normals Its split normals.
        /// \param m The metric.
        constexpr ann_forest(const ann_layout& layout, std::span<const Point> points, std::span<const ann_node> nodes, std::span<const std::uint32_t> items,
                             std::span<const Scalar> normals, const Metric& m = Metric{}) noexcept
            : layout_(layout), points_(points), nodes_(nodes), items_(items), normals_(normals), metric_(m)
        {
        }

        /// \brief The index's shape, to store beside its arrays. \return It.
        [[nodiscard]] constexpr const ann_layout& layout() const noexcept { return layout_; }
        /// \brief The nodes. \return They.
        [[nodiscard]] constexpr std::span<const ann_node> nodes() const noexcept { return nodes_; }
        /// \brief The item indices. \return They.
        [[nodiscard]] constexpr std::span<const std::uint32_t> items() const noexcept { return items_; }
        /// \brief The split normals. \return They.
        [[nodiscard]] constexpr std::span<const Scalar> normals() const noexcept { return normals_; }
        /// \brief The points indexed. \return Their number.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return layout_.points; }

        /// \brief The approximate `out.size()` nearest neighbours of `q`, nearest first, with exact distances.
        /// \tparam Q A `metric::point` the metric takes with `Point`.
        /// \param q The query.
        /// \param out Receives the neighbours.
        /// \param candidates Storage for the candidates; its size is the search budget (`trees · k` is a start).
        /// \param frontier Storage for the best-first queue (a few times the trees is plenty).
        /// \return How many neighbours were written.
        template <metric::point Q>
            requires metric::distance_for<Metric, Point, Q>
        constexpr std::size_t query(const Q& q, std::span<ann_neighbour> out, std::span<std::uint32_t> candidates, std::span<ann_frontier> frontier) const
        {
            if (out.empty() || candidates.empty() || frontier.empty() || layout_.points == 0)
                return 0;
            std::size_t heap = 0;
            auto less = [](const ann_frontier& a, const ann_frontier& b) { return a.priority < b.priority; };
            auto push = [&](float priority, std::int32_t node) {
                if (heap == frontier.size())
                    return;  // full: drop it (lower recall, never a wrong answer)
                frontier[heap++] = {priority, node};
                std::push_heap(frontier.begin(), frontier.begin() + static_cast<std::ptrdiff_t>(heap), less);
            };
            for (std::uint32_t t = 0; t < layout_.trees; ++t)
                push(std::numeric_limits<float>::infinity(), static_cast<std::int32_t>(t));
            std::size_t found = 0;
            while (heap > 0 && found < candidates.size()) {
                std::pop_heap(frontier.begin(), frontier.begin() + static_cast<std::ptrdiff_t>(heap), less);
                const ann_frontier top = frontier[--heap];
                const ann_node& node = nodes_[static_cast<std::size_t>(top.node)];
                if (node.count > 0) {
                    for (std::uint32_t i = 0; i < node.count && found < candidates.size(); ++i)
                        candidates[found++] = items_[node.first + i];
                    continue;
                }
                const float mg = static_cast<float>(margin(q, node, static_cast<std::size_t>(top.node)));
                push(top.priority < mg ? top.priority : mg, node.right);
                push(top.priority < -mg ? top.priority : -mg, node.left);
            }
            const auto cand = candidates.first(found);
            std::sort(cand.begin(), cand.end());
            const auto last = std::unique(cand.begin(), cand.end());
            const std::size_t unique = static_cast<std::size_t>(last - cand.begin());
            std::size_t have = 0;
            const std::size_t k = out.size();
            for (std::size_t c = 0; c < unique; ++c) {
                const double d = static_cast<double>(metric_(points_[cand[c]], q));
                if (have == k && !(d < out[k - 1].distance))
                    continue;
                std::size_t j = have < k ? have++ : k - 1;
                for (; j > 0 && out[j - 1].distance > d; --j)
                    out[j] = out[j - 1];
                out[j] = {cand[c], d};
            }
            return have;
        }

    private:
        static constexpr Scalar coord(const auto& p, std::size_t i) { return static_cast<Scalar>(metric::coordinate(p, i)); }

        static constexpr bool bit_of(const auto& p, std::uint32_t bit)
        {
            using W = metric::value_t<Point>;
            constexpr std::uint32_t w = sizeof(W) * 8;
            return ((metric::coordinate(p, bit / w) >> (bit % w)) & W(1)) != 0;
        }

        static constexpr void normalise(vec& v)
        {
            Scalar s(0);
            for (const Scalar x : v)
                s += x * x;
            if (s > Scalar(0)) {
                const Scalar r = static_cast<Scalar>(math::sqrt(s));
                for (Scalar& x : v)
                    x /= r;
            }
        }

        static constexpr vec load(const Point& p)
        {
            vec v{};
            for (std::size_t i = 0; i < D; ++i)
                v[i] = coord(p, i);
            if constexpr (through_origin)
                normalise(v);
            return v;
        }

        // The signed distance of q from a split: hyperplane margin, or ±1 by a bit
        template <typename Q>
        constexpr double margin(const Q& q, const ann_node& node, std::size_t index) const
        {
            if constexpr (bits) {
                return bit_of(q, node.split) ? 1.0 : -1.0;
            } else {
                double s = node.offset;
                for (std::size_t i = 0; i < D; ++i)
                    s += static_cast<double>(normals_[index * D + i]) * static_cast<double>(coord(q, i));
                return s;
            }
        }

        // Chooses the split of items[lo, hi) and partitions them: negative side first. Returns the boundary.
        template <typename G>
        static constexpr std::size_t split_range(std::span<const Point> points, std::span<std::uint32_t> items, std::size_t lo, std::size_t hi, std::size_t index, ann_node& node,
                                                 std::span<Scalar> normals, G& g, const Metric& m)
        {
            static_cast<void>(m);
            const std::size_t size = hi - lo;
            auto positive = [&](std::uint32_t item) -> int {  // 1 positive, 0 negative, -1 on the plane
                if constexpr (bits) {
                    return bit_of(points[item], node.split) ? 1 : 0;
                } else {
                    Scalar s = node.offset;
                    for (std::size_t i = 0; i < D; ++i)
                        s += normals[index * D + i] * coord(points[item], i);
                    return s > Scalar(0) ? 1 : s < Scalar(0) ? 0 : -1;
                }
            };
            if constexpr (bits) {
                constexpr std::uint32_t total = static_cast<std::uint32_t>(D * sizeof(metric::value_t<Point>) * 8);
                node.split = static_cast<std::uint32_t>(stats::uniform_index(g, total));
                for (int attempt = 0; attempt < 32; ++attempt) {
                    std::size_t ones = 0;
                    for (std::size_t i = lo; i < hi; ++i)
                        ones += bit_of(points[items[i]], node.split) ? 1u : 0u;
                    if (ones > 0 && ones < size)
                        break;
                    node.split = static_cast<std::uint32_t>(stats::uniform_index(g, total));
                }
            } else {
                // Two-means over random members: two centroids, each weighted by its count
                const std::size_t i0 = lo + static_cast<std::size_t>(stats::uniform_index(g, size));
                std::size_t j0 = lo + static_cast<std::size_t>(stats::uniform_index(g, size - 1));
                if (j0 >= i0)
                    ++j0;
                vec p = load(points[items[i0]]), q = load(points[items[j0]]);
                Scalar ic(1), jc(1);
                for (int step = 0; step < 200; ++step) {
                    const vec x = load(points[items[lo + static_cast<std::size_t>(stats::uniform_index(g, size))]]);
                    Scalar di(0), dj(0);
                    for (std::size_t c = 0; c < D; ++c) {
                        di += (p[c] - x[c]) * (p[c] - x[c]);
                        dj += (q[c] - x[c]) * (q[c] - x[c]);
                    }
                    di *= ic;
                    dj *= jc;
                    if (di < dj) {
                        for (std::size_t c = 0; c < D; ++c)
                            p[c] = (p[c] * ic + x[c]) / (ic + Scalar(1));
                        ic += Scalar(1);
                    } else if (dj < di) {
                        for (std::size_t c = 0; c < D; ++c)
                            q[c] = (q[c] * jc + x[c]) / (jc + Scalar(1));
                        jc += Scalar(1);
                    }
                }
                vec normal{};
                for (std::size_t c = 0; c < D; ++c)
                    normal[c] = p[c] - q[c];
                normalise(normal);
                Scalar offset(0);
                if constexpr (!through_origin)
                    for (std::size_t c = 0; c < D; ++c)
                        offset -= normal[c] * (p[c] + q[c]) / Scalar(2);
                for (std::size_t c = 0; c < D; ++c)
                    normals[index * D + c] = normal[c];
                node.offset = static_cast<float>(offset);
            }
            // Partition: negative side first; a point on the plane goes to a random side
            std::size_t i = lo, j = hi;
            while (i < j) {
                int s = positive(items[i]);
                if (s < 0)
                    s = static_cast<int>(stats::uniform_index(g, 2));
                if (s == 0)
                    ++i;
                else
                    std::swap(items[i], items[--j]);
            }
            if (i == lo || i == hi) {
                // Degenerate: shuffle and halve, and make the split neutral so a query explores both sides
                stats::shuffle(items.subspan(lo, size), g);
                if constexpr (!bits) {
                    for (std::size_t c = 0; c < D; ++c)
                        normals[index * D + c] = Scalar(0);
                    node.offset = 0;
                }
                return lo + size / 2;
            }
            return i;
        }

        ann_layout layout_;
        std::span<const Point> points_;
        std::span<const ann_node> nodes_;
        std::span<const std::uint32_t> items_;
        std::span<const Scalar> normals_;
        Metric metric_;
    };

} // namespace snicholls::containers

#endif /* snicholls_core_containers_ann_forest_hpp */
