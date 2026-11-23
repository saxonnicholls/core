//
//  concepts.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::metric::point`, `point_traits`, `distance_for`, `reducible`, `decomposable`: the plug-and-play contract between points, metrics and the structures that search with them.
/// \ingroup core
///
/// **Overview.** A search structure (a k-d tree, an approximate-nearest-neighbour forest, a set metric) needs two
/// things from its caller: how to read a point, and how far apart two points are. This header defines both as
/// customisation points, so any point type works with any metric, and any metric with any structure, as far as the
/// metric's declared capabilities allow.
///
/// **Points.** `point_traits<P>` says how to read a point: its `value_type`, its `dimension(p)`, its coordinate
/// `at(p, i)`, and its compile-time `extent` (`std::dynamic_extent` when only known at run time). It is provided for:
/// - anything with `operator[]` and `size()`: `std::array`, `std::vector`, `std::span`, `inplace_vector`,
///   `std::basic_string_view`, ...;
/// - C arrays;
/// - `linalg::fixed_vector` (and a one-row `fixed_matrix`).
///
/// A type of the caller's (a struct of `x`, `y`, `z`, a strided view into a tensor, a quantised row) becomes a point
/// by specialising `point_traits` for it, as `std::hash` or `math_limits` are specialised. The `point` concept says
/// whether a type is one.
///
/// **Metrics.** A metric is any value `m` with `m(a, b)` returning a distance. What else it offers is optional, and
/// a structure asks for what it needs:
///
/// | Capability | Members | Who uses it |
/// |---|---|---|
/// | `distance_for<M, A, B>` | `m(a, b)` | everything: brute force, re-ranking, set metrics |
/// | `reducible` | `m.reduced(a, b)`, `m.expand(r)`, `m.contract(d)` | searches that compare in a cheaper monotone form (squared Euclidean, Σ|d|ᵖ) |
/// | `decomposable` | `m.term(axis, diff)`, `m.combine(acc, t)` | k-d trees and early-exit scans: one coordinate's term bounds the whole |
/// | `true_metric` | `static constexpr bool is_metric = true` | trees that prune by the triangle inequality |
///
/// `reduced_distance(m, a, b)`, `expand(m, r)` and `contract(m, d)` work for every metric: a metric without a cheaper
/// form uses the distance itself. `accumulate(m, a, b)` and `accumulate_within(m, a, b, bound)` evaluate a
/// decomposable metric term by term, the second stopping as soon as the partial sum exceeds `bound`.
///
/// \snippet metric/concepts_demo.cpp concepts
///
/// **Relationships.** `metric/minkowski.hpp`, `metric/angular.hpp`, `metric/binary.hpp`, `metric/geodesic.hpp`,
/// `metric/statistical.hpp`, `metric/string.hpp` (the metrics), `metric/sets.hpp` (metrics between sets of points),
/// `containers/kd_tree.hpp` and `containers/ann_forest.hpp` (structures that search with them), `linalg/fixed.hpp`.
/// Group: \ref core.
///
/// **Guarantees.** Everything is `constexpr` and `noexcept` when the point's coordinate read and the metric's
/// operations are. No allocation.
///
/// **Errors.** None: a type that is not a point, or a metric that lacks a capability a structure needs, fails to
/// compile, with the concept named in the message.
///
/// **Performance.** Not benchmarked: the traits are inline reads; the metrics' benchmarks are in their headers.
///
/// **Tests and demos.** `tests/metric/concepts_test.cpp`, `demos/metric/concepts_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<span>`, `<type_traits>`, `<utility>`,
/// `linalg/fixed.hpp`.
///
/// **Known issues.** `std::vector<bool>` is not a point (its `operator[]` returns a proxy); pack bits into words and
/// use `metric/binary.hpp`.
///
/// **Status.** Experimental, new in 0.20.0.

#ifndef snicholls_core_metric_concepts_hpp
#define snicholls_core_metric_concepts_hpp

#include <snicholls/core/linalg/fixed.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls::metric {

    namespace detail {
        template <typename P>
        concept sized_indexable = requires(const P& p, std::size_t i) {
            { p.size() } -> std::convertible_to<std::size_t>;
            p[i];
        } && std::is_arithmetic_v<std::remove_cvref_t<decltype(std::declval<const P&>()[std::size_t{}])>>
          // a bool read by value is a packed-bit proxy (libstdc++'s const std::vector<bool>::operator[]): not a point
          && (!std::is_same_v<std::remove_cvref_t<decltype(std::declval<const P&>()[std::size_t{}])>, bool>
              || std::is_reference_v<decltype(std::declval<const P&>()[std::size_t{}])>);

        template <typename P>
        struct static_extent : std::integral_constant<std::size_t, std::dynamic_extent> {};
        template <typename T, std::size_t N>
        struct static_extent<std::array<T, N>> : std::integral_constant<std::size_t, N> {};
        template <typename T, std::size_t N>
        struct static_extent<std::span<T, N>> : std::integral_constant<std::size_t, N> {};
    } // namespace detail

    /// \brief How to read a point: `value_type`, `extent`, `dimension(p)`, `at(p, i)`. Specialise it for a point type
    /// of your own; the primary template is empty (not a point).
    /// \ingroup core
    /// \tparam P The point type.
    template <typename P>
    struct point_traits {};

    /// \brief A point read by `operator[]` and `size()`: `std::array`, `std::vector`, `std::span`, `inplace_vector`, ...
    /// \ingroup core
    /// \tparam P The point type.
    template <typename P>
        requires detail::sized_indexable<P>
    struct point_traits<P> {
        using value_type = std::remove_cvref_t<decltype(std::declval<const P&>()[std::size_t{}])>;  ///< The coordinate type.
        static constexpr std::size_t extent = detail::static_extent<std::remove_cv_t<P>>::value;     ///< The dimension, if fixed in the type.
        /// \brief The dimension. \param p The point. \return `p.size()`.
        [[nodiscard]] static constexpr std::size_t dimension(const P& p) noexcept(noexcept(p.size())) { return static_cast<std::size_t>(p.size()); }
        /// \brief One coordinate. \param p The point. \param i The axis. \return `p[i]`.
        [[nodiscard]] static constexpr value_type at(const P& p, std::size_t i) noexcept(noexcept(p[i])) { return p[i]; }
    };

    /// \brief A C array as a point.
    /// \ingroup core
    /// \tparam T The coordinate type.
    /// \tparam N The dimension.
    template <typename T, std::size_t N>
        requires std::is_arithmetic_v<T>
    struct point_traits<T[N]> {
        using value_type = std::remove_cv_t<T>;          ///< The coordinate type.
        static constexpr std::size_t extent = N;         ///< The dimension.
        /// \brief The dimension. \param p The point (unused). \return `N`.
        [[nodiscard]] static constexpr std::size_t dimension(const T (&p)[N]) noexcept
        {
            static_cast<void>(p);
            return N;
        }
        /// \brief One coordinate. \param p The point. \param i The axis. \return `p[i]`.
        [[nodiscard]] static constexpr value_type at(const T (&p)[N], std::size_t i) noexcept { return p[i]; }
    };

    /// \brief A `linalg::fixed_vector` (a column) as a point.
    /// \ingroup core
    /// \tparam T The coordinate type.
    /// \tparam N The dimension.
    template <linalg::scalar T, std::size_t N>
    struct point_traits<linalg::fixed_matrix<T, N, 1>> {
        using value_type = T;                            ///< The coordinate type.
        static constexpr std::size_t extent = N;         ///< The dimension.
        /// \brief The dimension. \return `N`.
        [[nodiscard]] static constexpr std::size_t dimension(const linalg::fixed_matrix<T, N, 1>&) noexcept { return N; }
        /// \brief One coordinate. \param p The point. \param i The axis. \return `p[i]`.
        [[nodiscard]] static constexpr T at(const linalg::fixed_matrix<T, N, 1>& p, std::size_t i) noexcept { return p[i]; }
    };

    /// \brief A one-row `linalg::fixed_matrix` as a point.
    /// \ingroup core
    /// \tparam T The coordinate type.
    /// \tparam N The dimension, above 1 (a 1 x 1 matrix is the column case).
    template <linalg::scalar T, std::size_t N>
        requires(N > 1)
    struct point_traits<linalg::fixed_matrix<T, 1, N>> {
        using value_type = T;                            ///< The coordinate type.
        static constexpr std::size_t extent = N;         ///< The dimension.
        /// \brief The dimension. \return `N`.
        [[nodiscard]] static constexpr std::size_t dimension(const linalg::fixed_matrix<T, 1, N>&) noexcept { return N; }
        /// \brief One coordinate. \param p The point. \param i The axis. \return `p[i]`.
        [[nodiscard]] static constexpr T at(const linalg::fixed_matrix<T, 1, N>& p, std::size_t i) noexcept { return p[i]; }
    };

    /// \brief A type that `point_traits` can read.
    /// \ingroup core
    /// \tparam P The type (cv and references ignored).
    template <typename P>
    concept point = requires(const std::remove_cvref_t<P>& p, std::size_t i) {
        typename point_traits<std::remove_cvref_t<P>>::value_type;
        { point_traits<std::remove_cvref_t<P>>::extent } -> std::convertible_to<std::size_t>;
        { point_traits<std::remove_cvref_t<P>>::dimension(p) } -> std::convertible_to<std::size_t>;
        point_traits<std::remove_cvref_t<P>>::at(p, i);
    };

    /// \brief A point's coordinate type. \tparam P A `point`.
    /// \ingroup core
    template <point P>
    using value_t = typename point_traits<std::remove_cvref_t<P>>::value_type;

    /// \brief A point's compile-time dimension, or `std::dynamic_extent`. \tparam P A `point`.
    /// \ingroup core
    template <point P>
    inline constexpr std::size_t extent_v = point_traits<std::remove_cvref_t<P>>::extent;

    /// \brief The type distances between `P` and `Q` are computed in: their common floating type, or `double` when
    /// either has integer coordinates.
    /// \ingroup core
    /// \tparam P A `point`.
    /// \tparam Q A `point`.
    template <point P, point Q = P>
    using distance_t = std::conditional_t<std::is_floating_point_v<value_t<P>> && std::is_floating_point_v<value_t<Q>>,
                                          std::common_type_t<value_t<P>, value_t<Q>>, double>;

    /// \brief A point's dimension. \tparam P A `point`. \param p The point. \return Its dimension.
    /// \ingroup core
    template <point P>
    [[nodiscard]] constexpr std::size_t dimension(const P& p) noexcept(noexcept(point_traits<std::remove_cvref_t<P>>::dimension(p)))
    {
        return point_traits<std::remove_cvref_t<P>>::dimension(p);
    }

    /// \brief One coordinate of a point. \tparam P A `point`. \param p The point. \param i The axis.
    /// \ingroup core
    /// \return The coordinate.
    template <point P>
    [[nodiscard]] constexpr value_t<P> coordinate(const P& p, std::size_t i) noexcept(noexcept(point_traits<std::remove_cvref_t<P>>::at(p, i)))
    {
        return point_traits<std::remove_cvref_t<P>>::at(p, i);
    }

    /// \brief Whether two point types can be compared coordinate by coordinate: their extents agree where both are
    /// fixed.
    /// \ingroup core
    /// \tparam P A type.
    /// \tparam Q A type.
    template <typename P, typename Q>
    concept compatible_points = point<P> && point<Q> && (extent_v<P> == std::dynamic_extent || extent_v<Q> == std::dynamic_extent || extent_v<P> == extent_v<Q>);

    /// \brief A distance between an `A` and a `B`: `m(a, b)` returns an arithmetic value.
    /// \ingroup core
    /// \tparam M The metric.
    /// \tparam A The first argument's type.
    /// \tparam B The second's.
    template <typename M, typename A, typename B = A>
    concept distance_for = std::invocable<const M&, const A&, const B&> && std::is_arithmetic_v<std::remove_cvref_t<std::invoke_result_t<const M&, const A&, const B&>>>;

    /// \brief A metric with a cheaper monotone form: `m.reduced(a, b)` (such as the squared Euclidean distance),
    /// `m.expand(r)` back to the distance and `m.contract(d)` from it.
    /// \ingroup core
    /// \tparam M The metric.
    /// \tparam A The first argument's type.
    /// \tparam B The second's.
    template <typename M, typename A, typename B = A>
    concept reducible = distance_for<M, A, B> && requires(const M& m, const A& a, const B& b, std::invoke_result_t<const M&, const A&, const B&> d) {
        { m.reduced(a, b) } -> std::convertible_to<decltype(d)>;
        { m.expand(d) } -> std::convertible_to<decltype(d)>;
        { m.contract(d) } -> std::convertible_to<decltype(d)>;
    };

    /// \brief A metric whose reduced form is built coordinate by coordinate: `m.term(axis, aᵢ - bᵢ)` combined by
    /// `m.combine(acc, t)` from zero, monotone in each term, so that one coordinate's term never exceeds the whole
    /// (what a k-d tree prunes with). Every decomposable metric is `reducible`.
    /// \ingroup core
    /// \tparam M The metric.
    /// \tparam T The number type of the differences.
    template <typename M, typename T>
    concept decomposable = requires(const M& m, std::size_t axis, T x) {
        { m.term(axis, x) } -> std::convertible_to<T>;
        { m.combine(x, x) } -> std::convertible_to<T>;
        { m.expand(x) } -> std::convertible_to<T>;
        { m.contract(x) } -> std::convertible_to<T>;
    };

    /// \brief A metric in the mathematical sense, declared by `static constexpr bool is_metric = true`: zero only
    /// for equal points, symmetric, and obeying the triangle inequality.
    /// \ingroup core
    /// \tparam M The metric.
    template <typename M>
    concept true_metric = requires {
        { M::is_metric } -> std::convertible_to<bool>;
    } && M::is_metric;

    /// \brief The reduced distance: `m.reduced(a, b)` for a `reducible` metric. \tparam M The metric.
    /// \ingroup core
    /// \tparam A The first argument's type. \tparam B The second's. \param m The metric. \param a One point.
    /// \param b The other. \return The reduced distance.
    template <typename M, typename A, typename B>
        requires reducible<M, A, B>
    [[nodiscard]] constexpr auto reduced_distance(const M& m, const A& a, const B& b) noexcept(noexcept(m.reduced(a, b)))
    {
        return m.reduced(a, b);
    }

    /// \brief The reduced distance of a metric with no cheaper form: the distance itself. \ingroup core
    /// \tparam M The metric. \tparam A The first argument's type. \tparam B The second's. \param m The metric.
    /// \param a One point. \param b The other. \return `m(a, b)`.
    template <typename M, typename A, typename B>
        requires(distance_for<M, A, B> && !reducible<M, A, B>)
    [[nodiscard]] constexpr auto reduced_distance(const M& m, const A& a, const B& b) noexcept(noexcept(m(a, b)))
    {
        return m(a, b);
    }

    /// \brief A reduced distance back to a distance. \tparam M The metric. \tparam T The number type.
    /// \ingroup core
    /// \param m The metric. \param r The reduced distance. \return `m.expand(r)`.
    template <typename M, typename T>
        requires requires(const M& m, T r) { m.expand(r); }
    [[nodiscard]] constexpr T expand(const M& m, T r) noexcept(noexcept(m.expand(r)))
    {
        return static_cast<T>(m.expand(r));
    }

    /// \brief A reduced distance back to a distance, for a metric with no cheaper form: the value itself.
    /// \tparam M The metric. \tparam T The number type. \param m The metric. \param r The value. \return `r`.
    /// \ingroup core
    template <typename M, typename T>
        requires(!requires(const M& m, T r) { m.expand(r); })
    [[nodiscard]] constexpr T expand(const M& m, T r) noexcept
    {
        static_cast<void>(m);
        return r;
    }

    /// \brief A distance in reduced form (a radius as a squared radius, say). \tparam M The metric.
    /// \ingroup core
    /// \tparam T The number type. \param m The metric. \param d The distance. \return `m.contract(d)`.
    template <typename M, typename T>
        requires requires(const M& m, T d) { m.contract(d); }
    [[nodiscard]] constexpr T contract(const M& m, T d) noexcept(noexcept(m.contract(d)))
    {
        return static_cast<T>(m.contract(d));
    }

    /// \brief A distance in reduced form, for a metric with no cheaper form: the value itself. \ingroup core
    /// \tparam M The metric. \tparam T The number type. \param m The metric. \param d The distance. \return `d`.
    template <typename M, typename T>
        requires(!requires(const M& m, T d) { m.contract(d); })
    [[nodiscard]] constexpr T contract(const M& m, T d) noexcept
    {
        static_cast<void>(m);
        return d;
    }

    /// \brief A decomposable metric's reduced distance, term by term.
    /// \ingroup core
    /// \tparam M A `decomposable` metric.
    /// \tparam A A `point`.
    /// \tparam B A `point` compatible with `A`.
    /// \param m The metric.
    /// \param a One point.
    /// \param b The other, of the same dimension.
    /// \return `combine` over the axes of `term(i, aᵢ - bᵢ)`.
    template <typename M, point A, point B>
        requires compatible_points<A, B> && decomposable<M, distance_t<A, B>>
    [[nodiscard]] constexpr distance_t<A, B> accumulate(const M& m, const A& a, const B& b) noexcept(noexcept(m.combine(distance_t<A, B>{}, m.term(std::size_t{}, distance_t<A, B>{}))) && noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
    {
        using T = distance_t<A, B>;
        T acc(0);
        const std::size_t n = dimension(a);
        for (std::size_t i = 0; i < n; ++i)
            acc = m.combine(acc, m.term(i, static_cast<T>(coordinate(a, i)) - static_cast<T>(coordinate(b, i))));
        return acc;
    }

    /// \brief A decomposable metric's reduced distance, stopping as soon as it exceeds `bound` (for scans that only
    /// need the points within a bound, or better than the best so far).
    /// \ingroup core
    /// \tparam M A `decomposable` metric.
    /// \tparam A A `point`.
    /// \tparam B A `point` compatible with `A`.
    /// \param m The metric.
    /// \param a One point.
    /// \param b The other, of the same dimension.
    /// \param bound The reduced bound.
    /// \return The reduced distance if it is at most `bound`; otherwise some partial value above `bound`.
    template <typename M, point A, point B>
        requires compatible_points<A, B> && decomposable<M, distance_t<A, B>>
    [[nodiscard]] constexpr distance_t<A, B> accumulate_within(const M& m, const A& a, const B& b, distance_t<A, B> bound) noexcept(noexcept(m.combine(distance_t<A, B>{}, m.term(std::size_t{}, distance_t<A, B>{}))) && noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
    {
        using T = distance_t<A, B>;
        T acc(0);
        const std::size_t n = dimension(a);
        for (std::size_t i = 0; i < n && !(acc > bound); ++i)
            acc = m.combine(acc, m.term(i, static_cast<T>(coordinate(a, i)) - static_cast<T>(coordinate(b, i))));
        return acc;
    }

} // namespace snicholls::metric

#endif /* snicholls_core_metric_concepts_hpp */
