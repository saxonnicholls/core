//
//  minkowski.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::metric::euclidean`, `manhattan`, `chebyshev`, `minkowski<P>`, `squared_euclidean` and `weighted_minkowski<P, W>`: the coordinate-wise distances, on any point type.
/// \ingroup core
///
/// **Overview.** The distances built from coordinate differences, each a small `constexpr` value with the full set
/// of capabilities of `metric/concepts.hpp`: `m(a, b)`, a reduced form for searching, a per-coordinate
/// decomposition (so a k-d tree can prune with them), and `is_metric`.
///
/// | Metric | `m(a, b)` | Reduced form | `is_metric` |
/// |---|---|---|---|
/// | `minkowski<P>` (`P ≥ 1`) | `(Σ |aᵢ - bᵢ|ᴾ)^(1/P)` | `Σ |aᵢ - bᵢ|ᴾ` | yes |
/// | `euclidean` = `minkowski<2>` | `√Σ (aᵢ - bᵢ)²` | `Σ (aᵢ - bᵢ)²` | yes |
/// | `manhattan` = `minkowski<1>` | `Σ |aᵢ - bᵢ|` | the same | yes |
/// | `chebyshev` | `maxᵢ |aᵢ - bᵢ|` | the same | yes |
/// | `squared_euclidean` | `Σ (aᵢ - bᵢ)²` | the same | no (no triangle inequality) |
/// | `weighted_minkowski<P, W>{w}` | `(Σ wᵢ |aᵢ - bᵢ|ᴾ)^(1/P)` | `Σ wᵢ |aᵢ - bᵢ|ᴾ` | yes, for `wᵢ > 0` |
///
/// The exponent is a template argument, so `|d|ᴾ` is `P` multiplications, and the root is `sqrt` for `P = 2` and
/// nothing for `P = 1`. Points are anything `point_traits` reads, mixed freely (a `std::array` against a `span`),
/// in their common floating type (`double` for integer coordinates). Both points must have the same dimension.
///
/// \snippet metric/minkowski_demo.cpp minkowski
///
/// **Relationships.** `metric/concepts.hpp`, `math/constexpr_math.hpp`. Used by `containers/kd_tree.hpp`,
/// `containers/ann_forest.hpp` and `metric/sets.hpp`; `tracking::euclidean` and `euclidean_weighted` are the same
/// distances on `fixed_vector`s. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` when the points' coordinate reads are, no allocation. Every metric is an
/// empty value (or, weighted, holds its weights), trivially copyable.
///
/// **Errors.** None. Points of different dimensions are a precondition violation (the shorter's extent is not
/// checked at run time).
///
/// **Performance.** Not benchmarked on their own: `containers/kd_tree.hpp`'s and `containers/ann_forest.hpp`'s
/// benchmarks time them inside searches.
///
/// **Tests and demos.** `tests/metric/minkowski_test.cpp`, `demos/metric/minkowski_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<type_traits>`, `math/constexpr_math.hpp`, `metric/concepts.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.20.0.

#ifndef snicholls_core_metric_minkowski_hpp
#define snicholls_core_metric_minkowski_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/metric/concepts.hpp>

#include <cstddef>
#include <type_traits>

namespace snicholls::metric {

    namespace detail {
        template <std::size_t P, typename T>
        constexpr T abs_power(T d) noexcept
        {
            const T a = d < T(0) ? -d : d;
            T r = a;
            for (std::size_t i = 1; i < P; ++i)
                r *= a;
            return r;
        }

        template <std::size_t P, typename T>
        constexpr T root(T r) noexcept
        {
            if constexpr (P == 1)
                return r;
            else if constexpr (P == 2)
                return static_cast<T>(math::sqrt(r));
            else
                return r > T(0) ? static_cast<T>(math::pow(r, T(1) / T(P))) : T(0);
        }
    } // namespace detail

    /// \brief The Minkowski distance of order `P`: `(Σ |aᵢ - bᵢ|ᴾ)^(1/P)`.
    /// \ingroup core
    /// \tparam P The order, at least 1.
    template <std::size_t P>
    struct minkowski {
        static_assert(P >= 1, "snicholls::metric::minkowski: the order is at least 1");
        static constexpr bool is_metric = true;  ///< It obeys the triangle inequality.

        /// \brief One coordinate's term. \tparam T The number type. \param axis The axis (unused).
        /// \param d The coordinate difference. \return `|d|ᴾ`.
        template <typename T>
        [[nodiscard]] constexpr T term(std::size_t axis, T d) const noexcept
        {
            static_cast<void>(axis);
            return detail::abs_power<P>(d);
        }
        /// \brief Terms combine by addition. \tparam T The number type. \param acc The sum so far. \param t A term.
        /// \return `acc + t`.
        template <typename T>
        [[nodiscard]] constexpr T combine(T acc, T t) const noexcept
        {
            return acc + t;
        }
        /// \brief The distance of a reduced value. \tparam T The number type. \param r `Σ |d|ᴾ`. \return `r^(1/P)`.
        template <typename T>
        [[nodiscard]] constexpr T expand(T r) const noexcept
        {
            return detail::root<P>(r);
        }
        /// \brief The reduced value of a distance. \tparam T The number type. \param d The distance. \return `dᴾ`.
        template <typename T>
        [[nodiscard]] constexpr T contract(T d) const noexcept
        {
            return detail::abs_power<P>(d);
        }
        /// \brief The reduced distance. \tparam A A `point`. \tparam B A compatible `point`. \param a One point.
        /// \param b The other. \return `Σ |aᵢ - bᵢ|ᴾ`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> reduced(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return accumulate(*this, a, b);
        }
        /// \brief The distance. \tparam A A `point`. \tparam B A compatible `point`. \param a One point.
        /// \param b The other. \return `(Σ |aᵢ - bᵢ|ᴾ)^(1/P)`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return expand(reduced(a, b));
        }
    };

    /// \brief The Euclidean distance, `√Σ (aᵢ - bᵢ)²`; its reduced form is the squared distance. \ingroup core
    using euclidean = minkowski<2>;
    /// \brief The Manhattan (taxicab, L1) distance, `Σ |aᵢ - bᵢ|`. \ingroup core
    using manhattan = minkowski<1>;

    /// \brief The Chebyshev (L∞) distance, `maxᵢ |aᵢ - bᵢ|`.
    /// \ingroup core
    struct chebyshev {
        static constexpr bool is_metric = true;  ///< It obeys the triangle inequality.

        /// \brief One coordinate's term. \tparam T The number type. \param axis The axis (unused).
        /// \param d The coordinate difference. \return `|d|`.
        template <typename T>
        [[nodiscard]] constexpr T term(std::size_t axis, T d) const noexcept
        {
            static_cast<void>(axis);
            return d < T(0) ? -d : d;
        }
        /// \brief Terms combine by the maximum. \tparam T The number type. \param acc The maximum so far.
        /// \param t A term. \return `max(acc, t)`.
        template <typename T>
        [[nodiscard]] constexpr T combine(T acc, T t) const noexcept
        {
            return t > acc ? t : acc;
        }
        /// \brief The identity: the reduced form is the distance. \tparam T The number type. \param r The value. \return `r`.
        template <typename T>
        [[nodiscard]] constexpr T expand(T r) const noexcept
        {
            return r;
        }
        /// \brief The identity. \tparam T The number type. \param d The distance. \return `d`.
        template <typename T>
        [[nodiscard]] constexpr T contract(T d) const noexcept
        {
            return d;
        }
        /// \brief The reduced distance, the distance itself. \tparam A A `point`. \tparam B A compatible `point`.
        /// \param a One point. \param b The other. \return `maxᵢ |aᵢ - bᵢ|`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> reduced(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return accumulate(*this, a, b);
        }
        /// \brief The distance. \tparam A A `point`. \tparam B A compatible `point`. \param a One point.
        /// \param b The other. \return `maxᵢ |aᵢ - bᵢ|`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return accumulate(*this, a, b);
        }
    };

    /// \brief The squared Euclidean distance, `Σ (aᵢ - bᵢ)²`: cheaper than `euclidean` and orders points the same
    /// way, but not a metric (no triangle inequality), so not for trees that prune by it.
    /// \ingroup core
    struct squared_euclidean {
        static constexpr bool is_metric = false;  ///< The triangle inequality fails.

        /// \brief One coordinate's term. \tparam T The number type. \param axis The axis (unused).
        /// \param d The coordinate difference. \return `d²`.
        template <typename T>
        [[nodiscard]] constexpr T term(std::size_t axis, T d) const noexcept
        {
            static_cast<void>(axis);
            return d * d;
        }
        /// \brief Terms combine by addition. \tparam T The number type. \param acc The sum so far. \param t A term.
        /// \return `acc + t`.
        template <typename T>
        [[nodiscard]] constexpr T combine(T acc, T t) const noexcept
        {
            return acc + t;
        }
        /// \brief The identity. \tparam T The number type. \param r The value. \return `r`.
        template <typename T>
        [[nodiscard]] constexpr T expand(T r) const noexcept
        {
            return r;
        }
        /// \brief The identity. \tparam T The number type. \param d The value. \return `d`.
        template <typename T>
        [[nodiscard]] constexpr T contract(T d) const noexcept
        {
            return d;
        }
        /// \brief The reduced distance, the value itself. \tparam A A `point`. \tparam B A compatible `point`.
        /// \param a One point. \param b The other. \return `Σ (aᵢ - bᵢ)²`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> reduced(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return accumulate(*this, a, b);
        }
        /// \brief The value. \tparam A A `point`. \tparam B A compatible `point`. \param a One point. \param b The other.
        /// \return `Σ (aᵢ - bᵢ)²`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return accumulate(*this, a, b);
        }
    };

    /// \brief The weighted Minkowski distance `(Σ wᵢ |aᵢ - bᵢ|ᴾ)^(1/P)`, with one weight per axis (a point of weights).
    /// \ingroup core
    /// \tparam P The order, at least 1.
    /// \tparam W The weights' type, a `point` (`fixed_vector`, `std::array`, a `span` over storage that outlives the metric).
    template <std::size_t P, point W>
    struct weighted_minkowski {
        static_assert(P >= 1, "snicholls::metric::weighted_minkowski: the order is at least 1");
        static constexpr bool is_metric = true;  ///< It obeys the triangle inequality, for positive weights.
        W weights{};                             ///< One positive weight per axis.

        /// \brief One coordinate's term. \tparam T The number type. \param axis The axis. \param d The difference.
        /// \return `w_axis |d|ᴾ`.
        template <typename T>
        [[nodiscard]] constexpr T term(std::size_t axis, T d) const noexcept(noexcept(coordinate(weights, axis)))
        {
            return static_cast<T>(coordinate(weights, axis)) * detail::abs_power<P>(d);
        }
        /// \brief Terms combine by addition. \tparam T The number type. \param acc The sum so far. \param t A term.
        /// \return `acc + t`.
        template <typename T>
        [[nodiscard]] constexpr T combine(T acc, T t) const noexcept
        {
            return acc + t;
        }
        /// \brief The distance of a reduced value. \tparam T The number type. \param r `Σ wᵢ |dᵢ|ᴾ`. \return `r^(1/P)`.
        template <typename T>
        [[nodiscard]] constexpr T expand(T r) const noexcept
        {
            return detail::root<P>(r);
        }
        /// \brief The reduced value of a distance. \tparam T The number type. \param d The distance. \return `dᴾ`.
        template <typename T>
        [[nodiscard]] constexpr T contract(T d) const noexcept
        {
            return detail::abs_power<P>(d);
        }
        /// \brief The reduced distance. \tparam A A `point`. \tparam B A compatible `point`. \param a One point.
        /// \param b The other. \return `Σ wᵢ |aᵢ - bᵢ|ᴾ`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> reduced(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)) && noexcept(coordinate(weights, 0)))
        {
            return accumulate(*this, a, b);
        }
        /// \brief The distance. \tparam A A `point`. \tparam B A compatible `point`. \param a One point.
        /// \param b The other. \return `(Σ wᵢ |aᵢ - bᵢ|ᴾ)^(1/P)`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)) && noexcept(coordinate(weights, 0)))
        {
            return expand(reduced(a, b));
        }
    };

    /// \brief A weighted Minkowski metric from its weights. \tparam P The order. \tparam W The weights' type.
    /// \ingroup core
    /// \param w The weights. \return The metric.
    template <std::size_t P, point W>
    [[nodiscard]] constexpr weighted_minkowski<P, W> weighted(const W& w) noexcept(std::is_nothrow_copy_constructible_v<W>)
    {
        return weighted_minkowski<P, W>{w};
    }

} // namespace snicholls::metric

#endif /* snicholls_core_metric_minkowski_hpp */
