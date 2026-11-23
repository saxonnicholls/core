//
//  angular.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::metric::dot`, `norm`, `cosine_similarity`, and the metrics `cosine_distance`, `angular`, `angle` and `inner_product`: distances between directions, for embeddings and normalised features.
/// \ingroup core
///
/// **Overview.** Embeddings (text, image, appearance) are compared by direction, not position. These work on any
/// two points `point_traits` reads, in their common floating type:
///
/// - `dot(a, b)`, `squared_norm(a)`, `norm(a)`: the inner product and length of any point, the one generic form
///   (`linalg::dot` and `linalg::norm` are the same on `fixed_vector`s).
/// - `cosine_similarity(a, b)` = `a·b / (|a| |b|)`, in `[-1, 1]`; 0 when either is the zero vector.
///
/// | Metric | `m(a, b)` | `is_metric` | Note |
/// |---|---|---|---|
/// | `cosine_distance` | `1 - cos θ`, in `[0, 2]` | no | the usual embedding distance |
/// | `angular` | `√(2 - 2 cos θ)`, in `[0, 2]` | yes | Annoy's "angular": the Euclidean distance between the unit vectors |
/// | `angle` | `θ = acos(cos θ)`, in `[0, π]` | yes | the great-circle distance on the unit sphere |
/// | `inner_product` | `-a·b` | no | maximum inner product search as a nearest-neighbour search (Annoy's "dot") |
///
/// All four order points the same way when the vectors are normalised; `inner_product` also counts length.
///
/// \snippet metric/angular_demo.cpp angular
///
/// **Relationships.** `metric/concepts.hpp`, `metric/minkowski.hpp` (`euclidean` on normalised vectors equals
/// `angular`), `containers/ann_forest.hpp` (its angular and dot splits), `math/constexpr_math.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` when the points' coordinate reads are, no allocation. The cosine is
/// clamped to `[-1, 1]` before `acos` and `√(2 - 2 cos)`, so rounding never yields NaN.
///
/// **Errors.** None. A zero vector has cosine similarity 0 with everything (so `cosine_distance` 1, `angular` √2,
/// `angle` π/2).
///
/// **Performance.** Not benchmarked on their own: one pass over the coordinates for the dot product and both norms.
///
/// **Tests and demos.** `tests/metric/angular_test.cpp`, `demos/metric/angular_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `math/constexpr_math.hpp`, `metric/concepts.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.20.0.

#ifndef snicholls_core_metric_angular_hpp
#define snicholls_core_metric_angular_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/metric/concepts.hpp>

#include <cstddef>

namespace snicholls::metric {

    /// \brief The inner product `Σ aᵢ bᵢ`. \tparam A A `point`. \tparam B A compatible `point`.
    /// \ingroup core
    /// \param a One point. \param b The other, of the same dimension. \return The inner product.
    template <point A, point B>
        requires compatible_points<A, B>
    [[nodiscard]] constexpr distance_t<A, B> dot(const A& a, const B& b) noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
    {
        using T = distance_t<A, B>;
        T s(0);
        const std::size_t n = dimension(a);
        for (std::size_t i = 0; i < n; ++i)
            s += static_cast<T>(coordinate(a, i)) * static_cast<T>(coordinate(b, i));
        return s;
    }

    /// \brief The squared length `Σ aᵢ²`. \tparam A A `point`. \param a The point. \return It.
    /// \ingroup core
    template <point A>
    [[nodiscard]] constexpr distance_t<A> squared_norm(const A& a) noexcept(noexcept(coordinate(a, 0)))
    {
        return dot(a, a);
    }

    /// \brief The length `√Σ aᵢ²`. \tparam A A `point`. \param a The point. \return It.
    /// \ingroup core
    template <point A>
    [[nodiscard]] constexpr distance_t<A> norm(const A& a) noexcept(noexcept(coordinate(a, 0)))
    {
        return static_cast<distance_t<A>>(math::sqrt(squared_norm(a)));
    }

    /// \brief The cosine of the angle between two vectors, `a·b / (|a| |b|)`, clamped to `[-1, 1]`.
    /// \tparam A A `point`. \tparam B A compatible `point`. \param a One vector. \param b The other.
    /// \ingroup core
    /// \return The cosine; 0 when either vector is zero.
    template <point A, point B>
        requires compatible_points<A, B>
    [[nodiscard]] constexpr distance_t<A, B> cosine_similarity(const A& a, const B& b) noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
    {
        using T = distance_t<A, B>;
        T ab(0), aa(0), bb(0);
        const std::size_t n = dimension(a);
        for (std::size_t i = 0; i < n; ++i) {
            const T x = static_cast<T>(coordinate(a, i)), y = static_cast<T>(coordinate(b, i));
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        if (!(aa > T(0)) || !(bb > T(0)))
            return T(0);
        const T c = ab / static_cast<T>(math::sqrt(aa * bb));
        return c > T(1) ? T(1) : c < T(-1) ? T(-1) : c;
    }

    /// \brief `1 - cos θ`, the cosine distance: in `[0, 2]`, not a metric.
    /// \ingroup core
    struct cosine_distance {
        static constexpr bool is_metric = false;  ///< The triangle inequality fails.
        /// \brief The distance. \tparam A A `point`. \tparam B A compatible `point`. \param a One vector.
        /// \param b The other. \return `1 - cos θ`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return distance_t<A, B>(1) - cosine_similarity(a, b);
        }
    };

    /// \brief `√(2 - 2 cos θ)`, the Euclidean distance between the unit vectors (Annoy's "angular"): a metric.
    /// \ingroup core
    struct angular {
        static constexpr bool is_metric = true;  ///< It is a Euclidean distance on the unit sphere.
        /// \brief The reduced form, `2 - 2 cos θ`. \tparam A A `point`. \tparam B A compatible `point`.
        /// \param a One vector. \param b The other. \return `2 - 2 cos θ`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> reduced(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            using T = distance_t<A, B>;
            return T(2) - T(2) * cosine_similarity(a, b);
        }
        /// \brief The distance of a reduced value. \tparam T The number type. \param r `2 - 2 cos θ`. \return `√r`.
        template <typename T>
        [[nodiscard]] constexpr T expand(T r) const noexcept
        {
            return static_cast<T>(math::sqrt(r > T(0) ? r : T(0)));
        }
        /// \brief The reduced value of a distance. \tparam T The number type. \param d The distance. \return `d²`.
        template <typename T>
        [[nodiscard]] constexpr T contract(T d) const noexcept
        {
            return d * d;
        }
        /// \brief The distance. \tparam A A `point`. \tparam B A compatible `point`. \param a One vector.
        /// \param b The other. \return `√(2 - 2 cos θ)`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return expand(reduced(a, b));
        }
    };

    /// \brief The angle between two vectors, `acos(cos θ)` in `[0, π]`: the great-circle distance on the unit sphere.
    /// \ingroup core
    struct angle {
        static constexpr bool is_metric = true;  ///< The angle is a metric on directions.
        /// \brief The angle. \tparam A A `point`. \tparam B A compatible `point`. \param a One vector.
        /// \param b The other. \return θ, radians.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return static_cast<distance_t<A, B>>(math::acos(cosine_similarity(a, b)));
        }
    };

    /// \brief `-a·b`, so that the nearest neighbour is the maximum inner product (Annoy's "dot"): not a metric.
    /// \ingroup core
    struct inner_product {
        static constexpr bool is_metric = false;  ///< Not a distance at all: it can be negative.
        /// \brief The negated inner product. \tparam A A `point`. \tparam B A compatible `point`. \param a One vector.
        /// \param b The other. \return `-a·b`.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr distance_t<A, B> operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            return -dot(a, b);
        }
    };

} // namespace snicholls::metric

#endif /* snicholls_core_metric_angular_hpp */
