//
//  sets.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::metric::set_metric_solver` with `ospa` and `gospa`, and `hausdorff`: distances between two sets of points under any metric.
/// \ingroup core
///
/// **Overview.** Comparing a set of estimates with a set of truths (a multi-target tracker's output against ground
/// truth, a detector's boxes against labels, two point clouds) needs a distance between sets of possibly different
/// sizes. Each of these takes the sets as any random-access ranges of any `point` type, and the base metric as a
/// value (`euclidean` by default; `mahalanobis`, `haversine`, `angular` or the caller's own plug in unchanged):
///
/// - **OSPA** (Schuhmacher, Vo and Vo 2008), cut-off `c`, order `p`: for `m ≤ n` points,
///   `((1/n) (min_π Σ min(d, c)^p + c^p (n - m)))^(1/p)`, in `[0, c]`, normalised per target.
/// - **GOSPA** (Rahmathullah, García-Fernández and Svensson 2017):
///   `(min_π Σ min(d, c)^p + (c^p / α) |n - m|)^(1/p)`. With `α = 2` (the default) it splits into localisation
///   error, missed targets and false tracks, which `gospa` reports separately.
/// - **Hausdorff**: `max(sup_x inf_y d(x, y), sup_y inf_x d(x, y))`, the worst nearest-neighbour distance either way.
///   No assignment; O(m n) distances.
///
/// OSPA and GOSPA solve an optimal assignment (`combinatorics/assignment.hpp`) over the points.
/// `set_metric_solver<T, MaxTruths, MaxEstimates>` holds the cost matrix and the solver's arrays, sized at compile
/// time; construct it once (on the heap if large) and reuse it.
///
/// \snippet metric/sets_demo.cpp metrics
///
/// **Relationships.** `metric/concepts.hpp` and the metrics, `combinatorics/assignment.hpp`. Stone Soup's
/// `OSPAMetric` and `GOSPAMetric`, whose generators over a whole scenario live in the framework above core. Group:
/// \ref core.
///
/// **Guarantees.** `constexpr`, no allocation; `ospa` and `gospa` are `noexcept` when the metric and `T`'s
/// arithmetic are. With a `true_metric` base, OSPA, GOSPA
/// and Hausdorff are metrics on sets. Complexity: one assignment, O(m² n), for OSPA and GOSPA.
///
/// **Errors.** `set_metric_error::too_large`: more points than the solver holds. `set_metric_error::invalid_parameters`:
/// `c ≤ 0`, `p < 1`, or `α` outside `(0, 2]`. `hausdorff` of an empty set and a non-empty one is infinity; of two empty
/// sets, zero.
///
/// **Performance.** Not benchmarked: the assignment dominates (see its benchmark).
///
/// **Tests and demos.** `tests/metric/sets_test.cpp`, `demos/metric/sets_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<iterator>`, `<limits>`, `<ranges>`, `<span>`, `<utility>`,
/// `combinatorics/assignment.hpp`, `math/constexpr_math.hpp`, `metric/concepts.hpp`, `metric/minkowski.hpp`,
/// `result.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.19.0 as `tracking/metrics.hpp` (Euclidean only); generic over the metric and
/// moved to `metric/` in 0.20.0.

#ifndef snicholls_core_metric_sets_hpp
#define snicholls_core_metric_sets_hpp

#include <snicholls/core/combinatorics/assignment.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <utility>

namespace snicholls::metric {

    /// \brief Why a set metric has no answer.
    /// \ingroup core
    enum class set_metric_error : std::uint8_t {
        too_large = 1,           ///< More points than the solver holds.
        invalid_parameters = 2,  ///< `c ≤ 0`, `p < 1`, or `α` outside `(0, 2]`.
    };

    /// \brief GOSPA and its parts.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct gospa_result {
        T distance{};                  ///< GOSPA.
        T localisation{};              ///< The p-th-power sum of the assigned pairs' distances (each under `c`).
        std::size_t missed = 0;        ///< Truths with no estimate within `c`.
        std::size_t false_tracks = 0;  ///< Estimates with no truth within `c`.
    };

    /// \brief A random-access range of points, the argument of the set metrics.
    /// \ingroup core
    /// \tparam R The range type.
    template <typename R>
    concept point_range = std::ranges::random_access_range<const R> && std::ranges::sized_range<const R> && point<std::ranges::range_value_t<const R>>;

    /// \brief OSPA and GOSPA between sets of up to `MaxTruths` and `MaxEstimates` points.
    /// \ingroup core
    /// \tparam T The number type of the costs.
    /// \tparam MaxTruths The most truths.
    /// \tparam MaxEstimates The most estimates.
    template <linalg::scalar T, std::size_t MaxTruths, std::size_t MaxEstimates>
    class set_metric_solver {
        static_assert(MaxTruths > 0 && MaxEstimates > 0, "snicholls::metric::set_metric_solver: hold at least one point on each side");

        // min(d, c)^p for every pair, and the optimal assignment of the smaller set into the larger
        template <typename X, typename Y, typename M>
        constexpr result<T, set_metric_error> assign(const X& x, const Y& y, T c, T p, const M& m)
        {
            const std::size_t nx = std::ranges::size(x), ny = std::ranges::size(y);
            const T cp = static_cast<T>(math::pow(c, p));
            for (std::size_t i = 0; i < nx; ++i)
                for (std::size_t j = 0; j < ny; ++j) {
                    const T d = static_cast<T>(m(x[i], y[j]));
                    cost_[i * ny + j] = d < c ? static_cast<T>(math::pow(d, p)) : cp;
                }
            const auto s = solver_.solve(std::span<const T>(cost_).first(nx * ny), nx, ny, std::span<std::int32_t>(row_to_col_).first(nx));
            if (!s)
                return err(set_metric_error::too_large);
            return ok(*s);
        }

    public:
        /// \brief OSPA.
        /// \tparam X A `point_range` of truths.
        /// \tparam Y A `point_range` of estimates.
        /// \tparam M The base metric.
        /// \param truths The true points.
        /// \param estimates The estimated points.
        /// \param c The cut-off, above zero.
        /// \param p The order, at least 1.
        /// \param m The base metric.
        /// \return OSPA in `[0, c]`, or a `set_metric_error`.
        template <point_range X, point_range Y, typename M = euclidean>
            requires distance_for<M, std::ranges::range_value_t<const X>, std::ranges::range_value_t<const Y>>
        [[nodiscard]] constexpr result<T, set_metric_error> ospa(const X& truths, const Y& estimates, T c, T p = T(1), const M& m = M{}) noexcept(linalg::detail::nothrow_scalar<T> && noexcept(std::declval<const M&>()(std::declval<std::ranges::range_reference_t<const X>>(), std::declval<std::ranges::range_reference_t<const Y>>())))
        {
            if (!(c > T(0)) || !(p >= T(1)))
                return err(set_metric_error::invalid_parameters);
            const std::size_t nx = std::ranges::size(truths), ny = std::ranges::size(estimates);
            if (nx > MaxTruths || ny > MaxEstimates)
                return err(set_metric_error::too_large);
            if (nx == 0 && ny == 0)
                return ok(T(0));
            if (nx == 0 || ny == 0)
                return ok(c);
            const auto s = assign(truths, estimates, c, p, m);
            if (!s)
                return s;
            const std::size_t big = nx > ny ? nx : ny, small = nx > ny ? ny : nx;
            const T total = *s + static_cast<T>(math::pow(c, p)) * static_cast<T>(big - small);
            return ok(static_cast<T>(math::pow(total / static_cast<T>(big), T(1) / p)));
        }

        /// \brief GOSPA, with its localisation, missed and false parts (meaningful for `α = 2`).
        /// \tparam X A `point_range` of truths.
        /// \tparam Y A `point_range` of estimates.
        /// \tparam M The base metric.
        /// \param truths The true points.
        /// \param estimates The estimated points.
        /// \param c The cut-off, above zero.
        /// \param p The order, at least 1.
        /// \param alpha The cardinality-mismatch weight, in `(0, 2]`.
        /// \param m The base metric.
        /// \return The result, or a `set_metric_error`.
        template <point_range X, point_range Y, typename M = euclidean>
            requires distance_for<M, std::ranges::range_value_t<const X>, std::ranges::range_value_t<const Y>>
        [[nodiscard]] constexpr result<gospa_result<T>, set_metric_error> gospa(const X& truths, const Y& estimates, T c, T p = T(1), T alpha = T(2), const M& m = M{}) noexcept(linalg::detail::nothrow_scalar<T> && noexcept(std::declval<const M&>()(std::declval<std::ranges::range_reference_t<const X>>(), std::declval<std::ranges::range_reference_t<const Y>>())))
        {
            if (!(c > T(0)) || !(p >= T(1)) || !(alpha > T(0)) || alpha > T(2))
                return err(set_metric_error::invalid_parameters);
            const std::size_t nx = std::ranges::size(truths), ny = std::ranges::size(estimates);
            if (nx > MaxTruths || ny > MaxEstimates)
                return err(set_metric_error::too_large);
            gospa_result<T> g{};
            const T cp = static_cast<T>(math::pow(c, p));
            T assigned(0);
            std::size_t paired = 0;
            if (nx > 0 && ny > 0) {
                const auto s = assign(truths, estimates, c, p, m);
                if (!s)
                    return err(s.unwrap_err_unchecked());
                assigned = *s;
                for (std::size_t i = 0; i < nx; ++i) {
                    const auto j = row_to_col_[i];
                    if (j < 0)
                        continue;
                    const T d = static_cast<T>(m(truths[i], estimates[static_cast<std::size_t>(j)]));
                    if (d < c) {
                        g.localisation += static_cast<T>(math::pow(d, p));
                        ++paired;
                    }
                }
            }
            g.missed = nx - paired;
            g.false_tracks = ny - paired;
            const std::size_t mismatch = nx > ny ? nx - ny : ny - nx;
            g.distance = static_cast<T>(math::pow(assigned + cp / alpha * static_cast<T>(mismatch), T(1) / p));
            return ok(g);
        }

    private:
        std::array<T, MaxTruths * MaxEstimates> cost_{};
        std::array<std::int32_t, MaxTruths> row_to_col_{};
        combinatorics::assignment_solver<T, MaxTruths, MaxEstimates> solver_{};
    };

    /// \brief The Hausdorff distance between two sets: the worst nearest-neighbour distance, either way.
    /// \ingroup core
    /// \tparam X A `point_range`.
    /// \tparam Y A `point_range`.
    /// \tparam M The base metric.
    /// \param x One set.
    /// \param y The other.
    /// \param m The base metric.
    /// \return `max(sup_x inf_y d, sup_y inf_x d)` as a `double`; 0 for two empty sets, infinity for one.
    template <point_range X, point_range Y, typename M = euclidean>
        requires distance_for<M, std::ranges::range_value_t<const X>, std::ranges::range_value_t<const Y>>
    [[nodiscard]] constexpr double hausdorff(const X& x, const Y& y, const M& m = M{})
    {
        const std::size_t nx = std::ranges::size(x), ny = std::ranges::size(y);
        if (nx == 0 && ny == 0)
            return 0.0;
        if (nx == 0 || ny == 0)
            return std::numeric_limits<double>::infinity();
        double worst = 0;
        for (std::size_t i = 0; i < nx; ++i) {
            double best = std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < ny; ++j) {
                const double d = static_cast<double>(m(x[i], y[j]));
                best = d < best ? d : best;
            }
            worst = best > worst ? best : worst;
        }
        for (std::size_t j = 0; j < ny; ++j) {
            double best = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < nx; ++i) {
                const double d = static_cast<double>(m(x[i], y[j]));
                best = d < best ? d : best;
            }
            worst = best > worst ? best : worst;
        }
        return worst;
    }

} // namespace snicholls::metric

#endif /* snicholls_core_metric_sets_hpp */
