//
//  metrics.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::set_metric_solver` with `ospa` and `gospa`: distances between a set of truths and a set of estimates that charge for localisation error, missed targets and false tracks.
/// \ingroup core
///
/// **Overview.** A multi-target tracker is scored by comparing, at each time, the set of true positions with the
/// set of estimated ones, sets of possibly different sizes. With a cut-off `c` (an error that counts as a miss) and
/// an order `p`:
///
/// - **OSPA** (Schuhmacher, Vo and Vo 2008): for `m ≤ n` points,
///   `((1/n) (min_π Σ min(d, c)^p + c^p (n - m)))^(1/p)`, between 0 and `c`, normalised per target.
/// - **GOSPA** (Rahmathullah, García-Fernández and Svensson 2017):
///   `(min_π Σ min(d, c)^p + (c^p / α) |n - m|)^(1/p)`. With `α = 2` (the default) it splits into localisation
///   error, missed targets and false tracks, which `gospa` reports separately.
///
/// Both solve an optimal assignment (`combinatorics/assignment.hpp`) over the points. `set_metric_solver<T, D,
/// MaxTruths, MaxEstimates>` holds the cost matrix and the solver's arrays, sized at compile time; construct it once
/// (on the heap if large).
///
/// \snippet tracking/metrics_demo.cpp metrics
///
/// **Relationships.** `combinatorics/assignment.hpp`, `linalg/fixed.hpp`. Stone Soup's `OSPAMetric` and
/// `GOSPAMetric`; the generators that run them over a whole scenario live in the framework above core. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. Both are metrics
/// (zero only for equal sets, symmetric, triangle inequality). Complexity is one assignment, O(m² n).
///
/// **Errors.** `metric_error::too_large`: more points than the solver holds. `metric_error::invalid_parameters`:
/// `c ≤ 0`, `p < 1`, or `α` outside `(0, 2]`.
///
/// **Performance.** Not benchmarked: the assignment dominates (see its benchmark).
///
/// **Tests and demos.** `tests/tracking/metrics_test.cpp`, `demos/tracking/metrics_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `combinatorics/assignment.hpp`,
/// `linalg/fixed.hpp`, `math/constexpr_math.hpp`, `result.hpp`.
///
/// **Known issues.** Distances are Euclidean on the given vectors; map states to positions first.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_metrics_hpp
#define snicholls_core_tracking_metrics_hpp

#include <snicholls/core/combinatorics/assignment.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls::tracking {

    /// \brief Why a set metric has no answer.
    /// \ingroup core
    enum class metric_error : std::uint8_t {
        too_large = 1,           ///< More points than the solver holds.
        invalid_parameters = 2,  ///< `c ≤ 0`, `p < 1`, or `α` outside `(0, 2]`.
    };

    /// \brief GOSPA and its parts.
    /// \ingroup core
    /// \tparam T The number type.
    template <linalg::scalar T>
    struct gospa_result {
        T distance{};             ///< GOSPA.
        T localisation{};         ///< The p-th-power sum of the assigned pairs' distances (each under `c`).
        std::size_t missed = 0;   ///< Truths with no estimate within `c`.
        std::size_t false_tracks = 0;  ///< Estimates with no truth within `c`.
    };

    /// \brief OSPA and GOSPA between point sets of up to `MaxTruths` and `MaxEstimates` points of dimension `D`.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam D The points' dimension.
    /// \tparam MaxTruths The most truths.
    /// \tparam MaxEstimates The most estimates.
    template <linalg::scalar T, std::size_t D, std::size_t MaxTruths, std::size_t MaxEstimates>
    class set_metric_solver {
        static constexpr std::size_t big = MaxTruths > MaxEstimates ? MaxTruths : MaxEstimates;

        // min(d, c)^p for every pair, and the optimal assignment of the smaller set into the larger
        constexpr result<T, metric_error> assign(std::span<const linalg::fixed_vector<T, D>> x, std::span<const linalg::fixed_vector<T, D>> y, T c, T p) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            if (x.size() > MaxTruths || y.size() > MaxEstimates)
                return err(metric_error::too_large);
            const T cp = static_cast<T>(math::pow(c, p));
            for (std::size_t i = 0; i < x.size(); ++i)
                for (std::size_t j = 0; j < y.size(); ++j) {
                    const T d = linalg::norm(x[i] - y[j]);
                    cost_[i * y.size() + j] = d < c ? static_cast<T>(math::pow(d, p)) : cp;
                }
            const auto s = solver_.solve(std::span<const T>(cost_).first(x.size() * y.size()), x.size(), y.size(), std::span<std::int32_t>(row_to_col_).first(x.size()));
            if (!s)
                return err(metric_error::too_large);
            return ok(*s);
        }

    public:
        /// \brief OSPA.
        /// \param truths The true positions.
        /// \param estimates The estimated positions.
        /// \param c The cut-off, above zero.
        /// \param p The order, at least 1.
        /// \return OSPA in `[0, c]`, or a `metric_error`.
        [[nodiscard]] constexpr result<T, metric_error> ospa(std::span<const linalg::fixed_vector<T, D>> truths, std::span<const linalg::fixed_vector<T, D>> estimates, T c, T p = T(1)) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            if (!(c > T(0)) || !(p >= T(1)))
                return err(metric_error::invalid_parameters);
            const std::size_t m = truths.size(), n = estimates.size();
            if (m == 0 && n == 0)
                return ok(T(0));
            if (m == 0 || n == 0)
                return ok(c);
            const auto s = assign(truths, estimates, c, p);
            if (!s)
                return s;
            const std::size_t big_n = m > n ? m : n, small_n = m > n ? n : m;
            const T total = *s + static_cast<T>(math::pow(c, p)) * static_cast<T>(big_n - small_n);
            return ok(static_cast<T>(math::pow(total / static_cast<T>(big_n), T(1) / p)));
        }

        /// \brief GOSPA, with its localisation, missed and false parts (meaningful for `α = 2`).
        /// \param truths The true positions.
        /// \param estimates The estimated positions.
        /// \param c The cut-off, above zero.
        /// \param p The order, at least 1.
        /// \param alpha The cardinality-mismatch weight, in `(0, 2]`.
        /// \return The result, or a `metric_error`.
        [[nodiscard]] constexpr result<gospa_result<T>, metric_error> gospa(std::span<const linalg::fixed_vector<T, D>> truths, std::span<const linalg::fixed_vector<T, D>> estimates, T c,
                                                                            T p = T(1), T alpha = T(2)) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            if (!(c > T(0)) || !(p >= T(1)) || !(alpha > T(0)) || alpha > T(2))
                return err(metric_error::invalid_parameters);
            const std::size_t m = truths.size(), n = estimates.size();
            gospa_result<T> g{};
            const T cp = static_cast<T>(math::pow(c, p));
            T assigned(0);
            std::size_t paired = 0;
            if (m > 0 && n > 0) {
                const auto s = assign(truths, estimates, c, p);
                if (!s)
                    return err(s.unwrap_err_unchecked());
                assigned = *s;
                for (std::size_t i = 0; i < m; ++i) {
                    const auto j = row_to_col_[i];
                    if (j < 0)
                        continue;
                    const T d = linalg::norm(truths[i] - estimates[static_cast<std::size_t>(j)]);
                    if (d < c) {
                        g.localisation += static_cast<T>(math::pow(d, p));
                        ++paired;
                    }
                }
            }
            g.missed = m - paired;
            g.false_tracks = n - paired;
            const std::size_t mismatch = m > n ? m - n : n - m;
            g.distance = static_cast<T>(math::pow(assigned + cp / alpha * static_cast<T>(mismatch), T(1) / p));
            return ok(g);
        }

    private:
        std::array<T, MaxTruths * MaxEstimates> cost_{};
        std::array<std::int32_t, MaxTruths> row_to_col_{};
        combinatorics::assignment_solver<T, MaxTruths, MaxEstimates> solver_{};
        static_assert(big > 0, "snicholls::tracking::set_metric_solver: hold at least one point");
    };

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_metrics_hpp */
