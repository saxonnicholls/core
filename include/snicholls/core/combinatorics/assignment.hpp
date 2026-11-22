//
//  assignment.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::combinatorics::linear_assignment` and `assignment_solver`: the minimum-cost matching of rows to columns of a cost matrix (the "Hungarian" problem), with forbidden pairs.
/// \ingroup core
///
/// **Overview.** Given an `R x C` matrix of costs, choose one column for each row (or one row for each column if
/// there are fewer columns), no column twice, minimising the total. That is global nearest-neighbour data
/// association (tracks x detections, costs from Mahalanobis distances or negative log-likelihoods), sensor-to-target
/// tasking and the matching inside OSPA and GOSPA. An entry of `+∞` forbids the pair (outside the gate).
///
/// - `linear_assignment(cost)` for a `fixed_matrix<T, R, C>`: everything on the stack, sized at compile time.
/// - `assignment_solver<T, MaxRows, MaxCols>`: a reusable solver whose working arrays are sized at compile time,
///   for matrices known only at run time (row-major span, `rows` and `cols` up to the maxima). Construct it once
///   (on the heap if it is large) and call `solve` every scan.
/// - `objective::maximise` turns it into a maximum-weight matching (costs are negated inside).
///
/// The result assigns every row when `R ≤ C` (and every column when `R > C`); the other side has unassigned
/// entries, `-1`. To let a row stay unassigned at a price (a missed detection), append one dummy column per row
/// with that price on its own row and `+∞` elsewhere, the standard construction.
///
/// \snippet combinatorics/assignment_demo.cpp assignment
///
/// **Relationships.** `linalg/fixed.hpp` (`fixed_matrix`), `result.hpp`. Used by `tracking/metrics.hpp` (OSPA,
/// GOSPA) and by global-nearest-neighbour association. Group: \ref core.
///
/// **Implementations.** Shortest augmenting paths with dual variables (Jonker-Volgenant style), as in Crouse,
/// "On implementing 2D rectangular assignment algorithms", IEEE TAES 2016, the algorithm of SciPy's
/// `linear_sum_assignment`. One Dijkstra-like pass per row of the smaller side; ties prefer an unassigned column. A
/// matrix with more rows than columns is solved through its transpose, read in place.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation, no recursion. The
/// result is optimal. Complexity O(n² m) for `n = min(R, C)`, `m = max(R, C)`; working storage O(R + C).
///
/// **Errors.** `assignment_error::infeasible`: no complete matching with finite total cost exists (some row's
/// allowed columns are all taken). `assignment_error::invalid_cost`: a NaN or `-∞`. `assignment_error::too_large`:
/// `rows` or `cols` beyond the solver's maxima, or a span shorter than `rows * cols`. On error, `row_to_col` is
/// unspecified.
///
/// **Performance.** Uniform costs (`benchmarks/combinatorics/assignment_benchmark.cpp`; Release, Apple clang 17,
/// Xeon W-3245, 2026-11-23): 64 x 64 in 100 µs, 256 x 256 in 1.66 ms, and 256 x 256 with 90% of the pairs forbidden
/// in 1.36 ms. Each solution is checked first (complete, and no pair of rows gains by swapping columns).
///
/// **Tests and demos.** `tests/combinatorics/assignment_test.cpp`, `demos/combinatorics/assignment_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `linalg/fixed.hpp`, `result.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_combinatorics_assignment_hpp
#define snicholls_core_combinatorics_assignment_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::combinatorics {

    /// \brief Why an assignment has no answer.
    /// \ingroup core
    enum class assignment_error : std::uint8_t {
        infeasible = 1,    ///< No complete matching with finite total cost exists.
        invalid_cost = 2,  ///< A cost is NaN or minus infinity.
        too_large = 3,     ///< The matrix exceeds the solver's maxima, or the cost span is too short.
    };

    /// \brief Whether to minimise the total cost or maximise the total weight.
    /// \ingroup core
    enum class objective : std::uint8_t {
        minimise = 0,  ///< The minimum total cost.
        maximise = 1,  ///< The maximum total weight; `-∞` forbids a pair.
    };

    /// \brief The result for a fixed-size matrix: each row's column (or -1) and the total.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam R The rows.
    template <typename T, std::size_t R>
    struct assignment {
        std::array<std::int32_t, R> row_to_col{};  ///< The column assigned to each row, or -1.
        T total{};                                 ///< The total cost (or weight) of the assigned pairs.
    };

    namespace detail::assignment {
        // Crouse's shortest augmenting path over the n x m problem (n <= m) read through at(i, j). Working arrays
        // are spans of at least n (u, col4row, sr) or m (the rest) elements.
        template <typename T, typename At>
        constexpr result<T, assignment_error> solve(std::size_t n, std::size_t m, At at, std::span<T> u, std::span<T> v, std::span<T> spc, std::span<std::int32_t> path,
                                                    std::span<std::int32_t> col4row, std::span<std::int32_t> row4col, std::span<std::int32_t> remaining,
                                                    std::span<std::uint8_t> sr, std::span<std::uint8_t> sc) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            const T inf = std::numeric_limits<T>::infinity();
            for (std::size_t i = 0; i < n; ++i) {
                u[i] = T(0);
                col4row[i] = -1;
            }
            for (std::size_t j = 0; j < m; ++j) {
                v[j] = T(0);
                row4col[j] = -1;
                path[j] = -1;
            }
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < m; ++j) {
                    const T c = at(i, j);
                    if (c != c || c == -inf)
                        return err(assignment_error::invalid_cost);
                }
            for (std::size_t cur = 0; cur < n; ++cur) {
                T min_val(0);
                auto i = static_cast<std::int32_t>(cur);
                std::size_t left = m;
                for (std::size_t j = 0; j < m; ++j) {
                    remaining[j] = static_cast<std::int32_t>(m - j - 1);
                    spc[j] = inf;
                    sc[j] = 0;
                }
                for (std::size_t r = 0; r < n; ++r)
                    sr[r] = 0;
                std::int32_t sink = -1;
                while (sink == -1) {
                    std::size_t index = m;
                    T lowest = inf;
                    sr[static_cast<std::size_t>(i)] = 1;
                    for (std::size_t it = 0; it < left; ++it) {
                        const auto j = static_cast<std::size_t>(remaining[it]);
                        const T r = min_val + at(static_cast<std::size_t>(i), j) - u[static_cast<std::size_t>(i)] - v[j];
                        if (r < spc[j]) {
                            path[j] = i;
                            spc[j] = r;
                        }
                        if (spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1)) {
                            lowest = spc[j];
                            index = it;
                        }
                    }
                    min_val = lowest;
                    if (min_val == inf || index == m)
                        return err(assignment_error::infeasible);
                    const auto j = static_cast<std::size_t>(remaining[index]);
                    if (row4col[j] == -1)
                        sink = static_cast<std::int32_t>(j);
                    else
                        i = row4col[j];
                    sc[j] = 1;
                    remaining[index] = remaining[--left];
                }
                u[cur] += min_val;
                for (std::size_t r = 0; r < n; ++r)
                    if (sr[r] && r != cur)
                        u[r] += min_val - spc[static_cast<std::size_t>(col4row[r])];
                for (std::size_t j = 0; j < m; ++j)
                    if (sc[j])
                        v[j] -= min_val - spc[j];
                auto j = sink;
                for (;;) {
                    const std::int32_t r = path[static_cast<std::size_t>(j)];
                    row4col[static_cast<std::size_t>(j)] = r;
                    const std::int32_t t = col4row[static_cast<std::size_t>(r)];
                    col4row[static_cast<std::size_t>(r)] = j;
                    j = t;
                    if (r == static_cast<std::int32_t>(cur))
                        break;
                }
            }
            T total(0);
            for (std::size_t r = 0; r < n; ++r)
                total += at(r, static_cast<std::size_t>(col4row[r]));
            return ok(total);
        }
    } // namespace detail::assignment

    /// \brief A linear-assignment solver for matrices of up to `MaxRows x MaxCols`, its working arrays inside it.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam MaxRows The most rows.
    /// \tparam MaxCols The most columns.
    template <linalg::scalar T, std::size_t MaxRows, std::size_t MaxCols>
    class assignment_solver {
        static constexpr std::size_t lo = MaxRows < MaxCols ? MaxRows : MaxCols;
        static constexpr std::size_t hi = MaxRows < MaxCols ? MaxCols : MaxRows;

    public:
        /// \brief Solves one problem.
        /// \param cost The costs, row-major, `rows * cols` of them.
        /// \param rows The rows, at most `MaxRows`.
        /// \param cols The columns, at most `MaxCols`.
        /// \param row_to_col Receives each row's column, or -1; at least `rows` long.
        /// \param goal Minimise cost or maximise weight.
        /// \return The total of the assigned pairs, or an `assignment_error`.
        [[nodiscard]] constexpr result<T, assignment_error> solve(std::span<const T> cost, std::size_t rows, std::size_t cols, std::span<std::int32_t> row_to_col,
                                                                  objective goal = objective::minimise) noexcept(linalg::detail::nothrow_scalar<T>)
        {
            if (rows > MaxRows || cols > MaxCols || cost.size() < rows * cols || row_to_col.size() < rows)
                return err(assignment_error::too_large);
            for (std::size_t r = 0; r < rows; ++r)
                row_to_col[r] = -1;
            if (rows == 0 || cols == 0)
                return ok(T(0));
            const T sign = goal == objective::maximise ? T(-1) : T(1);
            const bool transposed = rows > cols;
            const std::size_t n = transposed ? cols : rows, m = transposed ? rows : cols;
            auto at = [&](std::size_t i, std::size_t j) { return sign * (transposed ? cost[j * cols + i] : cost[i * cols + j]); };
            const auto s = detail::assignment::solve<T>(n, m, at, std::span<T>(u_).first(n), std::span<T>(v_).first(m), std::span<T>(spc_).first(m),
                                                        std::span<std::int32_t>(path_).first(m), std::span<std::int32_t>(col4row_).first(n),
                                                        std::span<std::int32_t>(row4col_).first(m), std::span<std::int32_t>(remaining_).first(m),
                                                        std::span<std::uint8_t>(sr_).first(n), std::span<std::uint8_t>(sc_).first(m));
            if (!s)
                return err(s.unwrap_err_unchecked());
            if (transposed) {
                for (std::size_t j = 0; j < m; ++j)
                    if (row4col_[j] >= 0)
                        row_to_col[j] = row4col_[j];
                // row4col_ maps the transposed problem's columns (our rows) to its rows (our columns)
            } else {
                for (std::size_t r = 0; r < n; ++r)
                    row_to_col[r] = col4row_[r];
            }
            return ok(sign * *s);
        }

    private:
        std::array<T, lo> u_{};
        std::array<T, hi> v_{}, spc_{};
        std::array<std::int32_t, hi> path_{}, row4col_{}, remaining_{};
        std::array<std::int32_t, lo> col4row_{};
        std::array<std::uint8_t, lo> sr_{};
        std::array<std::uint8_t, hi> sc_{};
    };

    /// \brief The optimal assignment for a fixed-size cost matrix.
    /// \ingroup core
    /// \tparam T The number type.
    /// \tparam R The rows.
    /// \tparam C The columns.
    /// \param cost The costs (or weights); `+∞` (or `-∞` when maximising) forbids a pair.
    /// \param goal Minimise cost or maximise weight.
    /// \return Each row's column and the total, or an `assignment_error`.
    template <linalg::scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr result<assignment<T, R>, assignment_error> linear_assignment(const linalg::fixed_matrix<T, R, C>& cost, objective goal = objective::minimise) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        assignment_solver<T, R, C> solver;
        assignment<T, R> a{};
        const auto s = solver.solve(std::span<const T>(cost.elements), R, C, std::span<std::int32_t>(a.row_to_col), goal);
        if (!s)
            return err(s.unwrap_err_unchecked());
        a.total = *s;
        return ok(a);
    }

} // namespace snicholls::combinatorics

#endif /* snicholls_core_combinatorics_assignment_hpp */
