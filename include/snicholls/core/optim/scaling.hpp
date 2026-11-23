//
//  scaling.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::optim::ruiz_equilibrate`: modified Ruiz equilibration of a quadratic program's data, as OSQP does it, and of a single matrix, as SCS does it, with the scaling and unscaling of solutions, all `constexpr`, in place.
/// \ingroup core
///
/// **Overview.** First-order and Krylov solvers converge at a rate set by the conditioning of their data. A
/// portfolio QP mixes weights near 1e-2, notionals near 1e8 and risk numbers near 1e-6; left alone, ADMM crawls.
/// Equilibration rescales rows and columns so that every row and column of the KKT matrix has ∞-norm near 1 (Ruiz,
/// "A scaling algorithm to equilibrate both rows and columns norms in matrices", 2001), which is cheap (a few passes
/// over the non-zeros) and changes neither the solution nor the sparsity.
///
/// The quadratic program, the form of OSQP:
///
///     minimise ½ xᵀ P x + qᵀ x   subject to   l ≤ A x ≤ u,     P (n x n) symmetric PSD, A (m x n)
///
/// is replaced by its scaled form, with `D = diag(d)` (variables, n), `E = diag(e)` (constraints, m) and the cost
/// scale `c > 0`:
///
///     P̄ = c D P D,   q̄ = c D q,   Ā = E A D,   l̄ = E l,   ū = E u,   x = D x̄,   y = E ȳ / c,   z = A x = E⁻¹ z̄
///
/// where `y` is the multiplier of the constraints with the sign of `P x + q + Aᵀ y = 0`, and the scaled objective is
/// `c` times the original. `ruiz_equilibrate(P, A, q, d, e, c, iterations, work)` computes `d`, `e`, `c` and rewrites
/// `P`, `A` and `q` in place (`P` as its upper triangle); `scale_bounds` rewrites `l` and `u`. After solving the scaled
/// problem, `unscale_solution` (or `unscale_primal`, `unscale_dual`, `unscale_constraint`, `unscale_objective`) gives
/// the original's; `scale_primal` and `scale_dual` take a warm start the other way; `unscale_data` restores `P`, `A`
/// and `q`.
///
/// Each iteration (all from the current, already scaled, data):
///
/// 1. The KKT matrix `[P Aᵀ; A 0]`'s column ∞-norms: `δⱼ = max(‖P₍:,ⱼ₎‖∞, ‖A₍:,ⱼ₎‖∞)` for the variables (P's full
///    symmetric column), `εᵢ = ‖A₍ᵢ,:₎‖∞` for the constraints.
/// 2. Each norm limited (`limit(v) = 1` if `v < min_scaling`, `max_scaling` if `v > max_scaling`, else `v`), then
///    `δ̂ⱼ = 1 / sqrt(limit(δⱼ))`, `ε̂ᵢ = 1 / sqrt(limit(εᵢ))` (rounded to the nearest power of two in the geometric
///    sense when `power_of_two`).
/// 3. `P ← Δ̂ P Δ̂`, `A ← Ê A Δ̂`, `q ← Δ̂ q`, `d ← Δ̂ d`, `e ← Ê e`.
/// 4. When `scale_cost`: `γ = 1 / limit(max(mean ‖P₍:,ⱼ₎‖∞, limit(‖q‖∞)))` (rounded likewise), `P ← γ P`,
///    `q ← γ q`, `c ← γ c`.
///
/// `d`, `e` and `c` start at 1. A zero row or column (norm below `min_scaling`) is left unscaled. This is OSQP's
/// procedure (Stellato et al., "OSQP: an operator splitting solver for quadratic programs", 2020, section 5.1), and its
/// defaults: 10 iterations, limits 1e-4 and 1e4.
///
/// `ruiz_equilibrate_rectangular(A, r, s, iterations, work)` is the same iteration for one `m x n` matrix, rows and
/// columns at once: `Ā = R A S`, `R = diag(r)` (rows, m), `S = diag(s)` (columns, n), `r̂ᵢ = 1 / sqrt(limit(‖A₍ᵢ,:₎‖∞))`,
/// `ŝⱼ = 1 / sqrt(limit(‖A₍:,ⱼ₎‖∞))` from the current matrix each iteration. It is the equilibration SCS applies to its
/// data matrix (O'Donoghue et al., 2016, section 5; SCS's `D` is `r`, its `E` is `s`), and what a Krylov method on
/// `AᵀA` wants. For a system `A x = b`: solve `Ā x̄ = R b`, then `x = S x̄`.
///
/// \snippet optim/scaling_demo.cpp scaling
///
/// **Relationships.** `sparse/csc.hpp` (`scale`, `column_inf_norms`, `row_inf_norms`, `symmetric_upper_inf_norms`),
/// `linalg/krylov.hpp` (solve after equilibrating), `math/constexpr_math.hpp` (`sqrt`). Group: \ref core.
///
/// **Implementations.** Plain passes over the CSC arrays: one per norm, one per `scale`. The power-of-two variant
/// rounds each factor to `2ᵏ`, so every scaling and unscaling is exact in binary floating point (no rounding,
/// barring overflow and underflow); its norms then approach 1 only to within a factor of `√2` or so.
///
/// **Guarantees.** `constexpr`, `noexcept` for arithmetic `T` and integer `I`, no allocation, no recursion. The
/// patterns of `P` and `A` are unchanged; only values change. `d`, `e` and `c` are positive and finite. Complexity:
/// O(iterations · (nnz(P) + nnz(A) + n + m)).
///
/// **Errors.** `sparse::sparse_error::dimension_mismatch`: `P` not `n x n`, `A` not `m x n`, `q`, `d` or `e` of the
/// wrong length. `sparse::sparse_error::insufficient_capacity`: `work` shorter than `ruiz_workspace(n, m)` (or
/// `ruiz_rectangular_workspace(m, n)`). On an error nothing is written.
///
/// **Performance.** Not benchmarked: it runs once per problem, before the solve.
///
/// **Tests and demos.** `tests/optim/scaling_test.cpp`, `demos/optim/scaling_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<span>`, `<type_traits>`, `sparse/csc.hpp`, `linalg/types.hpp`, `math/constexpr_math.hpp`,
/// `result.hpp`.
///
/// **Known issues.** Without `power_of_two`, scaling then unscaling a value returns it to within an ulp or two, not
/// exactly. Equilibration is a heuristic: it brings norms near 1 and conditioning down in practice, without a bound
/// on the condition number. Infinite bounds stay infinite through `scale_bounds`.
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_optim_scaling_hpp
#define snicholls_core_optim_scaling_hpp

#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/sparse/csc.hpp>

#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::optim {

    /// \brief The settings of Ruiz equilibration other than the iteration count.
    /// \ingroup core
    /// \tparam T The value type.
    template <linalg::scalar T>
    struct ruiz_options {
        T min_scaling = T(1e-4);     ///< A norm below this counts as zero: its row or column is left unscaled.
        T max_scaling = T(1e4);      ///< A norm above this is taken as this, bounding one iteration's factor.
        bool scale_cost = true;      ///< Also scale the cost by `c` (QP form only).
        bool power_of_two = false;   ///< Round every factor to a power of two, so scaling is exact.
    };

    /// \brief The working storage `ruiz_equilibrate` needs: `2n + m` values.
    /// \ingroup core
    /// \param n The variables.
    /// \param m The constraints.
    /// \return `2n + m`.
    [[nodiscard]] constexpr std::size_t ruiz_workspace(std::size_t n, std::size_t m) noexcept { return 2 * n + m; }

    /// \brief The working storage `ruiz_equilibrate_rectangular` needs: `m + n` values.
    /// \ingroup core
    /// \param m The rows.
    /// \param n The columns.
    /// \return `m + n`.
    [[nodiscard]] constexpr std::size_t ruiz_rectangular_workspace(std::size_t m, std::size_t n) noexcept { return m + n; }

    namespace detail {
        template <typename T>
        constexpr T ruiz_limit(T v, const ruiz_options<T>& o) noexcept
        {
            return v < o.min_scaling ? T(1) : (v > o.max_scaling ? o.max_scaling : v);
        }
        // The power of two nearest s > 0 in the geometric sense: 2ᵏ with 2ᵏ ≤ s < 2ᵏ⁺¹, or 2ᵏ⁺¹ if s / 2ᵏ ≥ √2
        template <typename T>
        constexpr T ruiz_round(T s, const ruiz_options<T>& o) noexcept
        {
            if (!o.power_of_two)
                return s;
            T p(1);
            if (s >= T(1))
                while (p * T(2) <= s)
                    p *= T(2);
            else
                while (p > s)
                    p /= T(2);
            return s / p < T(1.41421356237309504880L) ? p : p * T(2);
        }
        template <typename T>
        constexpr T ruiz_factor(T norm, const ruiz_options<T>& o) noexcept
        {
            return ruiz_round(T(1) / static_cast<T>(math::sqrt(ruiz_limit(norm, o))), o);
        }
    } // namespace detail

    /// \brief Modified Ruiz equilibration of a QP's data, as OSQP does it: `P̄ = c D P D`, `Ā = E A D`, `q̄ = c D q`,
    /// in place, with the conventions of the file's documentation.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param p The upper triangle of `P`, `n x n` (mutable values; entries below the diagonal are ignored).
    /// \param a `A`, `m x n` (mutable values).
    /// \param q The linear cost, `n` values, scaled in place.
    /// \param d Receives the variables' scaling, `n` values.
    /// \param e Receives the constraints' scaling, `m` values.
    /// \param c Receives the cost scale.
    /// \param iterations The Ruiz iterations (OSQP: 10; 0 leaves the data alone, with unit scalings).
    /// \param work At least `ruiz_workspace(n, m)` values.
    /// \param options The limits, cost scaling and power-of-two rounding.
    /// \return Nothing, or a `sparse::sparse_error`.
    template <linalg::scalar T, typename I>
    [[nodiscard]] constexpr result<void, sparse::sparse_error> ruiz_equilibrate(const sparse::csc_view<T, I>& p, const sparse::csc_view<T, I>& a, std::type_identity_t<std::span<T>> q,
                                                                               std::type_identity_t<std::span<T>> d, std::type_identity_t<std::span<T>> e, T& c,
                                                                               std::size_t iterations, std::type_identity_t<std::span<T>> work,
                                                                               const std::type_identity_t<ruiz_options<T>>& options = {}) noexcept
    {
        const std::size_t n = p.cols, m = a.rows;
        if (p.rows != n || a.cols != n || q.size() != n || d.size() != n || e.size() != m)
            return err(sparse::sparse_error::dimension_mismatch);
        if (work.size() < ruiz_workspace(n, m))
            return err(sparse::sparse_error::insufficient_capacity);
        const auto dt = work.subspan(0, n), tmp = work.subspan(n, n), et = work.subspan(2 * n, m);
        for (std::size_t j = 0; j < n; ++j)
            d[j] = T(1);
        for (std::size_t i = 0; i < m; ++i)
            e[i] = T(1);
        c = T(1);
        for (std::size_t it = 0; it < iterations; ++it) {
            sparse::symmetric_upper_inf_norms(p, dt);
            sparse::column_inf_norms(a, tmp);
            sparse::row_inf_norms(a, et);
            for (std::size_t j = 0; j < n; ++j)
                dt[j] = detail::ruiz_factor(dt[j] > tmp[j] ? dt[j] : tmp[j], options);
            for (std::size_t i = 0; i < m; ++i)
                et[i] = detail::ruiz_factor(et[i], options);
            sparse::scale(p, std::span<const T>(dt), std::span<const T>(dt));
            sparse::scale(a, std::span<const T>(et), std::span<const T>(dt));
            for (std::size_t j = 0; j < n; ++j) {
                q[j] *= dt[j];
                d[j] *= dt[j];
            }
            for (std::size_t i = 0; i < m; ++i)
                e[i] *= et[i];
            if (options.scale_cost) {
                sparse::symmetric_upper_inf_norms(p, tmp);
                T mean(0), qn(0);
                for (std::size_t j = 0; j < n; ++j) {
                    mean += tmp[j];
                    const T v = q[j] < T(0) ? -q[j] : q[j];
                    qn = v > qn ? v : qn;
                }
                mean = n > 0 ? mean / static_cast<T>(n) : T(0);
                qn = detail::ruiz_limit(qn, options);
                const T measure = detail::ruiz_limit(mean > qn ? mean : qn, options);
                const T gamma = detail::ruiz_round(T(1) / measure, options);
                for (std::size_t k = 0; k < p.nnz(); ++k)
                    p.values[k] *= gamma;
                for (std::size_t j = 0; j < n; ++j)
                    q[j] *= gamma;
                c *= gamma;
            }
        }
        return ok();
    }

    /// \brief Ruiz equilibration of one `m x n` matrix, rows and columns at once, as SCS does it: `Ā = R A S`, in place.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param a The matrix (mutable values).
    /// \param r Receives the row scaling, `m` values (SCS's `D`).
    /// \param s Receives the column scaling, `n` values (SCS's `E`).
    /// \param iterations The iterations (0 leaves `a` alone, with unit scalings).
    /// \param work At least `ruiz_rectangular_workspace(m, n)` values.
    /// \param options The limits and power-of-two rounding (`scale_cost` is ignored).
    /// \return Nothing, or a `sparse::sparse_error`.
    template <linalg::scalar T, typename I>
    [[nodiscard]] constexpr result<void, sparse::sparse_error> ruiz_equilibrate_rectangular(const sparse::csc_view<T, I>& a, std::type_identity_t<std::span<T>> r,
                                                                                           std::type_identity_t<std::span<T>> s, std::size_t iterations,
                                                                                           std::type_identity_t<std::span<T>> work, const std::type_identity_t<ruiz_options<T>>& options = {}) noexcept
    {
        const std::size_t m = a.rows, n = a.cols;
        if (r.size() != m || s.size() != n)
            return err(sparse::sparse_error::dimension_mismatch);
        if (work.size() < ruiz_rectangular_workspace(m, n))
            return err(sparse::sparse_error::insufficient_capacity);
        const auto rt = work.subspan(0, m), st = work.subspan(m, n);
        for (std::size_t i = 0; i < m; ++i)
            r[i] = T(1);
        for (std::size_t j = 0; j < n; ++j)
            s[j] = T(1);
        for (std::size_t it = 0; it < iterations; ++it) {
            sparse::row_inf_norms(a, rt);
            sparse::column_inf_norms(a, st);
            for (std::size_t i = 0; i < m; ++i) {
                rt[i] = detail::ruiz_factor(rt[i], options);
                r[i] *= rt[i];
            }
            for (std::size_t j = 0; j < n; ++j) {
                st[j] = detail::ruiz_factor(st[j], options);
                s[j] *= st[j];
            }
            sparse::scale(a, std::span<const T>(rt), std::span<const T>(st));
        }
        return ok();
    }

    /// \brief `l ← E l`, `u ← E u`: the constraint bounds into the scaled problem (infinite bounds stay infinite).
    /// \ingroup core
    /// \tparam T The value type.
    /// \param e The constraints' scaling, `m` values.
    /// \param l The lower bounds, `m` values, in place.
    /// \param u The upper bounds, `m` values, in place.
    template <linalg::scalar T>
    constexpr void scale_bounds(std::span<const T> e, std::type_identity_t<std::span<T>> l, std::type_identity_t<std::span<T>> u) noexcept
    {
        for (std::size_t i = 0; i < e.size(); ++i) {
            l[i] *= e[i];
            u[i] *= e[i];
        }
    }

    /// \brief `x̄ = D⁻¹ x`: a primal point (a warm start) into the scaled problem.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param d The variables' scaling, `n` values.
    /// \param x `n` values, in place.
    template <linalg::scalar T>
    constexpr void scale_primal(std::span<const T> d, std::type_identity_t<std::span<T>> x) noexcept
    {
        for (std::size_t j = 0; j < d.size(); ++j)
            x[j] /= d[j];
    }

    /// \brief `ȳ = c E⁻¹ y`: a constraint multiplier (a warm start) into the scaled problem.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param e The constraints' scaling, `m` values.
    /// \param c The cost scale.
    /// \param y `m` values, in place.
    template <linalg::scalar T>
    constexpr void scale_dual(std::span<const T> e, std::type_identity_t<T> c, std::type_identity_t<std::span<T>> y) noexcept
    {
        for (std::size_t i = 0; i < e.size(); ++i)
            y[i] = c * y[i] / e[i];
    }

    /// \brief `x = D x̄`: the scaled problem's primal solution back to the original's.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param d The variables' scaling, `n` values.
    /// \param x `n` values, in place.
    template <linalg::scalar T>
    constexpr void unscale_primal(std::span<const T> d, std::type_identity_t<std::span<T>> x) noexcept
    {
        for (std::size_t j = 0; j < d.size(); ++j)
            x[j] *= d[j];
    }

    /// \brief `y = E ȳ / c`: the scaled problem's constraint multiplier back to the original's.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param e The constraints' scaling, `m` values.
    /// \param c The cost scale.
    /// \param y `m` values, in place.
    template <linalg::scalar T>
    constexpr void unscale_dual(std::span<const T> e, std::type_identity_t<T> c, std::type_identity_t<std::span<T>> y) noexcept
    {
        for (std::size_t i = 0; i < e.size(); ++i)
            y[i] = e[i] * y[i] / c;
    }

    /// \brief `z = E⁻¹ z̄`: the scaled constraint values `Ā x̄` back to the original's `A x`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param e The constraints' scaling, `m` values.
    /// \param z `m` values, in place.
    template <linalg::scalar T>
    constexpr void unscale_constraint(std::span<const T> e, std::type_identity_t<std::span<T>> z) noexcept
    {
        for (std::size_t i = 0; i < e.size(); ++i)
            z[i] /= e[i];
    }

    /// \brief The scaled problem's objective value back to the original's: `f = f̄ / c`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param c The cost scale.
    /// \param value The scaled objective.
    /// \return The original objective.
    template <linalg::scalar T>
    [[nodiscard]] constexpr T unscale_objective(T c, T value) noexcept
    {
        return value / c;
    }

    /// \brief The scaled problem's primal and dual solution back to the original's: `x = D x̄`, `y = E ȳ / c`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param d The variables' scaling, `n` values.
    /// \param e The constraints' scaling, `m` values.
    /// \param c The cost scale.
    /// \param x `n` values, in place.
    /// \param y `m` values, in place.
    template <linalg::scalar T>
    constexpr void unscale_solution(std::span<const T> d, std::type_identity_t<std::span<const T>> e, std::type_identity_t<T> c, std::type_identity_t<std::span<T>> x,
                                    std::type_identity_t<std::span<T>> y) noexcept
    {
        unscale_primal<T>(d, x);
        unscale_dual<T>(e, c, y);
    }

    /// \brief Restores the original data in place: `P = D⁻¹ P̄ D⁻¹ / c`, `A = E⁻¹ Ā D⁻¹`, `q = D⁻¹ q̄ / c`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param p The scaled upper triangle of `P`, `n x n` (mutable values).
    /// \param a The scaled `A`, `m x n` (mutable values).
    /// \param q The scaled linear cost, `n` values, in place.
    /// \param d The variables' scaling, `n` values.
    /// \param e The constraints' scaling, `m` values.
    /// \param c The cost scale.
    template <linalg::scalar T, typename I>
    constexpr void unscale_data(const sparse::csc_view<T, I>& p, const sparse::csc_view<T, I>& a, std::type_identity_t<std::span<T>> q,
                                std::type_identity_t<std::span<const T>> d, std::type_identity_t<std::span<const T>> e, std::type_identity_t<T> c) noexcept
    {
        for (std::size_t j = 0; j < p.cols; ++j)
            for (std::size_t k = static_cast<std::size_t>(p.col_ptr[j]); k < static_cast<std::size_t>(p.col_ptr[j + 1]); ++k)
                p.values[k] = p.values[k] / d[static_cast<std::size_t>(p.row_idx[k])] / d[j] / c;
        for (std::size_t j = 0; j < a.cols; ++j)
            for (std::size_t k = static_cast<std::size_t>(a.col_ptr[j]); k < static_cast<std::size_t>(a.col_ptr[j + 1]); ++k)
                a.values[k] = a.values[k] / e[static_cast<std::size_t>(a.row_idx[k])] / d[j];
        for (std::size_t j = 0; j < q.size(); ++j)
            q[j] = q[j] / d[j] / c;
    }

} // namespace snicholls::optim

#endif /* snicholls_core_optim_scaling_hpp */
