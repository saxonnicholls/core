//
//  krylov.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::linalg::conjugate_gradient` and `minres`: Krylov solvers for `A x = b` over any linear operator (a CSC matrix, a symmetric upper triangle, a dense array, or a callable the GPU provides), on the caller's storage, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A convex solver's inner step is a linear system: the reduced KKT system of an interior-point method,
/// `P + σI + ρAᵀA` in ADMM, a Newton system in L-BFGS-B. When the matrix is large, or is never formed (a product the
/// GPU computes from factor loadings), a Krylov method solves it from matrix-vector products alone.
///
/// - `linear_operator<Op>`: `Op::value_type`, `op.rows()`, `op.cols()` and `op.apply(x, y)`, which writes `y = A x`.
///   Adapters: `csc_operator` (`sparse::spmv`), `symmetric_csc_operator` (a symmetric matrix stored as its upper
///   triangle, `sparse::spmv_symmetric_upper`), `dense_operator` (row-major), `function_operator` (any callable
///   `f(x, y)`); `make_csc_operator`, `make_symmetric_csc_operator`, `make_dense_operator`, `make_function_operator`.
/// - `preconditioner<M, T>`: `m.apply(r, z)`, which writes `z = M⁻¹ r` for a symmetric positive definite `M`.
///   `identity_preconditioner`, `jacobi_preconditioner` (from a diagonal; `csc_diagonal` extracts one).
/// - `conjugate_gradient(op, b, x, work, options, m)`: preconditioned CG (Hestenes and Stiefel) for symmetric positive
///   definite `A`. `minres(op, b, x, work, options, m)`: MINRES (Paige and Saunders, 1975) for symmetric `A`, definite
///   or not (a KKT matrix, a saddle point).
/// - `conjugate_gradient_workspace(n)`, `minres_workspace(n)`: the working storage, so the caller sizes it.
///
/// `x` is the warm start on entry and the solution on exit. Use CG when `A` is SPD (it is the cheaper of the two, one
/// product and two dot products an iteration); MINRES when `A` is symmetric but indefinite. Neither is for a
/// non-symmetric `A` (that is GMRES or BiCGStab, not here), nor for a small dense system (factorise it:
/// `linalg/fixed.hpp`, `linalg/backend.hpp`).
///
/// \snippet linalg/krylov_demo.cpp krylov
///
/// **Relationships.** `sparse/csc.hpp` (the products the CSC adapters call), `linalg/types.hpp` (`scalar`),
/// `math/constexpr_math.hpp` (`sqrt`), `optim/scaling.hpp` (equilibrate before solving), `optim/lbfgs.hpp`.
/// Group: \ref core.
///
/// **Implementations.** CG: `r = b - A x`, `z = M⁻¹ r`, `p = z`; each iteration `q = A p`, `α = rᵀz / pᵀq`,
/// `x += α p`, `r -= α q`, then the test on `‖r‖₂`, `z = M⁻¹ r`, `β = rᵀz_new / rᵀz_old`, `p = z + β p`. The residual
/// is the recurrence's, not recomputed. With `identity_preconditioner`, `z` is `r` (no copy, no call). MINRES: the
/// Lanczos process with the `M`-inner product, the tridiagonal system's QR factor updated by one Givens rotation an
/// iteration, and three search directions, rotated in place; the residual estimate `φ̄` is `‖b - A x‖` in the
/// `M⁻¹`-norm (`‖·‖₂` without a preconditioner). Dot products accumulate in index order.
///
/// **Guarantees.** `constexpr`; `noexcept` when the operator's and the preconditioner's `apply` are and `T`'s
/// arithmetic is. No allocation, no recursion. Convergence: `‖r‖ ≤ max(relative_tolerance · ‖b‖, absolute_tolerance)`,
/// in the norm stated above (CG: `‖·‖₂`; MINRES: `‖·‖_{M⁻¹}`, `‖b‖` too). `b = 0` returns `x = 0`, converged, with no
/// iterations. In exact arithmetic CG and MINRES finish in at most `k` iterations for `A` with `k` distinct
/// eigenvalues (at most `n`). An iteration is one product with `A`; the initial residual's product is not counted.
/// Complexity: per iteration one `apply`, one preconditioner `apply` and O(n) more.
///
/// **Errors.** Nothing throws unless the operator does. Every outcome is a `krylov_status` in the result:
/// `iteration_limit` (not converged; `x` is the last iterate), `breakdown` (CG: `pᵀAp ≤ 0` or `rᵀM⁻¹r ≤ 0`, so `A` or
/// `M` is not positive definite; MINRES: `rᵀM⁻¹r < 0`, `M` is not; `x` is the last iterate), `invalid_input` (shapes
/// that do not fit, or `work` too short; `x` untouched).
///
/// **Performance.** CG on the 2-D Laplacian of a 64 x 64 grid (n = 4096, 20 224 non-zeros), 200 iterations from
/// zero, through `csc_operator`, against the same CG written by hand over the same CSC arrays in the same arithmetic
/// order, after a bit-for-bit check (`benchmarks/linalg/krylov_benchmark.cpp`; Release, Apple clang 17, Xeon W-3245,
/// 2026-11-23, a loaded machine): 10.5 ms of CPU a solve, by hand 10.7 ms (19.0 and 18.7 thousand iterations a
/// second); the two are the same within the noise. Without a preconditioner `z` is `r` itself, so nothing is copied.
///
/// **Tests and demos.** `tests/linalg/krylov_test.cpp`, `demos/linalg/krylov_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`, `<utility>`,
/// `sparse/csc.hpp`, `linalg/types.hpp` (which brings `containers/md_span.hpp`), `math/constexpr_math.hpp`.
///
/// **Known issues.** `‖·‖₂` is `sqrt(xᵀx)`, without scaling, so it overflows for entries near `sqrt(max())`. In
/// floating point CG loses orthogonality and may need more than `n` iterations; `max_iterations = 0` allows `2n`.
/// The default relative tolerance, 1e-8, is out of reach in `float`: raise it.
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_linalg_krylov_hpp
#define snicholls_core_linalg_krylov_hpp

#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/sparse/csc.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls::linalg {

    /// \brief A linear map `A`: `Op::value_type`, `op.rows()`, `op.cols()`, and `op.apply(x, y)`, which writes
    /// `y = A x` (`x` of `cols()` values, `y` of `rows()`, not aliased).
    /// \ingroup core
    /// \tparam Op The operator type.
    template <typename Op>
    concept linear_operator = scalar<typename Op::value_type> && requires(const Op& op, std::span<const typename Op::value_type> x, std::span<typename Op::value_type> y) {
        { op.rows() } -> std::convertible_to<std::size_t>;
        { op.cols() } -> std::convertible_to<std::size_t>;
        op.apply(x, y);
    };

    /// \brief A preconditioner for values of `T`: `m.apply(r, z)` writes `z = M⁻¹ r` (not aliased), for a symmetric
    /// positive definite `M`.
    /// \ingroup core
    /// \tparam M The preconditioner type.
    /// \tparam T The value type.
    template <typename M, typename T>
    concept preconditioner = requires(const M& m, std::span<const T> r, std::span<T> z) { m.apply(r, z); };

    namespace detail {
        template <typename Op, typename T>
        inline constexpr bool nothrow_apply = noexcept(std::declval<const Op&>().apply(std::declval<std::span<const T>>(), std::declval<std::span<T>>()));

        template <typename T>
        constexpr T krylov_dot(std::span<const T> a, std::span<const T> b) noexcept(nothrow_scalar<T>)
        {
            T s(0);
            for (std::size_t i = 0; i < a.size(); ++i)
                s += a[i] * b[i];
            return s;
        }
        template <typename T>
        constexpr T krylov_norm(std::span<const T> a) noexcept(nothrow_scalar<T>)
        {
            return static_cast<T>(math::sqrt(krylov_dot(a, a)));
        }
    } // namespace detail

    /// \brief A CSC matrix as a `linear_operator`: `y = A x` by `sparse::spmv`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    template <scalar T, typename I = std::int32_t>
    struct csc_operator {
        using value_type = T;                          ///< The value type.
        sparse::csc_view<const T, const I> matrix{};   ///< The matrix, viewed.

        /// \brief The rows. \return `matrix.rows`.
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return matrix.rows; }
        /// \brief The columns. \return `matrix.cols`.
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return matrix.cols; }
        /// \brief `y = A x`. \param x `cols()` values. \param y `rows()` values, overwritten.
        constexpr void apply(std::span<const T> x, std::span<T> y) const noexcept { sparse::spmv(matrix, x, y); }
    };

    /// \brief A symmetric matrix stored as its upper triangle (CSC) as a `linear_operator`: `y = A x` by
    /// `sparse::spmv_symmetric_upper`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    template <scalar T, typename I = std::int32_t>
    struct symmetric_csc_operator {
        using value_type = T;                          ///< The value type.
        sparse::csc_view<const T, const I> upper{};    ///< The upper triangle, diagonal included, square.

        /// \brief The rows. \return `upper.rows`.
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return upper.rows; }
        /// \brief The columns. \return `upper.cols`.
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return upper.cols; }
        /// \brief `y = A x`. \param x `cols()` values. \param y `rows()` values, overwritten.
        constexpr void apply(std::span<const T> x, std::span<T> y) const noexcept { sparse::spmv_symmetric_upper(upper, x, y); }
    };

    /// \brief A dense row-major matrix as a `linear_operator`: `yᵢ = Σⱼ aᵢⱼ xⱼ`, `j` ascending.
    /// \ingroup core
    /// \tparam T The value type.
    template <scalar T>
    struct dense_operator {
        using value_type = T;              ///< The value type.
        std::span<const T> elements{};     ///< `row_count · col_count` values, row by row.
        std::size_t row_count = 0;         ///< The rows.
        std::size_t col_count = 0;         ///< The columns.

        /// \brief The rows. \return `row_count`.
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return row_count; }
        /// \brief The columns. \return `col_count`.
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return col_count; }
        /// \brief `y = A x`. \param x `cols()` values. \param y `rows()` values, overwritten.
        constexpr void apply(std::span<const T> x, std::span<T> y) const noexcept(detail::nothrow_scalar<T>)
        {
            for (std::size_t i = 0; i < row_count; ++i) {
                T s(0);
                for (std::size_t j = 0; j < col_count; ++j)
                    s += elements[i * col_count + j] * x[j];
                y[i] = s;
            }
        }
    };

    /// \brief A callable `f(x, y)` that writes `y = A x` as a `linear_operator`: the matrix-free form (a product on the
    /// GPU, a Hessian-vector product, an operator never formed).
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam F The callable, invocable as `f(std::span<const T>, std::span<T>)` through a `const` reference.
    template <scalar T, typename F>
    struct function_operator {
        using value_type = T;          ///< The value type.
        std::size_t row_count = 0;     ///< The rows.
        std::size_t col_count = 0;     ///< The columns.
        F function;                    ///< The product.

        /// \brief The rows. \return `row_count`.
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return row_count; }
        /// \brief The columns. \return `col_count`.
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return col_count; }
        /// \brief `y = A x`, by the callable. \param x `cols()` values. \param y `rows()` values, overwritten.
        constexpr void apply(std::span<const T> x, std::span<T> y) const noexcept(std::is_nothrow_invocable_v<const F&, std::span<const T>, std::span<T>>)
        {
            function(x, y);
        }
    };

    /// \brief A CSC matrix as a `csc_operator`.
    /// \ingroup core
    /// \tparam T The value type (possibly `const`).
    /// \tparam I The index type (possibly `const`).
    /// \param a The matrix; its storage must outlive the operator.
    /// \return The operator.
    template <typename T, typename I>
    [[nodiscard]] constexpr csc_operator<std::remove_const_t<T>, std::remove_const_t<I>> make_csc_operator(const sparse::csc_view<T, I>& a) noexcept
    {
        return {a.as_const()};
    }

    /// \brief A symmetric matrix's upper triangle as a `symmetric_csc_operator`.
    /// \ingroup core
    /// \tparam T The value type (possibly `const`).
    /// \tparam I The index type (possibly `const`).
    /// \param upper The upper triangle, square; its storage must outlive the operator.
    /// \return The operator.
    template <typename T, typename I>
    [[nodiscard]] constexpr symmetric_csc_operator<std::remove_const_t<T>, std::remove_const_t<I>> make_symmetric_csc_operator(const sparse::csc_view<T, I>& upper) noexcept
    {
        return {upper.as_const()};
    }

    /// \brief A dense row-major matrix as a `dense_operator`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param elements `rows · cols` values, row by row; they must outlive the operator.
    /// \param rows The rows.
    /// \param cols The columns.
    /// \return The operator.
    template <scalar T>
    [[nodiscard]] constexpr dense_operator<T> make_dense_operator(std::span<const T> elements, std::size_t rows, std::size_t cols) noexcept
    {
        return {elements, rows, cols};
    }

    /// \brief A callable as a `function_operator`.
    /// \ingroup core
    /// \tparam T The value type, given explicitly.
    /// \tparam F The callable.
    /// \param rows The rows.
    /// \param cols The columns.
    /// \param f The product, `f(x, y)` writing `y = A x`.
    /// \return The operator, holding a copy of `f`.
    template <scalar T, typename F>
    [[nodiscard]] constexpr function_operator<T, std::decay_t<F>> make_function_operator(std::size_t rows, std::size_t cols, F&& f) noexcept(std::is_nothrow_constructible_v<std::decay_t<F>, F&&>)
    {
        return {rows, cols, std::forward<F>(f)};
    }

    /// \brief No preconditioner: `z = r`.
    /// \ingroup core
    struct identity_preconditioner {
        /// \brief `z = r`. \tparam T The value type. \param r The residual. \param z Overwritten with `r`.
        template <typename T>
        constexpr void apply(std::span<const T> r, std::span<T> z) const noexcept(std::is_nothrow_copy_assignable_v<T>)
        {
            for (std::size_t i = 0; i < r.size(); ++i)
                z[i] = r[i];
        }
    };

    /// \brief The Jacobi (diagonal) preconditioner: `zᵢ = rᵢ / dᵢ`, and `zᵢ = rᵢ` where `dᵢ = 0`.
    /// \ingroup core
    /// \tparam T The value type.
    ///
    /// For CG and MINRES the diagonal must be positive where it is not zero (`M` positive definite); a negative entry is
    /// reported as `krylov_status::breakdown` when it makes `rᵀM⁻¹r` non-positive.
    template <scalar T>
    struct jacobi_preconditioner {
        std::span<const T> diagonal{};   ///< `dᵢ`, one per row; it must outlive the preconditioner.

        /// \brief `z = D⁻¹ r`. \param r The residual. \param z Overwritten.
        constexpr void apply(std::span<const T> r, std::span<T> z) const noexcept(detail::nothrow_scalar<T>)
        {
            for (std::size_t i = 0; i < r.size(); ++i)
                z[i] = diagonal[i] == T(0) ? r[i] : r[i] / diagonal[i];
        }
    };

    /// \brief The diagonal of a CSC matrix (a full matrix or an upper triangle): `out[j] = a(j, j)`, 0 where not stored.
    /// \ingroup core
    /// \tparam T The matrix's value type.
    /// \tparam I The index type.
    /// \tparam U The output's value type.
    /// \param a The matrix.
    /// \param out `min(a.rows, a.cols)` values, overwritten.
    template <typename T, typename I, typename U>
    constexpr void csc_diagonal(const sparse::csc_view<T, I>& a, std::span<U> out) noexcept
    {
        const std::size_t d = a.rows < a.cols ? a.rows : a.cols;
        for (std::size_t j = 0; j < d; ++j) {
            out[j] = U(0);
            for (std::size_t k = static_cast<std::size_t>(a.col_ptr[j]); k < static_cast<std::size_t>(a.col_ptr[j + 1]); ++k)
                if (static_cast<std::size_t>(a.row_idx[k]) == j)
                    out[j] += static_cast<U>(a.values[k]);
        }
    }

    /// \brief How a Krylov solve ended.
    /// \ingroup core
    enum class krylov_status : std::uint8_t {
        converged = 0,        ///< The residual met the tolerance.
        iteration_limit = 1,  ///< `max_iterations` ran out first; `x` is the last iterate.
        breakdown = 2,        ///< A curvature or preconditioner product was not positive: `A` (CG) or `M` is not positive definite.
        invalid_input = 3,    ///< Shapes that do not fit, or working storage too short; `x` is untouched.
    };

    /// \brief The stopping rule of a Krylov solve.
    /// \ingroup core
    /// \tparam T The value type.
    template <scalar T>
    struct krylov_options {
        T relative_tolerance = T(1e-8);  ///< Stop when `‖r‖ ≤ relative_tolerance · ‖b‖` ...
        T absolute_tolerance = T(0);     ///< ... or `‖r‖ ≤ absolute_tolerance`, whichever is larger.
        std::size_t max_iterations = 0;  ///< The most iterations; 0 for `2n`.
    };

    /// \brief The outcome of a Krylov solve.
    /// \ingroup core
    /// \tparam T The value type.
    template <scalar T>
    struct krylov_result {
        krylov_status status = krylov_status::invalid_input;  ///< How it ended.
        std::size_t iterations = 0;                           ///< Products with `A` after the initial residual's.
        T residual_norm = T(0);                               ///< The final residual's norm (CG: `‖r‖₂`; MINRES: `‖r‖_{M⁻¹}`).

        /// \brief Whether it converged. \return `status == krylov_status::converged`.
        [[nodiscard]] constexpr bool converged() const noexcept { return status == krylov_status::converged; }
    };

    /// \brief The working storage `conjugate_gradient` needs for `n` unknowns: `4n` values.
    /// \ingroup core
    /// \param n The unknowns.
    /// \return `4n`.
    [[nodiscard]] constexpr std::size_t conjugate_gradient_workspace(std::size_t n) noexcept { return 4 * n; }

    /// \brief The working storage `minres` needs for `n` unknowns: `7n` values.
    /// \ingroup core
    /// \param n The unknowns.
    /// \return `7n`.
    [[nodiscard]] constexpr std::size_t minres_workspace(std::size_t n) noexcept { return 7 * n; }

    namespace detail {
        template <typename T>
        constexpr T krylov_threshold(const krylov_options<T>& o, T bnorm) noexcept(nothrow_scalar<T>)
        {
            const T r = o.relative_tolerance * bnorm;
            return r > o.absolute_tolerance ? r : o.absolute_tolerance;
        }
        template <typename Op, typename M>
        inline constexpr bool nothrow_krylov =
            nothrow_apply<Op, typename Op::value_type> && nothrow_apply<M, typename Op::value_type> && nothrow_scalar<typename Op::value_type>;
    } // namespace detail

    /// \brief Solves `A x = b` for a symmetric positive definite `A` by the preconditioned conjugate gradient method.
    /// \ingroup core
    /// \tparam Op The operator.
    /// \tparam M The preconditioner.
    /// \param op `A`, square, `n x n`.
    /// \param b The right-hand side, `n` values.
    /// \param x The warm start on entry (zeros for none), the solution on exit; `n` values.
    /// \param work At least `conjugate_gradient_workspace(n)` values, not aliasing `b` or `x`.
    /// \param options The stopping rule.
    /// \param m The preconditioner, `M ≈ A`, SPD.
    /// \return The status, the iterations and `‖b - A x‖₂` by the recurrence.
    template <linear_operator Op, typename M = identity_preconditioner>
        requires preconditioner<M, typename Op::value_type>
    [[nodiscard]] constexpr krylov_result<typename Op::value_type> conjugate_gradient(const Op& op, std::span<const typename Op::value_type> b, std::span<typename Op::value_type> x,
                                                                                    std::span<typename Op::value_type> work, const krylov_options<typename Op::value_type>& options = {},
                                                                                    const M& m = {}) noexcept(detail::nothrow_krylov<Op, M>)
    {
        using T = typename Op::value_type;
        using result_t = krylov_result<T>;
        const std::size_t n = b.size();
        if (op.rows() != n || op.cols() != n || x.size() != n || work.size() < conjugate_gradient_workspace(n))
            return result_t{krylov_status::invalid_input, 0, T(0)};
        // Without a preconditioner z is r itself: no copy, the same numbers
        constexpr bool plain = std::is_same_v<M, identity_preconditioner>;
        const auto r = work.subspan(0, n), z = plain ? r : work.subspan(n, n), p = work.subspan(2 * n, n), q = work.subspan(3 * n, n);
        const T bnorm = detail::krylov_norm(b);
        if (bnorm == T(0)) {
            for (std::size_t i = 0; i < n; ++i)
                x[i] = T(0);
            return result_t{krylov_status::converged, 0, T(0)};
        }
        const T threshold = detail::krylov_threshold(options, bnorm);
        const std::size_t limit = options.max_iterations != 0 ? options.max_iterations : 2 * n;
        op.apply(std::span<const T>(x), q);
        for (std::size_t i = 0; i < n; ++i)
            r[i] = b[i] - q[i];
        T rnorm = detail::krylov_norm(std::span<const T>(r));
        if (rnorm <= threshold)
            return result_t{krylov_status::converged, 0, rnorm};
        if (!plain)
            m.apply(std::span<const T>(r), z);
        T rz = detail::krylov_dot(std::span<const T>(r), std::span<const T>(z));
        if (!(rz > T(0)))
            return result_t{krylov_status::breakdown, 0, rnorm};
        for (std::size_t i = 0; i < n; ++i)
            p[i] = z[i];
        for (std::size_t k = 1; k <= limit; ++k) {
            op.apply(std::span<const T>(p), q);
            const T pq = detail::krylov_dot(std::span<const T>(p), std::span<const T>(q));
            if (!(pq > T(0)))
                return result_t{krylov_status::breakdown, k - 1, rnorm};
            const T alpha = rz / pq;
            for (std::size_t i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
            }
            rnorm = detail::krylov_norm(std::span<const T>(r));
            if (rnorm <= threshold)
                return result_t{krylov_status::converged, k, rnorm};
            if (!plain)
                m.apply(std::span<const T>(r), z);
            const T rz_new = detail::krylov_dot(std::span<const T>(r), std::span<const T>(z));
            if (!(rz_new > T(0)))
                return result_t{krylov_status::breakdown, k, rnorm};
            const T beta = rz_new / rz;
            rz = rz_new;
            for (std::size_t i = 0; i < n; ++i)
                p[i] = z[i] + beta * p[i];
        }
        return result_t{krylov_status::iteration_limit, limit, rnorm};
    }

    /// \brief Solves `A x = b` for a symmetric `A`, definite or indefinite, by MINRES (Paige and Saunders): each
    /// iterate minimises `‖b - A x‖_{M⁻¹}` over the Krylov space.
    /// \ingroup core
    /// \tparam Op The operator.
    /// \tparam M The preconditioner.
    /// \param op `A`, square and symmetric, `n x n`.
    /// \param b The right-hand side, `n` values.
    /// \param x The warm start on entry (zeros for none), the solution on exit; `n` values.
    /// \param work At least `minres_workspace(n)` values, not aliasing `b` or `x`.
    /// \param options The stopping rule, against `‖b‖_{M⁻¹}`.
    /// \param m The preconditioner, SPD (even when `A` is indefinite).
    /// \return The status, the iterations and the residual estimate `φ̄ = ‖b - A x‖_{M⁻¹}` (`‖·‖₂` without a
    /// preconditioner).
    template <linear_operator Op, typename M = identity_preconditioner>
        requires preconditioner<M, typename Op::value_type>
    [[nodiscard]] constexpr krylov_result<typename Op::value_type> minres(const Op& op, std::span<const typename Op::value_type> b, std::span<typename Op::value_type> x,
                                                                        std::span<typename Op::value_type> work, const krylov_options<typename Op::value_type>& options = {},
                                                                        const M& m = {}) noexcept(detail::nothrow_krylov<Op, M>)
    {
        using T = typename Op::value_type;
        using result_t = krylov_result<T>;
        using cspan = std::span<const T>;
        const std::size_t n = b.size();
        if (op.rows() != n || op.cols() != n || x.size() != n || work.size() < minres_workspace(n))
            return result_t{krylov_status::invalid_input, 0, T(0)};
        auto v = work.subspan(0, n), y = work.subspan(n, n), r1 = work.subspan(2 * n, n), r2 = work.subspan(3 * n, n);
        auto w = work.subspan(4 * n, n), w1 = work.subspan(5 * n, n), w2 = work.subspan(6 * n, n);
        m.apply(b, y);
        const T bb = detail::krylov_dot(b, cspan(y));
        if (bb < T(0))
            return result_t{krylov_status::breakdown, 0, T(0)};
        if (bb == T(0)) {
            for (std::size_t i = 0; i < n; ++i)
                x[i] = T(0);
            return result_t{krylov_status::converged, 0, T(0)};
        }
        const T threshold = detail::krylov_threshold(options, static_cast<T>(math::sqrt(bb)));
        const std::size_t limit = options.max_iterations != 0 ? options.max_iterations : 2 * n;
        op.apply(cspan(x), y);
        for (std::size_t i = 0; i < n; ++i)
            r1[i] = r2[i] = b[i] - y[i];
        m.apply(cspan(r1), y);
        const T b1 = detail::krylov_dot(cspan(r1), cspan(y));
        if (b1 < T(0))
            return result_t{krylov_status::breakdown, 0, T(0)};
        const T beta1 = static_cast<T>(math::sqrt(b1));
        if (beta1 <= threshold)
            return result_t{krylov_status::converged, 0, beta1};
        for (std::size_t i = 0; i < n; ++i)
            w[i] = w2[i] = T(0);
        T oldb(0), beta = beta1, dbar(0), epsln(0), phibar = beta1, cs(-1), sn(0);
        for (std::size_t k = 1; k <= limit; ++k) {
            // Lanczos: v = y / β, y = A v - (β / β_old) r1 - (α / β) r2, then r1 ← r2, r2 ← y, y ← M⁻¹ r2
            const T s = T(1) / beta;
            for (std::size_t i = 0; i < n; ++i)
                v[i] = s * y[i];
            op.apply(cspan(v), y);
            if (k >= 2) {
                const T f = beta / oldb;
                for (std::size_t i = 0; i < n; ++i)
                    y[i] -= f * r1[i];
            }
            const T alfa = detail::krylov_dot(cspan(v), cspan(y));
            const T g = alfa / beta;
            for (std::size_t i = 0; i < n; ++i)
                y[i] -= g * r2[i];
            std::swap(r1, r2);   // r1 ← r2
            std::swap(r2, y);    // r2 ← the new vector; y takes the old r1's storage
            m.apply(cspan(r2), y);
            oldb = beta;
            const T bsq = detail::krylov_dot(cspan(r2), cspan(y));
            if (bsq < T(0))
                return result_t{krylov_status::breakdown, k - 1, phibar};
            beta = static_cast<T>(math::sqrt(bsq));
            // The tridiagonal's next column through the previous rotation, then a new rotation
            const T oldeps = epsln;
            const T delta = cs * dbar + sn * alfa;
            const T gbar = sn * dbar - cs * alfa;
            epsln = sn * beta;
            dbar = -cs * beta;
            T gamma = static_cast<T>(math::sqrt(gbar * gbar + beta * beta));
            gamma = gamma > std::numeric_limits<T>::epsilon() ? gamma : std::numeric_limits<T>::epsilon();
            cs = gbar / gamma;
            sn = beta / gamma;
            const T phi = cs * phibar;
            phibar = sn * phibar;
            // Directions: w1 ← w2, w2 ← w, w ← (v - ε w1 - δ w2) / γ
            const T denom = T(1) / gamma;
            std::swap(w1, w2);   // w1 ← w2
            std::swap(w2, w);    // w2 ← w; w takes the old w1's storage, overwritten next
            for (std::size_t i = 0; i < n; ++i) {
                w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) * denom;
                x[i] += phi * w[i];
            }
            if (phibar <= threshold)
                return result_t{krylov_status::converged, k, phibar};
        }
        return result_t{krylov_status::iteration_limit, limit, phibar};
    }

} // namespace snicholls::linalg

#endif /* snicholls_core_linalg_krylov_hpp */
