//
//  reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `linalg::reference_backend`: the standard-library-only backend, naive and correct, that every other backend is checked beside.
/// \ingroup core
///
/// **Overview.** Textbook algorithms written once, generic in the number type, over strided views: the BLAS
/// operations as loops, LU with partial pivoting, Cholesky and LDLT (Crout order), Householder QR, the cyclic
/// Jacobi method for the symmetric eigenproblem and the one-sided (Hestenes) Jacobi SVD. It models
/// `linalg::backend`, so it is what `linalg::lu_factor(linalg::reference, a, pivots)` and the batched solves run
/// when no other backend is chosen, and it is the baseline of the conformance suite
/// (`tests/support/snicholls/core/conformance/linalg.hpp`). Use it for small systems (the batched solves of
/// sizes up to about a hundred are its strength), on a device, at compile time, or where a dependency is not
/// allowed; use a tuned backend from a companion library for large dense matrices.
///
/// \snippet linalg/reference_demo.cpp reference
///
/// **Relationships.** Models `linalg::backend` (`linalg/backend.hpp`); its batched members run per system through
/// `linalg/batched.hpp`'s `for_each_system`. Group: \ref core.
///
/// **Implementations / variants.** One implementation per operation. Each member looks at its views' strides once
/// per call (once per system in a batch) and runs the same kernel on the accessor the compiler optimises best: a
/// packed square matrix (one run-time number for the order and the row stride, as a hand-written kernel has), rows
/// or columns with unit stride and a leading dimension, or any strides. A right-hand side prefers unit stride down
/// its columns, which the substitutions walk. A contiguous matrix therefore gets the code a hand-written loop gets.
///
/// **Guarantees.** Every member is `constexpr`, and `noexcept` when `T`'s arithmetic is (`float`, `double`; not an
/// allocating multiprecision type, whose `bad_alloc` propagates), except the batched ones, which call the caller's
/// executor; none allocates or uses workspace beyond its arguments; loops are bounded by the extents and, for the
/// iterative decompositions, by `max_sweeps`. The backend is a trivially copyable type holding one `double`
/// (`relative_tolerance`). **Determinism:** every sum runs in ascending index order in the argument's
/// own type, and nothing inside an operation runs in parallel, so a result is bit-for-bit the same on every call
/// and for every executor of a batch. Across compilers or flags it is the same only if floating-point contraction
/// (`-ffp-contract`) is.
///
/// | Operation | Algorithm | Cost |
/// |---|---|---|
/// | `dot`, `axpy` | one pass | n |
/// | `gemv`, `gemm`, `syrk` | inner products in index order | mn, mnk, n^2 k |
/// | `triangular_solve` | substitution | n^2 k |
/// | `lu_factor` / `lu_solve` | right-looking, partial pivoting (first largest), LAPACK's storage | 2n^3/3 / 2n^2 k |
/// | `cholesky_factor` / `cholesky_solve` | Cholesky-Crout, lower | n^3/3 / 2n^2 k |
/// | `ldlt_factor` / `ldlt_solve` | Crout, unpivoted, lower | n^3/3 / 2n^2 k |
/// | `qr_factor` / `qr_q` / `qr_solve` | Householder, LAPACK's storage (`v[0] = 1`, `H = I - tau v v^T`) | 2mn^2 |
/// | `symmetric_eigen` | cyclic Jacobi, at most `max_sweeps` sweeps | about 10 n^3 per sweep |
/// | `svd` | one-sided Jacobi, at most `max_sweeps` sweeps | about 6 m n^2 per sweep |
/// | `*_solve_batched` | the single-system kernels per system, through `for_each_system` | n times one system |
///
/// **Errors.** As `linalg/backend.hpp` documents for each operation: `singular` when a pivot magnitude is at or
/// below `pivot_tolerance` (a NaN pivot counts as zero), `not_positive_definite` when a Cholesky diagonal is,
/// `did_not_converge` when Jacobi uses `max_sweeps` or its input holds a NaN or an infinity (the Jacobi methods scale
/// by the largest element first, so no sum of squares overflows or underflows). A triangular solve reports `singular` only for an exact zero
/// on the diagonal, as LAPACK's does. On an error the matrix holds a partial factorisation and must not be used.
///
/// **Performance.** Against the same algorithms written by hand over raw pointers (`benchmarks/linalg/reference_benchmark.cpp`;
/// Release, AppleClang 17, Xeon W-3245, macOS 26, 2026-11-11; medians of five interleaved repetitions on a shared
/// machine with load average 7 to 9):
///
/// | Operation | Interface | By hand |
/// |---|---|---|
/// | `dot`, 4096 | 4.08 us | 4.06 us |
/// | `axpy`, 4096 | 1.82 us | 1.79 us |
/// | `gemv`, 64 x 64 | 2.64 us | 2.77 us |
/// | `gemm`, 64 x 64 | 168 us | 162 us |
/// | `syrk`, 64 x 64 | 82.3 us | 82.0 us |
/// | `triangular_solve`, 64 | 1.39 us | 1.38 us |
/// | `lu_factor` + `lu_solve`, 64 | 40.0 us | 46.7 us |
/// | `cholesky_factor` + `cholesky_solve`, 64 | 31.2 us | 31.0 us |
/// | `ldlt_factor` + `ldlt_solve`, 64 | 43.4 us | 40.7 us |
/// | `least_squares`, 96 x 32 | 82.1 us | 89.9 us |
/// | `qr_q`, 96 x 32 | 138 us | 134 us |
/// | `symmetric_eigen`, 16 | 59.1 us | 53.6 us |
/// | `svd`, 24 x 16 | 75.1 us | 96.1 us |
///
/// Not benchmarked: `lu_inverse` (one `lu_solve`) and the `log_det`s (one pass over a diagonal). Batched solves:
/// `linalg/batched.hpp`. See BENCHMARKS.md.
///
/// **Tests and demos.** `tests/linalg/reference_test.cpp` (the conformance suite, compile-time solves, the
/// layout contract), `demos/linalg/reference_demo.cpp`.
///
/// **Dependencies.** `<cmath>` (`fabs`), `<cstddef>`, `<limits>`, `<span>`, `<type_traits>`, `<utility>`, `linalg/backend.hpp`, `math/constexpr_math.hpp`.
///
/// **Known issues.** Naive by design: no blocking, no vectorised reductions (they would change the order of the
/// sums). LDLT is unpivoted. Its accuracy degrades with the element growth a small leading pivot causes (about
/// `epsilon / |d_min|` relative error, e.g. 2e-10 for `[[1e-6, 1, 0], [1, 0, 1], [0, 1, 1]]`, where LU gives
/// 4e-16), so it measures the growth and reports `singular` above `ldlt_growth_limit(n)`, and for an exactly zero
/// pivot; solve those with LU. Positive-definite matrices (growth at most 1) and bordered systems (growth about
/// `2 k` at most, `ldlt_growth_limit`) stay far below the limit. The growth check guards accuracy, not singularity, so
/// it applies whatever `relative_tolerance` is. The SVD's columns of U for zero singular values are zero, not
/// completed to an orthonormal basis. `symmetric_eigen` and `cholesky_factor` read only the lower triangle. QR scales `tau`
/// and `v`, but the reflector's inner product `w = b_k + sum v_i b_i` is not: inputs whose column sums exceed the
/// range (a tall column of values near the largest finite one) overflow there. An
/// eigenvalue or singular value beyond the type's range comes back as infinity, the correctly rounded value, with
/// success: `[[1e308, 1e308], [1e308, 1e308]]` has eigenvalues 0 and 2e308, returned as `{0, inf}`.
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_linalg_reference_hpp
#define snicholls_core_linalg_reference_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls::linalg {

    namespace detail {

        // A matrix whose rows are contiguous: element (i, j) at p[i * ld + j], the unit stride visible to the compiler.
        template <typename T>
        struct dense_rows {
            T* p;
            std::size_t rows, cols, ld;
            [[nodiscard]] constexpr T& operator()(std::size_t i, std::size_t j) const noexcept { return p[i * ld + j]; }
            [[nodiscard]] constexpr std::size_t extent(std::size_t d) const noexcept { return d == 0 ? rows : cols; }
        };

        // A matrix whose columns are contiguous: element (i, j) at p[i + j * ld].
        template <typename T>
        struct dense_cols {
            T* p;
            std::size_t rows, cols, ld;
            [[nodiscard]] constexpr T& operator()(std::size_t i, std::size_t j) const noexcept { return p[i + j * ld]; }
            [[nodiscard]] constexpr std::size_t extent(std::size_t d) const noexcept { return d == 0 ? rows : cols; }
        };

        // Run f on the fastest accessor for this matrix: one test per call, then a kernel whose inner stride is a constant.
        template <typename T, typename L, typename F>
        constexpr decltype(auto) dense(const md_span<T, 2, L>& a, F&& f)
        {
            if (a.stride(1) == 1)
                return f(dense_rows<T>{a.data_handle(), a.extent(0), a.extent(1), a.stride(0)});
            if (a.stride(0) == 1)
                return f(dense_cols<T>{a.data_handle(), a.extent(0), a.extent(1), a.stride(1)});
            return f(a);
        }

        // A vector with unit stride: element i at p[i].
        template <typename T>
        struct contiguous_vector {
            T* p;
            std::size_t n;
            [[nodiscard]] constexpr T& operator()(std::size_t i) const noexcept { return p[i]; }
            [[nodiscard]] constexpr std::size_t extent(std::size_t) const noexcept { return n; }
        };

        // As dense, for vectors.
        template <typename T, typename L, typename F>
        constexpr decltype(auto) dense_vector(const md_span<T, 1, L>& v, F&& f)
        {
            if (v.stride(0) == 1)
                return f(contiguous_vector<T>{v.data_handle(), v.extent(0)});
            return f(v);
        }

        // |x|: the hardware's at run time (one bit cleared), math::abs's when constant evaluated.
        template <typename T>
        constexpr T magnitude(T x)
        {
            if (std::is_constant_evaluated())
                return math::abs(x);
            using std::fabs;
            return fabs(x);
        }

        // A square matrix stored row after row with no padding: element (i, j) at p[i * n + j], one run-time number
        // for the order and the row stride, as a hand-written kernel has; the common case of the factorisations.
        template <typename T>
        struct packed_square {
            T* p;
            std::size_t n;
            [[nodiscard]] constexpr T& operator()(std::size_t i, std::size_t j) const noexcept { return p[i * n + j]; }
            [[nodiscard]] constexpr std::size_t extent(std::size_t) const noexcept { return n; }
        };

        // As dense, with the packed square case first: for the matrix a factorisation reads and writes.
        template <typename T, typename L, typename F>
        constexpr decltype(auto) dense_square(const md_span<T, 2, L>& a, F&& f)
        {
            if (a.extent(0) == a.extent(1) && a.stride(1) == 1 && a.stride(0) == a.extent(1))
                return f(packed_square<T>{a.data_handle(), a.extent(0)});
            return dense(a, std::forward<F>(f));
        }

        // As dense, for right-hand sides: the substitutions walk down a column, so a unit stride down the columns (a
        // vector, a column-major block) comes first.
        template <typename T, typename L, typename F>
        constexpr decltype(auto) dense_rhs(const md_span<T, 2, L>& b, F&& f)
        {
            if (b.stride(0) == 1)
                return f(dense_cols<T>{b.data_handle(), b.extent(0), b.extent(1), b.stride(1)});
            return dense(b, std::forward<F>(f));
        }

        // The largest |a(i, j)|, NaN if any element is NaN. No early exit: the loop stays branch-free and vectorises.
        template <typename T, typename M>
        constexpr T max_abs(M a)
        {
            T m = T(0);
            for (std::size_t i = 0; i < a.extent(0); ++i)
                for (std::size_t j = 0; j < a.extent(1); ++j) {
                    const T v = magnitude(a(i, j));
                    if (v > m || math::is_nan(v))   // once NaN, m stays NaN: no comparison with it is true
                        m = v;
                }
            return m;
        }

        template <typename T>
        constexpr void swap_values(T& a, T& b)
        {
            const T t = a;
            a = b;
            b = t;
        }

        // ---- BLAS -------------------------------------------------------------------------------------------

        template <typename T, typename X, typename Y>
        constexpr T dot(X x, Y y)
        {
            T s = T(0);
            for (std::size_t i = 0; i < x.extent(0); ++i)
                s += x(i) * y(i);
            return s;
        }

        template <typename T, typename A, typename X, typename Y>
        constexpr void gemv(T alpha, A a, X x, T beta, Y y)
        {
            for (std::size_t i = 0; i < a.extent(0); ++i) {
                T s = T(0);
                for (std::size_t j = 0; j < a.extent(1); ++j)
                    s += a(i, j) * x(j);
                y(i) = beta == T(0) ? alpha * s : alpha * s + beta * y(i);
            }
        }

        template <typename T, typename A, typename B, typename C>
        constexpr void gemm(T alpha, A a, B b, T beta, C c)
        {
            for (std::size_t i = 0; i < c.extent(0); ++i)
                for (std::size_t j = 0; j < c.extent(1); ++j) {
                    T s = T(0);
                    for (std::size_t p = 0; p < a.extent(1); ++p)
                        s += a(i, p) * b(p, j);
                    c(i, j) = beta == T(0) ? alpha * s : alpha * s + beta * c(i, j);
                }
        }

        template <typename T, typename A, typename C>
        constexpr void syrk(T alpha, A a, T beta, C c, triangle t)
        {
            const std::size_t n = c.extent(0);
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t first = t == triangle::lower ? 0 : i;
                const std::size_t last = t == triangle::lower ? i + 1 : n;
                for (std::size_t j = first; j < last; ++j) {
                    T s = T(0);
                    for (std::size_t p = 0; p < a.extent(1); ++p)
                        s += a(i, p) * a(j, p);
                    c(i, j) = beta == T(0) ? alpha * s : alpha * s + beta * c(i, j);
                }
            }
        }

        // Solve op(L) X = B in place, lower triangular, forward; unit or explicit diagonal.
        template <typename T, typename L, typename B>
        constexpr void lower_solve(L l, B b, bool unit)
        {
            const std::size_t n = l.extent(0);
            for (std::size_t c = 0; c < b.extent(1); ++c)
                for (std::size_t i = 0; i < n; ++i) {
                    T s = b(i, c);
                    for (std::size_t j = 0; j < i; ++j)
                        s -= l(i, j) * b(j, c);
                    b(i, c) = unit ? s : s / l(i, i);
                }
        }

        template <typename T, typename U, typename B>
        constexpr void upper_solve(U u, B b, bool unit)
        {
            const std::size_t n = u.extent(0);
            for (std::size_t c = 0; c < b.extent(1); ++c)
                for (std::size_t i = n; i-- > 0;) {
                    T s = b(i, c);
                    for (std::size_t j = i + 1; j < n; ++j)
                        s -= u(i, j) * b(j, c);
                    b(i, c) = unit ? s : s / u(i, i);
                }
        }

        // ---- LU ---------------------------------------------------------------------------------------------

        template <typename T, typename A>
        constexpr bool lu_factor(A a, std::span<pivot_index> pivots, T factor)
        {
            const std::size_t n = a.extent(0);
            const T tolerance = pivot_tolerance<T>(n, max_abs<T>(a)) * factor;
            for (std::size_t k = 0; k < n; ++k) {
                std::size_t p = k;
                T largest = magnitude(a(k, k));
                for (std::size_t i = k + 1; i < n; ++i) {
                    const T v = magnitude(a(i, k));
                    if (v > largest) {
                        largest = v;
                        p = i;
                    }
                }
                pivots[k] = static_cast<pivot_index>(p);
                if (!(largest > tolerance))   // zero, below the tolerance, or NaN
                    return false;
                if (p != k)
                    for (std::size_t j = 0; j < n; ++j)
                        swap_values(a(k, j), a(p, j));
                // Multiply by the reciprocal, as LAPACK does, unless the pivot is subnormal: its reciprocal overflows.
                const T pivot = a(k, k);
                if (magnitude(pivot) >= std::numeric_limits<T>::min()) {
                    const T inverse = T(1) / pivot;
                    for (std::size_t i = k + 1; i < n; ++i)
                        a(i, k) *= inverse;
                } else {
                    for (std::size_t i = k + 1; i < n; ++i)
                        a(i, k) = a(i, k) / pivot;
                }
                for (std::size_t i = k + 1; i < n; ++i) {
                    const T l = a(i, k);
                    for (std::size_t j = k + 1; j < n; ++j)
                        a(i, j) -= l * a(k, j);
                }
            }
            return true;
        }

        template <typename T, typename A, typename B>
        constexpr void lu_solve(A lu, std::span<const pivot_index> pivots, B b)
        {
            const std::size_t n = lu.extent(0);
            for (std::size_t k = 0; k < n; ++k) {
                const auto p = static_cast<std::size_t>(pivots[k]);
                if (p != k)
                    for (std::size_t c = 0; c < b.extent(1); ++c)
                        swap_values(b(k, c), b(p, c));
            }
            lower_solve<T>(lu, b, true);
            upper_solve<T>(lu, b, false);
        }

        // ---- Cholesky and LDLT ------------------------------------------------------------------------------

        template <typename T, typename A>
        constexpr bool cholesky_factor(A a, T factor)
        {
            const std::size_t n = a.extent(0);
            T scale = T(0);
            for (std::size_t i = 0; i < n; ++i) {
                const T d = a(i, i);
                if (d > scale || math::is_nan(d))
                    scale = d;
            }
            const T tolerance = pivot_tolerance<T>(n, scale) * factor;
            for (std::size_t j = 0; j < n; ++j) {
                T d = a(j, j);
                for (std::size_t k = 0; k < j; ++k)
                    d -= a(j, k) * a(j, k);
                if (!(d > tolerance))
                    return false;
                const T root = math::sqrt(d);
                a(j, j) = root;
                for (std::size_t i = j + 1; i < n; ++i) {
                    T s = a(i, j);
                    for (std::size_t k = 0; k < j; ++k)
                        s -= a(i, k) * a(j, k);
                    a(i, j) = s / root;
                }
            }
            return true;
        }

        // L^T X = B with L lower: back substitution reading L by columns.
        template <typename T, typename L, typename B>
        constexpr void lower_transpose_solve(L l, B b, bool unit)
        {
            const std::size_t n = l.extent(0);
            for (std::size_t c = 0; c < b.extent(1); ++c)
                for (std::size_t i = n; i-- > 0;) {
                    T s = b(i, c);
                    for (std::size_t j = i + 1; j < n; ++j)
                        s -= l(j, i) * b(j, c);
                    b(i, c) = unit ? s : s / l(i, i);
                }
        }

        template <typename T, typename A, typename B>
        constexpr void cholesky_solve(A l, B b)
        {
            lower_solve<T>(l, b, false);
            lower_transpose_solve<T>(l, b, false);
        }

        template <typename T, typename A>
        constexpr T max_abs_lower(A a)
        {
            T m = T(0);
            for (std::size_t i = 0; i < a.extent(0); ++i)
                for (std::size_t j = 0; j <= i; ++j) {
                    const T v = magnitude(a(i, j));
                    if (v > m || math::is_nan(v))   // once NaN, m stays NaN: no comparison with it is true
                        m = v;
                }
            return m;
        }

        template <typename T, typename A>
        constexpr bool ldlt_factor(A a, T factor)
        {
            const std::size_t n = a.extent(0);
            const T scale = max_abs_lower<T>(a);
            const T tolerance = pivot_tolerance<T>(n, scale) * factor;
            for (std::size_t j = 0; j < n; ++j) {
                T d = a(j, j);
                for (std::size_t k = 0; k < j; ++k)
                    d -= a(j, k) * (a(j, k) * a(k, k));
                if (!(magnitude(d) > tolerance))
                    return false;
                a(j, j) = d;
                for (std::size_t i = j + 1; i < n; ++i) {
                    T s = a(i, j);
                    for (std::size_t k = 0; k < j; ++k)
                        s -= a(i, k) * (a(j, k) * a(k, k));
                    a(i, j) = s / d;
                }
            }
            // Growth: sum_k l_ik^2 |d_k| bounds row i of |L| |D| |L^T|. Too large, and the answer would be inaccurate.
            T growth = T(0);
            for (std::size_t i = 0; i < n; ++i) {
                T g = magnitude(a(i, i));
                for (std::size_t k = 0; k < i; ++k)
                    g += a(i, k) * a(i, k) * magnitude(a(k, k));
                if (g > growth || math::is_nan(g))
                    growth = g;
            }
            return growth <= ldlt_growth_limit<T>(n) * scale;
        }

        template <typename T, typename A, typename B>
        constexpr void ldlt_solve(A ld, B b)
        {
            const std::size_t n = ld.extent(0);
            lower_solve<T>(ld, b, true);
            for (std::size_t c = 0; c < b.extent(1); ++c)
                for (std::size_t i = 0; i < n; ++i)
                    b(i, c) = b(i, c) / ld(i, i);
            lower_transpose_solve<T>(ld, b, true);
        }

        // ---- QR ---------------------------------------------------------------------------------------------

        // Whether x is neither NaN nor infinite.
        template <typename T>
        constexpr bool finite(T x)
        {
            return !math::is_nan(x) && !math::is_inf(x);
        }


        // sqrt(sum of squares) of a(first.., column), scaled so it neither overflows nor underflows.
        template <typename T, typename A>
        constexpr T column_norm(A a, std::size_t first, std::size_t column)
        {
            T scale = T(0);
            for (std::size_t i = first; i < a.extent(0); ++i)
                if (magnitude(a(i, column)) > scale)
                    scale = magnitude(a(i, column));
            if (scale == T(0))
                return T(0);
            T sum = T(0);
            for (std::size_t i = first; i < a.extent(0); ++i) {
                const T v = a(i, column) / scale;
                sum += v * v;
            }
            return scale * math::sqrt(sum);
        }

        template <typename T>
        constexpr T hypot2(T a, T b)
        {
            const T x = magnitude(a), y = magnitude(b);
            const T big = x > y ? x : y, small = x > y ? y : x;
            if (big == T(0))
                return T(0);
            const T r = small / big;
            return big * math::sqrt(T(1) + r * r);
        }

        // Apply H = I - tau v v^T (v[k] = 1, v[i] = h(i, k) below) to rows k.. of column j of b.
        template <typename T, typename H, typename B>
        constexpr void apply_reflector(H h, std::size_t k, T tau, B b, std::size_t j)
        {
            // b - tau v (v^T b), arranged so that no product exceeds the result: tau w can overflow (tau is up to 2)
            // where (b_k - w) + (1 - tau) w and (tau v_i) w do not, since |1 - tau| <= 1 and |tau v_i| <= 1.
            T w = b(k, j);
            for (std::size_t i = k + 1; i < h.extent(0); ++i)
                w += h(i, k) * b(i, j);
            b(k, j) = (b(k, j) - w) + (T(1) - tau) * w;
            for (std::size_t i = k + 1; i < h.extent(0); ++i)
                b(i, j) -= (tau * h(i, k)) * w;
        }

        template <typename T, typename A>
        constexpr bool qr_factor(A a, std::span<T> tau)
        {
            const std::size_t m = a.extent(0), n = a.extent(1);
            for (std::size_t i = 0; i < m; ++i)   // a NaN or an infinity is reported, never factored
                for (std::size_t j = 0; j < n; ++j)
                    if (!finite(a(i, j)))
                        return false;
            for (std::size_t k = 0; k < n; ++k) {
                const T x_norm = k + 1 < m ? column_norm<T>(a, k + 1, k) : T(0);
                const T alpha = a(k, k);
                if (x_norm == T(0)) {
                    tau[k] = T(0);
                    continue;
                }
                const T norm = hypot2(alpha, x_norm);
                const T beta = alpha >= T(0) ? -norm : norm;
                // tau = (beta - alpha) / beta and v = x / (alpha - beta), in forms that cannot overflow: |alpha / beta| <= 1,
                // and (alpha - beta) / norm is between 1 and 2, so neither a huge nor a subnormal column overflows.
                tau[k] = T(1) - alpha / beta;
                const T denominator = alpha / norm - beta / norm;
                for (std::size_t i = k + 1; i < m; ++i)
                    a(i, k) = a(i, k) / norm / denominator;
                a(k, k) = beta;
                for (std::size_t j = k + 1; j < n; ++j)
                    apply_reflector(a, k, tau[k], a, j);
            }
            return true;
        }

        template <typename T, typename A, typename Q>
        constexpr void qr_q(A qr, std::span<const T> tau, Q q)
        {
            const std::size_t m = q.extent(0), n = q.extent(1);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    q(i, j) = i == j ? T(1) : T(0);
            for (std::size_t k = n; k-- > 0;)
                for (std::size_t j = 0; j < n; ++j)
                    apply_reflector(qr, k, tau[k], q, j);
        }

        template <typename T, typename A, typename B>
        constexpr bool qr_solve(A qr, std::span<const T> tau, B b, T factor)
        {
            const std::size_t m = qr.extent(0), n = qr.extent(1);
            T largest = T(0);
            for (std::size_t k = 0; k < n; ++k)
                if (magnitude(qr(k, k)) > largest || math::is_nan(qr(k, k)))
                    largest = magnitude(qr(k, k));
            const T tolerance = pivot_tolerance<T>(m > n ? m : n, largest) * factor;
            for (std::size_t k = 0; k < n; ++k)
                if (!(magnitude(qr(k, k)) > tolerance))
                    return false;
            for (std::size_t i = 0; i < n; ++i)   // and R must be finite above its diagonal too
                for (std::size_t j = i + 1; j < n; ++j)
                    if (!finite(qr(i, j)))
                        return false;
            for (std::size_t k = 0; k < n; ++k)
                for (std::size_t c = 0; c < b.extent(1); ++c)
                    apply_reflector(qr, k, tau[k], b, c);
            for (std::size_t c = 0; c < b.extent(1); ++c)
                for (std::size_t i = n; i-- > 0;) {
                    T s = b(i, c);
                    for (std::size_t j = i + 1; j < n; ++j)
                        s -= qr(i, j) * b(j, c);
                    b(i, c) = s / qr(i, i);
                }
            return true;
        }

        // ---- Jacobi -----------------------------------------------------------------------------------------

        // The rotation (c, s) that the Jacobi methods use for t = sign(theta) / (|theta| + sqrt(theta^2 + 1)).
        template <typename T>
        constexpr void jacobi_rotation(T theta, T& c, T& s)
        {
            const T root = magnitude(theta) > T(1) ? magnitude(theta) * math::sqrt(T(1) + T(1) / (theta * theta))
                                                   : math::sqrt(T(1) + theta * theta);
            const T t = (theta >= T(0) ? T(1) : T(-1)) / (magnitude(theta) + root);
            c = T(1) / math::sqrt(T(1) + t * t);
            s = t * c;
        }

        // Swap columns i and j of a matrix.
        template <typename M>
        constexpr void swap_columns(M m, std::size_t i, std::size_t j)
        {
            for (std::size_t r = 0; r < m.extent(0); ++r)
                swap_values(m(r, i), m(r, j));
        }

        template <typename T, typename A, typename V>
        constexpr bool symmetric_eigen(A a, std::span<T> w, V v, int max_sweeps)
        {
            const std::size_t n = a.extent(0);
            // Scale by the largest element, so the sums of squares below can neither overflow nor underflow to zero
            // and pass the convergence test before any rotation. A NaN or an infinity is reported, never decomposed.
            T scale = T(0);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j <= i; ++j) {
                    const T x = magnitude(a(i, j));
                    if (!finite(x))
                        return false;
                    if (x > scale)
                        scale = x;
                }
            T norm2 = T(0);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    if (scale > T(0))
                        a(i, j) = j > i ? a(j, i) / scale : a(i, j) / scale;   // the lower triangle is the input
                    else if (j > i)
                        a(i, j) = a(j, i);
                    norm2 += a(i, j) * a(i, j);
                    v(i, j) = i == j ? T(1) : T(0);
                }
            const T eps = std::numeric_limits<T>::epsilon();
            bool converged = false;
            for (int sweep = 0; sweep <= max_sweeps && !converged; ++sweep) {
                T off = T(0);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = i + 1; j < n; ++j)
                        off += a(i, j) * a(i, j);
                if (!(off > eps * eps * norm2)) {   // also stops on an all-zero matrix
                    converged = !math::is_nan(off);
                    break;
                }
                if (sweep == max_sweeps)
                    break;
                for (std::size_t p = 0; p < n; ++p)
                    for (std::size_t q = p + 1; q < n; ++q) {
                        if (a(p, q) == T(0))
                            continue;
                        T c = T(1), s = T(0);
                        jacobi_rotation((a(q, q) - a(p, p)) / (T(2) * a(p, q)), c, s);
                        for (std::size_t k = 0; k < n; ++k) {   // A <- A J
                            const T akp = a(k, p), akq = a(k, q);
                            a(k, p) = c * akp - s * akq;
                            a(k, q) = s * akp + c * akq;
                        }
                        for (std::size_t k = 0; k < n; ++k) {   // A <- J^T A
                            const T apk = a(p, k), aqk = a(q, k);
                            a(p, k) = c * apk - s * aqk;
                            a(q, k) = s * apk + c * aqk;
                        }
                        for (std::size_t k = 0; k < n; ++k) {   // V <- V J
                            const T vkp = v(k, p), vkq = v(k, q);
                            v(k, p) = c * vkp - s * vkq;
                            v(k, q) = s * vkp + c * vkq;
                        }
                    }
            }
            for (std::size_t i = 0; i < n; ++i)
                w[i] = scale > T(0) ? a(i, i) * scale : a(i, i);
            for (std::size_t i = 0; i < n; ++i) {   // ascending; selection sort keeps equal values in index order
                std::size_t smallest = i;
                for (std::size_t j = i + 1; j < n; ++j)
                    if (w[j] < w[smallest])
                        smallest = j;
                if (smallest != i) {
                    swap_values(w[i], w[smallest]);
                    swap_columns(v, i, smallest);
                }
            }
            return converged;
        }

        template <typename T, typename A, typename V>
        constexpr bool svd(A a, std::span<T> s, V v, int max_sweeps)
        {
            const std::size_t m = a.extent(0), n = a.extent(1);
            // Scale by the largest element, as symmetric_eigen does, and report a NaN or an infinity.
            T scale = T(0);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    const T x = magnitude(a(i, j));
                    if (!finite(x))
                        return false;
                    if (x > scale)
                        scale = x;
                }
            if (scale > T(0))
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        a(i, j) = a(i, j) / scale;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    v(i, j) = i == j ? T(1) : T(0);
            const T eps = std::numeric_limits<T>::epsilon();
            bool converged = false;
            for (int sweep = 0; sweep < max_sweeps && !converged; ++sweep) {
                bool rotated = false;
                for (std::size_t p = 0; p < n; ++p)
                    for (std::size_t q = p + 1; q < n; ++q) {
                        T alpha = T(0), beta = T(0), gamma = T(0);
                        for (std::size_t i = 0; i < m; ++i) {
                            alpha += a(i, p) * a(i, p);
                            beta += a(i, q) * a(i, q);
                            gamma += a(i, p) * a(i, q);
                        }
                        if (math::is_nan(gamma) || math::is_nan(alpha) || math::is_nan(beta))
                            return false;
                        if (!(magnitude(gamma) > eps * math::sqrt(alpha) * math::sqrt(beta)))
                            continue;
                        rotated = true;
                        T c = T(1), sn = T(0);
                        jacobi_rotation((beta - alpha) / (T(2) * gamma), c, sn);
                        for (std::size_t i = 0; i < m; ++i) {
                            const T ap = a(i, p), aq = a(i, q);
                            a(i, p) = c * ap - sn * aq;
                            a(i, q) = sn * ap + c * aq;
                        }
                        for (std::size_t i = 0; i < n; ++i) {
                            const T vp = v(i, p), vq = v(i, q);
                            v(i, p) = c * vp - sn * vq;
                            v(i, q) = sn * vp + c * vq;
                        }
                    }
                converged = !rotated;
            }
            for (std::size_t j = 0; j < n; ++j) {
                const T norm = column_norm<T>(a, 0, j);
                for (std::size_t i = 0; i < m; ++i)
                    a(i, j) = norm > T(0) ? a(i, j) / norm : T(0);
                s[j] = norm * scale;
            }
            for (std::size_t i = 0; i < n; ++i) {   // descending
                std::size_t largest = i;
                for (std::size_t j = i + 1; j < n; ++j)
                    if (s[j] > s[largest])
                        largest = j;
                if (largest != i) {
                    swap_values(s[i], s[largest]);
                    swap_columns(a, i, largest);
                    swap_columns(v, i, largest);
                }
            }
            return converged;
        }

        // relative_tolerance as a factor in T: a negative value or a NaN counts as 0 (only an exact zero is singular),
        // so no setting can make a zero pivot pass; infinity reports every matrix singular.
        template <typename T>
        constexpr T tolerance_factor(double relative)
        {
            return relative > 0.0 ? static_cast<T>(relative) : T(0);
        }

        constexpr result<void, error> status_of(bool succeeded, error e) noexcept
        {
            if (succeeded)
                return snicholls::ok();
            return err(e);
        }

    } // namespace detail

    /// \brief The standard-library-only backend: naive, correct, `constexpr`, deterministic. Models `linalg::backend`.
    /// \ingroup core
    ///
    /// Members take the canonical views of `linalg::backend` (strided, `const` for inputs) and assume the
    /// extents were checked: call them through `linalg::lu_factor(linalg::reference, ...)` and the other free
    /// functions, which check.
    struct reference_backend {
        /// \brief The most Jacobi sweeps `symmetric_eigen` and `svd` run before reporting `did_not_converge`; a few
        ///        sweeps (about 6 to 10 for n below 100) are usual.
        static constexpr int max_sweeps = 64;

        /// \brief The multiple of `pivot_tolerance(n, scale)` below which a pivot counts as zero: 1 by default; 0 for
        ///        LAPACK's rule, where only an exact zero (or a NaN) is a singular pivot.
        ///
        /// The default judges a matrix and its multiple by 1e-30 alike but rejects one whose rows or columns differ in
        /// scale by more than about `1 / (n epsilon)`: `diag(1e16, 1)` is reported singular. Equilibrate such a matrix
        /// first (scale its rows and columns so their largest elements are 1, solve, and scale the solution back), or
        /// construct the backend with `reference_backend{.relative_tolerance = 0}`.
        ///
        /// A negative value or a NaN is clamped to 0, so no setting lets a zero pivot through (an unclamped -1 made
        /// `lu_factor` accept `diag(1, 0)` and `lu_solve` return infinities); infinity reports every matrix singular.
        ///
        /// It governs the zero-pivot tests only. LDLT's growth check (`ldlt_growth_limit`) applies in every mode,
        /// including 0: it guards the accuracy of a nonsingular matrix (`[[1e-6, 1, 0], [1, 0, 1], [0, 1, 1]]` would be
        /// solved to 2e-10), which no tolerance on pivots expresses, so LAPACK's rule is not a licence to answer
        /// inaccurately. Solve such a matrix with LU.
        double relative_tolerance = 1.0;

        /// \brief `sum x[i] y[i]`, in index order.
        /// \tparam T The number type.
        /// \param x A vector.
        /// \param y A vector of the same length.
        /// \return The sum.
        template <scalar T>
        [[nodiscard]] constexpr T dot(vector_view<const T> x, vector_view<const T> y) const noexcept(detail::nothrow_scalar<T>)
        {
            return detail::dense_vector(x, [&](const auto& xv) { return detail::dense_vector(y, [&](const auto& yv) { return detail::dot<T>(xv, yv); }); });
        }

        /// \brief `y += alpha x`.
        /// \tparam T The number type.
        /// \param alpha The multiplier.
        /// \param x     A vector.
        /// \param y     A vector of the same length, updated.
        template <scalar T>
        constexpr void axpy(T alpha, vector_view<const T> x, vector_view<T> y) const noexcept(detail::nothrow_scalar<T>)
        {
            detail::dense_vector(x, [&](const auto& xv) {
                detail::dense_vector(y, [&](const auto& yv) {
                    for (std::size_t i = 0; i < xv.extent(0); ++i)
                        yv(i) += alpha * xv(i);
                });
            });
        }

        /// \brief `y = alpha A x + beta y`; `y` is not read when `beta` is 0.
        /// \tparam T The number type.
        /// \param alpha The multiplier of `A x`.
        /// \param a     An `m x n` matrix.
        /// \param x     A vector of length `n`.
        /// \param beta  The multiplier of `y`.
        /// \param y     A vector of length `m`, overwritten.
        template <scalar T>
        constexpr void gemv(T alpha, matrix_view<const T> a, vector_view<const T> x, T beta, vector_view<T> y) const noexcept(detail::nothrow_scalar<T>)
        {
            detail::dense(a, [&](const auto& av) {
                detail::dense_vector(x, [&](const auto& xv) { detail::dense_vector(y, [&](const auto& yv) { detail::gemv<T>(alpha, av, xv, beta, yv); }); });
            });
        }

        /// \brief `C = alpha A B + beta C`; `C` is not read when `beta` is 0.
        /// \tparam T The number type.
        /// \param alpha The multiplier of `A B`.
        /// \param a     An `m x k` matrix.
        /// \param b     A `k x n` matrix.
        /// \param beta  The multiplier of `C`.
        /// \param c     An `m x n` matrix, overwritten.
        template <scalar T>
        constexpr void gemm(T alpha, matrix_view<const T> a, matrix_view<const T> b, T beta, matrix_view<T> c) const noexcept(detail::nothrow_scalar<T>)
        {
            detail::dense(a, [&](const auto& av) {
                detail::dense_rhs(b, [&](const auto& bv) { detail::dense(c, [&](const auto& cv) { detail::gemm<T>(alpha, av, bv, beta, cv); }); });
            });
        }

        /// \brief `C = alpha A A^T + beta C` on one triangle of `C`; the other is not touched.
        /// \tparam T The number type.
        /// \param alpha The multiplier of `A A^T`.
        /// \param a     An `n x k` matrix.
        /// \param beta  The multiplier of `C`.
        /// \param c     An `n x n` matrix, one triangle overwritten.
        /// \param t     Which triangle.
        template <scalar T>
        constexpr void syrk(T alpha, matrix_view<const T> a, T beta, matrix_view<T> c, triangle t) const noexcept(detail::nothrow_scalar<T>)
        {
            detail::dense(a, [&](const auto& av) { detail::dense(c, [&](const auto& cv) { detail::syrk<T>(alpha, av, beta, cv, t); }); });
        }

        /// \brief Solve `T X = B` in place for a triangular `T`; `singular` (and `B` unchanged) if an explicit diagonal holds a zero.
        /// \tparam T The number type.
        /// \param t     An `n x n` triangular matrix; only the triangle named is read.
        /// \param b     An `n x k` matrix, overwritten by `X`.
        /// \param which Which triangle `t` is.
        /// \param diag  Whether the diagonal is read or taken as ones.
        /// \return Nothing, or `error::singular`.
        template <scalar T>
        constexpr result<void, error> triangular_solve(matrix_view<const T> t, matrix_view<T> b, triangle which, diagonal diag) const noexcept(detail::nothrow_scalar<T>)
        {
            const bool unit = diag == diagonal::implicit_unit;
            if (!unit)
                for (std::size_t i = 0; i < t.extent(0); ++i)
                    if (t(i, i) == T(0))
                        return err(error::singular);
            detail::dense(t, [&](const auto& tv) {
                detail::dense_rhs(b, [&](const auto& bv) {
                    if (which == triangle::lower)
                        detail::lower_solve<T>(tv, bv, unit);
                    else
                        detail::upper_solve<T>(tv, bv, unit);
                });
            });
            return ok();
        }

        /// \brief LU with partial pivoting in place: `P A = L U`, unit `L` below the diagonal, `U` on and above, as LAPACK's `getrf`.
        /// \tparam T The number type.
        /// \param a      An `n x n` matrix, overwritten by its factors.
        /// \param pivots `n` row interchanges.
        /// \return Nothing, or `error::singular` when a pivot is at or below `pivot_tolerance(n, max |a_ij|)`.
        template <scalar T>
        constexpr result<void, error> lu_factor(matrix_view<T> a, std::span<pivot_index> pivots) const noexcept(detail::nothrow_scalar<T>)
        {
            return detail::status_of(detail::dense_square(a, [&](const auto& av) { return detail::lu_factor<T>(av, pivots, detail::tolerance_factor<T>(relative_tolerance)); }),
                                                         error::singular);
        }

        /// \brief Solve `A X = B` in place from `lu_factor`'s factors.
        /// \tparam T The number type.
        /// \param lu     The factors.
        /// \param pivots The interchanges.
        /// \param b      An `n x k` matrix, overwritten by `X`.
        template <scalar T>
        constexpr void lu_solve(matrix_view<const T> lu, std::span<const pivot_index> pivots, matrix_view<T> b) const noexcept(detail::nothrow_scalar<T>)
        {
            detail::dense_square(lu, [&](const auto& av) { detail::dense_rhs(b, [&](const auto& bv) { detail::lu_solve<T>(av, pivots, bv); }); });
        }

        /// \brief Cholesky in place, `A = L L^T`, reading and writing the lower triangle; the strictly upper triangle is not touched.
        /// \tparam T The number type.
        /// \param a An `n x n` symmetric matrix, its lower triangle overwritten by `L`.
        /// \return Nothing, or `error::not_positive_definite` when a diagonal is at or below `pivot_tolerance(n, max a_ii)`.
        template <scalar T>
        constexpr result<void, error> cholesky_factor(matrix_view<T> a) const noexcept(detail::nothrow_scalar<T>)
        {
            return detail::status_of(detail::dense_square(a, [&](const auto& av) { return detail::cholesky_factor<T>(av, detail::tolerance_factor<T>(relative_tolerance)); }),
                                                         error::not_positive_definite);
        }

        /// \brief Solve `A X = B` in place from `cholesky_factor`'s `L`.
        /// \tparam T The number type.
        /// \param l The factor (lower triangle read).
        /// \param b An `n x k` matrix, overwritten by `X`.
        template <scalar T>
        constexpr void cholesky_solve(matrix_view<const T> l, matrix_view<T> b) const noexcept(detail::nothrow_scalar<T>)
        {
            detail::dense_square(l, [&](const auto& lv) { detail::dense_rhs(b, [&](const auto& bv) { detail::cholesky_solve<T>(lv, bv); }); });
        }

        /// \brief LDLT without pivoting in place: unit `L` strictly below the diagonal, `D` on it; the strictly upper triangle is not touched.
        /// \tparam T The number type.
        /// \param a An `n x n` symmetric matrix, possibly indefinite; its lower triangle is read and overwritten.
        /// \return Nothing, or `error::singular` when `|d_j|` is at or below `pivot_tolerance(n, max |a_ij|)` or the
        ///         growth `max_i sum_k l_ik^2 |d_k|` exceeds `ldlt_growth_limit(n) * max |a_ij|`.
        template <scalar T>
        constexpr result<void, error> ldlt_factor(matrix_view<T> a) const noexcept(detail::nothrow_scalar<T>)
        {
            return detail::status_of(detail::dense_square(a, [&](const auto& av) { return detail::ldlt_factor<T>(av, detail::tolerance_factor<T>(relative_tolerance)); }),
                                                         error::singular);
        }

        /// \brief Solve `A X = B` in place from `ldlt_factor`'s factors.
        /// \tparam T The number type.
        /// \param ld The factors.
        /// \param b  An `n x k` matrix, overwritten by `X`.
        template <scalar T>
        constexpr void ldlt_solve(matrix_view<const T> ld, matrix_view<T> b) const noexcept(detail::nothrow_scalar<T>)
        {
            detail::dense_square(ld, [&](const auto& lv) { detail::dense_rhs(b, [&](const auto& bv) { detail::ldlt_solve<T>(lv, bv); }); });
        }

        /// \brief Householder QR in place, as LAPACK's `geqrf`: `R` on and above the diagonal, the reflectors below.
        /// \tparam T The number type.
        /// \param a   An `m x n` matrix, `m >= n`, overwritten.
        /// \param tau `n` reflector scales.
        /// \return Nothing, or `error::singular` when `A` holds a NaN or an infinity. A rank-deficient `A` factors,
        ///         and `qr_solve` reports it.
        template <scalar T>
        constexpr result<void, error> qr_factor(matrix_view<T> a, std::span<T> tau) const noexcept(detail::nothrow_scalar<T>)
        {
            return detail::status_of(detail::dense(a, [&](const auto& av) { return detail::qr_factor<T>(av, tau); }), error::singular);
        }

        /// \brief The thin `Q` (`m x n`, orthonormal columns) of `qr_factor`'s factors.
        /// \tparam T The number type.
        /// \param qr  The factors.
        /// \param tau The reflector scales.
        /// \param q   An `m x n` matrix, overwritten.
        template <scalar T>
        constexpr void qr_q(matrix_view<const T> qr, std::span<const T> tau, matrix_view<T> q) const noexcept(detail::nothrow_scalar<T>)
        {
            detail::dense(qr, [&](const auto& av) { detail::dense(q, [&](const auto& qv) { detail::qr_q<T>(av, tau, qv); }); });
        }

        /// \brief The least-squares solution of `A X = B` from `qr_factor`'s factors: `B`'s first `n` rows become `X`.
        /// \tparam T The number type.
        /// \param qr  The factors of an `m x n` `A`.
        /// \param tau The reflector scales.
        /// \param b   An `m x k` matrix, overwritten: rows `0..n` hold `X`, the rest `Q^T B`'s residual part.
        /// \return Nothing, or `error::singular` (and `B` unchanged) when a diagonal of `R` is at or below
        ///         `pivot_tolerance(max(m, n), max |r_ii|)`: `A` is rank deficient.
        template <scalar T>
        constexpr result<void, error> qr_solve(matrix_view<const T> qr, std::span<const T> tau, matrix_view<T> b) const noexcept(detail::nothrow_scalar<T>)
        {
            return detail::status_of(
                detail::dense(qr, [&](const auto& av) { return detail::dense_rhs(b, [&](const auto& bv) { return detail::qr_solve<T>(av, tau, bv, detail::tolerance_factor<T>(relative_tolerance)); }); }),
                error::singular);
        }

        /// \brief Symmetric eigendecomposition `A = V diag(w) V^T` by cyclic Jacobi: eigenvalues ascending, eigenvectors in `V`'s columns.
        /// \tparam T The number type.
        /// \param a An `n x n` symmetric matrix (lower triangle read); destroyed.
        /// \param w `n` eigenvalues, ascending.
        /// \param v An `n x n` matrix, overwritten by orthonormal eigenvectors.
        /// \return Nothing, or `error::did_not_converge` after `max_sweeps` sweeps (or on a NaN).
        template <scalar T>
        constexpr result<void, error> symmetric_eigen(matrix_view<T> a, std::span<T> w, matrix_view<T> v) const noexcept(detail::nothrow_scalar<T>)
        {
            return detail::status_of(
                detail::dense_square(a, [&](const auto& av) {
                    return detail::dense_square(v, [&](const auto& vv) { return detail::symmetric_eigen<T>(av, w, vv, max_sweeps); });
                }),
                error::did_not_converge);
        }

        /// \brief Thin singular value decomposition `A = U diag(s) V^T` by one-sided Jacobi; `A` becomes `U`.
        /// \tparam T The number type.
        /// \param a An `m x n` matrix, `m >= n`, overwritten by `U` (orthonormal columns where `s > 0`).
        /// \param s `n` singular values, descending.
        /// \param v An `n x n` matrix, overwritten by `V`.
        /// \return Nothing, or `error::did_not_converge` after `max_sweeps` sweeps (or on a NaN).
        template <scalar T>
        constexpr result<void, error> svd(matrix_view<T> a, std::span<T> s, matrix_view<T> v) const noexcept(detail::nothrow_scalar<T>)
        {
            return detail::status_of(detail::dense(a, [&](const auto& av) { return detail::dense_square(v, [&](const auto& vv) { return detail::svd<T>(av, s, vv, max_sweeps); }); }),
                                                         error::did_not_converge);
        }

        /// \brief Factorise every system by LU and solve it, through the executor; see `linalg::lu_solve_batched`.
        /// \tparam T The number type.
        /// \tparam E The executor.
        /// \param a      `n x m x m` matrices, overwritten.
        /// \param pivots `n * m` interchanges.
        /// \param b      `n x m x k` right-hand sides, overwritten by the solutions.
        /// \param status `n` statuses: `ok` or `singular`.
        /// \param ex     The executor.
        /// \return The number of systems not solved.
        template <scalar T, range_executor E>
        constexpr result<std::size_t, error> lu_solve_batched(batch_view<T> a, std::span<pivot_index> pivots, batch_view<T> b, std::span<status> status,
                                                              E ex) const
        {
            const std::size_t m = a.extent(1);
            for_each_system(ex, a.extent(0), [&](std::size_t i) {
                const matrix_view<T> ai = containers::submdspan(a, i, containers::full_extent, containers::full_extent);
                const matrix_view<T> bi = containers::submdspan(b, i, containers::full_extent, containers::full_extent);
                const std::span<pivot_index> pi = pivots.subspan(i * m, m);
                const bool solved = detail::dense_square(ai, [&](const auto& av) {
                    if (!detail::lu_factor<T>(av, pi, detail::tolerance_factor<T>(relative_tolerance)))
                        return false;
                    detail::dense_rhs(bi, [&](const auto& bv) { detail::lu_solve<T>(av, pi, bv); });
                    return true;
                });
                status[i] = solved ? linalg::status::ok : linalg::status::singular;
            });
            return ok(count_failures(status));
        }

        /// \brief Factorise every system by Cholesky and solve it, through the executor; see `linalg::cholesky_solve_batched`.
        /// \tparam T The number type.
        /// \tparam E The executor.
        /// \param a      `n x m x m` symmetric matrices, overwritten.
        /// \param b      `n x m x k` right-hand sides, overwritten by the solutions.
        /// \param status `n` statuses: `ok` or `not_positive_definite`.
        /// \param ex     The executor.
        /// \return The number of systems not solved.
        template <scalar T, range_executor E>
        constexpr result<std::size_t, error> cholesky_solve_batched(batch_view<T> a, batch_view<T> b, std::span<status> status, E ex) const
        {
            for_each_system(ex, a.extent(0), [&](std::size_t i) {
                const matrix_view<T> ai = containers::submdspan(a, i, containers::full_extent, containers::full_extent);
                const matrix_view<T> bi = containers::submdspan(b, i, containers::full_extent, containers::full_extent);
                const bool solved = detail::dense_square(ai, [&](const auto& av) {
                    if (!detail::cholesky_factor<T>(av, detail::tolerance_factor<T>(relative_tolerance)))
                        return false;
                    detail::dense_rhs(bi, [&](const auto& bv) { detail::cholesky_solve<T>(av, bv); });
                    return true;
                });
                status[i] = solved ? linalg::status::ok : linalg::status::not_positive_definite;
            });
            return ok(count_failures(status));
        }

        /// \brief Factorise every system by LDLT and solve it, through the executor; see `linalg::ldlt_solve_batched`.
        /// \tparam T The number type.
        /// \tparam E The executor.
        /// \param a      `n x m x m` symmetric matrices, overwritten.
        /// \param b      `n x m x k` right-hand sides, overwritten by the solutions.
        /// \param status `n` statuses: `ok` or `singular`.
        /// \param ex     The executor.
        /// \return The number of systems not solved.
        template <scalar T, range_executor E>
        constexpr result<std::size_t, error> ldlt_solve_batched(batch_view<T> a, batch_view<T> b, std::span<status> status, E ex) const
        {
            for_each_system(ex, a.extent(0), [&](std::size_t i) {
                const matrix_view<T> ai = containers::submdspan(a, i, containers::full_extent, containers::full_extent);
                const matrix_view<T> bi = containers::submdspan(b, i, containers::full_extent, containers::full_extent);
                const bool solved = detail::dense_square(ai, [&](const auto& av) {
                    if (!detail::ldlt_factor<T>(av, detail::tolerance_factor<T>(relative_tolerance)))
                        return false;
                    detail::dense_rhs(bi, [&](const auto& bv) { detail::ldlt_solve<T>(av, bv); });
                    return true;
                });
                status[i] = solved ? linalg::status::ok : linalg::status::singular;
            });
            return ok(count_failures(status));
        }
    };

    /// \brief The reference backend, as a value to pass: `linalg::lu_factor(linalg::reference, a, pivots)`.
    /// \ingroup core
    inline constexpr reference_backend reference{};

    static_assert(backend<reference_backend, float> && backend<reference_backend, double> && backend<reference_backend, long double>);
    static_assert(std::is_trivially_copyable_v<reference_backend> && sizeof(reference_backend) == sizeof(double));

} // namespace snicholls::linalg

#endif /* snicholls_core_linalg_reference_hpp */
