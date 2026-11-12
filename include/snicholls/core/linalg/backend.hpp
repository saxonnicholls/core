//
//  backend.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The linear-algebra interface: the `linalg::backend` concepts and the free functions that check extents and call a backend, resolved at compile time.
/// \ingroup core
///
/// **Overview.** Callers write against one set of free functions - `linalg::dot`, `gemm`, `lu_factor`,
/// `cholesky_solve`, `symmetric_eigen`, ... - and pass the backend as the first argument. A backend is any type
/// whose members model the concepts below: core's `reference_backend`, or one from a companion library binding
/// BLAS and LAPACK, Eigen or an accelerator. Everything is resolved at compile time: the free function is a
/// template, the backend's member is called directly and inlines, and there is no virtual call, function pointer
/// or type erasure anywhere, so a hot loop over a million small systems costs what the hand-written loop costs.
///
/// \snippet linalg/backend_demo.cpp backend
///
/// The free functions take any `md_span` layout (row-major, column-major, strided) and any right-hand side as a
/// vector or a matrix, check that the extents fit together (`error::dimension_mismatch` otherwise), and hand the
/// backend the canonical views: `matrix_view<T>` / `vector_view<T>` (strided), `const` for inputs. A backend
/// therefore implements one member per operation and never sees a layout it was not written for.
///
/// **Choosing a backend at run time.** There is deliberately no run-time adapter. A program that picks its
/// backend at run time (a GPU backend if a device is present, otherwise BLAS) branches once, at the top, between
/// two instantiations of its own generic code, and each instantiation is resolved at compile time:
///
/// \snippet linalg/backend_demo.cpp run_time_choice
///
/// **The contract a backend implements** (each member `const`; `T` is `float`, `double` or another `scalar`;
/// `M<T>` is `matrix_view<T>`, `V<T>` is `vector_view<T>`, `R` is `result<void, error>`). Extents are checked
/// before a member is called, so a member may assume them.
///
/// | Concept | Member | Effect |
/// |---|---|---|
/// | `blas_backend` | `T dot(V<const T> x, V<const T> y)` | `sum x_i y_i` |
/// | | `void axpy(T alpha, V<const T> x, V<T> y)` | `y += alpha x` |
/// | | `void gemv(T alpha, M<const T> a, V<const T> x, T beta, V<T> y)` | `y = alpha A x + beta y` |
/// | | `void gemm(T alpha, M<const T> a, M<const T> b, T beta, M<T> c)` | `C = alpha A B + beta C` |
/// | | `void syrk(T alpha, M<const T> a, T beta, M<T> c, triangle t)` | `C = alpha A A^T + beta C`, one triangle |
/// | | `R triangular_solve(M<const T> t, M<T> b, triangle, diagonal)` | `B = T^-1 B` |
/// | `factorisation_backend` | `R lu_factor(M<T> a, std::span<pivot_index> p)` | `P A = L U` in place |
/// | | `void lu_solve(M<const T> lu, std::span<const pivot_index> p, M<T> b)` | `B = A^-1 B` |
/// | | `R cholesky_factor(M<T> a)` | `A = L L^T`, lower |
/// | | `void cholesky_solve(M<const T> l, M<T> b)` | `B = A^-1 B` |
/// | | `R ldlt_factor(M<T> a)` | `A = L D L^T`, unit lower `L` |
/// | | `void ldlt_solve(M<const T> ld, M<T> b)` | `B = A^-1 B` |
/// | | `R qr_factor(M<T> a, std::span<T> tau)` | `A = Q R`, Householder |
/// | | `void qr_q(M<const T> qr, std::span<const T> tau, M<T> q)` | the thin `Q` |
/// | | `R qr_solve(M<const T> qr, std::span<const T> tau, M<T> b)` | least squares into `B`'s first `n` rows |
/// | `decomposition_backend` | `R symmetric_eigen(M<T> a, std::span<T> w, M<T> v)` | `A = V diag(w) V^T`, `w` ascending |
/// | | `R svd(M<T> a, std::span<T> s, M<T> v)` | `A = U diag(s) V^T`, `A` becomes `U`, `s` descending |
/// | `batched_backend` | see `linalg/batched.hpp` | factorise and solve `n` systems, one status each |
///
/// `backend<B, T>` is all four. Storage follows LAPACK so that `lu_log_det`, `cholesky_log_det`, `ldlt_log_det`
/// and the conformance suite read any backend's factors: LU as `getrf` (0-based pivots), Cholesky and LDLT in the
/// lower triangle, QR as `geqrf` (`R` on and above the diagonal). `log_det`s and `lu_inverse` are written once,
/// here, on top of those members.
///
/// **Relationships.** `linalg/types.hpp` (the views and codes), `linalg/reference.hpp` (core's backend),
/// `linalg/batched.hpp` (batches), `result.hpp`. The conformance suite every backend registers in is
/// `tests/support/snicholls/core/conformance/linalg.hpp` (CMake target `snicholls::core_conformance`).
/// Group: \ref core.
///
/// **Implementations / variants.** In core: `reference_backend` (standard library only, naive, `constexpr`,
/// deterministic). Third-party backends live in companion libraries, never in core.
///
/// **Guarantees.** The free functions add a few comparisons of extents and nothing else; they are `constexpr`,
/// not declared `noexcept`, and throw only what the backend's member throws (the reference backend's members are
/// `noexcept` for `float` and `double`). A backend's own guarantees (determinism, threads, allocation)
/// are its own: the reference backend's are in `linalg/reference.hpp`.
///
/// **Errors.** `error::dimension_mismatch` from the extent checks and `error::aliased` when an output view starts
/// where an input does (`gemm(.., a, .., a)`, `symmetric_eigen(a, w, a)`), both before the backend is called and
/// with every argument unchanged. The span outputs (`pivots`, `tau`, `w`, `s`) are checked the same way. Other overlap
/// (one view inside another, interleaved strides) is not detected: every function's `\pre` forbids it; otherwise what the member returns (see each function). A backend type that lacks a member
/// is a compile error naming the member it lacks (a `static_assert` with a `snicholls::linalg::` message).
///
/// **Performance.** The concept path compiles to the backend's member called directly. Every member of the
/// reference backend's concepts is benchmarked through these functions against the same algorithm written by hand,
/// after a bit-for-bit fairness check; `lu_inverse` (one `lu_solve`) and the three `log_det`s (one pass over a
/// diagonal) are not benchmarked: `benchmarks/linalg/reference_benchmark.cpp` and `batched_benchmark.cpp`; the figures
/// (Release, AppleClang 17, Xeon W-3245, macOS 26, 2026-11-11, on a shared machine with load average 7 to 10) are in `linalg/reference.hpp`, `linalg/batched.hpp` and BENCHMARKS.md.
///
/// **Tests and demos.** `tests/linalg/backend_test.cpp` (the concepts, a backend missing an operation, the extent
/// checks, a recording backend), the compile-fail case `tests/compile_fail/linalg_backend_missing_member.cpp`,
/// `demos/linalg/backend_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<limits>`, `<span>`, `<type_traits>`, `result.hpp`, `linalg/types.hpp`,
/// `linalg/batched.hpp`.
///
/// **Known issues.** Complex numbers are not supported yet (no conjugate transposes).
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_linalg_backend_hpp
#define snicholls_core_linalg_backend_hpp

#include <snicholls/core/linalg/batched.hpp>
#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/result.hpp>

#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::linalg {

    namespace detail {
        template <typename T>
        using cm = matrix_view<const T>;
        template <typename T>
        using mm = matrix_view<T>;
        template <typename T>
        using cv = vector_view<const T>;
        template <typename T>
        using mv = vector_view<T>;
        using r = result<void, error>;

        template <typename B, typename T>
        concept has_dot = requires(const B& b, cv<T> x) { { b.dot(x, x) } -> std::same_as<T>; };
        template <typename B, typename T>
        concept has_axpy = requires(const B& b, T s, cv<T> x, mv<T> y) { b.axpy(s, x, y); };
        template <typename B, typename T>
        concept has_gemv = requires(const B& b, T s, cm<T> a, cv<T> x, mv<T> y) { b.gemv(s, a, x, s, y); };
        template <typename B, typename T>
        concept has_gemm = requires(const B& b, T s, cm<T> a, cm<T> x, mm<T> c) { b.gemm(s, a, x, s, c); };
        template <typename B, typename T>
        concept has_syrk = requires(const B& b, T s, cm<T> a, mm<T> c, triangle t) { b.syrk(s, a, s, c, t); };
        template <typename B, typename T>
        concept has_triangular_solve = requires(const B& b, cm<T> a, mm<T> c, triangle t, diagonal d) {
            { b.triangular_solve(a, c, t, d) } -> std::same_as<r>;
        };
        template <typename B, typename T>
        concept has_lu_factor = requires(const B& b, mm<T> a, std::span<pivot_index> p) { { b.lu_factor(a, p) } -> std::same_as<r>; };
        template <typename B, typename T>
        concept has_lu_solve = requires(const B& b, cm<T> a, std::span<const pivot_index> p, mm<T> c) { b.lu_solve(a, p, c); };
        template <typename B, typename T>
        concept has_cholesky_factor = requires(const B& b, mm<T> a) { { b.cholesky_factor(a) } -> std::same_as<r>; };
        template <typename B, typename T>
        concept has_cholesky_solve = requires(const B& b, cm<T> a, mm<T> c) { b.cholesky_solve(a, c); };
        template <typename B, typename T>
        concept has_ldlt_factor = requires(const B& b, mm<T> a) { { b.ldlt_factor(a) } -> std::same_as<r>; };
        template <typename B, typename T>
        concept has_ldlt_solve = requires(const B& b, cm<T> a, mm<T> c) { b.ldlt_solve(a, c); };
        template <typename B, typename T>
        concept has_qr_factor = requires(const B& b, mm<T> a, std::span<T> tau) { { b.qr_factor(a, tau) } -> std::same_as<r>; };
        template <typename B, typename T>
        concept has_qr_q = requires(const B& b, cm<T> a, std::span<const T> tau, mm<T> q) { b.qr_q(a, tau, q); };
        template <typename B, typename T>
        concept has_qr_solve = requires(const B& b, cm<T> a, std::span<const T> tau, mm<T> c) { { b.qr_solve(a, tau, c) } -> std::same_as<r>; };
        template <typename B, typename T>
        concept has_symmetric_eigen = requires(const B& b, mm<T> a, std::span<T> w, mm<T> v) { { b.symmetric_eigen(a, w, v) } -> std::same_as<r>; };
        template <typename B, typename T>
        concept has_svd = requires(const B& b, mm<T> a, std::span<T> s, mm<T> v) { { b.svd(a, s, v) } -> std::same_as<r>; };

        template <typename E>
        using value_t = std::remove_const_t<E>;

        // Any rank-1 or rank-2 view as the canonical strided matrix, so a right-hand side may be a vector.
        template <typename E, typename L>
        constexpr matrix_view<E> as_matrix(const md_span<E, 2, L>& m) noexcept
        {
            return m;
        }
        template <typename E, typename L>
        constexpr matrix_view<E> as_matrix(const md_span<E, 1, L>& v) noexcept
        {
            return as_column(v);
        }

        constexpr bool is_square(std::size_t rows, std::size_t cols) noexcept { return rows == cols; }
    } // namespace detail

    /// \brief A backend with the BLAS operations for `T`: `dot`, `axpy`, `gemv`, `gemm`, `syrk`, `triangular_solve`.
    /// \ingroup core
    /// \tparam B The backend type: trivially copyable, its members `const`.
    /// \tparam T The number type.
    template <typename B, typename T>
    concept blas_backend = scalar<T> && std::is_trivially_copyable_v<B> && detail::has_dot<B, T> && detail::has_axpy<B, T> &&
                           detail::has_gemv<B, T> && detail::has_gemm<B, T> && detail::has_syrk<B, T> && detail::has_triangular_solve<B, T>;

    /// \brief A backend with the factorisations for `T`: LU, Cholesky, LDLT and QR, each with its solve.
    /// \ingroup core
    /// \tparam B The backend type.
    /// \tparam T The number type.
    template <typename B, typename T>
    concept factorisation_backend = scalar<T> && std::is_trivially_copyable_v<B> && detail::has_lu_factor<B, T> && detail::has_lu_solve<B, T> &&
                                    detail::has_cholesky_factor<B, T> && detail::has_cholesky_solve<B, T> && detail::has_ldlt_factor<B, T> &&
                                    detail::has_ldlt_solve<B, T> && detail::has_qr_factor<B, T> && detail::has_qr_q<B, T> &&
                                    detail::has_qr_solve<B, T>;

    /// \brief A backend with the decompositions for `T`: the symmetric eigenproblem and the SVD.
    /// \ingroup core
    /// \tparam B The backend type.
    /// \tparam T The number type.
    template <typename B, typename T>
    concept decomposition_backend = scalar<T> && std::is_trivially_copyable_v<B> && detail::has_symmetric_eigen<B, T> && detail::has_svd<B, T>;

    /// \brief A complete backend for `T`: every operation of the interface.
    /// \ingroup core
    /// \tparam B The backend type.
    /// \tparam T The number type.
    template <typename B, typename T>
    concept backend = blas_backend<B, T> && factorisation_backend<B, T> && decomposition_backend<B, T> && batched_backend<B, T>;

    // ---- BLAS -----------------------------------------------------------------------------------------------

    /// \brief `sum x_i y_i`.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EX `x`'s element type.
    /// \tparam LX `x`'s layout.
    /// \tparam EY `y`'s element type.
    /// \tparam LY `y`'s layout.
    /// \param b The backend.
    /// \param x A vector.
    /// \param y A vector.
    /// \return The sum, or `error::dimension_mismatch` if the lengths differ.
    template <typename B, typename EX, typename LX, typename EY, typename LY>
        requires(scalar<detail::value_t<EX>> && std::same_as<detail::value_t<EX>, detail::value_t<EY>>)
    [[nodiscard]] constexpr result<detail::value_t<EX>, error> dot(const B& b, const md_span<EX, 1, LX>& x, const md_span<EY, 1, LY>& y)
    {
        using T = detail::value_t<EX>;
        static_assert(detail::has_dot<B, T>, "snicholls::linalg::dot: the backend has no member T dot(vector_view<const T>, vector_view<const T>) const");
        if (x.extent(0) != y.extent(0))
            return err(error::dimension_mismatch);
        return ok(b.dot(vector_view<const T>(x), vector_view<const T>(y)));
    }

    /// \brief `y += alpha x`.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EX `x`'s element type.
    /// \tparam LX `x`'s layout.
    /// \tparam T  The number type.
    /// \tparam LY `y`'s layout.
    /// \param b     The backend.
    /// \param alpha The multiplier.
    /// \param x     A vector.
    /// \param y     A vector, updated.
    /// \pre  `x` is `y` itself (then `y = (1 + alpha) y`, as BLAS allows) or does not overlap it; any other overlap is
    ///       undefined.
    /// \return Nothing, or `error::dimension_mismatch` if the lengths differ.
    template <typename B, typename EX, typename LX, scalar T, typename LY>
        requires std::same_as<detail::value_t<EX>, T>
    constexpr result<void, error> axpy(const B& b, std::type_identity_t<T> alpha, const md_span<EX, 1, LX>& x, const md_span<T, 1, LY>& y)
    {
        static_assert(detail::has_axpy<B, T>, "snicholls::linalg::axpy: the backend has no member void axpy(T, vector_view<const T>, vector_view<T>) const");
        if (x.extent(0) != y.extent(0))
            return err(error::dimension_mismatch);
        b.axpy(alpha, vector_view<const T>(x), vector_view<T>(y));
        return ok();
    }

    /// \brief `y = alpha A x + beta y` (`y` not read when `beta` is 0).
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA `A`'s element type.
    /// \tparam LA `A`'s layout.
    /// \tparam EX `x`'s element type.
    /// \tparam LX `x`'s layout.
    /// \tparam T  The number type.
    /// \tparam LY `y`'s layout.
    /// \param b     The backend.
    /// \param alpha The multiplier of `A x`.
    /// \param a     An `m x n` matrix.
    /// \param x     A vector of length `n`.
    /// \param beta  The multiplier of `y`.
    /// \param y     A vector of length `m`.
    /// \pre  `y` overlaps neither `A` nor `x`. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <typename B, typename EA, typename LA, typename EX, typename LX, scalar T, typename LY>
        requires(std::same_as<detail::value_t<EA>, T> && std::same_as<detail::value_t<EX>, T>)
    constexpr result<void, error> gemv(const B& b, std::type_identity_t<T> alpha, const md_span<EA, 2, LA>& a, const md_span<EX, 1, LX>& x,
                                       std::type_identity_t<T> beta, const md_span<T, 1, LY>& y)
    {
        static_assert(detail::has_gemv<B, T>,
                      "snicholls::linalg::gemv: the backend has no member void gemv(T, matrix_view<const T>, vector_view<const T>, T, vector_view<T>) const");
        if (a.extent(1) != x.extent(0) || a.extent(0) != y.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(y, a) || detail::same_storage(y, x))
            return err(error::aliased);
        b.gemv(alpha, matrix_view<const T>(a), vector_view<const T>(x), beta, vector_view<T>(y));
        return ok();
    }

    /// \brief `C = alpha A B + beta C` (`C` not read when `beta` is 0).
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA `A`'s element type.
    /// \tparam LA `A`'s layout.
    /// \tparam EB `B`'s element type.
    /// \tparam LB `B`'s layout.
    /// \tparam T  The number type.
    /// \tparam LC `C`'s layout.
    /// \param b     The backend.
    /// \param alpha The multiplier of `A B`.
    /// \param a     An `m x k` matrix.
    /// \param bm    A `k x n` matrix.
    /// \param beta  The multiplier of `C`.
    /// \param c     An `m x n` matrix.
    /// \pre  `C` overlaps neither `A` nor `B`. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <typename B, typename EA, typename LA, typename EB, typename LB, scalar T, typename LC>
        requires(std::same_as<detail::value_t<EA>, T> && std::same_as<detail::value_t<EB>, T>)
    constexpr result<void, error> gemm(const B& b, std::type_identity_t<T> alpha, const md_span<EA, 2, LA>& a, const md_span<EB, 2, LB>& bm,
                                       std::type_identity_t<T> beta, const md_span<T, 2, LC>& c)
    {
        static_assert(detail::has_gemm<B, T>,
                      "snicholls::linalg::gemm: the backend has no member void gemm(T, matrix_view<const T>, matrix_view<const T>, T, matrix_view<T>) const");
        if (a.extent(1) != bm.extent(0) || a.extent(0) != c.extent(0) || bm.extent(1) != c.extent(1))
            return err(error::dimension_mismatch);
        if (detail::same_storage(c, a) || detail::same_storage(c, bm))
            return err(error::aliased);
        b.gemm(alpha, matrix_view<const T>(a), matrix_view<const T>(bm), beta, matrix_view<T>(c));
        return ok();
    }

    /// \brief `C = alpha A A^T + beta C` on one triangle of `C`.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA `A`'s element type.
    /// \tparam LA `A`'s layout.
    /// \tparam T  The number type.
    /// \tparam LC `C`'s layout.
    /// \param b     The backend.
    /// \param alpha The multiplier of `A A^T`.
    /// \param a     An `n x k` matrix.
    /// \param beta  The multiplier of `C`.
    /// \param c     An `n x n` matrix.
    /// \param t     The triangle of `C` written.
    /// \pre  `C` does not overlap `A`. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <typename B, typename EA, typename LA, scalar T, typename LC>
        requires std::same_as<detail::value_t<EA>, T>
    constexpr result<void, error> syrk(const B& b, std::type_identity_t<T> alpha, const md_span<EA, 2, LA>& a, std::type_identity_t<T> beta,
                                       const md_span<T, 2, LC>& c, triangle t)
    {
        static_assert(detail::has_syrk<B, T>,
                      "snicholls::linalg::syrk: the backend has no member void syrk(T, matrix_view<const T>, T, matrix_view<T>, triangle) const");
        if (c.extent(0) != c.extent(1) || a.extent(0) != c.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(c, a))
            return err(error::aliased);
        b.syrk(alpha, matrix_view<const T>(a), beta, matrix_view<T>(c), t);
        return ok();
    }

    /// \brief Solve `T X = B` in place for a triangular `T`.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam ET `T`'s element type.
    /// \tparam LT `T`'s layout.
    /// \tparam T  The number type.
    /// \tparam RB `B`'s rank: 1 (a vector) or 2.
    /// \tparam LB `B`'s layout.
    /// \param b     The backend.
    /// \param t     An `n x n` triangular matrix; only the named triangle is read.
    /// \param rhs   `n` rows of right-hand sides, overwritten by `X`.
    /// \param which Which triangle `t` is (`transposed(t)` with the other triangle solves `T^T X = B`).
    /// \param diag  Whether `t`'s diagonal is read or taken as ones.
    /// \pre  `B` does not overlap `T`. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, `error::dimension_mismatch`, or `error::singular` when an explicit diagonal holds a zero.
    template <typename B, typename ET, typename LT, scalar T, std::size_t RB, typename LB>
        requires(std::same_as<detail::value_t<ET>, T> && (RB == 1 || RB == 2))
    constexpr result<void, error> triangular_solve(const B& b, const md_span<ET, 2, LT>& t, const md_span<T, RB, LB>& rhs, triangle which,
                                                   diagonal diag = diagonal::explicit_diagonal)
    {
        static_assert(detail::has_triangular_solve<B, T>,
                      "snicholls::linalg::triangular_solve: the backend has no member result<void, error> triangular_solve(matrix_view<const T>, "
                      "matrix_view<T>, triangle, diagonal) const");
        if (t.extent(0) != t.extent(1) || rhs.extent(0) != t.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(rhs, t))
            return err(error::aliased);
        return b.triangular_solve(matrix_view<const T>(t), detail::as_matrix(rhs), which, diag);
    }

    // ---- LU -------------------------------------------------------------------------------------------------

    /// \brief LU with partial pivoting in place, `P A = L U`, stored as LAPACK's `getrf` stores it (pivots 0-based).
    /// \ingroup core
    /// \tparam B The backend.
    /// \tparam T The number type.
    /// \tparam L `A`'s layout.
    /// \param b      The backend.
    /// \param a      An `n x n` matrix, overwritten by its factors.
    /// \param pivots `n` interchanges, written.
    /// \return Nothing, `error::dimension_mismatch`, or `error::singular` (the backend's tolerance; the reference
    ///         backend's is `pivot_tolerance(n, max |a_ij|)`). After `singular`, `a` and `pivots` are unspecified.
    template <typename B, scalar T, typename L>
    constexpr result<void, error> lu_factor(const B& b, const md_span<T, 2, L>& a, std::span<pivot_index> pivots)
    {
        static_assert(detail::has_lu_factor<B, T>,
                      "snicholls::linalg::lu_factor: the backend has no member result<void, error> lu_factor(matrix_view<T>, std::span<pivot_index>) const");
        if (!detail::is_square(a.extent(0), a.extent(1)) || pivots.size() != a.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(pivots, a))
            return err(error::aliased);
        return b.lu_factor(matrix_view<T>(a), pivots);
    }

    /// \brief Solve `A X = B` in place from `lu_factor`'s factors.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA The factors' element type.
    /// \tparam LA The factors' layout.
    /// \tparam T  The number type.
    /// \tparam RB `B`'s rank: 1 or 2.
    /// \tparam LB `B`'s layout.
    /// \param b      The backend.
    /// \param lu     The factors of an `n x n` `A`.
    /// \param pivots Its `n` interchanges.
    /// \param rhs    `n` rows of right-hand sides, overwritten by `X`.
    /// \pre  `B` does not overlap the factors. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <typename B, typename EA, typename LA, scalar T, std::size_t RB, typename LB>
        requires(std::same_as<detail::value_t<EA>, T> && (RB == 1 || RB == 2))
    constexpr result<void, error> lu_solve(const B& b, const md_span<EA, 2, LA>& lu, std::span<const pivot_index> pivots, const md_span<T, RB, LB>& rhs)
    {
        static_assert(detail::has_lu_solve<B, T>,
                      "snicholls::linalg::lu_solve: the backend has no member void lu_solve(matrix_view<const T>, std::span<const pivot_index>, "
                      "matrix_view<T>) const");
        if (!detail::is_square(lu.extent(0), lu.extent(1)) || pivots.size() != lu.extent(0) || rhs.extent(0) != lu.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(rhs, lu) || detail::same_storage(rhs, pivots))
            return err(error::aliased);
        b.lu_solve(matrix_view<const T>(lu), pivots, detail::as_matrix(rhs));
        return ok();
    }

    /// \brief `A^-1` from `lu_factor`'s factors: an explicit inverse, only when one is asked for (a solve is cheaper and more accurate).
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA The factors' element type.
    /// \tparam LA The factors' layout.
    /// \tparam T  The number type.
    /// \tparam LI The inverse's layout.
    /// \param b       The backend.
    /// \param lu      The factors of an `n x n` `A`.
    /// \param pivots  Its interchanges.
    /// \param inverse An `n x n` matrix, overwritten by `A^-1`.
    /// \pre  `inverse` does not overlap the factors (equal starts are reported as `error::aliased`).
    /// \return Nothing, `error::dimension_mismatch` or `error::aliased`, with `inverse` unchanged.
    template <typename B, typename EA, typename LA, scalar T, typename LI>
        requires std::same_as<detail::value_t<EA>, T>
    constexpr result<void, error> lu_inverse(const B& b, const md_span<EA, 2, LA>& lu, std::span<const pivot_index> pivots, const md_span<T, 2, LI>& inverse)
    {
        // Every check before the identity is written, so a failure leaves every argument unchanged.
        if (!detail::is_square(lu.extent(0), lu.extent(1)) || pivots.size() != lu.extent(0) || inverse.extent(0) != lu.extent(0) ||
            inverse.extent(1) != lu.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(inverse, lu) || detail::same_storage(inverse, pivots))
            return err(error::aliased);
        for (std::size_t i = 0; i < inverse.extent(0); ++i)
            for (std::size_t j = 0; j < inverse.extent(1); ++j)
                inverse(i, j) = i == j ? T(1) : T(0);
        return lu_solve(b, lu, pivots, inverse);
    }

    /// \brief `log |det A|` and its sign from `lu_factor`'s factors: the product of `U`'s diagonal and the interchanges' parity.
    /// \ingroup core
    /// \tparam EA The factors' element type.
    /// \tparam LA The factors' layout.
    /// \param lu     The factors of an `n x n` `A`.
    /// \param pivots Its interchanges.
    /// \return The signed logarithm (sign 0 and minus infinity for a zero on `U`'s diagonal), or `error::dimension_mismatch`.
    template <typename EA, typename LA>
        requires scalar<detail::value_t<EA>>
    [[nodiscard]] constexpr result<signed_log_det<detail::value_t<EA>>, error> lu_log_det(const md_span<EA, 2, LA>& lu, std::span<const pivot_index> pivots)
    {
        using T = detail::value_t<EA>;
        if (!detail::is_square(lu.extent(0), lu.extent(1)) || pivots.size() != lu.extent(0))
            return err(error::dimension_mismatch);
        signed_log_det<T> d;
        for (std::size_t i = 0; i < lu.extent(0); ++i) {
            const T u = lu(i, i);
            if (static_cast<std::size_t>(pivots[i]) != i)
                d.sign = -d.sign;
            if (u == T(0))
                return ok(signed_log_det<T>{0, -std::numeric_limits<T>::infinity()});
            if (u < T(0))
                d.sign = -d.sign;
            d.log_abs += static_cast<T>(math::log(math::abs(u)));
        }
        return ok(d);
    }

    // ---- Cholesky and LDLT ----------------------------------------------------------------------------------

    /// \brief Cholesky in place, `A = L L^T`, in the lower triangle.
    /// \ingroup core
    /// \tparam B The backend.
    /// \tparam T The number type.
    /// \tparam L `A`'s layout.
    /// \param b The backend.
    /// \param a An `n x n` symmetric matrix; its lower triangle is read and overwritten by `L`.
    /// \return Nothing, `error::dimension_mismatch`, or `error::not_positive_definite`.
    template <typename B, scalar T, typename L>
    constexpr result<void, error> cholesky_factor(const B& b, const md_span<T, 2, L>& a)
    {
        static_assert(detail::has_cholesky_factor<B, T>,
                      "snicholls::linalg::cholesky_factor: the backend has no member result<void, error> cholesky_factor(matrix_view<T>) const");
        if (!detail::is_square(a.extent(0), a.extent(1)))
            return err(error::dimension_mismatch);
        return b.cholesky_factor(matrix_view<T>(a));
    }

    /// \brief Solve `A X = B` in place from `cholesky_factor`'s `L`.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA The factor's element type.
    /// \tparam LA The factor's layout.
    /// \tparam T  The number type.
    /// \tparam RB `B`'s rank: 1 or 2.
    /// \tparam LB `B`'s layout.
    /// \param b   The backend.
    /// \param l   The factor of an `n x n` `A`.
    /// \param rhs `n` rows of right-hand sides, overwritten by `X`.
    /// \pre  `B` does not overlap the factor. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <typename B, typename EA, typename LA, scalar T, std::size_t RB, typename LB>
        requires(std::same_as<detail::value_t<EA>, T> && (RB == 1 || RB == 2))
    constexpr result<void, error> cholesky_solve(const B& b, const md_span<EA, 2, LA>& l, const md_span<T, RB, LB>& rhs)
    {
        static_assert(detail::has_cholesky_solve<B, T>,
                      "snicholls::linalg::cholesky_solve: the backend has no member void cholesky_solve(matrix_view<const T>, matrix_view<T>) const");
        if (!detail::is_square(l.extent(0), l.extent(1)) || rhs.extent(0) != l.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(rhs, l))
            return err(error::aliased);
        b.cholesky_solve(matrix_view<const T>(l), detail::as_matrix(rhs));
        return ok();
    }

    /// \brief `log det A` from `cholesky_factor`'s `L`: `2 sum log l_ii`; the sign is always +1.
    /// \ingroup core
    /// \tparam EA The factor's element type.
    /// \tparam LA The factor's layout.
    /// \param l The factor.
    /// \return The signed logarithm, or `error::dimension_mismatch`.
    template <typename EA, typename LA>
        requires scalar<detail::value_t<EA>>
    [[nodiscard]] constexpr result<signed_log_det<detail::value_t<EA>>, error> cholesky_log_det(const md_span<EA, 2, LA>& l)
    {
        using T = detail::value_t<EA>;
        if (!detail::is_square(l.extent(0), l.extent(1)))
            return err(error::dimension_mismatch);
        signed_log_det<T> d;
        for (std::size_t i = 0; i < l.extent(0); ++i)
            d.log_abs += static_cast<T>(math::log(l(i, i)));
        d.log_abs = T(2) * d.log_abs;
        return ok(d);
    }

    /// \brief LDLT in place, `A = L D L^T` with unit lower `L` and diagonal `D`, for a symmetric matrix that may be indefinite.
    /// \ingroup core
    /// \tparam B The backend.
    /// \tparam T The number type.
    /// \tparam L `A`'s layout.
    /// \param b The backend.
    /// \param a An `n x n` symmetric matrix; its lower triangle is read and overwritten (`D` on the diagonal).
    /// \return Nothing, `error::dimension_mismatch`, or `error::singular` when a `d_j` is (numerically) zero or, for an
    ///         unpivoted backend such as the reference one, when the element growth would make the answer
    ///         inaccurate (`ldlt_growth_limit`). A pivoted backend may solve those instead; none may solve them
    ///         inaccurately.
    template <typename B, scalar T, typename L>
    constexpr result<void, error> ldlt_factor(const B& b, const md_span<T, 2, L>& a)
    {
        static_assert(detail::has_ldlt_factor<B, T>,
                      "snicholls::linalg::ldlt_factor: the backend has no member result<void, error> ldlt_factor(matrix_view<T>) const");
        if (!detail::is_square(a.extent(0), a.extent(1)))
            return err(error::dimension_mismatch);
        return b.ldlt_factor(matrix_view<T>(a));
    }

    /// \brief Solve `A X = B` in place from `ldlt_factor`'s factors.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA The factors' element type.
    /// \tparam LA The factors' layout.
    /// \tparam T  The number type.
    /// \tparam RB `B`'s rank: 1 or 2.
    /// \tparam LB `B`'s layout.
    /// \param b   The backend.
    /// \param ld  The factors of an `n x n` `A`.
    /// \param rhs `n` rows of right-hand sides, overwritten by `X`.
    /// \pre  `B` does not overlap the factors. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <typename B, typename EA, typename LA, scalar T, std::size_t RB, typename LB>
        requires(std::same_as<detail::value_t<EA>, T> && (RB == 1 || RB == 2))
    constexpr result<void, error> ldlt_solve(const B& b, const md_span<EA, 2, LA>& ld, const md_span<T, RB, LB>& rhs)
    {
        static_assert(detail::has_ldlt_solve<B, T>,
                      "snicholls::linalg::ldlt_solve: the backend has no member void ldlt_solve(matrix_view<const T>, matrix_view<T>) const");
        if (!detail::is_square(ld.extent(0), ld.extent(1)) || rhs.extent(0) != ld.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(rhs, ld))
            return err(error::aliased);
        b.ldlt_solve(matrix_view<const T>(ld), detail::as_matrix(rhs));
        return ok();
    }

    /// \brief `log |det A|` and its sign from `ldlt_factor`'s factors: the product of `D`.
    /// \ingroup core
    /// \tparam EA The factors' element type.
    /// \tparam LA The factors' layout.
    /// \param ld The factors.
    /// \return The signed logarithm (its sign is the parity of `D`'s negative entries), or `error::dimension_mismatch`.
    template <typename EA, typename LA>
        requires scalar<detail::value_t<EA>>
    [[nodiscard]] constexpr result<signed_log_det<detail::value_t<EA>>, error> ldlt_log_det(const md_span<EA, 2, LA>& ld)
    {
        using T = detail::value_t<EA>;
        if (!detail::is_square(ld.extent(0), ld.extent(1)))
            return err(error::dimension_mismatch);
        signed_log_det<T> d;
        for (std::size_t i = 0; i < ld.extent(0); ++i) {
            const T v = ld(i, i);
            if (v == T(0))
                return ok(signed_log_det<T>{0, -std::numeric_limits<T>::infinity()});
            if (v < T(0))
                d.sign = -d.sign;
            d.log_abs += static_cast<T>(math::log(math::abs(v)));
        }
        return ok(d);
    }

    // ---- QR and least squares -------------------------------------------------------------------------------

    /// \brief Householder QR in place, `A = Q R`, stored as LAPACK's `geqrf` stores it.
    /// \ingroup core
    /// \tparam B The backend.
    /// \tparam T The number type.
    /// \tparam L `A`'s layout.
    /// \param b   The backend.
    /// \param a   An `m x n` matrix with `m >= n`, overwritten.
    /// \param tau `n` reflector scales, written.
    /// \return Nothing, `error::dimension_mismatch` (also for `m < n`: factor `transposed(a)` instead), or
    ///         `error::singular` when `A` holds a NaN or an infinity.
    template <typename B, scalar T, typename L>
    constexpr result<void, error> qr_factor(const B& b, const md_span<T, 2, L>& a, std::span<std::type_identity_t<T>> tau)
    {
        static_assert(detail::has_qr_factor<B, T>,
                      "snicholls::linalg::qr_factor: the backend has no member result<void, error> qr_factor(matrix_view<T>, std::span<T>) const");
        if (a.extent(0) < a.extent(1) || tau.size() != a.extent(1))
            return err(error::dimension_mismatch);
        if (detail::same_storage(tau, a))
            return err(error::aliased);
        return b.qr_factor(matrix_view<T>(a), tau);
    }

    /// \brief The thin `Q` of `qr_factor`'s factors.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA The factors' element type.
    /// \tparam LA The factors' layout.
    /// \tparam T  The number type.
    /// \tparam LQ `Q`'s layout.
    /// \param b   The backend.
    /// \param qr  The factors of an `m x n` `A`.
    /// \param tau The reflector scales.
    /// \param q   An `m x n` matrix, overwritten by `Q` (orthonormal columns).
    /// \pre  `Q` does not overlap the factors. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, or `error::dimension_mismatch`.
    template <typename B, typename EA, typename LA, scalar T, typename LQ>
        requires std::same_as<detail::value_t<EA>, T>
    constexpr result<void, error> qr_q(const B& b, const md_span<EA, 2, LA>& qr, std::span<const std::type_identity_t<T>> tau, const md_span<T, 2, LQ>& q)
    {
        static_assert(detail::has_qr_q<B, T>,
                      "snicholls::linalg::qr_q: the backend has no member void qr_q(matrix_view<const T>, std::span<const T>, matrix_view<T>) const");
        if (qr.extent(0) < qr.extent(1) || tau.size() != qr.extent(1) || q.extent(0) != qr.extent(0) || q.extent(1) != qr.extent(1))
            return err(error::dimension_mismatch);
        if (detail::same_storage(q, qr) || detail::same_storage(q, tau))
            return err(error::aliased);
        b.qr_q(matrix_view<const T>(qr), tau, matrix_view<T>(q));
        return ok();
    }

    /// \brief Least squares from `qr_factor`'s factors: minimises `|A X - B|`; `B`'s first `n` rows become `X`.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam EA The factors' element type.
    /// \tparam LA The factors' layout.
    /// \tparam T  The number type.
    /// \tparam RB `B`'s rank: 1 or 2.
    /// \tparam LB `B`'s layout.
    /// \param b   The backend.
    /// \param qr  The factors of an `m x n` `A`.
    /// \param tau The reflector scales.
    /// \param rhs `m` rows of right-hand sides, overwritten.
    /// \pre  `B` does not overlap the factors. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, `error::dimension_mismatch`, or `error::singular` when `A` is (numerically) rank deficient.
    template <typename B, typename EA, typename LA, scalar T, std::size_t RB, typename LB>
        requires(std::same_as<detail::value_t<EA>, T> && (RB == 1 || RB == 2))
    constexpr result<void, error> qr_solve(const B& b, const md_span<EA, 2, LA>& qr, std::span<const std::type_identity_t<T>> tau, const md_span<T, RB, LB>& rhs)
    {
        static_assert(detail::has_qr_solve<B, T>,
                      "snicholls::linalg::qr_solve: the backend has no member result<void, error> qr_solve(matrix_view<const T>, std::span<const T>, "
                      "matrix_view<T>) const");
        if (qr.extent(0) < qr.extent(1) || tau.size() != qr.extent(1) || rhs.extent(0) != qr.extent(0))
            return err(error::dimension_mismatch);
        if (detail::same_storage(rhs, qr) || detail::same_storage(rhs, tau))
            return err(error::aliased);
        return b.qr_solve(matrix_view<const T>(qr), tau, detail::as_matrix(rhs));
    }

    /// \brief Least squares in one call: `qr_factor` then `qr_solve`; `A` is overwritten by its factors.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam T  The number type.
    /// \tparam LA `A`'s layout.
    /// \tparam RB `B`'s rank: 1 or 2.
    /// \tparam LB `B`'s layout.
    /// \param b   The backend.
    /// \param a   An `m x n` matrix, `m >= n`, overwritten by its factors.
    /// \param tau `n` reflector scales, written.
    /// \param rhs `m` rows of right-hand sides; the first `n` become the solution.
    /// \pre  `rhs` does not overlap `A` (equal starts are reported as `error::aliased`, before anything is written); any
    ///       other overlap is undefined.
    /// \return Nothing, `error::dimension_mismatch` or `error::aliased` (both with every argument unchanged), or
    ///         `error::singular`.
    template <typename B, scalar T, typename LA, std::size_t RB, typename LB>
        requires(RB == 1 || RB == 2)
    constexpr result<void, error> least_squares(const B& b, const md_span<T, 2, LA>& a, std::span<std::type_identity_t<T>> tau, const md_span<T, RB, LB>& rhs)
    {
        // Every check qr_factor and qr_solve make, before qr_factor writes A.
        if (rhs.extent(0) != a.extent(0) || a.extent(0) < a.extent(1) || tau.size() != a.extent(1))
            return err(error::dimension_mismatch);
        if (detail::same_storage(rhs, a) || detail::same_storage(tau, a) || detail::same_storage(tau, rhs))
            return err(error::aliased);
        const result<void, error> f = qr_factor(b, a, tau);
        if (!f)
            return f;
        return qr_solve(b, md_span<const T, 2, LA>(a), std::span<const T>(tau), rhs);
    }

    // ---- Decompositions -------------------------------------------------------------------------------------

    /// \brief Symmetric eigendecomposition `A = V diag(w) V^T`: eigenvalues ascending, orthonormal eigenvectors in `V`'s columns.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam T  The number type.
    /// \tparam LA `A`'s layout.
    /// \tparam LV `V`'s layout.
    /// \param b The backend.
    /// \param a An `n x n` symmetric matrix (lower triangle read); destroyed.
    /// \param w `n` eigenvalues, written.
    /// \param v An `n x n` matrix, overwritten.
    /// \pre  `V` does not overlap `A`. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, `error::dimension_mismatch`, or `error::did_not_converge`.
    template <typename B, scalar T, typename LA, typename LV>
    constexpr result<void, error> symmetric_eigen(const B& b, const md_span<T, 2, LA>& a, std::span<std::type_identity_t<T>> w, const md_span<T, 2, LV>& v)
    {
        static_assert(detail::has_symmetric_eigen<B, T>,
                      "snicholls::linalg::symmetric_eigen: the backend has no member result<void, error> symmetric_eigen(matrix_view<T>, std::span<T>, "
                      "matrix_view<T>) const");
        const std::size_t n = a.extent(0);
        if (a.extent(1) != n || w.size() != n || v.extent(0) != n || v.extent(1) != n)
            return err(error::dimension_mismatch);
        if (detail::same_storage(v, a) || detail::same_storage(w, a) || detail::same_storage(w, v))
            return err(error::aliased);
        return b.symmetric_eigen(matrix_view<T>(a), w, matrix_view<T>(v));
    }

    /// \brief Thin singular value decomposition `A = U diag(s) V^T`; `A` becomes `U`.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam T  The number type.
    /// \tparam LA `A`'s layout.
    /// \tparam LV `V`'s layout.
    /// \param b The backend.
    /// \param a An `m x n` matrix, `m >= n` (decompose `transposed(a)` otherwise), overwritten by `U`.
    /// \param s `n` singular values, descending, written.
    /// \param v An `n x n` matrix, overwritten by `V`.
    /// \pre  `V` does not overlap `A`. Views that start at the same element are reported as `error::aliased`; any other overlap is
    ///       undefined.
    /// \return Nothing, `error::dimension_mismatch`, or `error::did_not_converge`.
    template <typename B, scalar T, typename LA, typename LV>
    constexpr result<void, error> svd(const B& b, const md_span<T, 2, LA>& a, std::span<std::type_identity_t<T>> s, const md_span<T, 2, LV>& v)
    {
        static_assert(detail::has_svd<B, T>,
                      "snicholls::linalg::svd: the backend has no member result<void, error> svd(matrix_view<T>, std::span<T>, matrix_view<T>) const");
        const std::size_t n = a.extent(1);
        if (a.extent(0) < n || s.size() != n || v.extent(0) != n || v.extent(1) != n)
            return err(error::dimension_mismatch);
        if (detail::same_storage(v, a) || detail::same_storage(s, a) || detail::same_storage(s, v))
            return err(error::aliased);
        return b.svd(matrix_view<T>(a), s, matrix_view<T>(v));
    }

} // namespace snicholls::linalg

#endif /* snicholls_core_linalg_backend_hpp */
