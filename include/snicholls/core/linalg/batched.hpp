//
//  batched.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Batched factorise-and-solve: `n` systems of one size as strided views, one `status` per system, on parallelism the caller provides.
/// \ingroup core
///
/// **Overview.** Many small dense systems of the same size - a million to a hundred million small systems, as in
/// batched kriging-style solves, local regressions or per-cell Gaussian-process updates - are the case a
/// general-purpose solver serves worst: the call overhead and the threading are paid per system. Here a whole
/// batch is one call. The matrices are a rank-3 view `(system, row, column)` and the right-hand sides a rank-2
/// `(system, row)` or rank-3 `(system, row, column)` view, any strides, so the systems can sit in the caller's
/// own records with gaps between them. Each system is factorised and solved in place, and its outcome is written to
/// `status[i]`: a singular or indefinite system is reported for that system only, so the caller falls back for it
/// (a smaller neighbourhood, a nugget, a pivoted solver) and keeps the rest.
///
/// \snippet linalg/batched_demo.cpp batched
///
/// Core starts no threads. The caller passes an executor: any callable `ex(n, f)` that calls `f(begin, end)` on
/// disjoint ranges covering `[0, n)`, in any order and on any threads, and returns when all have run. That is the
/// shape of a thread pool's `parallel_for` over an index range; `inline_executor` runs `f(0, n)` on the calling
/// thread.
///
/// \snippet linalg/batched_demo.cpp executor
///
/// **Relationships.** `linalg/backend.hpp` (the backend concepts; `backend<B, T>` includes `batched_backend<B, T>`),
/// `linalg/reference.hpp` (whose batched members are `for_each_system` over its single-system members),
/// `containers::submdspan` (one system of a batch). Group: \ref core.
///
/// **Implementations / variants.** A backend implements the three members below. A backend that runs a whole
/// batch in one device call (a GPU's batched LU) implements them directly and may ignore the executor (it then
/// says so: it brings its own parallelism); one that solves system by system calls `for_each_system` with its
/// single-system members, as the reference backend does.
///
/// | Member (`const`, `template <range_executor E>`) | Effect |
/// |---|---|
/// | `result<std::size_t, error> lu_solve_batched(batch_view<T> a, std::span<pivot_index> p, batch_view<T> b, std::span<status> s, E ex)` | `P A_i = L_i U_i`, `B_i = A_i^-1 B_i` |
/// | `result<std::size_t, error> cholesky_solve_batched(batch_view<T> a, batch_view<T> b, std::span<status> s, E ex)` | `A_i = L_i L_i^T`, `B_i = A_i^-1 B_i` |
/// | `result<std::size_t, error> ldlt_solve_batched(batch_view<T> a, batch_view<T> b, std::span<status> s, E ex)` | `A_i = L_i D_i L_i^T`, `B_i = A_i^-1 B_i` |
///
/// Each returns the number of systems whose status is not `ok`. Pivots are `n * m` entries, system `i`'s at
/// `[i * m, (i + 1) * m)`, as cuBLAS's batched LU stores them.
///
/// **Guarantees.** No allocation, no workspace beyond the arguments, no threads of core's own. Each system's factors,
/// solution and status depend on that system's data alone, so with the reference backend the results are bit-for-bit
/// the same for every executor, chunking and thread count (**deterministic**; tested). The count of failures is taken
/// after the executor returns, from the status span. The executor must call `f` for every index exactly once; ranges
/// may run concurrently because they touch disjoint systems. A view whose systems overlap in memory is a precondition
/// violation.
///
/// **Errors.** `error::aliased` (nothing written) when `rhs`, `pivots` or `status` starts where another argument does, and `error::dimension_mismatch` (nothing
/// written) when the extents do not fit together: `a` is
/// `n x m x m`, `b` is `n x m` or `n x m x k`, `pivots` has `n * m` entries, `status` has `n`. Otherwise the call
/// succeeds and each system's error is its status: `singular`, `not_positive_definite`. A system with a non-`ok`
/// status has unspecified contents in `a` and `b`.
///
/// **Performance.** `benchmarks/linalg/batched_benchmark.cpp`, 10^6 systems with one right-hand side, double, reference
/// backend against the same algorithms written by hand over raw pointers (Release, AppleClang 17, Xeon W-3245, macOS 26, 2026-11-11, on a shared machine with load average 7 to 10;
/// single-threaded):
///
/// | Systems | Interface | By hand |
/// |---|---|---|
/// | LU, order 25 | 4.94 s | 4.54 s |
/// | LU, order 65 | 48.3 s | 51.0 s |
/// | Cholesky, order 25 | 2.94 s | 2.99 s |
/// | Cholesky, order 65 | 39.2 s | 39.5 s |
/// | LDLT, order 25 | 3.91 s | 3.84 s |
/// | LDLT, order 65 | 50.4 s | 49.0 s |
///
/// The differences (between -5 % and +9 %) are of the order of the run-to-run variation on that machine. The executor
/// spreads a batch across cores. See BENCHMARKS.md.
///
/// **Tests and demos.** `tests/linalg/batched_test.cpp`, the batched cases of the conformance suite,
/// `demos/linalg/batched_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<span>`, `<type_traits>`, `result.hpp`, `linalg/types.hpp`.
///
/// **Known issues.** The executor's ranges are as fine as it chooses: a pool should hand out ranges of some
/// thousands of small systems, not single ones.
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_linalg_batched_hpp
#define snicholls_core_linalg_batched_hpp

#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/result.hpp>

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::linalg {

    namespace detail {
        // The callable an executor is checked with: f(begin, end).
        struct range_probe {
            void operator()(std::size_t, std::size_t) const noexcept {}
        };
    } // namespace detail

    /// \brief A caller's parallelism: `ex(n, f)` calls `f(begin, end)` on disjoint ranges covering `[0, n)`, then returns.
    /// \ingroup core
    /// \tparam E The executor type; its call operator takes any callable (a template).
    template <typename E>
    concept range_executor = std::copy_constructible<E> && requires(E& e, std::size_t n, detail::range_probe f) { e(n, f); };

    /// \brief The executor that runs `f(0, n)` on the calling thread: no threads, the default.
    /// \ingroup core
    struct inline_executor {
        /// \brief Run the whole range at once.
        /// \tparam F The callable's type.
        /// \param n The number of indices.
        /// \param f The callable, `f(begin, end)`.
        template <typename F>
        constexpr void operator()(std::size_t n, F&& f) const
        {
            if (n != 0)
                f(std::size_t{0}, n);
        }
    };

    /// \brief Run `per_system(i)` for every `i` in `[0, n)` through the executor: the loop a backend's batched member is made of.
    /// \ingroup core
    /// \tparam E The executor.
    /// \tparam F The per-system callable's type.
    /// \param ex         The executor.
    /// \param n          The number of systems.
    /// \param per_system Called once per index; calls on different indices may run concurrently.
    template <range_executor E, typename F>
    constexpr void for_each_system(E& ex, std::size_t n, F&& per_system)
    {
        ex(n, [&per_system](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i)
                per_system(i);
        });
    }

    /// \brief The number of statuses that are not `ok`.
    /// \ingroup core
    /// \param statuses The statuses.
    /// \return The count.
    [[nodiscard]] constexpr std::size_t count_failures(std::span<const status> statuses) noexcept
    {
        std::size_t failures = 0;
        for (const auto s : statuses)
            failures += s == status::ok ? 0 : 1;
        return failures;
    }

    namespace detail {
        template <typename B, typename T>
        concept has_lu_solve_batched = requires(const B& b, batch_view<T> a, std::span<pivot_index> p, batch_view<T> x, std::span<status> s,
                                                inline_executor ex) {
            { b.lu_solve_batched(a, p, x, s, ex) } -> std::same_as<result<std::size_t, error>>;
        };
        template <typename B, typename T>
        concept has_cholesky_solve_batched = requires(const B& b, batch_view<T> a, batch_view<T> x, std::span<status> s, inline_executor ex) {
            { b.cholesky_solve_batched(a, x, s, ex) } -> std::same_as<result<std::size_t, error>>;
        };
        template <typename B, typename T>
        concept has_ldlt_solve_batched = requires(const B& b, batch_view<T> a, batch_view<T> x, std::span<status> s, inline_executor ex) {
            { b.ldlt_solve_batched(a, x, s, ex) } -> std::same_as<result<std::size_t, error>>;
        };

        // A right-hand side batch as rank 3: (system, row) becomes (system, row, 1).
        template <typename T, typename L>
        constexpr batch_view<T> as_batch(const md_span<T, 3, L>& b) noexcept
        {
            return b;
        }
        template <typename T, typename L>
        constexpr batch_view<T> as_batch(const md_span<T, 2, L>& b) noexcept
        {
            return {b.data_handle(), {b.extent(0), b.extent(1), 1}, {b.stride(0), b.stride(1), 1}};
        }

        template <typename T, std::size_t RB, typename LA, typename LB>
        constexpr bool batch_fits(const md_span<T, 3, LA>& a, const md_span<T, RB, LB>& b, std::size_t statuses) noexcept
        {
            return a.extent(1) == a.extent(2) && b.extent(0) == a.extent(0) && b.extent(1) == a.extent(1) && statuses == a.extent(0);
        }
    } // namespace detail

    /// \brief A backend that factorises and solves batches for `T`: `lu_solve_batched`, `cholesky_solve_batched`, `ldlt_solve_batched`.
    /// \ingroup core
    /// \tparam B The backend type.
    /// \tparam T The number type.
    template <typename B, typename T>
    concept batched_backend = scalar<T> && std::is_trivially_copyable_v<B> && detail::has_lu_solve_batched<B, T> &&
                              detail::has_cholesky_solve_batched<B, T> && detail::has_ldlt_solve_batched<B, T>;

    /// \brief Factorise `n` systems by LU with partial pivoting and solve them, in place; one status per system.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam T  The number type.
    /// \tparam LA The matrices' layout.
    /// \tparam RB The right-hand sides' rank: 2 `(system, row)` or 3 `(system, row, column)`.
    /// \tparam LB Their layout.
    /// \tparam E  The executor.
    /// \param b      The backend.
    /// \param a      `n x m x m` matrices, overwritten by their factors.
    /// \param pivots `n * m` interchanges, system `i`'s at `[i * m, (i + 1) * m)`.
    /// \param rhs    The right-hand sides, overwritten by the solutions.
    /// \param status `n` statuses, written: `ok` or `singular`.
    /// \param ex     The executor; `inline_executor` by default.
    /// \pre  `rhs` overlaps no system of `a`: views that start at the same element are reported as `error::aliased`;
    ///       any other overlap is undefined.
    /// \return The number of systems not solved, or `error::dimension_mismatch` (nothing written).
    template <typename B, scalar T, typename LA, std::size_t RB, typename LB, range_executor E = inline_executor>
        requires(RB == 2 || RB == 3)
    constexpr result<std::size_t, error> lu_solve_batched(const B& b, const md_span<T, 3, LA>& a, std::span<pivot_index> pivots,
                                                          const md_span<T, RB, LB>& rhs, std::span<status> status, E ex = {})
    {
        static_assert(detail::has_lu_solve_batched<B, T>,
                      "snicholls::linalg::lu_solve_batched: the backend has no member result<std::size_t, error> lu_solve_batched(batch_view<T>, "
                      "std::span<pivot_index>, batch_view<T>, std::span<status>, E) const for every range_executor E");
        if (!detail::batch_fits(a, rhs, status.size()) || pivots.size() != a.extent(0) * a.extent(1))
            return err(error::dimension_mismatch);
        if (detail::same_storage(a, rhs) || detail::same_storage(pivots, a) || detail::same_storage(pivots, rhs) ||
            detail::same_storage(status, a) || detail::same_storage(status, rhs) || detail::same_storage(status, pivots))
            return err(error::aliased);
        return b.lu_solve_batched(batch_view<T>(a), pivots, detail::as_batch(rhs), status, ex);
    }

    /// \brief Factorise `n` symmetric positive-definite systems by Cholesky and solve them, in place; one status per system.
    /// \ingroup core
    /// \tparam B  The backend.
    /// \tparam T  The number type.
    /// \tparam LA The matrices' layout.
    /// \tparam RB The right-hand sides' rank: 2 or 3.
    /// \tparam LB Their layout.
    /// \tparam E  The executor.
    /// \param b      The backend.
    /// \param a      `n x m x m` symmetric matrices (lower triangles read), overwritten by their factors.
    /// \param rhs    The right-hand sides, overwritten by the solutions.
    /// \param status `n` statuses, written: `ok` or `not_positive_definite`.
    /// \param ex     The executor.
    /// \pre  `rhs` overlaps no system of `a`: views that start at the same element are reported as `error::aliased`;
    ///       any other overlap is undefined.
    /// \return The number of systems not solved, or `error::dimension_mismatch` (nothing written).
    template <typename B, scalar T, typename LA, std::size_t RB, typename LB, range_executor E = inline_executor>
        requires(RB == 2 || RB == 3)
    constexpr result<std::size_t, error> cholesky_solve_batched(const B& b, const md_span<T, 3, LA>& a, const md_span<T, RB, LB>& rhs,
                                                                std::span<status> status, E ex = {})
    {
        static_assert(detail::has_cholesky_solve_batched<B, T>,
                      "snicholls::linalg::cholesky_solve_batched: the backend has no member result<std::size_t, error> cholesky_solve_batched("
                      "batch_view<T>, batch_view<T>, std::span<status>, E) const for every range_executor E");
        if (!detail::batch_fits(a, rhs, status.size()))
            return err(error::dimension_mismatch);
        if (detail::same_storage(a, rhs) || detail::same_storage(status, a) || detail::same_storage(status, rhs))
            return err(error::aliased);
        return b.cholesky_solve_batched(batch_view<T>(a), detail::as_batch(rhs), status, ex);
    }

    /// \brief Factorise `n` symmetric, possibly indefinite, systems by LDLT and solve them, in place; one status per system.
    /// \ingroup core
    ///
    /// The fit for bordered systems such as ordinary kriging's `[[C, 1], [1^T, 0]]`, which are indefinite but whose
    /// leading block is positive definite, at half the work of LU.
    /// \tparam B  The backend.
    /// \tparam T  The number type.
    /// \tparam LA The matrices' layout.
    /// \tparam RB The right-hand sides' rank: 2 or 3.
    /// \tparam LB Their layout.
    /// \tparam E  The executor.
    /// \param b      The backend.
    /// \param a      `n x m x m` symmetric matrices (lower triangles read), overwritten by their factors.
    /// \param rhs    The right-hand sides, overwritten by the solutions.
    /// \param status `n` statuses, written: `ok` or `singular`.
    /// \param ex     The executor.
    /// \pre  `rhs` overlaps no system of `a`: views that start at the same element are reported as `error::aliased`;
    ///       any other overlap is undefined.
    /// \return The number of systems not solved, or `error::dimension_mismatch` (nothing written).
    template <typename B, scalar T, typename LA, std::size_t RB, typename LB, range_executor E = inline_executor>
        requires(RB == 2 || RB == 3)
    constexpr result<std::size_t, error> ldlt_solve_batched(const B& b, const md_span<T, 3, LA>& a, const md_span<T, RB, LB>& rhs,
                                                            std::span<status> status, E ex = {})
    {
        static_assert(detail::has_ldlt_solve_batched<B, T>,
                      "snicholls::linalg::ldlt_solve_batched: the backend has no member result<std::size_t, error> ldlt_solve_batched("
                      "batch_view<T>, batch_view<T>, std::span<status>, E) const for every range_executor E");
        if (!detail::batch_fits(a, rhs, status.size()))
            return err(error::dimension_mismatch);
        if (detail::same_storage(a, rhs) || detail::same_storage(status, a) || detail::same_storage(status, rhs))
            return err(error::aliased);
        return b.ldlt_solve_batched(batch_view<T>(a), detail::as_batch(rhs), status, ex);
    }

} // namespace snicholls::linalg

#endif /* snicholls_core_linalg_batched_hpp */
