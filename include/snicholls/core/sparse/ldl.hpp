//
//  ldl.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::sparse::ldl_numeric` and friends: the sparse `A = L D Lᵀ` of a quasi-definite matrix without pivoting, on a pattern analysed once, with static regularisation, solves and iterative refinement, all `constexpr` over the caller's storage.
/// \ingroup core
///
/// **Overview.** An interior-point or ADMM solver for a QP solves, at every iteration, a KKT system
/// `[P + σI  Aᵀ; A  -ρ⁻¹I]`: symmetric, indefinite, but *quasi-definite* (a positive definite block, a negative
/// definite block), so it has an `L D Lᵀ` factorisation for every symmetric permutation, with `D` of known signs,
/// and no pivoting is needed (Vanderbei, 1995). The pattern is fixed for the whole solve; only the values (ρ, the
/// scaling, the barrier) change. So the work splits:
///
/// - once: an ordering (`amd`, `sparse/ordering.hpp`), the permuted matrix (`symmetric_permute_upper`), the
///   elimination tree and column counts (`ldl_symbolic`, which also gives `nnz(L)` to size `L`);
/// - per iteration: `ldl_numeric` (the values of `L` and `D`), then `ldl_solve` and, when accuracy matters,
///   `ldl_refine`. `ldl_refactor` is `ldl_numeric` again with new values on the same pattern.
///
/// `etree` alone gives the elimination tree. `ldl_fixed<T, N, MaxNnzL>` holds every array for a problem whose size
/// is fixed when the program is written, and `ldl_nnz<N>(a)` sizes it in a constant expression, so the symbolic
/// analysis of a fixed problem can be done by the compiler.
///
/// For a definite matrix, or with `signs` given, the factorisation always exists; without signs a zero (or
/// non-finite) pivot is reported as `ldl_error::zero_pivot`, and an indefinite matrix that is not quasi-definite may
/// factor inaccurately (there is no pivoting).
///
/// \snippet sparse/ldl_demo.cpp ldl
///
/// **Relationships.** `sparse/csc.hpp` (`csc_view`, `spmv_symmetric_upper`, `symmetric_permute_upper`),
/// `sparse/ordering.hpp` (`amd`, `permute`, `inverse_permute`), `linalg/reference.hpp` (the dense factorisations, for
/// small dense systems), `result.hpp`. Group: \ref core.
///
/// **Implementations.** The up-looking factorisation (Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006,
/// §4.4 and §4.7; and Davis, *Algorithm 849*, 2005): row `k` of `L` is a sparse triangular solve with the first `k`
/// rows, whose pattern is the reach of column `k` of `A` in the elimination tree, found by walking up the tree from
/// each entry and marking. `ldl_symbolic` walks the same paths once to count the entries of each column of `L`,
/// building the tree as it goes. `etree` is Liu's algorithm with path compression. Rows of `L` within each column come
/// out ascending. Static regularisation (as OSQP and Clarabel do): with `signs`, a pivot `d` whose sign is not
/// `signs[k]`, or whose magnitude is below `delta`, becomes `signs[k] · delta`; `ldl_refine` against the unregularised
/// matrix removes the perturbation from the solution.
///
/// **Guarantees.** `constexpr`, `noexcept`, no allocation, no recursion. `ldl_symbolic` is O(nnz(L) + n) time;
/// `ldl_numeric` is O(Σ |L_{*j}|²) flops, the same as a dense-in-the-pattern factorisation; `ldl_solve` is
/// O(nnz(L) + n). On failure the outputs written so far are unspecified; the inputs are not modified.
///
/// **Errors.** `etree`, `ldl_symbolic`: `sparse_error` (`dimension_mismatch` not square, `invalid_index` not valid,
/// `insufficient_capacity` short spans or an `nnz(L)` that does not fit the index type). `ldl_numeric`:
/// `ldl_error::dimension_mismatch` (a different order from the symbolic analysis), `insufficient_capacity` (short
/// spans), `zero_pivot` (without a sign for that row, a pivot of exactly zero; with or without signs, a pivot that is
/// not finite, from non-finite input).
///
/// **Performance.** `benchmarks/sparse/ldl_benchmark.cpp` (`-O3`, Apple clang 17, Xeon W-3245, a loaded machine,
/// 2026-11-23; CPU time). The 100 x 100 grid Laplacian: in AMD order `nnz(L)` = 186,098, symbolic 1.4 ms, numeric
/// 7.7 ms, solve 0.52 ms; in natural order `nnz(L)` = 990,099, numeric 52 ms, solve 3.6 ms. A random KKT system of
/// 5,000 rows (3,000 variables, 2,000 local constraints): in AMD order `nnz(L)` = 24,118, symbolic 0.28 ms, numeric
/// 0.60 ms, solve 0.09 ms; in natural order (the constraints last, so their block fills in) `nnz(L)` = 5.0 million,
/// numeric 5.6 s. Not compared against another implementation.
///
/// **Tests and demos.** `tests/sparse/ldl_test.cpp`, `demos/sparse/ldl_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`, `sparse/csc.hpp`,
/// `result.hpp`.
///
/// **Known issues.** Entries below the diagonal of the input are ignored: pass the upper triangle. A structurally
/// missing diagonal entry is a zero pivot unless a later update makes it non-zero. There is no pivoting: a matrix
/// that is indefinite but not quasi-definite needs signs (and refinement) or a pivoting factorisation.
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_sparse_ldl_hpp
#define snicholls_core_sparse_ldl_hpp

#include <snicholls/core/result.hpp>
#include <snicholls/core/sparse/csc.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::sparse {

    /// \brief Why a numeric factorisation has no answer.
    /// \ingroup core
    enum class ldl_error : std::uint8_t {
        insufficient_capacity = 1,  ///< An output's (or working) span is too short.
        invalid_index = 2,          ///< Reserved, as `sparse_error::invalid_index`.
        dimension_mismatch = 3,     ///< The matrix's order is not the symbolic analysis's.
        zero_pivot = 4,             ///< A pivot is zero (where no sign is expected) or not finite.
    };

    /// \brief The symbolic analysis of `L D Lᵀ`: the elimination tree, the column counts of `L`, its column offsets
    /// and `nnz(L)`, as views of the storage `ldl_symbolic` wrote.
    /// \ingroup core
    /// \tparam I The index type.
    template <typename I = std::int32_t>
    struct ldl_symbolic_result {
        std::size_t n = 0;              ///< The order.
        std::span<const I> parent{};    ///< The elimination tree: `parent[j]`, -1 for a root.
        std::span<const I> counts{};    ///< The entries in each column of `L` below the diagonal.
        std::span<const I> col_ptr{};   ///< `L`'s column offsets, `n + 1`: the `col_ptr` of the `L` to pass to `ldl_numeric`.
        std::size_t nnz = 0;            ///< `nnz(L)`, below the diagonal: the length `L`'s `row_idx` and `values` need.
    };

    /// \brief The result of `ldl_refine`.
    /// \ingroup core
    /// \tparam T The value type.
    template <typename T>
    struct ldl_refine_result {
        std::size_t iterations = 0;     ///< The corrections applied.
        T residual{};                   ///< `‖b - A x‖∞` at the end.
    };

    namespace detail::ldl {
        template <typename J>
        constexpr bool square_valid(std::size_t rows, std::size_t cols) noexcept
        {
            return rows == cols && cols < static_cast<std::size_t>(std::numeric_limits<J>::max());
        }
    } // namespace detail::ldl

    /// \brief The elimination tree of a symmetric matrix given as its upper triangle (Liu's algorithm, with path
    /// compression).
    /// \ingroup core
    /// \tparam T The value type (unused: only the pattern is read).
    /// \tparam I The index type.
    /// \param a The upper triangle, square and valid; entries below the diagonal are ignored.
    /// \param parent Receives `parent[j]`, the parent of `j` in the tree, -1 for a root; `a.cols` entries.
    /// \param work Working storage of `a.cols` indices.
    /// \return Nothing, or a `sparse_error`.
    template <typename T, typename I>
    [[nodiscard]] constexpr result<void, sparse_error> etree(const csc_view<T, I>& a, std::span<std::remove_const_t<I>> parent, std::span<std::remove_const_t<I>> work) noexcept
    {
        using J = std::remove_const_t<I>;
        const std::size_t n = a.cols;
        if (!detail::ldl::square_valid<J>(a.rows, n))
            return err(sparse_error::dimension_mismatch);
        if (!is_valid(a))
            return err(sparse_error::invalid_index);
        if (parent.size() < n || work.size() < n)
            return err(sparse_error::insufficient_capacity);
        const auto ancestor = work.first(n);
        for (std::size_t k = 0; k < n; ++k) {
            parent[k] = ancestor[k] = -1;
            for (std::size_t q = detail::at(a.col_ptr[k]); q < detail::at(a.col_ptr[k + 1]); ++q) {
                // Climb from i to the root of its current subtree, pointing every node on the way at k
                for (J i = a.row_idx[q]; i != -1 && detail::at(i) < k;) {
                    const J up = ancestor[i];
                    ancestor[i] = static_cast<J>(k);
                    if (up == -1)
                        parent[i] = static_cast<J>(k);
                    i = up;
                }
            }
        }
        return ok();
    }

    /// \brief The symbolic analysis: the elimination tree, the column counts of `L` and its column offsets.
    /// \ingroup core
    /// \tparam T The value type (unused: only the pattern is read).
    /// \tparam I The index type.
    /// \param a The upper triangle, square and valid; entries below the diagonal are ignored.
    /// \param parent Receives the elimination tree, `a.cols` entries.
    /// \param counts Receives the entries of each column of `L` below the diagonal, `a.cols` entries.
    /// \param col_ptr Receives `L`'s column offsets, `a.cols + 1` entries.
    /// \param work Working storage of `a.cols` indices.
    /// \return The analysis (views of `parent`, `counts`, `col_ptr`, and `nnz(L)`), or a `sparse_error`.
    template <typename T, typename I>
    [[nodiscard]] constexpr result<ldl_symbolic_result<std::remove_const_t<I>>, sparse_error>
    ldl_symbolic(const csc_view<T, I>& a, std::span<std::remove_const_t<I>> parent, std::span<std::remove_const_t<I>> counts, std::span<std::remove_const_t<I>> col_ptr,
                 std::span<std::remove_const_t<I>> work) noexcept
    {
        using J = std::remove_const_t<I>;
        const std::size_t n = a.cols;
        if (!detail::ldl::square_valid<J>(a.rows, n))
            return err(sparse_error::dimension_mismatch);
        if (!is_valid(a))
            return err(sparse_error::invalid_index);
        if (parent.size() < n || counts.size() < n || col_ptr.size() < n + 1 || work.size() < n)
            return err(sparse_error::insufficient_capacity);
        const auto flag = work.first(n);
        for (std::size_t k = 0; k < n; ++k) {
            const J kk = static_cast<J>(k);
            parent[k] = -1;
            flag[k] = kk;
            counts[k] = 0;
            for (std::size_t q = detail::at(a.col_ptr[k]); q < detail::at(a.col_ptr[k + 1]); ++q) {
                // Row k of L has an entry in every column on the tree path from i up to k
                for (J i = a.row_idx[q]; detail::at(i) < k && flag[i] != kk; i = parent[i]) {
                    if (parent[i] == -1)
                        parent[i] = kk;
                    ++counts[i];
                    flag[i] = kk;
                }
            }
        }
        std::size_t total = 0;
        col_ptr[0] = 0;
        for (std::size_t k = 0; k < n; ++k) {
            total += detail::at(counts[k]);
            if (total > static_cast<std::size_t>(std::numeric_limits<J>::max()))
                return err(sparse_error::insufficient_capacity);
            col_ptr[k + 1] = static_cast<J>(total);
        }
        return ok(ldl_symbolic_result<J>{n, parent.first(n), counts.first(n), col_ptr.first(n + 1), total});
    }

    /// \brief The numeric factorisation `A = L D Lᵀ`, up-looking, without pivoting, on the pattern of a symbolic
    /// analysis; with `signs`, static regularisation of the pivots.
    /// \ingroup core
    /// \tparam T The matrix's value type (may be `const`).
    /// \tparam I The matrix's index type (may be `const`).
    /// \tparam U The factor's value type.
    /// \param a The upper triangle, the same pattern the analysis saw (values may differ); entries below the diagonal
    /// are ignored.
    /// \param symbolic The analysis, from `ldl_symbolic`.
    /// \param l Receives the strictly lower part of the unit lower triangular `L`: `row_idx` and `values` at least
    /// `symbolic.nnz`, `col_ptr` at least `n + 1` (`symbolic.col_ptr`'s storage, or other storage it is copied to);
    /// its `rows` and `cols` are set.
    /// \param d Receives the pivots `D`, `n` values.
    /// \param dinv Receives `D⁻¹`, `n` values (what `ldl_solve` uses).
    /// \param iwork Working storage of `3n` indices.
    /// \param xwork Working storage of `n` values.
    /// \param signs Optional, `n` entries: the expected sign of each pivot (+1, -1; 0 for no expectation). Where
    /// given, a pivot of the other sign or of magnitude below `delta` becomes `signs[k] · delta`.
    /// \param delta The static regularisation, positive.
    /// \return The number of pivots regularised, or an `ldl_error`.
    template <typename T, typename I, typename U>
    [[nodiscard]] constexpr result<std::size_t, ldl_error> ldl_numeric(const csc_view<T, I>& a, const ldl_symbolic_result<std::remove_const_t<I>>& symbolic,
                                                                       csc_view<U, std::remove_const_t<I>>& l, std::span<std::type_identity_t<U>> d,
                                                                       std::span<std::type_identity_t<U>> dinv, std::span<std::remove_const_t<I>> iwork,
                                                                       std::span<std::type_identity_t<U>> xwork, std::span<const std::int8_t> signs = {},
                                                                       std::type_identity_t<U> delta = static_cast<U>(1e-8)) noexcept
    {
        using J = std::remove_const_t<I>;
        const std::size_t n = symbolic.n;
        if (a.rows != n || a.cols != n || symbolic.col_ptr.size() < n + 1 || symbolic.parent.size() < n)
            return err(ldl_error::dimension_mismatch);
        if (l.col_ptr.size() < n + 1 || l.row_idx.size() < symbolic.nnz || l.values.size() < symbolic.nnz || d.size() < n || dinv.size() < n || iwork.size() < 3 * n
            || xwork.size() < n || (!signs.empty() && signs.size() < n))
            return err(ldl_error::insufficient_capacity);
        l.rows = l.cols = n;
        if (l.col_ptr.data() != symbolic.col_ptr.data())
            for (std::size_t k = 0; k <= n; ++k)
                l.col_ptr[k] = symbolic.col_ptr[k];
        const auto lp = symbolic.col_ptr;
        const auto parent = symbolic.parent;
        const auto pattern = iwork.first(n), flag = iwork.subspan(n, n), lnz = iwork.subspan(2 * n, n);
        const auto y = xwork.first(n);
        std::size_t regularised = 0;
        for (std::size_t k = 0; k < n; ++k) {
            const J kk = static_cast<J>(k);
            // The pattern of row k (the reach of column k in the tree), topologically ordered in pattern[top, n)
            y[k] = U(0);
            std::size_t top = n;
            flag[k] = kk;
            lnz[k] = 0;
            for (std::size_t q = detail::at(a.col_ptr[k]); q < detail::at(a.col_ptr[k + 1]); ++q) {
                J i = a.row_idx[q];
                if (detail::at(i) > k)
                    continue;
                y[detail::at(i)] += static_cast<U>(a.values[q]);
                std::size_t depth = 0;
                for (; i != -1 && flag[i] != kk; i = parent[i]) {
                    pattern[depth++] = i;
                    flag[i] = kk;
                }
                while (depth > 0)
                    pattern[--top] = pattern[--depth];
            }
            // The sparse triangular solve for row k, and the pivot
            U dk = y[k];
            y[k] = U(0);
            for (; top < n; ++top) {
                const std::size_t i = detail::at(pattern[top]);
                const U yi = y[i];
                y[i] = U(0);
                const std::size_t p0 = detail::at(lp[i]), p1 = p0 + detail::at(lnz[i]);
                for (std::size_t p = p0; p < p1; ++p)
                    y[detail::at(l.row_idx[p])] -= l.values[p] * yi;
                const U lki = yi * dinv[i];
                dk -= lki * yi;
                l.row_idx[p1] = kk;
                l.values[p1] = lki;
                ++lnz[i];
            }
            if (!(dk - dk == U(0)))
                return err(ldl_error::zero_pivot);   // not finite
            const int s = signs.empty() ? 0 : signs[k] > 0 ? 1 : signs[k] < 0 ? -1 : 0;
            if (s != 0 && !(static_cast<U>(s) * dk >= delta)) {
                dk = static_cast<U>(s) * delta;
                ++regularised;
            }
            if (dk == U(0))
                return err(ldl_error::zero_pivot);
            d[k] = dk;
            dinv[k] = U(1) / dk;
        }
        return ok(regularised);
    }

    /// \brief The numeric factorisation again, with new values on the same pattern: this is `ldl_numeric`, under the
    /// name a solver loop reads best with. The symbolic analysis is not repeated.
    /// \ingroup core
    /// \tparam T The matrix's value type.
    /// \tparam I The matrix's index type.
    /// \tparam U The factor's value type.
    /// \param a The upper triangle with the new values, on the analysed pattern.
    /// \param symbolic The analysis, from `ldl_symbolic`.
    /// \param l The factor, overwritten.
    /// \param d The pivots, overwritten.
    /// \param dinv Their inverses, overwritten.
    /// \param iwork Working storage of `3n` indices.
    /// \param xwork Working storage of `n` values.
    /// \param signs Optional expected signs, as `ldl_numeric`.
    /// \param delta The static regularisation.
    /// \return As `ldl_numeric`.
    template <typename T, typename I, typename U>
    [[nodiscard]] constexpr result<std::size_t, ldl_error> ldl_refactor(const csc_view<T, I>& a, const ldl_symbolic_result<std::remove_const_t<I>>& symbolic,
                                                                        csc_view<U, std::remove_const_t<I>>& l, std::span<std::type_identity_t<U>> d,
                                                                        std::span<std::type_identity_t<U>> dinv, std::span<std::remove_const_t<I>> iwork,
                                                                        std::span<std::type_identity_t<U>> xwork, std::span<const std::int8_t> signs = {},
                                                                        std::type_identity_t<U> delta = static_cast<U>(1e-8)) noexcept
    {
        return ldl_numeric(a, symbolic, l, d, dinv, iwork, xwork, signs, delta);
    }

    /// \brief Solves `L D Lᵀ x = b` in place.
    /// \ingroup core
    /// \tparam T The factor's value type (may be `const`).
    /// \tparam I The index type (may be `const`).
    /// \tparam U The vector's value type.
    /// \param l The strictly lower part of `L`, from `ldl_numeric`.
    /// \param dinv `D⁻¹`, from `ldl_numeric`.
    /// \param x `b` on entry, `x` on return; `l.cols` values.
    template <typename T, typename I, typename U>
    constexpr void ldl_solve(const csc_view<T, I>& l, std::span<const std::type_identity_t<U>> dinv, std::span<U> x) noexcept
    {
        const std::size_t n = l.cols;
        for (std::size_t j = 0; j < n; ++j) {
            const U xj = x[j];
            for (std::size_t p = detail::at(l.col_ptr[j]); p < detail::at(l.col_ptr[j + 1]); ++p)
                x[detail::at(l.row_idx[p])] -= static_cast<U>(l.values[p]) * xj;
        }
        for (std::size_t j = 0; j < n; ++j)
            x[j] *= dinv[j];
        for (std::size_t j = n; j-- > 0;) {
            U s = x[j];
            for (std::size_t p = detail::at(l.col_ptr[j]); p < detail::at(l.col_ptr[j + 1]); ++p)
                s -= static_cast<U>(l.values[p]) * x[detail::at(l.row_idx[p])];
            x[j] = s;
        }
    }

    /// \brief Iterative refinement: `x ← x + (L D Lᵀ)⁻¹ (b - A x)` until `‖b - A x‖∞ ≤ tolerance` or `iterations`
    /// corrections, against `A` itself (so a regularised factorisation still converges to `A`'s solution when `A` is
    /// nonsingular).
    /// \ingroup core
    /// \tparam T The matrix's value type (may be `const`).
    /// \tparam I The matrix's index type (may be `const`).
    /// \tparam TL The factor's value type (may be `const`).
    /// \tparam IL The factor's index type (may be `const`).
    /// \tparam U The vectors' value type.
    /// \param a The upper triangle of `A` (the same numbering as `L`).
    /// \param l The factor's strictly lower part.
    /// \param dinv `D⁻¹`.
    /// \param b The right-hand side, `n` values.
    /// \param x The current solution (from `ldl_solve`), improved in place.
    /// \param residual Working storage of `n` values.
    /// \param iterations The most corrections.
    /// \param tolerance Stop when the residual's largest magnitude is at or below it.
    /// \return The corrections applied and the final residual norm.
    template <typename T, typename I, typename TL, typename IL, typename U>
    constexpr ldl_refine_result<U> ldl_refine(const csc_view<T, I>& a, const csc_view<TL, IL>& l, std::span<const std::type_identity_t<U>> dinv,
                                              std::span<const std::type_identity_t<U>> b, std::span<U> x, std::span<std::type_identity_t<U>> residual,
                                              std::size_t iterations, std::type_identity_t<U> tolerance) noexcept
    {
        const std::size_t n = a.cols;
        for (std::size_t it = 0;; ++it) {
            spmv_symmetric_upper(a, std::span<const U>(x.data(), n), residual.first(n));
            U norm(0);
            for (std::size_t i = 0; i < n; ++i) {
                residual[i] = b[i] - residual[i];
                const U m = residual[i] < U(0) ? -residual[i] : residual[i];
                norm = m > norm ? m : norm;
            }
            if (norm <= tolerance || it == iterations)
                return {it, norm};
            ldl_solve(l, dinv, residual.first(n));
            for (std::size_t i = 0; i < n; ++i)
                x[i] += residual[i];
        }
    }

    /// \brief `nnz(L)` for a matrix of order `N`, with internal storage: to size an `ldl_fixed` in a constant
    /// expression.
    /// \ingroup core
    /// \tparam N The order.
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param a The upper triangle, `N x N`.
    /// \return `nnz(L)`, or a `sparse_error` (as `ldl_symbolic`; `dimension_mismatch` if `a` is not `N x N`).
    template <std::size_t N, typename T, typename I>
    [[nodiscard]] constexpr result<std::size_t, sparse_error> ldl_nnz(const csc_view<T, I>& a) noexcept
    {
        using J = std::remove_const_t<I>;
        if (a.rows != N || a.cols != N)
            return err(sparse_error::dimension_mismatch);
        std::array<J, N + 1> parent{}, counts{}, col_ptr{}, work{};
        const auto s = ldl_symbolic(a, std::span<J>(parent), std::span<J>(counts), std::span<J>(col_ptr), std::span<J>(work));
        if (!s)
            return err(s.unwrap_err());
        return ok(s->nnz);
    }

    /// \brief Every array of an `L D Lᵀ` of fixed order and fixed `nnz(L)`, inside one object: for a problem whose
    /// size is known when the program is written, and for a factorisation in a constant expression.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam N The order.
    /// \tparam MaxNnzL The most entries of `L` below the diagonal (`ldl_nnz<N>(a)` gives the exact need).
    /// \tparam I The index type.
    template <typename T, std::size_t N, std::size_t MaxNnzL, typename I = std::int32_t>
    struct ldl_fixed {
        std::array<I, N> parent{};             ///< The elimination tree.
        std::array<I, N> counts{};             ///< The column counts of `L`.
        std::array<I, N + 1> col_ptr{};        ///< `L`'s column offsets.
        std::array<I, MaxNnzL> row_idx{};      ///< `L`'s row indices.
        std::array<T, MaxNnzL> values{};       ///< `L`'s values.
        std::array<T, N> d{};                  ///< The pivots.
        std::array<T, N> dinv{};               ///< Their inverses.
        std::array<T, N> xwork{};              ///< Working values.
        std::array<I, 3 * N> iwork{};          ///< Working indices.
        std::size_t nnz = 0;                   ///< `nnz(L)` after `analyse`.

        /// \brief The symbolic analysis.
        /// \tparam TA The matrix's value type.
        /// \tparam IA The matrix's index type.
        /// \param a The upper triangle, `N x N`.
        /// \return `nnz(L)`, or a `sparse_error` (`insufficient_capacity` if it exceeds `MaxNnzL`).
        template <typename TA, typename IA>
        [[nodiscard]] constexpr result<std::size_t, sparse_error> analyse(const csc_view<TA, IA>& a) noexcept
        {
            if (a.rows != N || a.cols != N)
                return err(sparse_error::dimension_mismatch);
            const auto s = ldl_symbolic(a, std::span<I>(parent), std::span<I>(counts), std::span<I>(col_ptr), std::span<I>(iwork).first(N));
            if (!s)
                return err(s.unwrap_err());
            if (s->nnz > MaxNnzL)
                return err(sparse_error::insufficient_capacity);
            nnz = s->nnz;
            return ok(nnz);
        }

        /// \brief The analysis as `ldl_numeric` takes it. \return Views of the tree, the counts and the offsets.
        [[nodiscard]] constexpr ldl_symbolic_result<I> symbolic() const noexcept { return {N, parent, counts, col_ptr, nnz}; }

        /// \brief The numeric factorisation (and every refactorisation) after `analyse`.
        /// \tparam TA The matrix's value type.
        /// \tparam IA The matrix's index type.
        /// \param a The upper triangle, on the analysed pattern.
        /// \param signs Optional expected signs, as `ldl_numeric`.
        /// \param delta The static regularisation.
        /// \return The number of pivots regularised, or an `ldl_error`.
        template <typename TA, typename IA>
        [[nodiscard]] constexpr result<std::size_t, ldl_error> factor(const csc_view<TA, IA>& a, std::span<const std::int8_t> signs = {}, T delta = static_cast<T>(1e-8)) noexcept
        {
            csc_view<T, I> l{N, N, col_ptr, row_idx, values};
            return ldl_numeric(a, symbolic(), l, std::span<T>(d), std::span<T>(dinv), std::span<I>(iwork), std::span<T>(xwork), signs, delta);
        }

        /// \brief The factor's strictly lower part. \return A read-only view.
        [[nodiscard]] constexpr csc_view<const T, const I> l() const noexcept { return {N, N, col_ptr, row_idx, values}; }

        /// \brief Solves `L D Lᵀ x = b` in place. \param x `b` on entry, `x` on return; `N` values.
        constexpr void solve(std::span<T> x) const noexcept { ldl_solve(l(), std::span<const T>(dinv), x); }
    };

} // namespace snicholls::sparse

#endif /* snicholls_core_sparse_ldl_hpp */
