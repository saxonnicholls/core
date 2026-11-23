//
//  csc.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::sparse::csc_view` and `csc_matrix`: compressed sparse column matrices over the caller's storage, with assembly from triplets, products, transposes, scaling, norms and symmetric permutation, all `constexpr`.
/// \ingroup core
///
/// **Overview.** Optimisation problems are large and sparse: a portfolio QP's constraint matrix has a few non-zeros per
/// column, a KKT system a few per row. Compressed sparse column (CSC) storage keeps, for an `m x n` matrix with `nnz`
/// stored entries, three arrays: `col_ptr` (`n + 1` offsets), `row_idx` (`nnz` row indices, ascending within each
/// column) and `values` (`nnz` numbers). It is the layout of SuiteSparse, OSQP, QDLDL, SCS and Clarabel.
///
/// - `csc_view<T, I>`: dimensions plus three spans; `T` and `I` may be `const` for a read-only view. It owns nothing.
/// - `csc_matrix<T, Rows, Cols, MaxNnz, I>`: the three arrays inside, sized at compile time, for problems whose
///   structure is fixed when the program is written (and for building one in a constant expression); `view()`.
/// - Assembly: `from_triplets` (any order, duplicates summed), `is_valid`.
/// - Products: `spmv` (`y ← αAx + βy`), `spmv_transpose` (`y ← αAᵀx + βy`), `spmv_symmetric_upper` (for a symmetric
///   matrix stored as its upper triangle, the form of KKT matrices).
/// - Structure: `transpose`, `symmetric_permute_upper` (`P A Pᵀ` of an upper-triangle symmetric matrix, for a
///   fill-reducing ordering), `to_dense`.
/// - Scaling and norms: `scale` (`A ← diag(r) A diag(c)`), `column_inf_norms`, `row_inf_norms`,
///   `symmetric_upper_inf_norms` (equilibration needs them).
///
/// Indices are `std::int32_t` by default (as QDLDL's `c_int`); pass `std::int64_t` for more than 2³¹ entries. Working
/// storage, where a function needs any, is a span the caller passes, so nothing allocates.
///
/// \snippet sparse/csc_demo.cpp csc
///
/// **Relationships.** `sparse/ordering.hpp` (AMD), `sparse/ldl.hpp` (the factorisation), `linalg/krylov.hpp`
/// (iterative solvers over these products), `optim/scaling.hpp` (equilibration), `linalg/` (dense). Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for arithmetic `T` and integer `I`, no allocation, no recursion. Every
/// function that writes a matrix writes row indices ascending within each column, without duplicates. Products are
/// O(nnz); `from_triplets`, `transpose` and `symmetric_permute_upper` are O(nnz + n).
///
/// **Errors.** `sparse_error::insufficient_capacity`: an output's spans are too short. `sparse_error::invalid_index`:
/// a triplet's or permutation's index is out of range. `sparse_error::dimension_mismatch`: shapes that do not fit.
///
/// **Performance.** Not yet benchmarked.
///
/// **Tests and demos.** `tests/sparse/csc_test.cpp`, `demos/sparse/csc_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `result.hpp`.
///
/// **Known issues.** Values of explicitly stored zeros are kept (a structural non-zero may be numerically zero);
/// pattern-reusing factorisations rely on it.
///
/// **Status.** Experimental, new in 0.21.0.

#ifndef snicholls_core_sparse_csc_hpp
#define snicholls_core_sparse_csc_hpp

#include <snicholls/core/result.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::sparse {

    /// \brief Why a sparse operation has no answer.
    /// \ingroup core
    enum class sparse_error : std::uint8_t {
        insufficient_capacity = 1,  ///< An output's (or working) span is too short.
        invalid_index = 2,          ///< An index is out of range (a triplet, a permutation entry).
        dimension_mismatch = 3,     ///< Shapes that do not fit together.
    };

    /// \brief A compressed sparse column matrix over spans the caller owns.
    /// \ingroup core
    /// \tparam T The value type (`const T` for a read-only view).
    /// \tparam I The index type (`const I` for a read-only view).
    template <typename T, typename I = std::int32_t>
    struct csc_view {
        std::size_t rows = 0;          ///< The rows.
        std::size_t cols = 0;          ///< The columns.
        std::span<I> col_ptr{};        ///< `cols + 1` offsets: column `j` is entries `[col_ptr[j], col_ptr[j + 1])`.
        std::span<I> row_idx{};        ///< The row of each stored entry (at least `nnz()` long).
        std::span<T> values{};         ///< The value of each stored entry (at least `nnz()` long).

        /// \brief The stored entries. \return `col_ptr[cols]`, 0 for an empty `col_ptr`.
        [[nodiscard]] constexpr std::size_t nnz() const noexcept { return col_ptr.empty() ? 0 : static_cast<std::size_t>(col_ptr[cols]); }

        /// \brief A read-only view of the same storage. \return It.
        [[nodiscard]] constexpr csc_view<const std::remove_const_t<T>, const std::remove_const_t<I>> as_const() const noexcept
        {
            return {rows, cols, col_ptr, row_idx, values};
        }
        /// \brief A read-only view, implicitly. \return It.
        constexpr operator csc_view<const std::remove_const_t<T>, const std::remove_const_t<I>>() const noexcept
            requires(!std::is_const_v<T> || !std::is_const_v<I>)
        {
            return as_const();
        }
    };

    /// \brief A CSC matrix whose storage is inside it, sized at compile time.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam Rows The rows.
    /// \tparam Cols The columns.
    /// \tparam MaxNnz The most stored entries.
    /// \tparam I The index type.
    template <typename T, std::size_t Rows, std::size_t Cols, std::size_t MaxNnz, typename I = std::int32_t>
    struct csc_matrix {
        std::array<I, Cols + 1> col_ptr{};  ///< The column offsets.
        std::array<I, MaxNnz> row_idx{};    ///< The row indices.
        std::array<T, MaxNnz> values{};     ///< The values.

        /// \brief A view of the storage. \return It.
        [[nodiscard]] constexpr csc_view<T, I> view() noexcept { return {Rows, Cols, col_ptr, row_idx, values}; }
        /// \brief A read-only view of the storage. \return It.
        [[nodiscard]] constexpr csc_view<const T, const I> view() const noexcept { return {Rows, Cols, col_ptr, row_idx, values}; }
        /// \brief The stored entries. \return `col_ptr[Cols]`.
        [[nodiscard]] constexpr std::size_t nnz() const noexcept { return static_cast<std::size_t>(col_ptr[Cols]); }
    };

    namespace detail {
        template <typename I>
        constexpr std::size_t at(I i) noexcept
        {
            return static_cast<std::size_t>(i);
        }
    } // namespace detail

    /// \brief Whether a view is a well-formed CSC matrix: offsets from 0, non-decreasing, spans long enough, row indices
    /// in range and strictly ascending within each column.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param a The matrix.
    /// \return True if it is well formed.
    template <typename T, typename I>
    [[nodiscard]] constexpr bool is_valid(const csc_view<T, I>& a) noexcept
    {
        if (a.col_ptr.size() < a.cols + 1 || a.col_ptr[0] != 0)
            return false;
        for (std::size_t j = 0; j < a.cols; ++j)
            if (a.col_ptr[j + 1] < a.col_ptr[j])
                return false;
        const std::size_t nnz = a.nnz();
        if (a.row_idx.size() < nnz || a.values.size() < nnz)
            return false;
        for (std::size_t j = 0; j < a.cols; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                if (a.row_idx[k] < 0 || detail::at(a.row_idx[k]) >= a.rows)
                    return false;
                if (k > detail::at(a.col_ptr[j]) && !(a.row_idx[k - 1] < a.row_idx[k]))
                    return false;
            }
        return true;
    }

    /// \brief Assembles a CSC matrix from triplets `(i, j, v)` in any order, summing duplicates.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param rows The rows.
    /// \param cols The columns.
    /// \param ti The triplets' rows.
    /// \param tj The triplets' columns.
    /// \param tv The triplets' values.
    /// \param out The matrix to write: `col_ptr` at least `cols + 1`, `row_idx` and `values` at least the triplets; its
    /// `rows` and `cols` are set.
    /// \param work Working storage of at least `max(rows, cols)` indices.
    /// \return The stored entries (after summing duplicates), or a `sparse_error`.
    template <typename T, typename I>
    [[nodiscard]] constexpr result<std::size_t, sparse_error> from_triplets(std::size_t rows, std::size_t cols, std::span<const I> ti, std::span<const I> tj, std::span<const T> tv,
                                                                           csc_view<T, I>& out, std::span<I> work) noexcept
    {
        const std::size_t n = ti.size();
        if (tj.size() != n || tv.size() != n)
            return err(sparse_error::dimension_mismatch);
        if (out.col_ptr.size() < cols + 1 || out.row_idx.size() < n || out.values.size() < n || work.size() < (rows > cols ? rows : cols))
            return err(sparse_error::insufficient_capacity);
        for (std::size_t k = 0; k < n; ++k)
            if (ti[k] < 0 || tj[k] < 0 || detail::at(ti[k]) >= rows || detail::at(tj[k]) >= cols)
                return err(sparse_error::invalid_index);
        out.rows = rows;
        out.cols = cols;
        // Count per column, then place (a counting sort by column, stable in triplet order)
        for (std::size_t j = 0; j <= cols; ++j)
            out.col_ptr[j] = 0;
        for (std::size_t k = 0; k < n; ++k)
            ++out.col_ptr[detail::at(tj[k]) + 1];
        for (std::size_t j = 0; j < cols; ++j)
            out.col_ptr[j + 1] += out.col_ptr[j];
        for (std::size_t j = 0; j < cols; ++j)
            work[j] = out.col_ptr[j];
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t p = detail::at(work[detail::at(tj[k])]++);
            out.row_idx[p] = ti[k];
            out.values[p] = tv[k];
        }
        // Within each column: sort by row (insertion: columns are short) and sum duplicates, compacting in place
        std::size_t write = 0;
        for (std::size_t j = 0; j < cols; ++j) {
            const std::size_t lo = detail::at(out.col_ptr[j]), hi = detail::at(out.col_ptr[j + 1]);
            for (std::size_t a = lo + 1; a < hi; ++a)
                for (std::size_t b = a; b > lo && out.row_idx[b] < out.row_idx[b - 1]; --b) {
                    const I ri = out.row_idx[b];
                    out.row_idx[b] = out.row_idx[b - 1];
                    out.row_idx[b - 1] = ri;
                    const T vi = out.values[b];
                    out.values[b] = out.values[b - 1];
                    out.values[b - 1] = vi;
                }
            out.col_ptr[j] = static_cast<I>(write);
            for (std::size_t k = lo; k < hi; ++k) {
                if (write > detail::at(out.col_ptr[j]) && out.row_idx[write - 1] == out.row_idx[k])
                    out.values[write - 1] += out.values[k];
                else {
                    out.row_idx[write] = out.row_idx[k];
                    out.values[write] = out.values[k];
                    ++write;
                }
            }
        }
        out.col_ptr[cols] = static_cast<I>(write);
        return ok(write);
    }

    /// \brief `y ← α A x + β y`.
    /// \ingroup core
    /// \tparam T The matrix's value type.
    /// \tparam I The index type.
    /// \tparam U The vectors' value type.
    /// \param a The matrix.
    /// \param x A vector of `a.cols` values.
    /// \param y A vector of `a.rows` values, updated.
    /// \param alpha The product's scale.
    /// \param beta `y`'s scale (0 overwrites `y`, ignoring what was there, NaN included).
    template <typename T, typename I, typename U>
    constexpr void spmv(const csc_view<T, I>& a, std::span<const U> x, std::span<U> y, U alpha = U(1), U beta = U(0)) noexcept
    {
        for (std::size_t i = 0; i < a.rows; ++i)
            y[i] = beta == U(0) ? U(0) : beta * y[i];
        for (std::size_t j = 0; j < a.cols; ++j) {
            const U xj = alpha * x[j];
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k)
                y[detail::at(a.row_idx[k])] += static_cast<U>(a.values[k]) * xj;
        }
    }

    /// \brief `y ← α Aᵀ x + β y`.
    /// \ingroup core
    /// \tparam T The matrix's value type.
    /// \tparam I The index type.
    /// \tparam U The vectors' value type.
    /// \param a The matrix.
    /// \param x A vector of `a.rows` values.
    /// \param y A vector of `a.cols` values, updated.
    /// \param alpha The product's scale.
    /// \param beta `y`'s scale (0 overwrites `y`).
    template <typename T, typename I, typename U>
    constexpr void spmv_transpose(const csc_view<T, I>& a, std::span<const U> x, std::span<U> y, U alpha = U(1), U beta = U(0)) noexcept
    {
        for (std::size_t j = 0; j < a.cols; ++j) {
            U s(0);
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k)
                s += static_cast<U>(a.values[k]) * x[detail::at(a.row_idx[k])];
            y[j] = (beta == U(0) ? U(0) : beta * y[j]) + alpha * s;
        }
    }

    /// \brief `y ← α A x + β y` for a symmetric `A` stored as its upper triangle (diagonal included; entries below
    /// the diagonal, if any, are ignored).
    /// \ingroup core
    /// \tparam T The matrix's value type.
    /// \tparam I The index type.
    /// \tparam U The vectors' value type.
    /// \param a The upper triangle, square.
    /// \param x A vector of `a.cols` values.
    /// \param y A vector of `a.rows` values, updated.
    /// \param alpha The product's scale.
    /// \param beta `y`'s scale (0 overwrites `y`).
    template <typename T, typename I, typename U>
    constexpr void spmv_symmetric_upper(const csc_view<T, I>& a, std::span<const U> x, std::span<U> y, U alpha = U(1), U beta = U(0)) noexcept
    {
        for (std::size_t i = 0; i < a.rows; ++i)
            y[i] = beta == U(0) ? U(0) : beta * y[i];
        for (std::size_t j = 0; j < a.cols; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                const std::size_t i = detail::at(a.row_idx[k]);
                if (i > j)
                    continue;
                const U v = alpha * static_cast<U>(a.values[k]);
                y[i] += v * x[j];
                if (i != j)
                    y[j] += v * x[i];
            }
    }

    /// \brief `Aᵀ`.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \tparam U The output's value type.
    /// \tparam J The output's index type.
    /// \param a The matrix.
    /// \param out The transpose: `col_ptr` at least `a.rows + 1`, `row_idx` and `values` at least `a.nnz()`; its
    /// `rows` and `cols` are set.
    /// \param work Working storage of at least `a.rows` indices.
    /// \return Nothing, or `sparse_error::insufficient_capacity`.
    template <typename T, typename I, typename U, typename J>
    [[nodiscard]] constexpr result<void, sparse_error> transpose(const csc_view<T, I>& a, csc_view<U, J>& out, std::span<J> work) noexcept
    {
        const std::size_t nnz = a.nnz();
        if (out.col_ptr.size() < a.rows + 1 || out.row_idx.size() < nnz || out.values.size() < nnz || work.size() < a.rows)
            return err(sparse_error::insufficient_capacity);
        out.rows = a.cols;
        out.cols = a.rows;
        for (std::size_t i = 0; i <= a.rows; ++i)
            out.col_ptr[i] = 0;
        for (std::size_t k = 0; k < nnz; ++k)
            ++out.col_ptr[detail::at(a.row_idx[k]) + 1];
        for (std::size_t i = 0; i < a.rows; ++i)
            out.col_ptr[i + 1] += out.col_ptr[i];
        for (std::size_t i = 0; i < a.rows; ++i)
            work[i] = out.col_ptr[i];
        // Columns in order, so each output column's rows come out ascending
        for (std::size_t j = 0; j < a.cols; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                const std::size_t p = detail::at(work[detail::at(a.row_idx[k])]++);
                out.row_idx[p] = static_cast<J>(j);
                out.values[p] = static_cast<U>(a.values[k]);
            }
        return ok();
    }

    /// \brief The upper triangle of `C = P A Pᵀ` for a symmetric `A` stored as its upper triangle: `C(i, j) = A(p⁻¹…)`,
    /// with `perm[k]` the original index of the new index `k` (the form orderings return).
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \param a The upper triangle of `A`, square.
    /// \param perm The permutation: `perm[new] = old`.
    /// \param out The upper triangle of `C`: `col_ptr` at least `n + 1`, `row_idx` and `values` at least `a.nnz()`.
    /// \param work Working storage of at least `2n` indices.
    /// \param map Optional: receives, for each of `a`'s entries, its position in `out` (for refactorising with new
    /// values on the same pattern); empty to skip.
    /// \return Nothing, or a `sparse_error`.
    template <typename T, typename I>
    [[nodiscard]] constexpr result<void, sparse_error> symmetric_permute_upper(const csc_view<const T, const I>& a, std::span<const I> perm, csc_view<T, I>& out,
                                                                              std::span<I> work, std::span<I> map = {}) noexcept
    {
        const std::size_t n = a.cols, nnz = a.nnz();
        if (a.rows != n || perm.size() != n)
            return err(sparse_error::dimension_mismatch);
        if (out.col_ptr.size() < n + 1 || out.row_idx.size() < nnz || out.values.size() < nnz || work.size() < 2 * n || (!map.empty() && map.size() < nnz))
            return err(sparse_error::insufficient_capacity);
        const auto inverse = work.first(n);   // inverse[old] = new
        const auto next = work.subspan(n, n);
        for (std::size_t k = 0; k < n; ++k)
            inverse[k] = -1;
        for (std::size_t k = 0; k < n; ++k) {
            if (perm[k] < 0 || detail::at(perm[k]) >= n || inverse[detail::at(perm[k])] != -1)
                return err(sparse_error::invalid_index);
            inverse[detail::at(perm[k])] = static_cast<I>(k);
        }
        out.rows = out.cols = n;
        for (std::size_t j = 0; j <= n; ++j)
            out.col_ptr[j] = 0;
        // Count: entry (i, j), i ≤ j, goes to column max(new i, new j)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                const std::size_t i = detail::at(a.row_idx[k]);
                if (i > j)
                    continue;
                const auto ni = inverse[i], nj = inverse[j];
                ++out.col_ptr[detail::at(ni > nj ? ni : nj) + 1];
            }
        for (std::size_t j = 0; j < n; ++j)
            out.col_ptr[j + 1] += out.col_ptr[j];
        for (std::size_t j = 0; j < n; ++j)
            next[j] = out.col_ptr[j];
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                const std::size_t i = detail::at(a.row_idx[k]);
                if (i > j)
                    continue;
                const auto ni = inverse[i], nj = inverse[j];
                const auto col = ni > nj ? ni : nj, row = ni > nj ? nj : ni;
                const std::size_t p = detail::at(next[detail::at(col)]++);
                out.row_idx[p] = row;
                out.values[p] = a.values[k];
                if (!map.empty())
                    map[k] = static_cast<I>(p);
            }
        // Rows ascending within each column (insertion, carrying the map's back-pointers by a second pass)
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t lo = detail::at(out.col_ptr[j]), hi = detail::at(out.col_ptr[j + 1]);
            for (std::size_t s = lo + 1; s < hi; ++s)
                for (std::size_t b = s; b > lo && out.row_idx[b] < out.row_idx[b - 1]; --b) {
                    const I r = out.row_idx[b];
                    out.row_idx[b] = out.row_idx[b - 1];
                    out.row_idx[b - 1] = r;
                    const T v = out.values[b];
                    out.values[b] = out.values[b - 1];
                    out.values[b - 1] = v;
                }
        }
        if (!map.empty()) {
            // Recompute the map against the sorted columns: find each original entry's row in its target column
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                    const std::size_t i = detail::at(a.row_idx[k]);
                    if (i > j) {
                        map[k] = -1;
                        continue;
                    }
                    const auto ni = inverse[i], nj = inverse[j];
                    const auto col = detail::at(ni > nj ? ni : nj);
                    const auto row = ni > nj ? nj : ni;
                    for (std::size_t p = detail::at(out.col_ptr[col]); p < detail::at(out.col_ptr[col + 1]); ++p)
                        if (out.row_idx[p] == row) {
                            map[k] = static_cast<I>(p);
                            break;
                        }
                }
        }
        return ok();
    }

    /// \brief `A ← diag(r) A diag(c)`, in place on the values.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam I The index type.
    /// \tparam U The scales' type.
    /// \param a The matrix (mutable values).
    /// \param r The row scales, `a.rows` of them; empty for none.
    /// \param c The column scales, `a.cols` of them; empty for none.
    template <typename T, typename I, typename U>
    constexpr void scale(const csc_view<T, I>& a, std::span<const U> r, std::span<const U> c) noexcept
    {
        for (std::size_t j = 0; j < a.cols; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                T v = a.values[k];
                if (!r.empty())
                    v *= static_cast<T>(r[detail::at(a.row_idx[k])]);
                if (!c.empty())
                    v *= static_cast<T>(c[j]);
                a.values[k] = v;
            }
    }

    /// \brief The largest magnitude in each column.
    /// \ingroup core
    /// \tparam T The value type. \tparam I The index type.
    /// \tparam U The output's type. \param a The matrix. \param out `a.cols` values: `maxᵢ |aᵢⱼ|` (0 for an empty column).
    template <typename T, typename I, typename U>
    constexpr void column_inf_norms(const csc_view<T, I>& a, std::span<U> out) noexcept
    {
        for (std::size_t j = 0; j < a.cols; ++j) {
            U m(0);
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                const U v = static_cast<U>(a.values[k] < T(0) ? -a.values[k] : a.values[k]);
                m = v > m ? v : m;
            }
            out[j] = m;
        }
    }

    /// \brief The largest magnitude in each row.
    /// \ingroup core
    /// \tparam T The value type. \tparam I The index type.
    /// \tparam U The output's type. \param a The matrix. \param out `a.rows` values: `maxⱼ |aᵢⱼ|` (0 for an empty row).
    template <typename T, typename I, typename U>
    constexpr void row_inf_norms(const csc_view<T, I>& a, std::span<U> out) noexcept
    {
        for (std::size_t i = 0; i < a.rows; ++i)
            out[i] = U(0);
        for (std::size_t j = 0; j < a.cols; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                const U v = static_cast<U>(a.values[k] < T(0) ? -a.values[k] : a.values[k]);
                U& m = out[detail::at(a.row_idx[k])];
                m = v > m ? v : m;
            }
    }

    /// \brief The largest magnitude in each column of a symmetric matrix stored as its upper triangle (each stored
    /// off-diagonal entry counts for its row's column too).
    /// \ingroup core
    /// \tparam T The value type. \tparam I The index type. \tparam U The output's type.
    /// \param a The upper triangle, square. \param out `a.cols` values.
    template <typename T, typename I, typename U>
    constexpr void symmetric_upper_inf_norms(const csc_view<T, I>& a, std::span<U> out) noexcept
    {
        for (std::size_t j = 0; j < a.cols; ++j)
            out[j] = U(0);
        for (std::size_t j = 0; j < a.cols; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k) {
                const std::size_t i = detail::at(a.row_idx[k]);
                if (i > j)
                    continue;
                const U v = static_cast<U>(a.values[k] < T(0) ? -a.values[k] : a.values[k]);
                out[j] = v > out[j] ? v : out[j];
                out[i] = v > out[i] ? v : out[i];
            }
    }

    /// \brief The matrix as dense, row-major.
    /// \ingroup core
    /// \tparam T The value type. \tparam I The index type.
    /// \tparam U The output's type. \param a The matrix. \param out `a.rows · a.cols` values, row-major.
    template <typename T, typename I, typename U>
    constexpr void to_dense(const csc_view<T, I>& a, std::span<U> out) noexcept
    {
        for (std::size_t k = 0; k < a.rows * a.cols; ++k)
            out[k] = U(0);
        for (std::size_t j = 0; j < a.cols; ++j)
            for (std::size_t k = detail::at(a.col_ptr[j]); k < detail::at(a.col_ptr[j + 1]); ++k)
                out[detail::at(a.row_idx[k]) * a.cols + j] += static_cast<U>(a.values[k]);
    }

} // namespace snicholls::sparse

#endif /* snicholls_core_sparse_csc_hpp */
