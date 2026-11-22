//
//  fixed.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::linalg::fixed_matrix<T, R, C>`: a matrix whose size is part of its type, with the arithmetic and factorisations of state estimation, all `constexpr`.
/// \ingroup core
///
/// **Overview.** Tracking and estimation work on small matrices whose sizes are known when the program is written:
/// a 4-state target, a 2-component measurement, a 4 x 2 gain. `fixed_matrix<T, R, C>` is one `std::array` of
/// `R * C` elements, row-major, an aggregate, trivially copyable when `T` is. Its operations are free functions on
/// values, each a loop of compile-time length that the compiler unrolls: no allocation, no views, no error codes for
/// shapes (a mismatched product does not compile). The factorisations that can fail on a value (Cholesky of an
/// indefinite matrix, the inverse of a singular one) return `result<V, linalg::error>`.
///
/// - Construction: aggregate, `fixed_matrix<double, 2, 2>{{1, 2, 3, 4}}` (row by row); `fixed_vector<T, N>` is
///   `fixed_matrix<T, N, 1>`; `make_vector(1.0, 2.0)`; `zero()`, `identity()`, `diagonal_matrix(v)`.
/// - Arithmetic: `+`, `-`, `*` (matrix product, and by a scalar), `/` by a scalar, `transpose`, `trace`, `dot`,
///   `squared_norm`, `norm`, `outer`, `sandwich(A, P)` = `A P Aᵀ`, `symmetrise`.
/// - Blocks: `block<I, J, BR, BC>(a)`, `set_block<I, J>(a, b)`, `block_diagonal(a, b)`, `stack(x, y)`, `diagonal_of(a)`.
/// - Factorisations: `cholesky` (lower, `L Lᵀ = A`), `lower_solve`, `lower_transpose_solve`, `cholesky_solve`,
///   `cholesky_log_det`, `spd_inverse`; `solve`, `inverse` and `determinant` by LU with partial pivoting;
///   `symmetric_eigen` (cyclic Jacobi, ascending); `tria` (the lower square-root factor of `A Aᵀ`, by Householder QR).
/// - Interoperation: `view(m)` is a `matrix_view` over the same storage, for the run-time-sized functions of
///   `linalg/backend.hpp`.
///
/// Use it for anything whose size is fixed and small (up to a few tens). For sizes known only at run time, or large
/// ones, use `linalg/backend.hpp` over `md_span`.
///
/// \snippet linalg/fixed_demo.cpp fixed
///
/// **Relationships.** `linalg/types.hpp` (`scalar`, `error`, `pivot_tolerance`, `matrix_view`),
/// `math/constexpr_math.hpp` (`sqrt`, `log`), `result.hpp`. The tracking primitives (`tracking/gaussian.hpp`,
/// `tracking/kalman.hpp` and the rest) are written on it. Group: \ref core.
///
/// **Implementations.** Plain loops, row-major. The product accumulates `a(i, k) * b(k, j)` in `k` order. Cholesky
/// is the column (Cholesky-Banachiewicz) form; LU is Doolittle with partial pivoting; the eigen decomposition is cyclic
/// Jacobi, at most 64 sweeps; `tria` is Householder QR of `Aᵀ`, with the signs of the factor's columns made positive.
///
/// **Guarantees.** Every function is `constexpr`, and `noexcept` exactly when `T`'s arithmetic and copies are (they
/// are for `float` and `double`). No allocation, no recursion. `fixed_matrix<T, R, C>` has the size and alignment of
/// `std::array<T, R * C>` and is trivially copyable when `T` is. The factorisations judge a pivot against
/// `pivot_tolerance(n, scale)` (relative, as `linalg/backend.hpp` does), so a matrix and its multiple by 1e-30 are
/// judged alike. Complexity: products and solves O(n³) for n x n, Jacobi O(n³) per sweep.
///
/// **Errors.** `cholesky` and `spd_inverse`: `error::not_positive_definite` when a pivot is at or below the
/// tolerance, or not a number. `solve` and `inverse`: `error::singular`. `symmetric_eigen`: `error::did_not_converge`
/// after 64 sweeps (not seen for a finite symmetric input). Nothing throws unless `T`'s arithmetic does.
///
/// **Performance.** Against the same loops written by hand over raw arrays, after a bit-for-bit check, 4096
/// matrices (`benchmarks/linalg/fixed_benchmark.cpp`; Release, Apple clang 17, Xeon W-3245, 2026-11-23): the 4 x 4
/// product 53.1 million a second (by hand 58.9), the 4 x 4 Cholesky 55.8 million (by hand, reporting failure the same
/// way, 68.2). The factor is built inside the `result` it returns, so it is not copied; the remaining gap is the
/// `result`'s flag and the caller's copy out of it.
///
/// **Tests and demos.** `tests/linalg/fixed_test.cpp`, `demos/linalg/fixed_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<type_traits>`, `linalg/types.hpp`, `math/constexpr_math.hpp`,
/// `result.hpp`.
///
/// **Known issues.** At run time `sqrt` and `log` are the standard library's, so results agree with a constant
/// evaluation only to the last bit or so. Elements are read with `operator()(i, j)`; a vector also has `operator[]`.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_linalg_fixed_hpp
#define snicholls_core_linalg_fixed_hpp

#include <snicholls/core/linalg/types.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace snicholls::linalg {

    /// \brief A matrix of `R` rows and `C` columns of `T`, stored row-major in one `std::array`: an aggregate.
    /// \ingroup core
    /// \tparam T The element type, a `linalg::scalar` (`float`, `double`, or a type of the caller's with `math_limits`).
    /// \tparam R The rows, at least 1.
    /// \tparam C The columns, at least 1.
    template <scalar T, std::size_t R, std::size_t C>
    struct fixed_matrix {
        static_assert(R > 0 && C > 0, "snicholls::linalg::fixed_matrix: a matrix has at least one row and one column");

        using value_type = T;                        ///< The element type.
        static constexpr std::size_t rows = R;       ///< The rows.
        static constexpr std::size_t cols = C;       ///< The columns.
        static constexpr std::size_t size = R * C;   ///< The elements.

        std::array<T, R * C> elements{};             ///< The elements, row by row.

        /// \brief The element in row `i`, column `j`. \param i The row. \param j The column. \return A reference to it.
        [[nodiscard]] constexpr T& operator()(std::size_t i, std::size_t j) noexcept { return elements[i * C + j]; }
        /// \brief The element in row `i`, column `j`. \param i The row. \param j The column. \return A reference to it.
        [[nodiscard]] constexpr const T& operator()(std::size_t i, std::size_t j) const noexcept { return elements[i * C + j]; }
        /// \brief Element `i` of a vector (one row or one column). \param i The index. \return A reference to it.
        [[nodiscard]] constexpr T& operator[](std::size_t i) noexcept
            requires(R == 1 || C == 1)
        {
            return elements[i];
        }
        /// \brief Element `i` of a vector (one row or one column). \param i The index. \return A reference to it.
        [[nodiscard]] constexpr const T& operator[](std::size_t i) const noexcept
            requires(R == 1 || C == 1)
        {
            return elements[i];
        }

        /// \brief The zero matrix. \return It.
        [[nodiscard]] static constexpr fixed_matrix zero() noexcept(detail::nothrow_scalar<T>) { return fixed_matrix{}; }
        /// \brief The identity. \return It.
        [[nodiscard]] static constexpr fixed_matrix identity() noexcept(detail::nothrow_scalar<T>)
            requires(R == C)
        {
            fixed_matrix m{};
            for (std::size_t i = 0; i < R; ++i)
                m(i, i) = T(1);
            return m;
        }

        /// \brief Element-wise equality. \return Whether every element is equal.
        friend constexpr bool operator==(const fixed_matrix&, const fixed_matrix&) = default;
    };

    /// \brief A column vector of `N` elements.
    /// \ingroup core
    /// \tparam T The element type.
    /// \tparam N The elements.
    template <scalar T, std::size_t N>
    using fixed_vector = fixed_matrix<T, N, 1>;

    namespace detail::fixed {
        template <typename T>
        inline constexpr bool nt = linalg::detail::nothrow_scalar<T>;

        template <typename T>
        constexpr T abs(T x) noexcept(nt<T>)
        {
            return x < T(0) ? -x : x;
        }
    } // namespace detail::fixed

    /// \brief A vector of the given elements, in their common type.
    /// \ingroup core
    /// \tparam X The element types.
    /// \param x The elements.
    /// \return The vector.
    template <typename... X>
    [[nodiscard]] constexpr auto make_vector(X... x) noexcept(detail::fixed::nt<std::common_type_t<X...>>)
    {
        using T = std::common_type_t<X...>;
        return fixed_vector<T, sizeof...(X)>{{static_cast<T>(x)...}};
    }

    /// \brief The sum. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param a The left operand. \param b The right operand. \return `a + b`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R, C> operator+(const fixed_matrix<T, R, C>& a, const fixed_matrix<T, R, C>& b) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R, C> r = a;
        for (std::size_t i = 0; i < R * C; ++i)
            r.elements[i] += b.elements[i];
        return r;
    }

    /// \brief The difference. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param a The left operand. \param b The right operand. \return `a - b`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R, C> operator-(const fixed_matrix<T, R, C>& a, const fixed_matrix<T, R, C>& b) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R, C> r = a;
        for (std::size_t i = 0; i < R * C; ++i)
            r.elements[i] -= b.elements[i];
        return r;
    }

    /// \brief The negation. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param a The operand. \return `-a`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R, C> operator-(const fixed_matrix<T, R, C>& a) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R, C> r = a;
        for (auto& e : r.elements)
            e = -e;
        return r;
    }

    /// \brief The product by a scalar. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param a The matrix. \param s The scalar. \return `a s`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R, C> operator*(const fixed_matrix<T, R, C>& a, std::type_identity_t<T> s) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R, C> r = a;
        for (auto& e : r.elements)
            e *= s;
        return r;
    }

    /// \brief The product by a scalar. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param s The scalar. \param a The matrix. \return `s a`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R, C> operator*(std::type_identity_t<T> s, const fixed_matrix<T, R, C>& a) noexcept(detail::fixed::nt<T>)
    {
        return a * s;
    }

    /// \brief The quotient by a scalar. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param a The matrix. \param s The scalar. \return `a / s`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R, C> operator/(const fixed_matrix<T, R, C>& a, std::type_identity_t<T> s) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R, C> r = a;
        for (auto& e : r.elements)
            e /= s;
        return r;
    }

    /// \brief The matrix product. \tparam T The element type. \tparam R The rows of `a`.
    /// \ingroup core
    /// \tparam K The columns of `a`, the rows of `b`. \tparam C The columns of `b`.
    /// \param a The left operand. \param b The right operand. \return `a b`.
    template <scalar T, std::size_t R, std::size_t K, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R, C> operator*(const fixed_matrix<T, R, K>& a, const fixed_matrix<T, K, C>& b) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R, C> r{};
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t k = 0; k < K; ++k) {
                const T aik = a(i, k);
                for (std::size_t j = 0; j < C; ++j)
                    r(i, j) += aik * b(k, j);
            }
        return r;
    }

    /// \brief Adds in place. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param a The matrix to change. \param b The addend. \return `a`.
    template <scalar T, std::size_t R, std::size_t C>
    constexpr fixed_matrix<T, R, C>& operator+=(fixed_matrix<T, R, C>& a, const fixed_matrix<T, R, C>& b) noexcept(detail::fixed::nt<T>)
    {
        for (std::size_t i = 0; i < R * C; ++i)
            a.elements[i] += b.elements[i];
        return a;
    }

    /// \brief Subtracts in place. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param a The matrix to change. \param b The subtrahend. \return `a`.
    template <scalar T, std::size_t R, std::size_t C>
    constexpr fixed_matrix<T, R, C>& operator-=(fixed_matrix<T, R, C>& a, const fixed_matrix<T, R, C>& b) noexcept(detail::fixed::nt<T>)
    {
        for (std::size_t i = 0; i < R * C; ++i)
            a.elements[i] -= b.elements[i];
        return a;
    }

    /// \brief The transpose. \tparam T The element type. \tparam R The rows. \tparam C The columns.
    /// \ingroup core
    /// \param a The matrix. \return `aᵀ`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, C, R> transpose(const fixed_matrix<T, R, C>& a) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, C, R> r{};
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j)
                r(j, i) = a(i, j);
        return r;
    }

    /// \brief The trace. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \param a The matrix. \return The sum of the diagonal.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr T trace(const fixed_matrix<T, N, N>& a) noexcept(detail::fixed::nt<T>)
    {
        T s(0);
        for (std::size_t i = 0; i < N; ++i)
            s += a(i, i);
        return s;
    }

    /// \brief The inner product of two vectors. \tparam T The element type. \tparam N The elements.
    /// \ingroup core
    /// \param x The left operand. \param y The right operand. \return `xᵀ y`.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr T dot(const fixed_vector<T, N>& x, const fixed_vector<T, N>& y) noexcept(detail::fixed::nt<T>)
    {
        T s(0);
        for (std::size_t i = 0; i < N; ++i)
            s += x[i] * y[i];
        return s;
    }

    /// \brief The sum of the squares of the elements (the squared Frobenius norm). \tparam T The element type.
    /// \ingroup core
    /// \tparam R The rows. \tparam C The columns. \param a The matrix. \return It.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr T squared_norm(const fixed_matrix<T, R, C>& a) noexcept(detail::fixed::nt<T>)
    {
        T s(0);
        for (const auto& e : a.elements)
            s += e * e;
        return s;
    }

    /// \brief The Euclidean (Frobenius) norm. \tparam T The element type. \tparam R The rows.
    /// \ingroup core
    /// \tparam C The columns. \param a The matrix. \return Its square root of `squared_norm`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr T norm(const fixed_matrix<T, R, C>& a) noexcept(detail::fixed::nt<T>)
    {
        return static_cast<T>(math::sqrt(squared_norm(a)));
    }

    /// \brief The outer product. \tparam T The element type. \tparam R The elements of `x`.
    /// \ingroup core
    /// \tparam C The elements of `y`. \param x The left operand. \param y The right operand. \return `x yᵀ`.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R, C> outer(const fixed_vector<T, R>& x, const fixed_vector<T, C>& y) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R, C> r{};
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j)
                r(i, j) = x[i] * y[j];
        return r;
    }

    /// \brief `A P Aᵀ`, the covariance of `A x` for `x` of covariance `P`. \tparam T The element type.
    /// \ingroup core
    /// \tparam R The rows of `A`. \tparam N The order of `P`. \param a The map. \param p The covariance.
    /// \return `A P Aᵀ`, made exactly symmetric.
    template <scalar T, std::size_t R, std::size_t N>
    [[nodiscard]] constexpr fixed_matrix<T, R, R> sandwich(const fixed_matrix<T, R, N>& a, const fixed_matrix<T, N, N>& p) noexcept(detail::fixed::nt<T>)
    {
        const fixed_matrix<T, R, N> ap = a * p;
        fixed_matrix<T, R, R> r{};
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j <= i; ++j) {
                T s(0);
                for (std::size_t k = 0; k < N; ++k)
                    s += ap(i, k) * a(j, k);
                r(i, j) = s;
                r(j, i) = s;
            }
        return r;
    }

    /// \brief `(A + Aᵀ) / 2`. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \param a The matrix. \return Its symmetric part.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr fixed_matrix<T, N, N> symmetrise(const fixed_matrix<T, N, N>& a) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, N, N> r = a;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < i; ++j)
                r(i, j) = r(j, i) = (a(i, j) + a(j, i)) / T(2);
        return r;
    }

    /// \brief The diagonal matrix of a vector. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \param v The diagonal. \return The matrix.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr fixed_matrix<T, N, N> diagonal_matrix(const fixed_vector<T, N>& v) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, N, N> r{};
        for (std::size_t i = 0; i < N; ++i)
            r(i, i) = v[i];
        return r;
    }

    /// \brief The diagonal of a square matrix. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \param a The matrix. \return Its diagonal as a vector.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr fixed_vector<T, N> diagonal_of(const fixed_matrix<T, N, N>& a) noexcept(detail::fixed::nt<T>)
    {
        fixed_vector<T, N> r{};
        for (std::size_t i = 0; i < N; ++i)
            r[i] = a(i, i);
        return r;
    }

    /// \brief The block of `BR` rows and `BC` columns at row `I`, column `J`. \tparam I The first row.
    /// \ingroup core
    /// \tparam J The first column. \tparam BR The block's rows. \tparam BC The block's columns. \tparam T The element type.
    /// \tparam R The rows. \tparam C The columns. \param a The matrix. \return The block, copied.
    template <std::size_t I, std::size_t J, std::size_t BR, std::size_t BC, scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, BR, BC> block(const fixed_matrix<T, R, C>& a) noexcept(detail::fixed::nt<T>)
    {
        static_assert(I + BR <= R && J + BC <= C, "snicholls::linalg::block: the block is not inside the matrix");
        fixed_matrix<T, BR, BC> r{};
        for (std::size_t i = 0; i < BR; ++i)
            for (std::size_t j = 0; j < BC; ++j)
                r(i, j) = a(I + i, J + j);
        return r;
    }

    /// \brief Writes `b` into `a` at row `I`, column `J`. \tparam I The first row. \tparam J The first column.
    /// \ingroup core
    /// \tparam T The element type. \tparam R The rows of `a`. \tparam C The columns of `a`. \tparam BR The rows of `b`.
    /// \tparam BC The columns of `b`. \param a The matrix to change. \param b The block.
    template <std::size_t I, std::size_t J, scalar T, std::size_t R, std::size_t C, std::size_t BR, std::size_t BC>
    constexpr void set_block(fixed_matrix<T, R, C>& a, const fixed_matrix<T, BR, BC>& b) noexcept(detail::fixed::nt<T>)
    {
        static_assert(I + BR <= R && J + BC <= C, "snicholls::linalg::set_block: the block is not inside the matrix");
        for (std::size_t i = 0; i < BR; ++i)
            for (std::size_t j = 0; j < BC; ++j)
                a(I + i, J + j) = b(i, j);
    }

    /// \brief The block-diagonal matrix `[[a, 0], [0, b]]`. \tparam T The element type.
    /// \ingroup core
    /// \tparam R1 The rows of `a`. \tparam C1 The columns of `a`. \tparam R2 The rows of `b`. \tparam C2 The columns of `b`.
    /// \param a The upper-left block. \param b The lower-right block. \return The matrix.
    template <scalar T, std::size_t R1, std::size_t C1, std::size_t R2, std::size_t C2>
    [[nodiscard]] constexpr fixed_matrix<T, R1 + R2, C1 + C2> block_diagonal(const fixed_matrix<T, R1, C1>& a, const fixed_matrix<T, R2, C2>& b) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R1 + R2, C1 + C2> r{};
        set_block<0, 0>(r, a);
        set_block<R1, C1>(r, b);
        return r;
    }

    /// \brief `[a; b]`: `b`'s rows below `a`'s. \tparam T The element type. \tparam R1 The rows of `a`.
    /// \ingroup core
    /// \tparam R2 The rows of `b`. \tparam C The columns. \param a The upper block. \param b The lower block. \return The matrix.
    template <scalar T, std::size_t R1, std::size_t R2, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, R1 + R2, C> stack(const fixed_matrix<T, R1, C>& a, const fixed_matrix<T, R2, C>& b) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, R1 + R2, C> r{};
        set_block<0, 0>(r, a);
        set_block<R1, 0>(r, b);
        return r;
    }

    /// \brief The lower Cholesky factor: `L` with `L Lᵀ = A`, positive diagonal. Reads the lower triangle of `A` only.
    /// \ingroup core
    /// \tparam T The element type.
    /// \tparam N The order.
    /// \param a A symmetric positive-definite matrix.
    /// \return `L`, or `error::not_positive_definite` when a pivot is at or below `pivot_tolerance(N, max a_ii)` or NaN.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr result<fixed_matrix<T, N, N>, error> cholesky(const fixed_matrix<T, N, N>& a) noexcept(detail::fixed::nt<T>)
    {
        T scale(0);
        for (std::size_t i = 0; i < N; ++i)
            if (detail::fixed::abs(a(i, i)) > scale)
                scale = detail::fixed::abs(a(i, i));
        const T tol = pivot_tolerance<T>(N, scale);
        // The factor is built inside the result and the result is the only object returned, so it is never copied
        result<fixed_matrix<T, N, N>, error> out{std::in_place};
        fixed_matrix<T, N, N>& l = *out;
        for (std::size_t j = 0; j < N; ++j) {
            T d = a(j, j);
            for (std::size_t k = 0; k < j; ++k)
                d -= l(j, k) * l(j, k);
            if (!(d > tol)) {
                out = err(error::not_positive_definite);
                return out;
            }
            const T ljj = static_cast<T>(math::sqrt(d));
            l(j, j) = ljj;
            for (std::size_t i = j + 1; i < N; ++i) {
                T s = a(i, j);
                for (std::size_t k = 0; k < j; ++k)
                    s -= l(i, k) * l(j, k);
                l(i, j) = s / ljj;
            }
        }
        return out;
    }

    /// \brief Solves `L X = B` by forward substitution. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \tparam C The columns of `B`. \param l A lower-triangular matrix with a nonzero diagonal. \param b The right-hand side.
    /// \return `X = L⁻¹ B`.
    template <scalar T, std::size_t N, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, N, C> lower_solve(const fixed_matrix<T, N, N>& l, const fixed_matrix<T, N, C>& b) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, N, C> x = b;
        for (std::size_t j = 0; j < C; ++j)
            for (std::size_t i = 0; i < N; ++i) {
                T s = x(i, j);
                for (std::size_t k = 0; k < i; ++k)
                    s -= l(i, k) * x(k, j);
                x(i, j) = s / l(i, i);
            }
        return x;
    }

    /// \brief Solves `Lᵀ X = B` by back substitution. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \tparam C The columns of `B`. \param l A lower-triangular matrix with a nonzero diagonal. \param b The right-hand side.
    /// \return `X = L⁻ᵀ B`.
    template <scalar T, std::size_t N, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, N, C> lower_transpose_solve(const fixed_matrix<T, N, N>& l, const fixed_matrix<T, N, C>& b) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, N, C> x = b;
        for (std::size_t j = 0; j < C; ++j)
            for (std::size_t ii = N; ii-- > 0;) {
                T s = x(ii, j);
                for (std::size_t k = ii + 1; k < N; ++k)
                    s -= l(k, ii) * x(k, j);
                x(ii, j) = s / l(ii, ii);
            }
        return x;
    }

    /// \brief Solves `A X = B` given `A`'s Cholesky factor. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \tparam C The columns of `B`. \param l `cholesky(A)`. \param b The right-hand side. \return `X = A⁻¹ B`.
    template <scalar T, std::size_t N, std::size_t C>
    [[nodiscard]] constexpr fixed_matrix<T, N, C> cholesky_solve(const fixed_matrix<T, N, N>& l, const fixed_matrix<T, N, C>& b) noexcept(detail::fixed::nt<T>)
    {
        return lower_transpose_solve(l, lower_solve(l, b));
    }

    /// \brief `log det A` from `A`'s Cholesky factor: `2 Σ log l_ii`. \tparam T The element type.
    /// \ingroup core
    /// \tparam N The order. \param l `cholesky(A)`. \return The log-determinant.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr T cholesky_log_det(const fixed_matrix<T, N, N>& l) noexcept(detail::fixed::nt<T>)
    {
        T s(0);
        for (std::size_t i = 0; i < N; ++i)
            s += static_cast<T>(math::log(l(i, i)));
        return T(2) * s;
    }

    /// \brief The inverse of a symmetric positive-definite matrix, by Cholesky, made exactly symmetric.
    /// \tparam T The element type. \tparam N The order. \param a The matrix.
    /// \ingroup core
    /// \return `A⁻¹`, or `error::not_positive_definite`.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr result<fixed_matrix<T, N, N>, error> spd_inverse(const fixed_matrix<T, N, N>& a) noexcept(detail::fixed::nt<T>)
    {
        const auto l = cholesky(a);
        if (!l)
            return err(l.unwrap_err_unchecked());
        return ok(symmetrise(cholesky_solve(*l, fixed_matrix<T, N, N>::identity())));
    }

    namespace detail::fixed {
        template <typename T, std::size_t N>
        struct lu {
            fixed_matrix<T, N, N> f{};
            std::array<std::size_t, N> perm{};
            bool odd = false;
            bool singular = false;
        };

        // Doolittle with partial pivoting: f holds L (unit, below) and U (on and above); perm[i] is the row of A in row i
        template <typename T, std::size_t N>
        constexpr lu<T, N> lu_factor(const fixed_matrix<T, N, N>& a) noexcept(nt<T>)
        {
            lu<T, N> r{a, {}, false, false};
            T scale(0);
            for (const auto& e : a.elements)
                if (abs(e) > scale)
                    scale = abs(e);
            const T tol = pivot_tolerance<T>(N, scale);
            for (std::size_t i = 0; i < N; ++i)
                r.perm[i] = i;
            for (std::size_t k = 0; k < N; ++k) {
                std::size_t p = k;
                for (std::size_t i = k + 1; i < N; ++i)
                    if (abs(r.f(i, k)) > abs(r.f(p, k)))
                        p = i;
                if (!(abs(r.f(p, k)) > tol)) {
                    r.singular = true;
                    return r;
                }
                if (p != k) {
                    for (std::size_t j = 0; j < N; ++j) {
                        const T t = r.f(k, j);
                        r.f(k, j) = r.f(p, j);
                        r.f(p, j) = t;
                    }
                    const std::size_t t = r.perm[k];
                    r.perm[k] = r.perm[p];
                    r.perm[p] = t;
                    r.odd = !r.odd;
                }
                for (std::size_t i = k + 1; i < N; ++i) {
                    const T m = r.f(i, k) / r.f(k, k);
                    r.f(i, k) = m;
                    for (std::size_t j = k + 1; j < N; ++j)
                        r.f(i, j) -= m * r.f(k, j);
                }
            }
            return r;
        }
    } // namespace detail::fixed

    /// \brief Solves `A X = B` by LU with partial pivoting. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \tparam C The columns of `B`. \param a The matrix. \param b The right-hand side.
    /// \return `X`, or `error::singular` when a pivot is at or below `pivot_tolerance(N, max |a_ij|)`.
    template <scalar T, std::size_t N, std::size_t C>
    [[nodiscard]] constexpr result<fixed_matrix<T, N, C>, error> solve(const fixed_matrix<T, N, N>& a, const fixed_matrix<T, N, C>& b) noexcept(detail::fixed::nt<T>)
    {
        const auto lu = detail::fixed::lu_factor(a);
        if (lu.singular)
            return err(error::singular);
        fixed_matrix<T, N, C> x{};
        for (std::size_t j = 0; j < C; ++j) {
            for (std::size_t i = 0; i < N; ++i) {
                T s = b(lu.perm[i], j);
                for (std::size_t k = 0; k < i; ++k)
                    s -= lu.f(i, k) * x(k, j);
                x(i, j) = s;
            }
            for (std::size_t ii = N; ii-- > 0;) {
                T s = x(ii, j);
                for (std::size_t k = ii + 1; k < N; ++k)
                    s -= lu.f(ii, k) * x(k, j);
                x(ii, j) = s / lu.f(ii, ii);
            }
        }
        return ok(x);
    }

    /// \brief The inverse, by LU with partial pivoting. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \param a The matrix. \return `A⁻¹`, or `error::singular`.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr result<fixed_matrix<T, N, N>, error> inverse(const fixed_matrix<T, N, N>& a) noexcept(detail::fixed::nt<T>)
    {
        return solve(a, fixed_matrix<T, N, N>::identity());
    }

    /// \brief The determinant, by LU with partial pivoting. \tparam T The element type. \tparam N The order.
    /// \ingroup core
    /// \param a The matrix. \return `det A`; zero when a pivot is at or below the tolerance.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr T determinant(const fixed_matrix<T, N, N>& a) noexcept(detail::fixed::nt<T>)
    {
        const auto lu = detail::fixed::lu_factor(a);
        if (lu.singular)
            return T(0);
        T d = lu.odd ? T(-1) : T(1);
        for (std::size_t i = 0; i < N; ++i)
            d *= lu.f(i, i);
        return d;
    }

    /// \brief The eigenvalues of a symmetric matrix, ascending, and an orthonormal matrix of its eigenvectors.
    /// \ingroup core
    /// \tparam T The element type.
    /// \tparam N The order.
    template <scalar T, std::size_t N>
    struct eigen_decomposition {
        fixed_vector<T, N> values{};      ///< The eigenvalues, ascending.
        fixed_matrix<T, N, N> vectors{};  ///< Column `i` is the unit eigenvector of `values[i]`.
    };

    /// \brief The eigen decomposition of a symmetric matrix by cyclic Jacobi rotations: `A = V diag(w) Vᵀ`.
    /// \ingroup core
    /// \tparam T The element type.
    /// \tparam N The order.
    /// \param a A symmetric matrix (the upper triangle is read).
    /// \return The decomposition, eigenvalues ascending; or `error::did_not_converge` after 64 sweeps.
    template <scalar T, std::size_t N>
    [[nodiscard]] constexpr result<eigen_decomposition<T, N>, error> symmetric_eigen(const fixed_matrix<T, N, N>& a) noexcept(detail::fixed::nt<T>)
    {
        fixed_matrix<T, N, N> m = a;
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < i; ++j)
                m(i, j) = m(j, i);
        auto v = fixed_matrix<T, N, N>::identity();
        const T eps = std::numeric_limits<T>::epsilon();
        bool converged = false;
        for (int sweep = 0; sweep < 64 && !converged; ++sweep) {
            T off(0), diag(0);
            for (std::size_t i = 0; i < N; ++i) {
                diag += m(i, i) * m(i, i);
                for (std::size_t j = i + 1; j < N; ++j)
                    off += m(i, j) * m(i, j);
            }
            if (off <= eps * eps * diag || off == T(0)) {
                converged = true;
                break;
            }
            for (std::size_t p = 0; p < N; ++p)
                for (std::size_t q = p + 1; q < N; ++q) {
                    const T apq = m(p, q);
                    if (apq == T(0))
                        continue;
                    const T theta = (m(q, q) - m(p, p)) / (T(2) * apq);
                    const T at = detail::fixed::abs(theta);
                    // t = tan of the rotation, the smaller root; for a huge theta, 1 / (2 theta) without squaring it
                    T t = at > T(1e150) ? T(1) / (T(2) * at) : T(1) / (at + static_cast<T>(math::sqrt(theta * theta + T(1))));
                    if (theta < T(0))
                        t = -t;
                    const T c = T(1) / static_cast<T>(math::sqrt(t * t + T(1)));
                    const T s = t * c;
                    for (std::size_t k = 0; k < N; ++k) {
                        const T mkp = m(k, p), mkq = m(k, q);
                        m(k, p) = c * mkp - s * mkq;
                        m(k, q) = s * mkp + c * mkq;
                    }
                    for (std::size_t k = 0; k < N; ++k) {
                        const T mpk = m(p, k), mqk = m(q, k);
                        m(p, k) = c * mpk - s * mqk;
                        m(q, k) = s * mpk + c * mqk;
                    }
                    for (std::size_t k = 0; k < N; ++k) {
                        const T vkp = v(k, p), vkq = v(k, q);
                        v(k, p) = c * vkp - s * vkq;
                        v(k, q) = s * vkp + c * vkq;
                    }
                }
        }
        if (!converged)
            return err(error::did_not_converge);
        eigen_decomposition<T, N> r{diagonal_of(m), v};
        // Insertion sort, ascending, moving each eigenvector with its value
        for (std::size_t i = 1; i < N; ++i)
            for (std::size_t j = i; j > 0 && r.values[j] < r.values[j - 1]; --j) {
                const T t = r.values[j];
                r.values[j] = r.values[j - 1];
                r.values[j - 1] = t;
                for (std::size_t k = 0; k < N; ++k) {
                    const T u = r.vectors(k, j);
                    r.vectors(k, j) = r.vectors(k, j - 1);
                    r.vectors(k, j - 1) = u;
                }
            }
        return ok(r);
    }

    /// \brief The lower-triangular `S` with `S Sᵀ = A Aᵀ` and a non-negative diagonal: the square-root filters'
    /// triangularisation, by Householder QR of `Aᵀ`.
    /// \ingroup core
    /// \tparam T The element type.
    /// \tparam N The rows of `A`.
    /// \tparam K The columns of `A`, at least `N`.
    /// \param a The matrix, such as `[F S, Q^(1/2)]`.
    /// \return `S`.
    template <scalar T, std::size_t N, std::size_t K>
    [[nodiscard]] constexpr fixed_matrix<T, N, N> tria(const fixed_matrix<T, N, K>& a) noexcept(detail::fixed::nt<T>)
    {
        static_assert(K >= N, "snicholls::linalg::tria: A needs at least as many columns as rows");
        fixed_matrix<T, K, N> m = transpose(a);
        for (std::size_t j = 0; j < N; ++j) {
            T sigma(0);
            for (std::size_t i = j; i < K; ++i)
                sigma += m(i, j) * m(i, j);
            if (sigma == T(0))
                continue;
            const T nrm = static_cast<T>(math::sqrt(sigma));
            const T alpha = m(j, j) > T(0) ? -nrm : nrm;
            // v = x - alpha e1, stored in a copy; H = I - 2 v vᵀ / (vᵀ v)
            std::array<T, K> v{};
            for (std::size_t i = j; i < K; ++i)
                v[i] = m(i, j);
            v[j] -= alpha;
            T vv(0);
            for (std::size_t i = j; i < K; ++i)
                vv += v[i] * v[i];
            if (vv == T(0))
                continue;
            for (std::size_t c = j; c < N; ++c) {
                T s(0);
                for (std::size_t i = j; i < K; ++i)
                    s += v[i] * m(i, c);
                s = T(2) * s / vv;
                for (std::size_t i = j; i < K; ++i)
                    m(i, c) -= s * v[i];
            }
        }
        fixed_matrix<T, N, N> s{};
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j <= i; ++j)
                s(i, j) = m(j, i);
        for (std::size_t j = 0; j < N; ++j)
            if (s(j, j) < T(0))
                for (std::size_t i = j; i < N; ++i)
                    s(i, j) = -s(i, j);
        return s;
    }

    /// \brief A `matrix_view` over the matrix's storage, for `linalg/backend.hpp`. \tparam T The element type.
    /// \ingroup core
    /// \tparam R The rows. \tparam C The columns. \param m The matrix, which outlives the view. \return The view.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr matrix_view<T, layout_right> view(fixed_matrix<T, R, C>& m) noexcept
    {
        return {m.elements.data(), {R, C}};
    }

    /// \brief A read-only `matrix_view` over the matrix's storage. \tparam T The element type.
    /// \ingroup core
    /// \tparam R The rows. \tparam C The columns. \param m The matrix, which outlives the view. \return The view.
    template <scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr matrix_view<const T, layout_right> view(const fixed_matrix<T, R, C>& m) noexcept
    {
        return {m.elements.data(), {R, C}};
    }

} // namespace snicholls::linalg

#endif /* snicholls_core_linalg_fixed_hpp */
