//
//  types.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The vocabulary of `snicholls::linalg`: matrix and vector views, the error and status codes, triangles, pivots, tolerances.
/// \ingroup core
///
/// **Overview.** Every linear-algebra operation in core speaks these types and nothing else: matrices, vectors and
/// batches are `containers::md_span` views (row-major, column-major or strided, never owned), a failure is an
/// `error` inside a `snicholls::result`, and a batch reports one `status` per system. A backend from another
/// library (a BLAS, LAPACK, Eigen or GPU binding) sees the same types, so callers never see the backend's.
///
/// \snippet linalg/types_demo.cpp views
///
/// **Relationships.** `containers/md_array.hpp` (the views), `result.hpp` (the error channel),
/// `linalg/backend.hpp` (the operations), `linalg/batched.hpp`, `linalg/reference.hpp`. Group: \ref core.
///
/// **Guarantees.** Everything here is `constexpr` and trivially copyable, and nothing allocates. The views and codes are
/// `noexcept`; what computes in `T` (`pivot_tolerance`, `ldlt_growth_limit`, `signed_log_det::value`) is `noexcept`
/// exactly when `T`'s arithmetic is, so a number type that throws (an allocating multiprecision type) propagates.
///
/// **Errors.** None of its own: `error` is the value the operations return.
///
/// **Performance.** No run-time cost beyond the views' (see `containers/md_array.hpp`).
///
/// **Tests and demos.** `tests/linalg/types_test.cpp`, `demos/linalg/types_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `<type_traits>`,
/// `<utility>`, `containers/md_array.hpp`, `math/constexpr_math.hpp`.
///
/// **Known issues.** Tolerances are relative to `std::numeric_limits<T>::epsilon()`, and the kernels' `abs`, `sqrt`
/// and `log` are `snicholls::math`'s; a number type of the caller's (a half-precision or quantised type) takes part
/// by specialising `std::numeric_limits` and `math::math_limits`, as for every other numeric header in core.
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_linalg_types_hpp
#define snicholls_core_linalg_types_hpp

#include <snicholls/core/containers/md_array.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls::linalg {

    using containers::layout_left;
    using containers::layout_right;
    using containers::layout_stride;
    using containers::md_span;

    /// \brief A number the operations accept: `std::numeric_limits` and `math::math_limits` specialised, not an integer (`float`, `double`; later half and quantised types).
    /// \ingroup core
    /// \tparam T The type.
    template <typename T>
    concept scalar = std::numeric_limits<T>::is_specialized && !std::numeric_limits<T>::is_integer && math::math_number<T> && std::regular<T>;

    /// \brief A matrix: a rank-2 view, strided by default (every layout converts to it).
    /// \ingroup core
    /// \tparam T The element type, const for an input.
    /// \tparam L The layout.
    template <typename T, typename L = layout_stride>
    using matrix_view = md_span<T, 2, L>;

    /// \brief A vector: a rank-1 view, strided by default.
    /// \ingroup core
    /// \tparam T The element type, const for an input.
    /// \tparam L The layout.
    template <typename T, typename L = layout_stride>
    using vector_view = md_span<T, 1, L>;

    /// \brief A batch of matrices of one size: a rank-3 view, `(system, row, column)`, strided by default.
    /// \ingroup core
    /// \tparam T The element type, const for an input.
    /// \tparam L The layout.
    template <typename T, typename L = layout_stride>
    using batch_view = md_span<T, 3, L>;

    /// \brief A row interchange recorded by LU: at step `k`, rows `k` and `pivots[k]` were swapped (0-based, as LAPACK's are 1-based).
    /// \ingroup core
    using pivot_index = std::int32_t;

    /// \brief Why an operation failed. A singular or indefinite matrix is a normal outcome, reported, never undefined behaviour.
    /// \ingroup core
    enum class error : std::uint8_t {
        singular = 1,               ///< A pivot (LU, LDLT, a triangular diagonal, R's diagonal) is zero or below the tolerance.
        not_positive_definite = 2,  ///< Cholesky met a diagonal at or below the tolerance: the matrix is not (numerically) positive definite.
        did_not_converge = 3,       ///< An iterative decomposition (eigen, SVD) used its sweeps without converging.
        dimension_mismatch = 4,     ///< The views' extents do not fit together.
        aliased = 5,                ///< An output view starts where an input view does: the operation would read what it wrote.
    };

    /// \brief One system's outcome in a batch: `ok`, or the `error` it would have returned, with the same number.
    /// \ingroup core
    enum class status : std::uint8_t {
        ok = 0,                     ///< Solved.
        singular = 1,               ///< As `error::singular`.
        not_positive_definite = 2,  ///< As `error::not_positive_definite`.
        did_not_converge = 3,       ///< As `error::did_not_converge`.
        dimension_mismatch = 4,     ///< As `error::dimension_mismatch`.
        aliased = 5,                ///< As `error::aliased`.
    };

    namespace detail {
        // Whether T's arithmetic, comparisons and copies cannot throw: true for float and double, false for an
        // allocating multiprecision type, whose bad_alloc must propagate rather than terminate.
        template <typename T>
        inline constexpr bool nothrow_scalar = std::is_nothrow_copy_constructible_v<T> && std::is_nothrow_copy_assignable_v<T> &&
                                               noexcept(std::declval<T>() + std::declval<T>()) && noexcept(std::declval<T>() - std::declval<T>()) &&
                                               noexcept(std::declval<T>() * std::declval<T>()) && noexcept(std::declval<T>() / std::declval<T>()) &&
                                               noexcept(std::declval<T>() < std::declval<T>()) && noexcept(T(0));

        // Where a view or a span starts, for the aliasing test below.
        template <typename T, std::size_t R, typename L>
        constexpr const void* start_of(const md_span<T, R, L>& x) noexcept
        {
            return x.data_handle();
        }
        template <typename T, std::size_t E>
        constexpr const void* start_of(const std::span<T, E>& x) noexcept
        {
            return x.data();
        }

        // Whether two non-empty views or spans start at the same element: the cheap aliasing test of the free functions.
        // Views that overlap some other way (a sub-view, interleaved strides) are a precondition violation, not detected.
        template <typename X, typename Y>
        constexpr bool same_storage(const X& x, const Y& y) noexcept
        {
            return !x.empty() && !y.empty() && start_of(x) == start_of(y);
        }
    } // namespace detail

    /// \brief The status that reports this error.
    /// \ingroup core
    /// \param e The error.
    /// \return The status with the same number.
    [[nodiscard]] constexpr status to_status(error e) noexcept { return static_cast<status>(e); }

    /// \brief Which triangle of a square matrix an operation reads or writes.
    /// \ingroup core
    enum class triangle : std::uint8_t {
        lower,   ///< On and below the diagonal.
        upper,   ///< On and above the diagonal.
    };

    /// \brief Whether a triangular matrix's diagonal is read or taken to be ones.
    /// \ingroup core
    enum class diagonal : std::uint8_t {
        explicit_diagonal,   ///< Read the diagonal (`std::linalg::explicit_diagonal`).
        implicit_unit,       ///< Take every diagonal element as 1 without reading it (`std::linalg::implicit_unit_diagonal`).
    };

    /// \brief A determinant as a sign and a logarithm, so it neither overflows nor underflows: `det = sign * exp(log_abs)`.
    /// \ingroup core
    /// \tparam T The number type.
    template <scalar T>
    struct signed_log_det {
        int sign = 1;       ///< +1, -1, or 0 for a singular factor (then `log_abs` is minus infinity).
        T log_abs = T(0);   ///< `log(|det|)`.

        /// \brief The determinant itself, which may overflow to infinity or underflow to zero.
        /// \return `sign * exp(log_abs)`.
        [[nodiscard]] constexpr T value() const noexcept(detail::nothrow_scalar<T>) { return static_cast<T>(sign) * static_cast<T>(math::exp(log_abs)); }
    };

    /// \brief The tolerance below which a pivot counts as zero: `epsilon * n * scale`, the usual rank tolerance.
    /// \ingroup core
    ///
    /// `scale` is the largest magnitude the factorisation started from (the largest element for LU, LDLT and QR,
    /// the largest diagonal for Cholesky), so the test is relative and a matrix and its multiple by 1e-30 are
    /// judged alike. A matrix whose condition number is beyond about `1 / (n * epsilon)` is reported singular rather
    /// than solved into noise.
    ///
    /// **This differs from LAPACK**, whose factorisations call only an exact zero singular, and from
    /// `triangular_solve`, which does the same. It also rejects a nonsingular but badly scaled matrix such as
    /// `diag(1e16, 1)`: equilibrate it first (scale rows and columns to a largest element of 1), or choose LAPACK's
    /// rule with the reference backend's `relative_tolerance = 0`.
    /// \tparam T The number type; its `std::numeric_limits<T>::epsilon()` is used.
    /// \param n     The order of the matrix.
    /// \param scale The largest magnitude.
    /// \return The tolerance.
    template <scalar T>
    [[nodiscard]] constexpr T pivot_tolerance(std::size_t n, T scale) noexcept(detail::nothrow_scalar<T>)
    {
        return std::numeric_limits<T>::epsilon() * static_cast<T>(n) * scale;
    }

    /// \brief The largest element growth an unpivoted LDLT accepts: `n / epsilon^(1/4)` (8192 n for `double`, 54 n for `float`).
    /// \ingroup core
    ///
    /// Without pivoting, a small leading pivot `d` creates multipliers of size `1 / d`, and the factorisation's
    /// backward error grows with `max_i sum_k l_ik^2 |d_k|` relative to the largest element of `A`: the growth. A
    /// positive-definite matrix has growth at most 1. A bordered (kriging-style) system `[[C, 1], [1^T, 0]]` with `k`
    /// points has growth about `2 * 1^T C^-1 1 / max |a_ij|`, which is largest when the points are uncorrelated:
    /// measured, `2 k` for `C = I` (16 for `k = 8`), and from 2.1 (`k = 2`) to 33 (`k = 64`) for exponential
    /// covariances of points one unit apart with range 2 to 10. All of these are far below the limit, which grows
    /// with `n` too. Beyond this limit the factorisation would lose more than about a quarter of the digits a pivoted one
    /// keeps, so `ldlt_factor` reports `error::singular` (solve with LU instead) rather than answer inaccurately.
    /// \tparam T The number type.
    /// \param n The order of the matrix.
    /// \return The limit.
    template <scalar T>
    [[nodiscard]] constexpr T ldlt_growth_limit(std::size_t n) noexcept(detail::nothrow_scalar<T>)
    {
        return static_cast<T>(n) / static_cast<T>(math::sqrt(math::sqrt(std::numeric_limits<T>::epsilon())));
    }

    /// \brief The transpose of a row-major matrix: the same storage read column-major, `std::linalg::transposed`.
    /// \ingroup core
    /// \tparam T The element type.
    /// \param a The matrix.
    /// \return Its transpose; no element is copied.
    template <typename T>
    [[nodiscard]] constexpr matrix_view<T, layout_left> transposed(const matrix_view<T, layout_right>& a) noexcept
    {
        return {a.data_handle(), {a.extent(1), a.extent(0)}};
    }

    /// \brief The transpose of a column-major matrix: the same storage read row-major.
    /// \ingroup core
    /// \tparam T The element type.
    /// \param a The matrix.
    /// \return Its transpose; no element is copied.
    template <typename T>
    [[nodiscard]] constexpr matrix_view<T, layout_right> transposed(const matrix_view<T, layout_left>& a) noexcept
    {
        return {a.data_handle(), {a.extent(1), a.extent(0)}};
    }

    /// \brief The transpose of a strided matrix: extents and strides swapped.
    /// \ingroup core
    /// \tparam T The element type.
    /// \param a The matrix.
    /// \return Its transpose; no element is copied.
    template <typename T>
    [[nodiscard]] constexpr matrix_view<T> transposed(const matrix_view<T>& a) noexcept
    {
        return {a.data_handle(), {a.extent(1), a.extent(0)}, {a.stride(1), a.stride(0)}};
    }

    /// \brief A contiguous span as a vector view.
    /// \ingroup core
    /// \tparam T The element type.
    /// \tparam E The span's extent.
    /// \param s The span.
    /// \return The view.
    template <typename T, std::size_t E>
    [[nodiscard]] constexpr vector_view<T, layout_right> as_vector(std::span<T, E> s) noexcept
    {
        return {s.data(), {s.size()}};
    }

    /// \brief A vector as a one-column matrix, so a right-hand side can be a vector or a matrix.
    /// \ingroup core
    /// \tparam T The element type.
    /// \tparam L The vector's layout.
    /// \param v The vector.
    /// \return The `n x 1` matrix over the same elements.
    template <typename T, typename L>
    [[nodiscard]] constexpr matrix_view<T> as_column(const vector_view<T, L>& v) noexcept
    {
        return {v.data_handle(), {v.extent(0), 1}, {v.stride(0), 1}};
    }

} // namespace snicholls::linalg

#endif /* snicholls_core_linalg_types_hpp */
