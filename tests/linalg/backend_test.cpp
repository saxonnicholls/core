//
//  backend_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/linalg/backend.hpp>: which types model the backend concepts (and which do
//  not), the extent checks of every free function (error::dimension_mismatch, arguments unchanged), and
//  what a backend receives: canonical strided views whatever the caller's layout, a vector right-hand
//  side as one column. The behaviour of the operations themselves is the conformance suite's
//  (tests/support/snicholls/core/conformance/linalg.hpp).
//

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace la = snicholls::linalg;
using la::error;
using la::layout_left;
using la::layout_right;
using la::matrix_view;
using la::pivot_index;
using la::vector_view;

// ---- Which types model the concepts ---------------------------------------------------------------------

// The BLAS half of a backend only.
struct blas_only {
    template <la::scalar T>
    T dot(vector_view<const T> x, vector_view<const T> y) const noexcept
    {
        return la::reference.dot(x, y);
    }
    template <la::scalar T>
    void axpy(T a, vector_view<const T> x, vector_view<T> y) const noexcept
    {
        la::reference.axpy(a, x, y);
    }
    template <la::scalar T>
    void gemv(T a, matrix_view<const T> m, vector_view<const T> x, T b, vector_view<T> y) const noexcept
    {
        la::reference.gemv(a, m, x, b, y);
    }
    template <la::scalar T>
    void gemm(T a, matrix_view<const T> x, matrix_view<const T> y, T b, matrix_view<T> c) const noexcept
    {
        la::reference.gemm(a, x, y, b, c);
    }
    template <la::scalar T>
    void syrk(T a, matrix_view<const T> x, T b, matrix_view<T> c, la::triangle t) const noexcept
    {
        la::reference.syrk(a, x, b, c, t);
    }
    template <la::scalar T>
    snicholls::result<void, error> triangular_solve(matrix_view<const T> t, matrix_view<T> b, la::triangle w, la::diagonal d) const noexcept
    {
        return la::reference.triangular_solve(t, b, w, d);
    }
};

// Everything, but lu_factor returns a bool instead of a result: the concept rejects it.
struct wrong_return : la::reference_backend {
    template <la::scalar T>
    bool lu_factor(matrix_view<T>, std::span<pivot_index>) const noexcept
    {
        return true;
    }
};

// Everything, but it is not trivially copyable: a backend is a value passed into hot loops, so it must be.
struct not_trivially_copyable : la::reference_backend {
    not_trivially_copyable() = default;
    not_trivially_copyable(const not_trivially_copyable&) {}
};

// Only for double.
struct double_only : la::reference_backend {
    template <typename T>
    T dot(vector_view<const T>, vector_view<const T>) const noexcept = delete;
    double dot(vector_view<const double> x, vector_view<const double> y) const noexcept { return la::reference.dot(x, y); }
};

static_assert(la::backend<la::reference_backend, double> && la::backend<la::reference_backend, float>);
static_assert(la::blas_backend<blas_only, double> && !la::factorisation_backend<blas_only, double> && !la::backend<blas_only, double>);
static_assert(la::blas_backend<wrong_return, double> && !la::factorisation_backend<wrong_return, double> && !la::backend<wrong_return, double>);
static_assert(!la::backend<not_trivially_copyable, double>);
static_assert(la::backend<double_only, double> && !la::blas_backend<double_only, float>);
static_assert(!la::backend<la::reference_backend, int>);   // not a scalar
static_assert(!la::backend<int, double>);

// A backend lacking a member is a compile error naming it: tests/compile_fail/linalg_backend_missing_member.cpp.

// ---- What a backend receives ----------------------------------------------------------------------------

struct seen {
    std::size_t rows = 0, cols = 0, row_stride = 0, column_stride = 0, calls = 0;
};
seen last_seen;   // a backend must be trivially copyable, so it records into a global

struct recording_backend : la::reference_backend {
    template <la::scalar T>
    void lu_solve(matrix_view<const T> lu, std::span<const pivot_index> p, matrix_view<T> b) const noexcept
    {
        last_seen = {b.extent(0), b.extent(1), b.stride(0), b.stride(1), last_seen.calls + 1};
        la::reference.lu_solve(lu, p, b);
    }
    template <la::scalar T>
    snicholls::result<void, error> lu_factor(matrix_view<T> a, std::span<pivot_index> p) const noexcept
    {
        last_seen = {a.extent(0), a.extent(1), a.stride(0), a.stride(1), last_seen.calls + 1};
        return la::reference.lu_factor(a, p);
    }
};
static_assert(la::backend<recording_backend, double>);

TEST(LinalgBackendViews, ColumnMajorMatrixArrivesAsItsStrides)
{
    std::array<double, 4> a{4, 6, 3, 3};   // column-major [[4, 3], [6, 3]]
    std::array<pivot_index, 2> p{};
    last_seen = {};
    ASSERT_TRUE(la::lu_factor(recording_backend{}, matrix_view<double, layout_left>(a.data(), {2, 2}), std::span<pivot_index>(p)));
    EXPECT_EQ(last_seen.calls, 1u);
    EXPECT_EQ(last_seen.row_stride, 1u);
    EXPECT_EQ(last_seen.column_stride, 2u);
}

TEST(LinalgBackendViews, VectorRightHandSideArrivesAsOneColumn)
{
    std::array<double, 4> a{4, 3, 6, 3};
    std::array<pivot_index, 2> p{};
    std::array<double, 2> b{10, 12};
    const matrix_view<double, layout_right> m(a.data(), {2, 2});
    ASSERT_TRUE(la::lu_factor(recording_backend{}, m, std::span<pivot_index>(p)));
    ASSERT_TRUE(la::lu_solve(recording_backend{}, m, std::span<const pivot_index>(p), la::as_vector(std::span<double>(b))));
    EXPECT_EQ(last_seen.rows, 2u);
    EXPECT_EQ(last_seen.cols, 1u);
    EXPECT_EQ(last_seen.row_stride, 1u);
    EXPECT_DOUBLE_EQ(b[0], 1.0);
    EXPECT_DOUBLE_EQ(b[1], 2.0);
}

// ---- Extent checks: dimension_mismatch, before the backend runs, arguments unchanged ---------------------

// A backend whose QR always fails.
struct failing_qr : la::reference_backend {
    template <la::scalar T>
    snicholls::result<void, error> qr_factor(matrix_view<T>, std::span<T>) const noexcept
    {
        return snicholls::err(error::did_not_converge);
    }
};

template <typename R>
bool mismatch(const R& r)
{
    return !r && r.err() == error::dimension_mismatch;
}

TEST(LinalgBackendExtents, BlasOperationsRejectShapesThatDoNotFit)
{
    std::vector<double> s(12, 1.0);
    const auto copy = s;
    const vector_view<double, layout_right> v2(s.data(), {2}), v3(s.data(), {3});
    const matrix_view<double, layout_right> m23(s.data(), {2, 3}), m33(s.data(), {3, 3}), m22(s.data(), {2, 2});
    EXPECT_TRUE(mismatch(la::dot(la::reference, v2, v3)));
    EXPECT_TRUE(mismatch(la::axpy(la::reference, 1.0, v2, v3)));
    EXPECT_TRUE(mismatch(la::gemv(la::reference, 1.0, m23, v2, 0.0, v2)));   // x must have 3
    EXPECT_TRUE(mismatch(la::gemv(la::reference, 1.0, m23, v3, 0.0, v3)));   // y must have 2
    EXPECT_TRUE(mismatch(la::gemm(la::reference, 1.0, m23, m23, 0.0, m22)));
    EXPECT_TRUE(mismatch(la::gemm(la::reference, 1.0, m23, m33, 0.0, m22)));
    EXPECT_TRUE(mismatch(la::gemm(la::reference, 1.0, m22, m23, 0.0, m22)));
    EXPECT_TRUE(mismatch(la::syrk(la::reference, 1.0, m23, 0.0, m23, la::triangle::lower)));
    EXPECT_TRUE(mismatch(la::syrk(la::reference, 1.0, m33, 0.0, m22, la::triangle::lower)));
    EXPECT_TRUE(mismatch(la::triangular_solve(la::reference, m23, v2, la::triangle::lower)));
    EXPECT_TRUE(mismatch(la::triangular_solve(la::reference, m22, v3, la::triangle::upper)));
    EXPECT_EQ(s, copy);
}

TEST(LinalgBackendExtents, FactorisationsRejectShapesThatDoNotFit)
{
    std::vector<double> s(12, 1.0);
    const auto copy = s;
    std::array<pivot_index, 3> p{};
    std::array<double, 3> tau{};
    const vector_view<double, layout_right> v2(s.data(), {2});
    const matrix_view<double, layout_right> m23(s.data(), {2, 3}), m32(s.data(), {3, 2}), m33(s.data(), {3, 3}), m22(s.data(), {2, 2});
    const std::span<pivot_index> p3(p), p2(p.data(), 2);
    EXPECT_TRUE(mismatch(la::lu_factor(la::reference, m23, p3)));
    EXPECT_TRUE(mismatch(la::lu_factor(la::reference, m33, p2)));
    EXPECT_TRUE(mismatch(la::lu_solve(la::reference, m33, std::span<const pivot_index>(p3), v2)));
    EXPECT_TRUE(mismatch(la::lu_solve(la::reference, m23, std::span<const pivot_index>(p3), m33)));
    EXPECT_TRUE(mismatch(la::lu_solve(la::reference, m33, std::span<const pivot_index>(p2), m33)));
    EXPECT_TRUE(mismatch(la::lu_inverse(la::reference, m33, std::span<const pivot_index>(p3), m22)));
    EXPECT_TRUE(mismatch(la::lu_inverse(la::reference, m33, std::span<const pivot_index>(p3), m32)));
    std::vector<double> four(12, 2.0);   // 3 x 4 factors (not square) with a 3 x 3 inverse: nothing written
    const auto four_copy = four;
    EXPECT_TRUE(mismatch(la::lu_inverse(la::reference, matrix_view<double, layout_right>(s.data(), {3, 4}), std::span<const pivot_index>(p3),
                                        matrix_view<double, layout_right>(four.data(), {3, 3}))));
    EXPECT_EQ(four, four_copy);
    EXPECT_TRUE(mismatch(la::lu_log_det(m23, std::span<const pivot_index>(p3))));
    EXPECT_TRUE(mismatch(la::lu_log_det(m33, std::span<const pivot_index>(p2))));
    EXPECT_TRUE(mismatch(la::cholesky_factor(la::reference, m23)));
    EXPECT_TRUE(mismatch(la::cholesky_solve(la::reference, m23, m23)));
    EXPECT_TRUE(mismatch(la::cholesky_solve(la::reference, m33, v2)));
    EXPECT_TRUE(mismatch(la::cholesky_log_det(m23)));
    EXPECT_TRUE(mismatch(la::ldlt_factor(la::reference, m32)));
    EXPECT_TRUE(mismatch(la::ldlt_solve(la::reference, m32, m32)));
    EXPECT_TRUE(mismatch(la::ldlt_solve(la::reference, m33, v2)));
    EXPECT_TRUE(mismatch(la::ldlt_log_det(m32)));
    EXPECT_TRUE(mismatch(la::qr_factor(la::reference, m23, std::span<double>(tau))));            // m < n
    EXPECT_TRUE(mismatch(la::qr_factor(la::reference, m32, std::span<double>(tau))));            // tau must have 2
    EXPECT_TRUE(mismatch(la::qr_q(la::reference, m32, std::span<const double>(tau.data(), 2), m33)));
    EXPECT_TRUE(mismatch(la::qr_q(la::reference, m32, std::span<const double>(tau), m32)));
    EXPECT_TRUE(mismatch(la::qr_q(la::reference, m23, std::span<const double>(tau), m23)));
    EXPECT_TRUE(mismatch(la::qr_solve(la::reference, m32, std::span<const double>(tau.data(), 2), v2)));
    EXPECT_TRUE(mismatch(la::qr_solve(la::reference, m32, std::span<const double>(tau), m32)));
    EXPECT_TRUE(mismatch(la::qr_solve(la::reference, m23, std::span<const double>(tau), m23)));
    EXPECT_TRUE(mismatch(la::least_squares(la::reference, m32, std::span<double>(tau.data(), 2), v2)));
    EXPECT_TRUE(mismatch(la::least_squares(la::reference, m23, std::span<double>(tau), m23)));
    EXPECT_EQ(s, copy);
}

TEST(LinalgBackendExtents, DecompositionsRejectShapesThatDoNotFit)
{
    std::vector<double> s(12, 1.0);
    const auto copy = s;
    std::array<double, 3> w{};
    const matrix_view<double, layout_right> m23(s.data(), {2, 3}), m32(s.data(), {3, 2}), m33(s.data(), {3, 3}), m22(s.data(), {2, 2});
    EXPECT_TRUE(mismatch(la::symmetric_eigen(la::reference, m23, std::span<double>(w), m33)));
    EXPECT_TRUE(mismatch(la::symmetric_eigen(la::reference, m33, std::span<double>(w.data(), 2), m33)));
    EXPECT_TRUE(mismatch(la::symmetric_eigen(la::reference, m33, std::span<double>(w), m22)));
    EXPECT_TRUE(mismatch(la::svd(la::reference, m23, std::span<double>(w), m33)));   // m < n
    EXPECT_TRUE(mismatch(la::svd(la::reference, m32, std::span<double>(w), m22)));   // s must have 2
    EXPECT_TRUE(mismatch(la::svd(la::reference, m32, std::span<double>(w.data(), 2), m33)));
    EXPECT_EQ(s, copy);
}

TEST(LinalgBackendAliasing, AnOutputThatStartsAtAnInputIsReportedAndNothingIsWritten)
{
    const auto aliased = [](const auto& r) { return !r && r.err() == error::aliased; };
    std::vector<double> s{1, 1, 3, 3, 0, 0, 0, 0, 0}, rhs_store(3, 1.0);
    const auto copy = s;
    const matrix_view<double, layout_right> a(s.data(), {2, 2}), a3(s.data(), {3, 3});
    const vector_view<double, layout_right> x(s.data(), {2});
    std::array<pivot_index, 3> p{0, 1, 2};
    std::array<double, 3> w{};
    // The review's cases: gemm with C = A (gave {2, 2, 4, 4} for {2, 1, 4, 3}) and symmetric_eigen(a, w, a).
    EXPECT_TRUE(aliased(la::gemm(la::reference, 1.0, a, a, 0.0, a)));
    EXPECT_TRUE(aliased(la::symmetric_eigen(la::reference, a, std::span<double>(w.data(), 2), a)));
    EXPECT_TRUE(aliased(la::svd(la::reference, a, std::span<double>(w.data(), 2), a)));
    EXPECT_TRUE(aliased(la::gemv(la::reference, 1.0, a, la::vector_view<double, layout_right>(rhs_store.data(), {2}), 0.0, x)));
    EXPECT_TRUE(aliased(la::syrk(la::reference, 1.0, a, 0.0, a, la::triangle::lower)));
    EXPECT_TRUE(aliased(la::triangular_solve(la::reference, a, a, la::triangle::lower)));
    EXPECT_TRUE(aliased(la::lu_solve(la::reference, a, std::span<const pivot_index>(p.data(), 2), a)));
    EXPECT_TRUE(aliased(la::lu_inverse(la::reference, a, std::span<const pivot_index>(p.data(), 2), a)));
    EXPECT_TRUE(aliased(la::cholesky_solve(la::reference, a, x)));
    EXPECT_TRUE(aliased(la::ldlt_solve(la::reference, a, x)));
    EXPECT_TRUE(aliased(la::qr_q(la::reference, a3, std::span<const double>(w), a3)));
    EXPECT_TRUE(aliased(la::qr_solve(la::reference, a3, std::span<const double>(w), a3)));
    EXPECT_EQ(s, copy);
    // Without the alias, the same gemm gives A A.
    std::vector<double> c(4);
    const std::vector<double> m{1, 0, 1, 1};   // [[1, 0], [1, 1]]^2 = [[1, 0], [2, 1]]
    ASSERT_TRUE(la::gemm(la::reference, 1.0, matrix_view<const double, layout_right>(m.data(), {2, 2}),
                         matrix_view<const double, layout_right>(m.data(), {2, 2}), 0.0, matrix_view<double, layout_right>(c.data(), {2, 2})));
    EXPECT_EQ(c, (std::vector<double>{1, 0, 2, 1}));
}

TEST(LinalgBackendAliasing, LeastSquaresReportsAnAliasBeforeWritingA)
{
    // The review's case: A as both the matrix and the right-hand side. qr_factor used to write A first.
    std::vector<double> a{1, 1, 1, 2, 1, 3};
    const auto copy = a;
    std::array<double, 2> tau{7, 7};
    const matrix_view<double, layout_right> m(a.data(), {3, 2});
    const auto r = la::least_squares(la::reference, m, std::span<double>(tau), m);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.err(), error::aliased);
    EXPECT_EQ(a, copy);
    EXPECT_EQ(tau[0], 7.0);
}

TEST(LinalgBackendAliasing, SpanOutputsThatStartAtAMatrixAreReported)
{
    // The review's case: qr_factor(a, span(a.data(), 2)) returned ok with tau overwritten by R.
    std::vector<double> a{1, 2, 3, 4, 5, 6};
    const auto copy = a;
    const matrix_view<double, layout_right> m32(a.data(), {3, 2}), m22(a.data(), {2, 2});
    const auto aliased = [](const auto& r) { return !r && r.err() == error::aliased; };
    EXPECT_TRUE(aliased(la::qr_factor(la::reference, m32, std::span<double>(a.data(), 2))));
    std::vector<double> y(3, 1.0);
    EXPECT_TRUE(aliased(la::least_squares(la::reference, m32, std::span<double>(a.data(), 2), la::as_vector(std::span<double>(y)))));
    EXPECT_TRUE(aliased(la::least_squares(la::reference, m32, std::span<double>(y.data(), 2), la::as_vector(std::span<double>(y)))));
    EXPECT_TRUE(aliased(la::qr_solve(la::reference, m32, std::span<const double>(y.data(), 2), la::as_vector(std::span<double>(y)))));
    std::vector<double> q(6);
    EXPECT_TRUE(aliased(la::qr_q(la::reference, m32, std::span<const double>(q.data(), 2), matrix_view<double, layout_right>(q.data(), {3, 2}))));
    std::vector<double> v(4);
    EXPECT_TRUE(aliased(la::symmetric_eigen(la::reference, m22, std::span<double>(a.data(), 2), matrix_view<double, layout_right>(v.data(), {2, 2}))));
    EXPECT_TRUE(aliased(la::symmetric_eigen(la::reference, matrix_view<double, layout_right>(q.data(), {2, 2}), std::span<double>(v.data(), 2),
                                            matrix_view<double, layout_right>(v.data(), {2, 2}))));
    EXPECT_TRUE(aliased(la::svd(la::reference, m32, std::span<double>(a.data(), 2), matrix_view<double, layout_right>(v.data(), {2, 2}))));
    EXPECT_TRUE(aliased(la::svd(la::reference, matrix_view<double, layout_right>(q.data(), {3, 2}), std::span<double>(v.data(), 2),
                                matrix_view<double, layout_right>(v.data(), {2, 2}))));
    // Pivots are integers; a span of them starting at the matrix is still the same storage.
    // (Never written through: the call is refused before the backend runs.)
    const std::span<pivot_index> on_matrix(reinterpret_cast<pivot_index*>(a.data()), 2);
    EXPECT_TRUE(aliased(la::lu_factor(la::reference, m22, on_matrix)));
    EXPECT_EQ(a, copy);
}

TEST(LinalgBackendExtents, LeastSquaresStopsAtAFactorisationError)
{
    // A backend whose QR always fails: least_squares returns that error and does not solve.
    std::array<double, 6> a{1, 0, 0, 1, 1, 1};
    std::array<double, 3> b{1, 2, 3};
    std::array<double, 2> tau{};
    const auto r = la::least_squares(failing_qr{}, matrix_view<double, layout_right>(a.data(), {3, 2}), std::span<double>(tau),
                                     la::as_vector(std::span<double>(b)));
    ASSERT_FALSE(r);
    EXPECT_EQ(r.err(), error::did_not_converge);
    EXPECT_EQ(b[0], 1.0);
}

} // namespace
