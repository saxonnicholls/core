//
//  reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/linalg/reference.hpp>: the layout contract (a trivially copyable backend
//  holding one double, relative_tolerance, that models every concept), solves and decompositions evaluated at compile time, bit-for-bit
//  the same answer through every layout, and the behaviour particular to this backend (unpivoted LDLT,
//  exact-zero triangular test, NaN reported, sweep limit). Correctness of every operation is checked by
//  the conformance suite (tests/support/snicholls/core/conformance/linalg.hpp), which this file
//  registers the backend in, for float and double.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/conformance/linalg.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

// A number type whose division can throw, as an allocating multiprecision type's arithmetic can: it must reach
// the caller, not terminate. Dividing by 0.25 throws.
namespace reference_test_numbers {
    struct throwing_real {
        double v = 0;
        constexpr throwing_real() = default;
        template <typename A>
            requires std::is_arithmetic_v<A>
        constexpr throwing_real(A x) : v(static_cast<double>(x)) {}   // NOLINT: implicit, as a number's is
        constexpr explicit operator double() const { return v; }
        friend constexpr throwing_real operator+(throwing_real a, throwing_real b) { return a.v + b.v; }
        friend constexpr throwing_real operator-(throwing_real a, throwing_real b) { return a.v - b.v; }
        friend constexpr throwing_real operator*(throwing_real a, throwing_real b) { return a.v * b.v; }
        friend throwing_real operator/(throwing_real a, throwing_real b)
        {
            if (b.v == 0.25)
                throw std::runtime_error("throwing_real: division by 0.25");
            return a.v / b.v;
        }
        friend constexpr throwing_real operator-(throwing_real a) { return -a.v; }
        constexpr throwing_real& operator+=(throwing_real b) { return *this = *this + b; }
        constexpr throwing_real& operator-=(throwing_real b) { return *this = *this - b; }
        constexpr throwing_real& operator*=(throwing_real b) { return *this = *this * b; }
        throwing_real& operator/=(throwing_real b) { return *this = *this / b; }
        friend constexpr bool operator==(throwing_real a, throwing_real b) { return a.v == b.v; }
        friend constexpr bool operator<(throwing_real a, throwing_real b) { return a.v < b.v; }
        friend constexpr bool operator>(throwing_real a, throwing_real b) { return a.v > b.v; }
        friend constexpr bool operator<=(throwing_real a, throwing_real b) { return a.v <= b.v; }
        friend constexpr bool operator>=(throwing_real a, throwing_real b) { return a.v >= b.v; }
    };
    inline throwing_real fabs(throwing_real x) { return std::fabs(x.v); }
    inline throwing_real sqrt(throwing_real x) { return std::sqrt(x.v); }
    inline throwing_real log(throwing_real x) { return std::log(x.v); }
    inline throwing_real exp(throwing_real x) { return std::exp(x.v); }
} // namespace reference_test_numbers

template <>
struct std::numeric_limits<reference_test_numbers::throwing_real> : std::numeric_limits<double> {
    using T = reference_test_numbers::throwing_real;
    static constexpr T epsilon() noexcept { return std::numeric_limits<double>::epsilon(); }
    static constexpr T min() noexcept { return std::numeric_limits<double>::min(); }
    static constexpr T max() noexcept { return std::numeric_limits<double>::max(); }
    static constexpr T infinity() noexcept { return std::numeric_limits<double>::infinity(); }
    static constexpr T quiet_NaN() noexcept { return std::numeric_limits<double>::quiet_NaN(); }
};
template <>
struct snicholls::math::math_limits<reference_test_numbers::throwing_real> {
    using compute_type = double;
};

namespace {

namespace la = snicholls::linalg;
using la::error;
using la::layout_left;
using la::layout_right;
using la::matrix_view;
using la::pivot_index;

static_assert(la::backend<la::reference_backend, float> && la::backend<la::reference_backend, double>);
static_assert(std::is_trivially_copyable_v<la::reference_backend> && la::reference_backend{}.relative_tolerance == 1.0);
static_assert(noexcept(la::reference.dot(std::declval<la::vector_view<const double>>(), std::declval<la::vector_view<const double>>())));

static_assert(la::detail::nothrow_scalar<double> && la::detail::nothrow_scalar<float>);
static_assert(noexcept(la::reference.lu_factor(std::declval<matrix_view<float>>(), std::declval<std::span<pivot_index>>())));
// The free functions are not declared noexcept: they throw what the backend's member throws.
static_assert(!noexcept(la::lu_factor(la::reference, std::declval<matrix_view<double>>(), std::declval<std::span<pivot_index>>())));

// ---- At compile time ------------------------------------------------------------------------------------

// 4x + 3y = 10, 6x + 3y = 12: x = 1, y = 2; det = -6.
consteval bool lu_solves_at_compile_time()
{
    std::array<double, 4> a{4, 3, 6, 3};
    std::array<double, 2> b{10, 12};
    std::array<pivot_index, 2> p{};
    const matrix_view<double, layout_right> m(a.data(), {2, 2});
    if (!la::lu_factor(la::reference, m, std::span<pivot_index>(p)))
        return false;
    if (!la::lu_solve(la::reference, m, std::span<const pivot_index>(p), la::as_vector(std::span<double>(b))))
        return false;
    const auto d = la::lu_log_det(m, std::span<const pivot_index>(p));
    return b[0] == 1.0 && b[1] == 2.0 && d && d->sign == -1;
}
static_assert(lu_solves_at_compile_time());

// The 2 x 2 matrix [[2, 1], [1, 2]] has eigenvalues 1 and 3.
consteval bool eigen_at_compile_time()
{
    std::array<double, 4> a{2, 1, 1, 2};
    std::array<double, 4> v{};
    std::array<double, 2> w{};
    const bool ok = static_cast<bool>(la::symmetric_eigen(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}), std::span<double>(w),
                                                          matrix_view<double, layout_right>(v.data(), {2, 2})));
    return ok && w[0] > 1.0 - 1e-12 && w[0] < 1.0 + 1e-12 && w[1] > 3.0 - 1e-12 && w[1] < 3.0 + 1e-12;
}
static_assert(eigen_at_compile_time());

consteval bool cholesky_rejects_indefinite_at_compile_time()
{
    std::array<float, 4> a{1, 2, 2, 1};
    const auto r = la::cholesky_factor(la::reference, matrix_view<float, layout_right>(a.data(), {2, 2}));
    return !r && r.err() == error::not_positive_definite;
}
static_assert(cholesky_rejects_indefinite_at_compile_time());

// ---- The same bits through every layout -----------------------------------------------------------------

std::vector<double> sample(std::size_t n, std::size_t salt)
{
    std::vector<double> a(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a[i * n + j] = std::sin(static_cast<double>(1 + i * 7 + j * 13 + salt)) + (i == j ? 3.0 : 0.0);
    return a;
}

TEST(LinalgReferenceLayouts, LuSolveIsBitIdenticalInEveryLayout)
{
    constexpr std::size_t n = 9;
    const std::vector<double> a = sample(n, 0);
    std::vector<double> rhs(n);
    for (std::size_t i = 0; i < n; ++i)
        rhs[i] = static_cast<double>(i) - 4.0;

    // Row-major.
    std::vector<double> r = a, xr = rhs;
    std::vector<pivot_index> pr(n);
    ASSERT_TRUE(la::lu_factor(la::reference, matrix_view<double, layout_right>(r.data(), {n, n}), std::span<pivot_index>(pr)));
    ASSERT_TRUE(la::lu_solve(la::reference, matrix_view<double, layout_right>(r.data(), {n, n}), std::span<const pivot_index>(pr),
                             la::as_vector(std::span<double>(xr))));

    // Column-major: store the transpose's elements, read through layout_left.
    std::vector<double> c(n * n), xc = rhs;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            c[i + j * n] = a[i * n + j];
    std::vector<pivot_index> pc(n);
    ASSERT_TRUE(la::lu_factor(la::reference, matrix_view<double, layout_left>(c.data(), {n, n}), std::span<pivot_index>(pc)));
    ASSERT_TRUE(la::lu_solve(la::reference, matrix_view<double, layout_left>(c.data(), {n, n}), std::span<const pivot_index>(pc),
                             la::as_vector(std::span<double>(xc))));

    // Strided with no unit stride: rows of 2n with every other element, the right-hand side every third.
    std::vector<double> s(2 * n * n * 2), xs_storage(3 * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            s[i * 4 * n + 2 * j] = a[i * n + j];
    for (std::size_t i = 0; i < n; ++i)
        xs_storage[3 * i] = rhs[i];
    const matrix_view<double> sv(s.data(), {n, n}, {4 * n, 2});
    const la::vector_view<double> xs(xs_storage.data(), {n}, {3});
    std::vector<pivot_index> ps(n);
    ASSERT_TRUE(la::lu_factor(la::reference, sv, std::span<pivot_index>(ps)));
    ASSERT_TRUE(la::lu_solve(la::reference, sv, std::span<const pivot_index>(ps), xs));

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(xr[i], xc[i]) << i;
        EXPECT_EQ(xr[i], xs(i)) << i;
    }
    EXPECT_EQ(pr, pc);
    EXPECT_EQ(pr, ps);
}

TEST(LinalgReferenceLayouts, GemmIsBitIdenticalInEveryLayout)
{
    constexpr std::size_t n = 7;
    const std::vector<double> a = sample(n, 1), b = sample(n, 2);
    std::vector<double> c1(n * n), c2(n * n);
    std::vector<double> at(n * n), bt(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            at[j * n + i] = a[i * n + j];
            bt[j * n + i] = b[i * n + j];
        }
    ASSERT_TRUE(la::gemm(la::reference, 1.5, matrix_view<const double, layout_right>(a.data(), {n, n}),
                         matrix_view<const double, layout_right>(b.data(), {n, n}), 0.0, matrix_view<double, layout_right>(c1.data(), {n, n})));
    ASSERT_TRUE(la::gemm(la::reference, 1.5, matrix_view<const double, layout_left>(at.data(), {n, n}),
                         matrix_view<const double, layout_left>(bt.data(), {n, n}), 0.0, matrix_view<double, layout_left>(c2.data(), {n, n})));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            EXPECT_EQ(c1[i * n + j], c2[j * n + i]);
}

// ---- Behaviour particular to the reference backend ------------------------------------------------------

TEST(LinalgReference, LdltIsUnpivotedSoAZeroLeadingDiagonalIsSingular)
{
    // [[0, 1], [1, 0]] is nonsingular but needs pivoting; LU solves it.
    std::array<double, 4> a{0, 1, 1, 0};
    const auto r = la::ldlt_factor(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}));
    ASSERT_FALSE(r);
    EXPECT_EQ(r.err(), error::singular);
    std::array<double, 4> b{0, 1, 1, 0};
    std::array<pivot_index, 2> p{};
    EXPECT_TRUE(la::lu_factor(la::reference, matrix_view<double, layout_right>(b.data(), {2, 2}), std::span<pivot_index>(p)));
}

using reference_test_numbers::throwing_real;
static_assert(la::scalar<throwing_real> && !la::detail::nothrow_scalar<throwing_real>);
static_assert(!noexcept(la::reference.lu_factor(std::declval<matrix_view<throwing_real>>(), std::declval<std::span<pivot_index>>())));
static_assert(!noexcept(la::pivot_tolerance<throwing_real>(1, throwing_real(1))) && !noexcept(la::ldlt_growth_limit<throwing_real>(1)));
static_assert(!noexcept(la::signed_log_det<throwing_real>{}.value()) && noexcept(la::signed_log_det<double>{}.value()));

TEST(LinalgReference, ANumberTypeThatThrowsPropagatesItsException)
{
    // LU of diag(0.25, 1) divides by the pivot 0.25: the exception must reach the caller, not std::terminate.
    std::array<throwing_real, 4> a{throwing_real(0.25), throwing_real(0), throwing_real(0), throwing_real(1)};
    std::array<pivot_index, 2> p{};
    EXPECT_THROW(static_cast<void>(la::lu_factor(la::reference, matrix_view<throwing_real, layout_right>(a.data(), {2, 2}), std::span<pivot_index>(p))),
                 std::runtime_error);
    // The same type works where nothing throws.
    std::array<throwing_real, 4> b{throwing_real(2), throwing_real(0), throwing_real(0), throwing_real(4)};
    EXPECT_TRUE(la::lu_factor(la::reference, matrix_view<throwing_real, layout_right>(b.data(), {2, 2}), std::span<pivot_index>(p)));
}

TEST(LinalgReference, LdltReportsGrowthThatWouldMakeItInaccurate)
{
    // A small leading pivot: unpivoted LDLT would lose about 6 digits (forward error 2e-10 where LU gives 4e-16).
    std::array<double, 9> a{1e-6, 1, 0, 1, 0, 1, 0, 1, 1};
    const auto r = la::ldlt_factor(la::reference, matrix_view<double, layout_right>(a.data(), {3, 3}));
    ASSERT_FALSE(r);
    EXPECT_EQ(r.err(), error::singular);
    // LU solves it to the last digit.
    std::array<double, 9> b{1e-6, 1, 0, 1, 0, 1, 0, 1, 1};
    std::array<double, 3> x{1e-6 + 2.0, 4, 5};
    std::array<pivot_index, 3> p{};
    ASSERT_TRUE(la::lu_factor(la::reference, matrix_view<double, layout_right>(b.data(), {3, 3}), std::span<pivot_index>(p)));
    ASSERT_TRUE(la::lu_solve(la::reference, matrix_view<double, layout_right>(b.data(), {3, 3}), std::span<const pivot_index>(p),
                             la::as_vector(std::span<double>(x))));
    EXPECT_NEAR(x[0], 1.0, 1e-14);
    EXPECT_NEAR(x[2], 3.0, 1e-14);
    // The limit: 8192 n for double, about 54 n for float.
    EXPECT_NEAR(la::ldlt_growth_limit<double>(3), 3.0 / std::sqrt(std::sqrt(std::numeric_limits<double>::epsilon())), 1e-9);
    EXPECT_GT(la::ldlt_growth_limit<float>(1), 50.0f);
}

TEST(LinalgReference, TheDefaultToleranceRejectsBadlyScaledMatricesAndZeroAcceptsThem)
{
    const la::reference_backend exact{.relative_tolerance = 0.0};   // LAPACK's rule: only an exact zero is singular
    std::array<pivot_index, 3> p{};
    {
        std::array<double, 4> a{1e16, 0, 0, 1}, c = a;
        EXPECT_EQ(la::lu_factor(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}), std::span<pivot_index>(p.data(), 2)).err(),
                  error::singular);
        EXPECT_EQ(la::cholesky_factor(la::reference, matrix_view<double, layout_right>(c.data(), {2, 2})).err(), error::not_positive_definite);
        std::array<double, 4> a2{1e16, 0, 0, 1}, c2 = a2;
        std::array<double, 2> x{1e16, 1}, y{1e16, 1};
        ASSERT_TRUE(la::lu_factor(exact, matrix_view<double, layout_right>(a2.data(), {2, 2}), std::span<pivot_index>(p.data(), 2)));
        ASSERT_TRUE(la::lu_solve(exact, matrix_view<double, layout_right>(a2.data(), {2, 2}), std::span<const pivot_index>(p.data(), 2),
                                 la::as_vector(std::span<double>(x))));
        ASSERT_TRUE(la::cholesky_factor(exact, matrix_view<double, layout_right>(c2.data(), {2, 2})));
        ASSERT_TRUE(la::cholesky_solve(exact, matrix_view<double, layout_right>(c2.data(), {2, 2}), la::as_vector(std::span<double>(y))));
        EXPECT_EQ(x[0], 1.0);
        EXPECT_EQ(x[1], 1.0);
        EXPECT_EQ(y[0], 1.0);
        EXPECT_EQ(y[1], 1.0);
    }
    {
        std::array<float, 4> c{1e7f, 0, 0, 1}, c2 = c;
        EXPECT_EQ(la::cholesky_factor(la::reference, matrix_view<float, layout_right>(c.data(), {2, 2})).err(), error::not_positive_definite);
        EXPECT_TRUE(la::cholesky_factor(exact, matrix_view<float, layout_right>(c2.data(), {2, 2})));
    }
    {
        // Two tiny pivots bordered by ones: nonsingular (det -2e-16) but below eps * n * max |a|. Its condition number
        // is about 1e16, so with relative_tolerance = 0 the answer is the backward-stable one, a small residual.
        const std::array<double, 9> a0{1e-16, 0, 1, 0, 1e-16, 1, 1, 1, 0};
        std::array<double, 9> a = a0, a2 = a0;
        std::array<double, 3> x{1, 1, 1};
        EXPECT_EQ(la::lu_factor(la::reference, matrix_view<double, layout_right>(a.data(), {3, 3}), std::span<pivot_index>(p)).err(),
                  error::singular);
        ASSERT_TRUE(la::lu_factor(exact, matrix_view<double, layout_right>(a2.data(), {3, 3}), std::span<pivot_index>(p)));
        ASSERT_TRUE(la::lu_solve(exact, matrix_view<double, layout_right>(a2.data(), {3, 3}), std::span<const pivot_index>(p),
                                 la::as_vector(std::span<double>(x))));
        for (std::size_t i = 0; i < 3; ++i) {
            const double r = a0[i * 3] * x[0] + a0[i * 3 + 1] * x[1] + a0[i * 3 + 2] * x[2] - 1.0;
            EXPECT_LE(std::abs(r), 1e-14) << i;
        }
    }
}

TEST(LinalgReference, ANegativeOrNanRelativeToleranceIsClampedToZero)
{
    for (const double setting : {-1.0, std::numeric_limits<double>::quiet_NaN(), 0.0}) {
        const la::reference_backend backend{.relative_tolerance = setting};
        std::array<double, 4> singular{1, 0, 0, 0}, regular{1, 0, 0, 1e-300};
        std::array<pivot_index, 2> p{};
        EXPECT_EQ(la::lu_factor(backend, matrix_view<double, layout_right>(singular.data(), {2, 2}), std::span<pivot_index>(p)).err(),
                  error::singular)
            << setting;
        EXPECT_TRUE(la::lu_factor(backend, matrix_view<double, layout_right>(regular.data(), {2, 2}), std::span<pivot_index>(p))) << setting;
        std::array<double, 4> zero_diagonal{0, 0, 0, 1};
        EXPECT_EQ(la::cholesky_factor(backend, matrix_view<double, layout_right>(zero_diagonal.data(), {2, 2})).err(), error::not_positive_definite)
            << setting;
    }
    const la::reference_backend everything{.relative_tolerance = std::numeric_limits<double>::infinity()};
    std::array<double, 4> identity{1, 0, 0, 1};
    std::array<pivot_index, 2> p{};
    EXPECT_EQ(la::lu_factor(everything, matrix_view<double, layout_right>(identity.data(), {2, 2}), std::span<pivot_index>(p)).err(), error::singular);
}

TEST(LinalgReference, TheGrowthCheckAppliesWhateverTheTolerance)
{
    const la::reference_backend lapack_rule{.relative_tolerance = 0.0};
    std::array<double, 9> a{1e-6, 1, 0, 1, 0, 1, 0, 1, 1};
    EXPECT_EQ(la::ldlt_factor(lapack_rule, matrix_view<double, layout_right>(a.data(), {3, 3})).err(), error::singular);
    // A bordered identity, k = 8: growth 2 k = 16, far below ldlt_growth_limit(9), so it factors.
    std::array<double, 81> bordered{};
    for (std::size_t i = 0; i < 9; ++i)
        for (std::size_t j = 0; j < 9; ++j)
            bordered[i * 9 + j] = i < 8 && j < 8 ? (i == j ? 1.0 : 0.0) : (i == 8 && j == 8 ? 0.0 : 1.0);
    EXPECT_TRUE(la::ldlt_factor(la::reference, matrix_view<double, layout_right>(bordered.data(), {9, 9})));
    EXPECT_EQ(bordered[80], -8.0);   // d = -k; growth max_i sum_k l_ik^2 |d_k| = 8 + 8 = 16
}

TEST(LinalgReference, AnEigenvalueBeyondTheRangeIsInfinityWithSuccess)
{
    // Documented: [[1e308, 1e308], [1e308, 1e308]] has eigenvalues 0 and 2e308; 2e308 rounds to infinity.
    std::array<double, 4> a{1e308, 1e308, 1e308, 1e308}, v{};
    std::array<double, 2> w{};
    ASSERT_TRUE(la::symmetric_eigen(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}), std::span<double>(w),
                                    matrix_view<double, layout_right>(v.data(), {2, 2})));
    EXPECT_NEAR(w[0], 0.0, 1e293);
    EXPECT_EQ(w[1], std::numeric_limits<double>::infinity());
}

TEST(LinalgReference, TriangularSolveIsSingularOnlyForAnExactZero)
{
    std::array<double, 4> t{1, 0, 5, 1e-300};
    std::array<double, 2> b{1, 1};
    EXPECT_TRUE(la::triangular_solve(la::reference, matrix_view<const double, layout_right>(t.data(), {2, 2}), la::as_vector(std::span<double>(b)),
                                     la::triangle::lower));
    t[3] = 0.0;
    b = {1, 1};
    const auto r = la::triangular_solve(la::reference, matrix_view<const double, layout_right>(t.data(), {2, 2}), la::as_vector(std::span<double>(b)),
                                        la::triangle::lower);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.err(), error::singular);
    EXPECT_EQ(b[0], 1.0);   // unchanged
    // With an implicit unit diagonal the zero is never read.
    EXPECT_TRUE(la::triangular_solve(la::reference, matrix_view<const double, layout_right>(t.data(), {2, 2}), la::as_vector(std::span<double>(b)),
                                     la::triangle::lower, la::diagonal::implicit_unit));
    EXPECT_EQ(b[1], 1.0 - 5.0);
}

TEST(LinalgReference, NanIsReportedNeverSolved)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 4> a{1, nan, 0, 1};
    std::array<pivot_index, 2> p{};
    EXPECT_EQ(la::lu_factor(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}), std::span<pivot_index>(p)).err(), error::singular);
    a = {nan, 0, 0, 1};
    EXPECT_EQ(la::cholesky_factor(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2})).err(), error::not_positive_definite);
    a = {1, 0, nan, 1};
    EXPECT_EQ(la::ldlt_factor(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2})).err(), error::singular);
    std::array<double, 4> v{};
    std::array<double, 2> w{};
    a = {1, 0, nan, 1};
    EXPECT_EQ(la::symmetric_eigen(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}), std::span<double>(w),
                                  matrix_view<double, layout_right>(v.data(), {2, 2}))
                  .err(),
              error::did_not_converge);
    a = {1, 0, nan, 1};
    EXPECT_EQ(la::svd(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}), std::span<double>(w),
                      matrix_view<double, layout_right>(v.data(), {2, 2}))
                  .err(),
              error::did_not_converge);
}

TEST(LinalgReference, ZeroMatrixDecomposesToZeros)
{
    std::array<double, 4> a{}, v{};
    std::array<double, 2> w{1, 1};
    ASSERT_TRUE(la::symmetric_eigen(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}), std::span<double>(w),
                                    matrix_view<double, layout_right>(v.data(), {2, 2})));
    EXPECT_EQ(w[0], 0.0);
    EXPECT_EQ(w[1], 0.0);
    a = {};
    ASSERT_TRUE(la::svd(la::reference, matrix_view<double, layout_right>(a.data(), {2, 2}), std::span<double>(w),
                        matrix_view<double, layout_right>(v.data(), {2, 2})));
    EXPECT_EQ(w[0], 0.0);
    EXPECT_EQ(a[0], 0.0);   // U's columns for zero singular values are zero
}

TEST(LinalgReference, EmptyMatricesAreTrivial)
{
    std::array<double, 1> a{};
    std::array<pivot_index, 1> p{};
    const matrix_view<double, layout_right> m(a.data(), {0, 0});
    EXPECT_TRUE(la::lu_factor(la::reference, m, std::span<pivot_index>(p.data(), 0)));
    EXPECT_TRUE(la::cholesky_factor(la::reference, m));
    EXPECT_TRUE(la::ldlt_factor(la::reference, m));
    const auto d = la::lu_log_det(m, std::span<const pivot_index>(p.data(), 0));
    ASSERT_TRUE(d);
    EXPECT_EQ(d->value(), 1.0);   // the empty product
}

} // namespace

// The reference backend in every conformance suite, float and double.
SNICHOLLS_LINALG_CONFORMANCE(Reference, snicholls::conformance::linalg_case<snicholls::linalg::reference_backend, float>,
                             snicholls::conformance::linalg_case<snicholls::linalg::reference_backend, double>);
