//
//  fixed_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/linalg/fixed.hpp>: the layout contract of fixed_matrix, construction, the arithmetic
//  and block operations, Cholesky and its solves, LU's solve, inverse and determinant (with pivoting and the singular
//  error), the Jacobi eigen decomposition (and its did_not_converge error on NaN), tria, the views, and constant
//  evaluation of each factorisation.
//

#include <snicholls/core/linalg/fixed.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace {

namespace la = snicholls::linalg;
using m22 = la::fixed_matrix<double, 2, 2>;
using m33 = la::fixed_matrix<double, 3, 3>;
using m44 = la::fixed_matrix<double, 4, 4>;
using v3 = la::fixed_vector<double, 3>;

// The layout contract: one std::array, row-major, an aggregate, trivially copyable
static_assert(sizeof(m44) == 16 * sizeof(double) && alignof(m44) == alignof(double));
static_assert(sizeof(la::fixed_matrix<float, 3, 2>) == sizeof(std::array<float, 6>));
static_assert(std::is_aggregate_v<m44> && std::is_trivially_copyable_v<m44> && std::is_standard_layout_v<m44>);
static_assert(std::is_trivially_copyable_v<la::eigen_decomposition<double, 3>>);
static_assert(std::is_same_v<la::fixed_vector<double, 3>, la::fixed_matrix<double, 3, 1>>);
static_assert(m33::rows == 3 && m33::cols == 3 && m33::size == 9 && la::fixed_matrix<double, 2, 5>::size == 10);
static_assert(std::is_same_v<m33::value_type, double>);

// operator[] exists for vectors (a row or a column) only
template <typename M>
concept indexable = requires(M m) { m[0]; };
static_assert(indexable<v3> && indexable<la::fixed_matrix<double, 1, 4>> && !indexable<m22>);

// noexcept for double
static_assert(noexcept(std::declval<m33>() * std::declval<m33>()) && noexcept(std::declval<m33>() + std::declval<m33>()));
static_assert(noexcept(la::cholesky(std::declval<const m33&>())) && noexcept(la::solve(std::declval<const m33&>(), std::declval<const v3&>())));
static_assert(noexcept(la::symmetric_eigen(std::declval<const m33&>())) && noexcept(la::tria(std::declval<const la::fixed_matrix<double, 3, 5>&>())));
static_assert(noexcept(la::determinant(std::declval<const m33&>())) && noexcept(la::spd_inverse(std::declval<const m33&>())));
static_assert(noexcept(la::make_vector(1.0, 2.0)) && noexcept(m33::identity()) && noexcept(la::view(std::declval<m33&>())));

// A = L Lᵀ with an integer factor (the textbook example): L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]], det A = 36
constexpr m33 spd{{4, 12, -16, 12, 37, -43, -16, -43, 98}};
constexpr m33 spd_l{{2, 0, 0, 6, 1, 0, -8, 5, 3}};

// The factorisations run in a constant expression; these inputs give exact results
static_assert(*la::cholesky(spd) == spd_l);
static_assert(la::cholesky(m22{{1, 2, 2, 1}}).unwrap_err_unchecked() == la::error::not_positive_definite);
static_assert(la::determinant(m33{{2, -1, 0, 1, 3, 2, 0, 1, 4}}) == 24);
static_assert(*la::solve(m22{{0, 2, 4, 0}}, la::make_vector(6.0, 8.0)) == la::make_vector(2.0, 3.0));
static_assert(la::solve(m22{{1, 2, 2, 4}}, la::make_vector(1.0, 1.0)).unwrap_err_unchecked() == la::error::singular);
static_assert(*la::inverse(m22{{2, 0, 0, 4}}) == m22{{0.5, 0, 0, 0.25}});
static_assert(la::cholesky_solve(spd_l, la::fixed_vector<double, 3>{{4, 12, -16}}) == la::fixed_vector<double, 3>{{1, 0, 0}});
static_assert(la::symmetric_eigen(m22{{3, 0, 0, 1}})->values == la::make_vector(1.0, 3.0));
static_assert(la::tria(la::fixed_matrix<double, 2, 3>{{3, 0, 4, 0, 2, 0}}) == m22{{5, 0, 0, 2}});
static_assert(la::trace(spd) == 139 && la::dot(la::make_vector(1.0, 2.0), la::make_vector(3.0, 4.0)) == 11);

template <std::size_t R, std::size_t C>
void expect_near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol)
{
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol) << "(" << i << ", " << j << ")";
}

TEST(LinalgFixedConstruction, AggregateIsRowByRowAndZeroInitialised)
{
    const la::fixed_matrix<double, 2, 3> a{{1, 2, 3, 4, 5, 6}};
    EXPECT_EQ(a(0, 2), 3);
    EXPECT_EQ(a(1, 0), 4);
    EXPECT_EQ(a.elements[5], 6);
    const m22 z{};
    EXPECT_EQ(z, m22::zero());
    for (const double e : z.elements)
        EXPECT_EQ(e, 0);
    la::fixed_matrix<double, 2, 3> b = a;
    b(1, 2) = -1;
    EXPECT_EQ(b.elements[5], -1);
    EXPECT_NE(a, b);
}

TEST(LinalgFixedConstruction, IdentityHasOnesOnTheDiagonal)
{
    const auto i = m33::identity();
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_EQ(i(r, c), r == c ? 1.0 : 0.0);
    EXPECT_EQ((la::fixed_matrix<double, 1, 1>::identity()(0, 0)), 1);
}

TEST(LinalgFixedConstruction, MakeVectorUsesTheCommonType)
{
    const auto v = la::make_vector(1, 2.5, 3.0f);
    static_assert(std::is_same_v<std::remove_const_t<decltype(v)>, la::fixed_vector<double, 3>>);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.5);
    EXPECT_EQ(v[2], 3.0);
    const auto f = la::make_vector(1.0f, 2.0f);
    static_assert(std::is_same_v<std::remove_const_t<decltype(f)>, la::fixed_vector<float, 2>>);
    EXPECT_EQ(f(1, 0), 2.0f);
}

TEST(LinalgFixedArithmetic, SumDifferenceNegationAndScalarProducts)
{
    const m22 a{{1, 2, 3, 4}}, b{{10, 20, 30, 40}};
    EXPECT_EQ(a + b, (m22{{11, 22, 33, 44}}));
    EXPECT_EQ(b - a, (m22{{9, 18, 27, 36}}));
    EXPECT_EQ(-a, (m22{{-1, -2, -3, -4}}));
    EXPECT_EQ(a * 2.0, (m22{{2, 4, 6, 8}}));
    EXPECT_EQ(2.0 * a, a * 2.0);
    EXPECT_EQ(b / 10.0, a);
    m22 c = a;
    c += b;
    EXPECT_EQ(c, a + b);
    c -= b;
    EXPECT_EQ(c, a);
    EXPECT_EQ(&(c += b), &c);
    EXPECT_EQ(&(c -= b), &c);
}

TEST(LinalgFixedArithmetic, ProductMatchesHandCalculation)
{
    const la::fixed_matrix<double, 2, 3> a{{1, 2, 3, 4, 5, 6}};
    const la::fixed_matrix<double, 3, 2> b{{7, 8, 9, 10, 11, 12}};
    // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    EXPECT_EQ(a * b, (m22{{58, 64, 139, 154}}));
    EXPECT_EQ(a * la::make_vector(1.0, 0.0, -1.0), la::make_vector(-2.0, -2.0));
    EXPECT_EQ(m33::identity() * spd, spd);
    EXPECT_EQ(spd * m33::identity(), spd);
    EXPECT_EQ(m33::zero() * spd, m33::zero());
}

TEST(LinalgFixedArithmetic, TransposeTraceDotAndNorms)
{
    const la::fixed_matrix<double, 2, 3> a{{1, 2, 3, 4, 5, 6}};
    const auto at = la::transpose(a);
    static_assert(std::is_same_v<std::remove_const_t<decltype(at)>, la::fixed_matrix<double, 3, 2>>);
    EXPECT_EQ(at(2, 1), 6);
    EXPECT_EQ(at(0, 1), 4);
    EXPECT_EQ(la::transpose(at), a);
    EXPECT_EQ(la::trace(m22{{1, 2, 3, 4}}), 5);
    EXPECT_EQ(la::dot(la::make_vector(1.0, 2.0, 3.0), la::make_vector(4.0, -5.0, 6.0)), 12);
    EXPECT_EQ(la::squared_norm(a), 91);
    EXPECT_EQ(la::norm(la::make_vector(3.0, 4.0)), 5);
    EXPECT_EQ(la::norm(m22::zero()), 0);
}

TEST(LinalgFixedArithmetic, OuterSandwichAndSymmetrise)
{
    const auto o = la::outer(la::make_vector(1.0, 2.0), la::make_vector(3.0, 4.0, 5.0));
    EXPECT_EQ(o, (la::fixed_matrix<double, 2, 3>{{3, 4, 5, 6, 8, 10}}));
    const la::fixed_matrix<double, 2, 3> a{{1, 2, 0, 0, 1, -1}};
    const auto s = la::sandwich(a, spd);
    const auto ref = a * spd * la::transpose(a);
    EXPECT_EQ(s(0, 1), s(1, 0));
    expect_near(s, ref, 1e-12);
    const m22 n{{1, 2, 4, 3}};
    EXPECT_EQ(la::symmetrise(n), (m22{{1, 3, 3, 3}}));
    EXPECT_EQ(la::symmetrise(spd), spd);
}

TEST(LinalgFixedBlocks, DiagonalBlockSetBlockBlockDiagonalAndStack)
{
    const auto d = la::diagonal_matrix(la::make_vector(1.0, 2.0, 3.0));
    EXPECT_EQ(d, (m33{{1, 0, 0, 0, 2, 0, 0, 0, 3}}));
    EXPECT_EQ(la::diagonal_of(spd), la::make_vector(4.0, 37.0, 98.0));
    EXPECT_EQ((la::block<1, 1, 2, 2>(spd)), (m22{{37, -43, -43, 98}}));
    EXPECT_EQ((la::block<0, 2, 3, 1>(spd)), la::make_vector(-16.0, -43.0, 98.0));
    m33 m{};
    la::set_block<1, 0>(m, m22{{1, 2, 3, 4}});
    EXPECT_EQ(m, (m33{{0, 0, 0, 1, 2, 0, 3, 4, 0}}));
    const auto bd = la::block_diagonal(m22{{1, 2, 3, 4}}, la::fixed_matrix<double, 1, 1>{{5}});
    EXPECT_EQ(bd, (m33{{1, 2, 0, 3, 4, 0, 0, 0, 5}}));
    const auto st = la::stack(la::make_vector(1.0, 2.0), la::make_vector(3.0));
    EXPECT_EQ(st, la::make_vector(1.0, 2.0, 3.0));
}

TEST(LinalgFixedCholesky, FactorsTheTextbookMatrixExactly)
{
    const auto l = la::cholesky(spd);
    ASSERT_TRUE(l.is_ok());
    expect_near(*l, spd_l, 1e-14);
    expect_near(*l * la::transpose(*l), spd, 1e-12);
}

TEST(LinalgFixedCholesky, ReadsOnlyTheLowerTriangle)
{
    m33 a = spd;
    a(0, 1) = 1e9;
    a(0, 2) = -7;
    a(1, 2) = 42;
    EXPECT_EQ(*la::cholesky(a), *la::cholesky(spd));
}

TEST(LinalgFixedCholesky, IsScaleInvariant)
{
    const auto l = la::cholesky(spd * 1e-30);
    ASSERT_TRUE(l.is_ok());
    expect_near(*l * 1e15, spd_l, 1e-13);
    EXPECT_TRUE(la::cholesky(spd * 1e30).is_ok());
}

TEST(LinalgFixedCholesky, RejectsIndefiniteSingularZeroAndNaN)
{
    EXPECT_EQ(la::cholesky(m22{{1, 2, 2, 1}}).unwrap_err_unchecked(), la::error::not_positive_definite);    // eigenvalues -1, 3
    EXPECT_EQ(la::cholesky(m22{{1, 1, 1, 1}}).unwrap_err_unchecked(), la::error::not_positive_definite);    // singular
    EXPECT_EQ(la::cholesky(m22::zero()).unwrap_err_unchecked(), la::error::not_positive_definite);
    EXPECT_EQ(la::cholesky(m22{{-1, 0, 0, 1}}).unwrap_err_unchecked(), la::error::not_positive_definite);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(la::cholesky(m22{{1, 0, 0, nan}}).unwrap_err_unchecked(), la::error::not_positive_definite);
    EXPECT_EQ(la::spd_inverse(m22{{1, 2, 2, 1}}).unwrap_err_unchecked(), la::error::not_positive_definite);
}

TEST(LinalgFixedCholesky, SolvesLogDeterminantAndInverse)
{
    const auto l = *la::cholesky(spd);
    const v3 b{{1, -2, 3}};
    const auto y = la::lower_solve(l, b);
    expect_near(l * y, b, 1e-12);
    const auto z = la::lower_transpose_solve(l, b);
    expect_near(la::transpose(l) * z, b, 1e-12);
    const auto x = la::cholesky_solve(l, b);
    expect_near(spd * x, b, 1e-10);
    // Several right-hand sides at once
    const auto xs = la::cholesky_solve(l, m33::identity());
    expect_near(spd * xs, m33::identity(), 1e-10);
    EXPECT_NEAR(la::cholesky_log_det(l), std::log(36.0), 1e-14);
    const auto inv = la::spd_inverse(spd);
    ASSERT_TRUE(inv.is_ok());
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ((*inv)(i, j), (*inv)(j, i));   // exactly symmetric
    expect_near(spd * *inv, m33::identity(), 1e-10);
}

TEST(LinalgFixedLu, SolvesWithPivotingAndInverts)
{
    const m33 a{{2, -1, 0, 1, 3, 2, 0, 1, 4}};
    const v3 b{{1, 2, 3}};
    const auto x = la::solve(a, b);
    ASSERT_TRUE(x.is_ok());
    expect_near(a * *x, b, 1e-14);
    // A zero leading pivot needs a row exchange
    const m22 p{{0, 1, 1, 0}};
    EXPECT_EQ(*la::solve(p, la::make_vector(5.0, 7.0)), la::make_vector(7.0, 5.0));
    const auto inv = la::inverse(a);
    ASSERT_TRUE(inv.is_ok());
    expect_near(a * *inv, m33::identity(), 1e-14);
    expect_near(*inv * a, m33::identity(), 1e-14);
}

TEST(LinalgFixedLu, DeterminantCountsRowExchanges)
{
    EXPECT_EQ(la::determinant(m22{{0, 1, 1, 0}}), -1);
    EXPECT_NEAR(la::determinant(m33{{2, -1, 0, 1, 3, 2, 0, 1, 4}}), 24, 1e-13);    // by cofactors: 2*10 + 1*4
    EXPECT_NEAR(la::determinant(spd), 36, 1e-11);
    EXPECT_EQ(la::determinant(m33::identity()), 1);
    EXPECT_EQ((la::determinant(la::fixed_matrix<double, 1, 1>{{-3}})), -3);
    const m33 cyc{{0, 1, 0, 0, 0, 1, 1, 0, 0}};   // an even permutation
    EXPECT_EQ(la::determinant(cyc), 1);
}

TEST(LinalgFixedLu, SingularIsAnErrorAndAZeroDeterminant)
{
    const m33 s{{1, 2, 3, 4, 5, 6, 7, 8, 9}};    // rank 2
    EXPECT_EQ(la::solve(s, v3{{1, 1, 1}}).unwrap_err_unchecked(), la::error::singular);
    EXPECT_EQ(la::inverse(s).unwrap_err_unchecked(), la::error::singular);
    EXPECT_EQ(la::determinant(s), 0);
    EXPECT_EQ(la::inverse(m22::zero()).unwrap_err_unchecked(), la::error::singular);
    EXPECT_EQ(la::determinant(m22::zero()), 0);
    // Relative tolerance: a well-conditioned matrix scaled by 1e-30 is not singular
    EXPECT_TRUE(la::inverse(m22{{1, 2, 3, 4}} * 1e-30).is_ok());
}

TEST(LinalgFixedEigen, MatchesNumpyAndReconstructs)
{
    const m33 a{{4, 1, 2, 1, 3, 0, 2, 0, 5}};
    const auto e = la::symmetric_eigen(a);
    ASSERT_TRUE(e.is_ok());
    // numpy.linalg.eigh([[4,1,2],[1,3,0],[2,0,5]])
    EXPECT_NEAR(e->values[0], 1.854897308799577, 1e-13);
    EXPECT_NEAR(e->values[1], 3.476023602918134, 1e-13);
    EXPECT_NEAR(e->values[2], 6.669079088282288, 1e-13);
    expect_near(la::transpose(e->vectors) * e->vectors, m33::identity(), 1e-14);
    expect_near(e->vectors * la::diagonal_matrix(e->values) * la::transpose(e->vectors), a, 1e-13);
    for (std::size_t i = 0; i < 3; ++i) {
        const v3 col{{e->vectors(0, i), e->vectors(1, i), e->vectors(2, i)}};
        expect_near(a * col, col * e->values[i], 1e-13);
    }
}

TEST(LinalgFixedEigen, SortsAscendingAndHandlesDiagonalZeroAndRepeated)
{
    const auto d = la::symmetric_eigen(la::diagonal_matrix(la::make_vector(3.0, -1.0, 2.0)));
    ASSERT_TRUE(d.is_ok());
    EXPECT_EQ(d->values, la::make_vector(-1.0, 2.0, 3.0));
    // The eigenvectors moved with their values: column 0 is e_1
    EXPECT_EQ(d->vectors(1, 0), 1);
    const auto z = la::symmetric_eigen(m33::zero());
    ASSERT_TRUE(z.is_ok());
    EXPECT_EQ(z->values, v3{});
    EXPECT_EQ(z->vectors, m33::identity());
    const auto r = la::symmetric_eigen(m22{{2, 1, 1, 2}});   // 1 and 3, vectors (1, -1)/√2, (1, 1)/√2
    EXPECT_NEAR(r->values[0], 1, 1e-15);
    EXPECT_NEAR(r->values[1], 3, 1e-15);
    EXPECT_NEAR(std::abs(r->vectors(0, 0)), std::sqrt(0.5), 1e-15);
    EXPECT_EQ(la::symmetric_eigen(m22{{5, 0, 0, 5}})->values, la::make_vector(5.0, 5.0));
}

TEST(LinalgFixedEigen, ReadsTheUpperTriangle)
{
    const m22 a{{2, 1, -100, 2}};   // the lower element is ignored
    EXPECT_NEAR(la::symmetric_eigen(a)->values[1], 3, 1e-15);
}

TEST(LinalgFixedEigen, NaNDoesNotConverge)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(la::symmetric_eigen(m22{{1, nan, nan, 1}}).unwrap_err_unchecked(), la::error::did_not_converge);
}

TEST(LinalgFixedTria, IsTheCholeskyFactorOfAAt)
{
    // [F S, Q^(1/2)]: 3 x 5
    const la::fixed_matrix<double, 3, 5> a{{1, 2, 0, 0.5, -1, 0, 1, 3, 1, 0, 2, -1, 1, 0, 0.25}};
    const auto s = la::tria(a);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_GE(s(i, i), 0);
        for (std::size_t j = i + 1; j < 3; ++j)
            EXPECT_EQ(s(i, j), 0);
    }
    const auto aat = a * la::transpose(a);
    expect_near(s * la::transpose(s), aat, 1e-13);
    expect_near(s, *la::cholesky(aat), 1e-13);   // unique with a positive diagonal
}

TEST(LinalgFixedTria, SquareZeroAndRankDeficientInputs)
{
    const auto sq = la::tria(m22{{0, 3, 4, 0}});   // A Aᵀ = diag(9, 16)
    expect_near(sq, m22{{3, 0, 0, 4}}, 1e-15);
    EXPECT_EQ(la::tria(la::fixed_matrix<double, 2, 3>{}), m22::zero());
    // Rank 1: rows parallel
    const la::fixed_matrix<double, 2, 3> r1{{1, 2, 2, 2, 4, 4}};
    const auto s = la::tria(r1);
    expect_near(s * la::transpose(s), r1 * la::transpose(r1), 1e-13);
    EXPECT_GE(s(1, 1), 0);
    EXPECT_NEAR(s(1, 1), 0, 1e-7);
}

TEST(LinalgFixedView, SharesTheStorageRowMajor)
{
    la::fixed_matrix<double, 2, 3> a{{1, 2, 3, 4, 5, 6}};
    const auto v = la::view(a);
    EXPECT_EQ(v.extent(0), 2u);
    EXPECT_EQ(v.extent(1), 3u);
    EXPECT_EQ(v(1, 2), 6);
    v(0, 1) = 20;
    EXPECT_EQ(a(0, 1), 20);
    const auto& ca = a;
    const auto cv = la::view(ca);
    static_assert(std::is_same_v<std::remove_const_t<decltype(cv)>, la::matrix_view<const double, la::layout_right>>);
    EXPECT_EQ(cv(1, 0), 4);
    EXPECT_EQ(cv.data_handle(), a.elements.data());
}

TEST(LinalgFixedFloat, FactorisationsWorkInFloat)
{
    using f33 = la::fixed_matrix<float, 3, 3>;
    const f33 a{{4, 12, -16, 12, 37, -43, -16, -43, 98}};
    const auto l = la::cholesky(a);
    ASSERT_TRUE(l.is_ok());
    EXPECT_NEAR((*l)(2, 2), 3.0f, 1e-4f);
    EXPECT_NEAR(la::determinant(a), 36.0f, 1e-2f);
    EXPECT_NEAR(la::symmetric_eigen(la::fixed_matrix<float, 2, 2>{{2, 1, 1, 2}})->values[1], 3.0f, 1e-6f);
}

} // namespace
