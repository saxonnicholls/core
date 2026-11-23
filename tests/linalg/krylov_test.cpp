//
//  krylov_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/linalg/krylov.hpp>: the operator adapters agree with one another; CG and MINRES against a
//  dense solve on SPD and indefinite matrices (MINRES on a KKT matrix too); CG's iteration count on a matrix with k
//  distinct eigenvalues; the Jacobi preconditioner's effect; breakdown, iteration limit, invalid input, b = 0 and a
//  warm start at the solution; the concepts, noexcept and constant evaluation.
//

#include <snicholls/core/linalg/krylov.hpp>

#include <snicholls/core/sparse/csc.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace sp = snicholls::sparse;
namespace st = snicholls::stats;

// A CSC matrix held in vectors, built from a dense row-major one (zeros dropped)
struct csc_owner {
    std::size_t rows = 0, cols = 0;
    std::vector<std::int32_t> col_ptr, row_idx;
    std::vector<double> values;
    sp::csc_view<const double, const std::int32_t> view() const { return {rows, cols, col_ptr, row_idx, values}; }
};

csc_owner from_dense(const std::vector<double>& a, std::size_t rows, std::size_t cols, bool upper_only = false)
{
    csc_owner c{rows, cols, {0}, {}, {}};
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i)
            if (a[i * cols + j] != 0 && (!upper_only || i <= j)) {
                c.row_idx.push_back(static_cast<std::int32_t>(i));
                c.values.push_back(a[i * cols + j]);
            }
        c.col_ptr.push_back(static_cast<std::int32_t>(c.values.size()));
    }
    return c;
}

// Dense solve by Gaussian elimination with partial pivoting
std::vector<double> dense_solve(std::vector<double> a, std::vector<double> b)
{
    const std::size_t n = b.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(a[i * n + k]) > std::abs(a[p * n + k]))
                p = i;
        for (std::size_t j = 0; j < n; ++j)
            std::swap(a[k * n + j], a[p * n + j]);
        std::swap(b[k], b[p]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = a[i * n + k] / a[k * n + k];
            for (std::size_t j = k; j < n; ++j)
                a[i * n + j] -= f * a[k * n + j];
            b[i] -= f * b[k];
        }
    }
    std::vector<double> x(n);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= a[i * n + j] * x[j];
        x[i] = s / a[i * n + i];
    }
    return x;
}

// Q diag(eig) Qᵀ with Q a Householder reflector I - 2 v vᵀ / vᵀv of a random v: symmetric with the given eigenvalues
std::vector<double> with_eigenvalues(const std::vector<double>& eig, std::uint64_t seed)
{
    const std::size_t n = eig.size();
    st::wyrand g(seed);
    std::vector<double> v(n), q(n * n), a(n * n, 0.0);
    double vv = 0;
    for (auto& e : v) {
        e = st::uniform_real(g) - 0.5;
        vv += e * e;
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            q[i * n + j] = (i == j ? 1.0 : 0.0) - 2 * v[i] * v[j] / vv;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                a[i * n + j] += q[i * n + k] * eig[k] * q[j * n + k];
    return a;
}

// A random SPD matrix BᵀB + n I
std::vector<double> random_spd(std::size_t n, std::uint64_t seed)
{
    st::wyrand g(seed);
    std::vector<double> b(n * n), a(n * n, 0.0);
    for (auto& e : b)
        e = st::uniform_real(g) - 0.5;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k)
                a[i * n + j] += b[k * n + i] * b[k * n + j];
            if (i == j)
                a[i * n + j] += static_cast<double>(n) / 4;
        }
    return a;
}

std::vector<double> random_vector(std::size_t n, std::uint64_t seed)
{
    st::wyrand g(seed);
    std::vector<double> v(n);
    for (auto& e : v)
        e = 2 * st::uniform_real(g) - 1;
    return v;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

// The concepts
using csc_op = la::csc_operator<double>;
using sym_op = la::symmetric_csc_operator<double>;
using dense_op = la::dense_operator<double>;
struct throwing_product {
    void operator()(std::span<const double>, std::span<double>) const { throw std::runtime_error("no"); }
};
struct quiet_product {
    void operator()(std::span<const double> x, std::span<double> y) const noexcept { y[0] = x[0]; }
};
using throwing_op = la::function_operator<double, throwing_product>;
using quiet_op = la::function_operator<double, quiet_product>;
static_assert(la::linear_operator<csc_op> && la::linear_operator<sym_op> && la::linear_operator<dense_op> && la::linear_operator<throwing_op>);
static_assert(!la::linear_operator<int> && !la::linear_operator<std::vector<double>>);
static_assert(la::preconditioner<la::identity_preconditioner, double> && la::preconditioner<la::jacobi_preconditioner<double>, double>);
static_assert(!la::preconditioner<int, double>);

// noexcept follows the operator's apply
static_assert(noexcept(la::conjugate_gradient(std::declval<const csc_op&>(), {}, {}, {})));
static_assert(noexcept(la::minres(std::declval<const sym_op&>(), {}, {}, {})));
static_assert(noexcept(la::conjugate_gradient(std::declval<const quiet_op&>(), {}, {}, {}, {}, la::jacobi_preconditioner<double>{})));
static_assert(!noexcept(la::conjugate_gradient(std::declval<const throwing_op&>(), {}, {}, {})));
static_assert(!noexcept(la::minres(std::declval<const throwing_op&>(), {}, {}, {})));
static_assert(la::conjugate_gradient_workspace(10) == 40 && la::minres_workspace(10) == 70);

// Constant evaluation: CG and MINRES on the 3 x 3 tridiagonal [2 -1 0; -1 2 -1; 0 -1 2], b = (1, 0, 1), x = (1, 1, 1)
struct small_solve {
    bool converged;
    double error;
};
template <bool Minres>
constexpr small_solve solve_tridiagonal()
{
    sp::csc_matrix<double, 3, 3, 7> a{{0, 2, 5, 7}, {0, 1, 0, 1, 2, 1, 2}, {2, -1, -1, 2, -1, -1, 2}};
    const auto op = la::make_csc_operator(a.view());
    const std::array<double, 3> b{1, 0, 1};
    std::array<double, 3> x{};
    std::array<double, 21> work{};
    const auto r = Minres ? la::minres(op, b, x, work) : la::conjugate_gradient(op, b, x, work);
    double e = 0;
    for (double v : x)
        e += (v - 1) * (v - 1);
    return {r.converged() && r.iterations <= 3, e};
}
static_assert(solve_tridiagonal<false>().converged && solve_tridiagonal<false>().error < 1e-20);
static_assert(solve_tridiagonal<true>().converged && solve_tridiagonal<true>().error < 1e-20);

TEST(LinalgKrylovOperators, EveryAdapterComputesTheSameProduct)
{
    constexpr std::size_t n = 7;
    const auto a = random_spd(n, 3);
    const auto full = from_dense(a, n, n);
    const auto upper = from_dense(a, n, n, true);
    const auto x = random_vector(n, 4);
    std::vector<double> y1(n), y2(n), y3(n), y4(n);
    la::make_csc_operator(full.view()).apply(x, y1);
    la::make_symmetric_csc_operator(upper.view()).apply(x, y2);
    la::make_dense_operator(std::span<const double>(a), n, n).apply(x, y3);
    const auto f = la::make_function_operator<double>(n, n, [&](std::span<const double> in, std::span<double> out) { la::make_dense_operator(std::span<const double>(a), n, n).apply(in, out); });
    f.apply(x, y4);
    EXPECT_EQ(f.rows(), n);
    EXPECT_EQ(f.cols(), n);
    EXPECT_LT(max_abs_diff(y1, y2), 1e-13);
    EXPECT_LT(max_abs_diff(y1, y3), 1e-13);
    EXPECT_EQ(y3, y4);
}

TEST(LinalgKrylovOperators, CscDiagonalReadsStoredDiagonalEntriesAndZeroElsewhere)
{
    const std::vector<double> a{4, 1, 0, 0, 0, 2, 0, 3, 0};   // 3 x 3, a(1, 1) not stored
    const auto c = from_dense(a, 3, 3);
    std::vector<double> d(3, -1);
    la::csc_diagonal(c.view(), std::span<double>(d));
    EXPECT_EQ(d, (std::vector<double>{4, 0, 0}));
}

TEST(LinalgKrylovCg, MatchesADenseSolveOnRandomSpdMatrices)
{
    for (std::size_t n : {1u, 2u, 5u, 20u, 60u}) {
        const auto a = random_spd(n, 10 + n);
        const auto b = random_vector(n, 20 + n);
        const auto c = from_dense(a, n, n);
        std::vector<double> x(n, 0.0), work(la::conjugate_gradient_workspace(n));
        const auto r = la::conjugate_gradient(la::make_csc_operator(c.view()), b, x, work, {1e-12});
        EXPECT_TRUE(r.converged()) << n;
        EXPECT_LE(r.iterations, n + 2) << n;
        EXPECT_LT(r.residual_norm, 1e-11 * std::sqrt(static_cast<double>(n)) * 2) << n;
        EXPECT_LT(max_abs_diff(x, dense_solve(a, b)), 1e-9) << n;
    }
}

TEST(LinalgKrylovCg, ConvergesInAtMostKIterationsForKDistinctEigenvalues)
{
    for (std::size_t k = 1; k <= 5; ++k) {
        std::vector<double> eig;
        for (std::size_t i = 0; i < 24; ++i)
            eig.push_back(static_cast<double>((i % k + 1) * (i % k + 1)));
        const auto a = with_eigenvalues(eig, 100 + k);
        const auto b = random_vector(24, 200 + k);
        std::vector<double> x(24, 0.0), work(la::conjugate_gradient_workspace(24)), wm(la::minres_workspace(24));
        const auto op = la::make_dense_operator(std::span<const double>(a), 24, 24);
        const auto r = la::conjugate_gradient(op, b, x, work, {1e-10});
        EXPECT_TRUE(r.converged()) << k;
        EXPECT_LE(r.iterations, k) << k;
        EXPECT_LT(max_abs_diff(x, dense_solve(a, b)), 1e-8) << k;
        std::vector<double> xm(24, 0.0);
        const auto rm = la::minres(op, b, xm, wm, {1e-10});
        EXPECT_TRUE(rm.converged()) << k;
        EXPECT_LE(rm.iterations, k) << k;
    }
}

TEST(LinalgKrylovCg, JacobiPreconditioningCutsTheIterationsOnABadlyScaledMatrix)
{
    constexpr std::size_t n = 40;
    // A = D S D: S well conditioned, D spanning 1 to 1e3
    const auto s = random_spd(n, 7);
    std::vector<double> d(n), a(n * n);
    for (std::size_t i = 0; i < n; ++i)
        d[i] = std::pow(10.0, 3.0 * static_cast<double>(i) / (n - 1));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a[i * n + j] = d[i] * s[i * n + j] * d[j];
    const auto c = from_dense(a, n, n);
    const auto b = random_vector(n, 8);
    std::vector<double> diag(n), work(la::conjugate_gradient_workspace(n)), x0(n, 0.0), x1(n, 0.0);
    la::csc_diagonal(c.view(), std::span<double>(diag));
    const la::krylov_options<double> o{1e-10, 0, 2000};
    const auto plain = la::conjugate_gradient(la::make_csc_operator(c.view()), b, x0, work, o);
    const auto jac = la::conjugate_gradient(la::make_csc_operator(c.view()), b, x1, work, o, la::jacobi_preconditioner<double>{diag});
    EXPECT_TRUE(jac.converged());
    EXPECT_LT(jac.iterations * 3, plain.iterations) << "plain " << plain.iterations << ", jacobi " << jac.iterations;
    EXPECT_LT(max_abs_diff(x1, dense_solve(a, b)), 1e-9);
}

TEST(LinalgKrylovCg, ReportsBreakdownOnAnIndefiniteMatrix)
{
    // diag(1, -1) and b = (1, 1): the first direction has pᵀAp = 0
    const std::vector<double> a{1, 0, 0, -1}, b{1, 1};
    std::vector<double> x(2, 0.0), work(8);
    const auto r = la::conjugate_gradient(la::make_dense_operator(std::span<const double>(a), 2, 2), b, x, work);
    EXPECT_EQ(r.status, la::krylov_status::breakdown);
    EXPECT_FALSE(r.converged());
    EXPECT_EQ(r.iterations, 0u);
    // ... and on a preconditioner that is not positive definite
    const std::vector<double> spd{2, 0, 0, 3}, neg{1, -0.5};
    const auto r2 = la::conjugate_gradient(la::make_dense_operator(std::span<const double>(spd), 2, 2), std::span<const double>(b), x, work, {}, la::jacobi_preconditioner<double>{neg});
    EXPECT_EQ(r2.status, la::krylov_status::breakdown);
    const auto r3 = la::minres(la::make_dense_operator(std::span<const double>(spd), 2, 2), std::span<const double>(b), x, std::span<double>(std::vector<double>(14).data(), 14), {},
                               la::jacobi_preconditioner<double>{neg});
    EXPECT_EQ(r3.status, la::krylov_status::breakdown);
}

TEST(LinalgKrylovCg, StopsAtTheIterationLimitWithTheLastIterate)
{
    constexpr std::size_t n = 30;
    const auto a = random_spd(n, 5);
    const auto b = random_vector(n, 6);
    std::vector<double> x(n, 0.0), work(4 * n);
    const auto r = la::conjugate_gradient(la::make_dense_operator(std::span<const double>(a), n, n), b, x, work, {1e-14, 0, 3});
    EXPECT_EQ(r.status, la::krylov_status::iteration_limit);
    EXPECT_EQ(r.iterations, 3u);
    EXPECT_GT(r.residual_norm, 0.0);
    std::vector<double> wm(7 * n), xm(n, 0.0);
    const auto rm = la::minres(la::make_dense_operator(std::span<const double>(a), n, n), b, xm, wm, {1e-14, 0, 3});
    EXPECT_EQ(rm.status, la::krylov_status::iteration_limit);
}

TEST(LinalgKrylovCg, RejectsMismatchedShapesAndShortWorkspaceLeavingXUntouched)
{
    const std::vector<double> a{2, 0, 0, 2}, b{1, 1}, b3{1, 1, 1};
    std::vector<double> x{7, 7}, work(7), wm(13);
    const auto op = la::make_dense_operator(std::span<const double>(a), 2, 2);
    EXPECT_EQ(la::conjugate_gradient(op, b, x, work).status, la::krylov_status::invalid_input);
    EXPECT_EQ(la::minres(op, b, x, wm).status, la::krylov_status::invalid_input);
    std::vector<double> w8(8), w14(14);
    EXPECT_EQ(la::conjugate_gradient(op, b3, x, w8).status, la::krylov_status::invalid_input);
    EXPECT_EQ(la::minres(op, b3, x, w14).status, la::krylov_status::invalid_input);
    EXPECT_EQ(x, (std::vector<double>{7, 7}));
}

TEST(LinalgKrylovCg, AZeroRightHandSideGivesZeroAndAWarmStartAtTheSolutionNoIterations)
{
    const std::vector<double> a{4, 1, 1, 3}, zero{0, 0}, b{1, 2};
    const auto op = la::make_dense_operator(std::span<const double>(a), 2, 2);
    std::vector<double> x{5, 5}, work(8), wm(14);
    auto r = la::conjugate_gradient(op, zero, x, work);
    EXPECT_TRUE(r.converged());
    EXPECT_EQ(r.iterations, 0u);
    EXPECT_EQ(x, (std::vector<double>{0, 0}));
    x = {5, 5};
    r = la::minres(op, zero, x, wm);
    EXPECT_TRUE(r.converged());
    EXPECT_EQ(x, (std::vector<double>{0, 0}));
    x = dense_solve(a, b);
    r = la::conjugate_gradient(op, b, x, work, {1e-12});
    EXPECT_TRUE(r.converged());
    EXPECT_EQ(r.iterations, 0u);
    r = la::minres(op, b, x, wm, {1e-12});
    EXPECT_TRUE(r.converged());
    EXPECT_EQ(r.iterations, 0u);
    // the absolute tolerance alone
    x = {0, 0};
    r = la::conjugate_gradient(op, b, x, work, {0, 10});
    EXPECT_TRUE(r.converged());
    EXPECT_EQ(r.iterations, 0u);
}

TEST(LinalgKrylovMinres, MatchesADenseSolveOnSymmetricIndefiniteMatrices)
{
    for (std::size_t n : {2u, 6u, 25u, 50u}) {
        std::vector<double> eig(n);
        for (std::size_t i = 0; i < n; ++i)
            eig[i] = (i % 2 == 0 ? -1.0 : 1.0) * (1.0 + static_cast<double>(i));
        const auto a = with_eigenvalues(eig, 30 + n);
        const auto b = random_vector(n, 40 + n);
        const auto upper = from_dense(a, n, n, true);
        std::vector<double> x(n, 0.0), work(la::minres_workspace(n));
        const auto r = la::minres(la::make_symmetric_csc_operator(upper.view()), b, x, work, {1e-12});
        EXPECT_TRUE(r.converged()) << n;
        EXPECT_LT(max_abs_diff(x, dense_solve(a, b)), 1e-9) << n;
    }
}

TEST(LinalgKrylovMinres, SolvesAQuadraticProgramsKktSystem)
{
    // [P Aᵀ; A 0] with P = SPD 6 x 6, A 3 x 6: symmetric, indefinite (3 negative eigenvalues)
    constexpr std::size_t n = 6, m = 3, nk = n + m;
    const auto p = random_spd(n, 50);
    const auto am = random_vector(m * n, 51);
    std::vector<double> k(nk * nk, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            k[i * nk + j] = p[i * n + j];
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            k[(n + i) * nk + j] = k[j * nk + n + i] = am[i * n + j];
    const auto upper = from_dense(k, nk, nk, true);
    const auto b = random_vector(nk, 52);
    std::vector<double> x(nk, 0.0), work(la::minres_workspace(nk)), xj(nk, 0.0), diag(nk);
    const auto r = la::minres(la::make_symmetric_csc_operator(upper.view()), b, x, work, {1e-12});
    EXPECT_TRUE(r.converged());
    EXPECT_LE(r.iterations, nk + 2);
    EXPECT_LT(max_abs_diff(x, dense_solve(k, b)), 1e-9);
    // a Jacobi preconditioner of |diag| (SPD, with 1 for the zero block) also solves it
    la::csc_diagonal(upper.view(), std::span<double>(diag));
    for (auto& d : diag)
        d = d == 0 ? 1 : std::abs(d);
    const auto rj = la::minres(la::make_symmetric_csc_operator(upper.view()), b, xj, work, {1e-12}, la::jacobi_preconditioner<double>{diag});
    EXPECT_TRUE(rj.converged());
    EXPECT_LT(max_abs_diff(xj, dense_solve(k, b)), 1e-9);
}

TEST(LinalgKrylovMinres, MatchesCgOnAnSpdMatrixAndWarmStartsCount)
{
    constexpr std::size_t n = 30;
    const auto a = random_spd(n, 60);
    const auto b = random_vector(n, 61);
    const auto op = la::make_dense_operator(std::span<const double>(a), n, n);
    std::vector<double> xc(n, 0.0), xm(n, 0.0), wc(4 * n), wm(7 * n);
    const auto rc = la::conjugate_gradient(op, b, xc, wc, {1e-12});
    const auto rm = la::minres(op, b, xm, wm, {1e-12});
    EXPECT_TRUE(rc.converged() && rm.converged());
    EXPECT_LT(max_abs_diff(xc, xm), 1e-9);
    // a warm start near the solution needs fewer iterations
    std::vector<double> xw = dense_solve(a, b);
    for (auto& e : xw)
        e *= 1 + 1e-6;
    const auto rw = la::minres(op, b, xw, wm, {1e-12});
    EXPECT_TRUE(rw.converged());
    EXPECT_LT(rw.iterations, rm.iterations);
}

TEST(LinalgKrylovCg, WorksInFloat)
{
    const std::vector<float> a{4, 1, 0, 1, 3, 1, 0, 1, 2}, b{1, 2, 3};
    std::vector<float> x(3, 0.0f), work(12);
    const auto r = la::conjugate_gradient(la::make_dense_operator(std::span<const float>(a), 3, 3), b, x, work, {1e-5f});
    EXPECT_TRUE(r.converged());
    std::vector<float> y(3);
    la::make_dense_operator(std::span<const float>(a), 3, 3).apply(x, y);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(y[i], b[i], 1e-4f);
}

} // namespace
