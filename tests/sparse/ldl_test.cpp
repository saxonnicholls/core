//
//  ldl_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/sparse/ldl.hpp>: etree and ldl_symbolic against a dense symbolic elimination; ldl_numeric
//  reproducing A = L D Lᵀ and solving like a dense LU (the reference backend, and linalg/fixed.hpp's solve) on random
//  SPD and random quasi-definite KKT matrices in natural and AMD order; refinement's residual; refactorisation with
//  new values on the same analysis; static regularisation by signs and its counts; the zero-pivot and non-finite
//  errors; every capacity error; 64-bit indices; and the ordering, the symbolic and the numeric factorisation of a
//  small KKT system in a static_assert, with ldl_fixed sized by ldl_nnz at compile time.
//

#include <snicholls/core/sparse/ldl.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/sparse/csc.hpp>
#include <snicholls/core/sparse/ordering.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace sp = snicholls::sparse;
namespace st = snicholls::stats;
namespace la = snicholls::linalg;
using I = std::int32_t;

struct matrix {
    std::size_t n = 0;
    std::vector<I> cp, ri;
    std::vector<double> vx;
    [[nodiscard]] sp::csc_view<const double, const I> view() const { return {n, n, cp, ri, vx}; }
    // The full symmetric matrix, row-major
    [[nodiscard]] std::vector<double> dense() const
    {
        std::vector<double> d(n * n, 0.0);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t q = static_cast<std::size_t>(cp[j]); q < static_cast<std::size_t>(cp[j + 1]); ++q) {
                const std::size_t i = static_cast<std::size_t>(ri[q]);
                if (i <= j)
                    d[i * n + j] = d[j * n + i] = vx[q];
            }
        return d;
    }
};

struct triplets {
    std::size_t n = 0;
    std::vector<I> i, j;
    std::vector<double> v;
    void add(std::size_t r, std::size_t c, double x)
    {
        if (r > c)
            std::swap(r, c);
        i.push_back(static_cast<I>(r));
        j.push_back(static_cast<I>(c));
        v.push_back(x);
    }
    [[nodiscard]] matrix assemble() const
    {
        matrix m;
        m.n = n;
        m.cp.resize(n + 1);
        m.ri.resize(i.size());
        m.vx.resize(i.size());
        std::vector<I> work(n);
        sp::csc_view<double, I> out{0, 0, m.cp, m.ri, m.vx};
        EXPECT_TRUE(sp::from_triplets(n, n, std::span<const I>(i), std::span<const I>(j), std::span<const double>(v), out, std::span<I>(work)).is_ok());
        m.ri.resize(out.nnz());
        m.vx.resize(out.nnz());
        return m;
    }
};

triplets random_spd(std::size_t n, std::size_t per_column, std::uint64_t seed)
{
    st::wyrand g(seed);
    triplets t;
    t.n = n;
    std::vector<double> diag(n, 0.5);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t e = 0; e < per_column; ++e) {
            const std::size_t i = static_cast<std::size_t>(st::uniform_index(g, n));
            if (i == j)
                continue;
            const double x = st::uniform_real(g) - 0.5;
            t.add(i, j, x);
            diag[i] += std::abs(x);
            diag[j] += std::abs(x);
        }
    for (std::size_t j = 0; j < n; ++j)
        t.add(j, j, diag[j]);
    return t;
}

// [P + σI  Aᵀ; A  -ρ⁻¹I], P positive semidefinite (a sum of rank-one terms, so possibly singular), A m x nx
triplets random_kkt(std::size_t nx, std::size_t m, std::uint64_t seed, double sigma, double rho)
{
    st::wyrand g(seed);
    triplets t;
    t.n = nx + m;
    for (std::size_t k = 0; k < nx / 2; ++k) {
        // P += v vᵀ for a v with two entries
        const std::size_t a = static_cast<std::size_t>(st::uniform_index(g, nx)), b = static_cast<std::size_t>(st::uniform_index(g, nx));
        const double va = st::uniform_real(g) - 0.5, vb = st::uniform_real(g) - 0.5;
        t.add(a, a, va * va);
        t.add(b, b, vb * vb);
        if (a != b)
            t.add(a, b, va * vb);
        else
            t.add(a, a, 2 * va * vb);
    }
    for (std::size_t i = 0; i < nx; ++i)
        t.add(i, i, sigma);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t e = 0; e < 3; ++e)
            t.add(static_cast<std::size_t>(st::uniform_index(g, nx)), nx + r, st::uniform_real(g) - 0.5);
        t.add(nx + r, nx + r, -1.0 / rho);
    }
    return t;
}

std::vector<double> random_vector(std::size_t n, std::uint64_t seed)
{
    st::wyrand g(seed);
    std::vector<double> x(n);
    for (double& v : x)
        v = st::uniform_real(g) * 2 - 1;
    return x;
}

// The dense solution by LU with partial pivoting (the reference backend)
std::vector<double> dense_solve(const matrix& a, std::vector<double> b)
{
    auto d = a.dense();
    std::vector<la::pivot_index> piv(a.n);
    const la::matrix_view<double, la::layout_right> m(d.data(), {a.n, a.n});
    EXPECT_TRUE(la::lu_factor(la::reference, m, std::span<la::pivot_index>(piv)).is_ok());
    EXPECT_TRUE(la::lu_solve(la::reference, m, std::span<const la::pivot_index>(piv), la::as_vector(std::span<double>(b))).is_ok());
    return b;
}

// Everything a factorisation needs, owned
struct factor {
    std::size_t n = 0;
    std::vector<I> parent, counts, lp, li, iwork;
    std::vector<double> lx, d, dinv, xwork;
    sp::ldl_symbolic_result<I> sym{};
    sp::csc_view<double, I> l{};

    explicit factor(const matrix& a) : n(a.n), parent(n), counts(n), lp(n + 1), iwork(3 * n), d(n), dinv(n), xwork(n)
    {
        const auto s = sp::ldl_symbolic(a.view(), std::span<I>(parent), std::span<I>(counts), std::span<I>(lp), std::span<I>(iwork).first(n));
        EXPECT_TRUE(s.is_ok());
        sym = *s;
        li.resize(sym.nnz);
        lx.resize(sym.nnz);
        l = {n, n, lp, li, lx};
    }
    snicholls::result<std::size_t, sp::ldl_error> numeric(const matrix& a, std::span<const std::int8_t> signs = {}, double delta = 1e-8)
    {
        return sp::ldl_numeric(a.view(), sym, l, d, dinv, std::span<I>(iwork), xwork, signs, delta);
    }
    std::vector<double> solve(std::vector<double> b) const
    {
        sp::ldl_solve(l.as_const(), std::span<const double>(dinv), std::span<double>(b));
        return b;
    }
};

matrix permuted(const matrix& a, const std::vector<I>& perm)
{
    matrix c;
    c.n = a.n;
    c.cp.resize(a.n + 1);
    c.ri.resize(a.vx.size());
    c.vx.resize(a.vx.size());
    std::vector<I> work(2 * a.n);
    sp::csc_view<double, I> v{0, 0, c.cp, c.ri, c.vx};
    EXPECT_TRUE(sp::symmetric_permute_upper(a.view(), std::span<const I>(perm), v, std::span<I>(work)).is_ok());
    return c;
}

std::vector<I> amd_of(const matrix& a)
{
    std::vector<I> perm(a.n), work(sp::amd_workspace_size(a.n, a.vx.size()));
    EXPECT_TRUE(sp::amd(a.view(), std::span<I>(perm), std::span<I>(work)).is_ok());
    return perm;
}

double max_abs_diff(const std::vector<double>& x, const std::vector<double>& y)
{
    double m = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::abs(x[i] - y[i]));
    return m;
}

// The pattern of L by dense symbolic elimination: column counts and parents (the first entry below the diagonal)
void dense_symbolic(const matrix& a, std::vector<std::size_t>& counts, std::vector<I>& parent)
{
    const std::size_t n = a.n;
    std::vector<char> nz(n * n, 0);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t q = static_cast<std::size_t>(a.cp[j]); q < static_cast<std::size_t>(a.cp[j + 1]); ++q) {
            const std::size_t i = static_cast<std::size_t>(a.ri[q]);
            nz[i * n + j] = nz[j * n + i] = 1;
        }
    counts.assign(n, 0);
    parent.assign(n, -1);
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = k + 1; i < n; ++i)
            if (nz[i * n + k]) {
                ++counts[k];
                if (parent[k] == -1)
                    parent[k] = static_cast<I>(i);
                for (std::size_t j = k + 1; j < n; ++j)
                    if (nz[j * n + k])
                        nz[i * n + j] = 1;
            }
    }
}

TEST(SparseLdlEtree, IsTheTreeOfTheFirstSubdiagonalEntryOfL)
{
    for (std::uint64_t seed = 1; seed <= 25; ++seed) {
        const matrix a = random_spd(4 + seed * 3, 1 + seed % 3, seed).assemble();
        std::vector<std::size_t> counts;
        std::vector<I> expect;
        dense_symbolic(a, counts, expect);
        std::vector<I> parent(a.n), work(a.n);
        ASSERT_TRUE(sp::etree(a.view(), std::span<I>(parent), std::span<I>(work)).is_ok());
        EXPECT_EQ(parent, expect) << seed;
    }
    // A path 0 - 1 - 2 and an isolated 3: the tree is the path, 3 a root
    triplets t;
    t.n = 4;
    for (std::size_t i = 0; i < 4; ++i)
        t.add(i, i, 2);
    t.add(0, 1, 1);
    t.add(1, 2, 1);
    std::vector<I> parent(4), work(4);
    ASSERT_TRUE(sp::etree(t.assemble().view(), std::span<I>(parent), std::span<I>(work)).is_ok());
    EXPECT_EQ(parent, (std::vector<I>{1, 2, -1, -1}));
}

TEST(SparseLdlSymbolic, CountsMatchADenseSymbolicElimination)
{
    for (std::uint64_t seed = 1; seed <= 25; ++seed) {
        const matrix a = random_spd(5 + seed * 4, 1 + seed % 4, seed).assemble();
        std::vector<std::size_t> expect_counts;
        std::vector<I> expect_parent;
        dense_symbolic(a, expect_counts, expect_parent);
        const factor f(a);
        std::size_t total = 0;
        for (std::size_t k = 0; k < a.n; ++k) {
            EXPECT_EQ(static_cast<std::size_t>(f.counts[k]), expect_counts[k]) << seed;
            EXPECT_EQ(static_cast<std::size_t>(f.lp[k]), total);
            total += expect_counts[k];
        }
        EXPECT_EQ(f.sym.nnz, total);
        EXPECT_EQ(f.parent, expect_parent);
        EXPECT_EQ(f.sym.n, a.n);
        EXPECT_EQ(f.sym.col_ptr.data(), f.lp.data());
    }
}

TEST(SparseLdlSymbolic, ReportsErrors)
{
    const matrix a = random_spd(6, 2, 1).assemble();
    std::vector<I> p(6), c(6), lp(7), w(6);
    sp::csc_view<const double, const I> rect{6, 5, std::span<const I>(a.cp).first(6), a.ri, a.vx};
    EXPECT_EQ(sp::ldl_symbolic(rect, std::span<I>(p), std::span<I>(c), std::span<I>(lp), std::span<I>(w)).unwrap_err(), sp::sparse_error::dimension_mismatch);
    EXPECT_EQ(sp::etree(rect, std::span<I>(p), std::span<I>(w)).unwrap_err(), sp::sparse_error::dimension_mismatch);
    std::vector<I> bad = a.ri;
    bad[0] = 99;
    sp::csc_view<const double, const I> invalid{6, 6, a.cp, bad, a.vx};
    EXPECT_EQ(sp::ldl_symbolic(invalid, std::span<I>(p), std::span<I>(c), std::span<I>(lp), std::span<I>(w)).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(sp::etree(invalid, std::span<I>(p), std::span<I>(w)).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(sp::ldl_symbolic(a.view(), std::span<I>(p), std::span<I>(c), std::span<I>(lp).first(6), std::span<I>(w)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(sp::ldl_symbolic(a.view(), std::span<I>(p).first(5), std::span<I>(c), std::span<I>(lp), std::span<I>(w)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(sp::ldl_symbolic(a.view(), std::span<I>(p), std::span<I>(c), std::span<I>(lp), std::span<I>(w).first(5)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(sp::etree(a.view(), std::span<I>(p), std::span<I>(w).first(5)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    // nnz(L) that does not fit the index type: an arrow, head first, of order 300 has 44,850 entries in L
    triplets arrow;
    arrow.n = 300;
    for (std::size_t i = 0; i < 300; ++i) {
        arrow.add(i, i, 400);
        if (i > 0)
            arrow.add(0, i, 1);
    }
    const matrix am = arrow.assemble();
    std::vector<std::int16_t> cp16(am.cp.begin(), am.cp.end()), ri16(am.ri.begin(), am.ri.end()), p16(300), c16(300), lp16(301), w16(300);
    const sp::csc_view<const double, const std::int16_t> a16{300, 300, cp16, ri16, am.vx};
    EXPECT_EQ(sp::ldl_symbolic(a16, std::span<std::int16_t>(p16), std::span<std::int16_t>(c16), std::span<std::int16_t>(lp16), std::span<std::int16_t>(w16)).unwrap_err(),
              sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(factor(am).sym.nnz, 300u * 299u / 2u);   // with 32-bit indices it fits
}

TEST(SparseLdlNumeric, ReproducesTheMatrixAndSolvesRandomSpdSystems)
{
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        const matrix a = random_spd(8 + seed * 5, 1 + seed % 4, seed).assemble();
        for (const bool ordered : {false, true}) {
            const matrix b = ordered ? permuted(a, amd_of(a)) : a;
            factor f(b);
            const auto r = f.numeric(b);
            ASSERT_TRUE(r.is_ok());
            EXPECT_EQ(*r, 0u);
            // L D Lᵀ = A, entry by entry
            const std::size_t n = b.n;
            std::vector<double> l(n * n, 0.0);
            for (std::size_t j = 0; j < n; ++j) {
                l[j * n + j] = 1;
                for (std::size_t p = static_cast<std::size_t>(f.lp[j]); p < static_cast<std::size_t>(f.lp[j + 1]); ++p) {
                    EXPECT_GT(static_cast<std::size_t>(f.li[p]), j);
                    if (p > static_cast<std::size_t>(f.lp[j])) {
                        EXPECT_LT(f.li[p - 1], f.li[p]);
                    }
                    l[static_cast<std::size_t>(f.li[p]) * n + j] = f.lx[p];
                }
            }
            const auto dense = b.dense();
            double worst = 0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    double s = 0;
                    for (std::size_t k = 0; k < n; ++k)
                        s += l[i * n + k] * f.d[k] * l[j * n + k];
                    worst = std::max(worst, std::abs(s - dense[i * n + j]));
                }
            EXPECT_LT(worst, 1e-12) << seed;
            for (std::size_t k = 0; k < n; ++k) {
                EXPECT_GT(f.d[k], 0.0);
                EXPECT_DOUBLE_EQ(f.dinv[k], 1.0 / f.d[k]);
            }
            const auto rhs = random_vector(n, seed + 1000);
            EXPECT_LT(max_abs_diff(f.solve(rhs), dense_solve(b, rhs)), 1e-11) << seed;
        }
    }
}

TEST(SparseLdlNumeric, SolvesRandomQuasiDefiniteKktSystemsInAnyOrder)
{
    for (std::uint64_t seed = 1; seed <= 12; ++seed) {
        const std::size_t nx = 10 + seed * 4, m = 4 + seed * 2;
        const matrix kkt = random_kkt(nx, m, seed, 1e-3, 10.0).assemble();
        const auto perm = amd_of(kkt);
        std::vector<I> natural(kkt.n);
        for (std::size_t i = 0; i < kkt.n; ++i)
            natural[i] = static_cast<I>(i);
        for (const auto& p : {natural, perm}) {
            const matrix b = permuted(kkt, p);
            // The expected signs, permuted: + for the first nx rows, - for the rest
            std::vector<std::int8_t> signs(kkt.n);
            for (std::size_t k = 0; k < kkt.n; ++k)
                signs[k] = static_cast<std::size_t>(p[k]) < nx ? 1 : -1;
            factor f(b);
            const auto r = f.numeric(b, signs);
            ASSERT_TRUE(r.is_ok());
            EXPECT_EQ(*r, 0u) << seed;   // σ and ρ keep it well away from the threshold
            for (std::size_t k = 0; k < kkt.n; ++k)
                EXPECT_EQ(f.d[k] > 0, signs[k] > 0) << seed;
            const auto rhs = random_vector(kkt.n, seed);
            const auto x = f.solve(rhs), expect = dense_solve(b, rhs);
            EXPECT_LT(max_abs_diff(x, expect), 1e-9 * (1 + max_abs_diff(expect, std::vector<double>(kkt.n, 0.0)))) << seed;
            // Without signs the same factorisation
            factor g(b);
            ASSERT_TRUE(g.numeric(b).is_ok());
            EXPECT_EQ(g.lx, f.lx);
            EXPECT_EQ(g.d, f.d);
        }
    }
}

TEST(SparseLdlNumeric, CheckedAgainstTheFixedSizeDenseSolve)
{
    // A 6 x 6 KKT system: P 4 x 4, A 2 x 4; compared with linalg/fixed.hpp's LU solve
    const matrix kkt = random_kkt(4, 2, 3, 0.5, 2.0).assemble();
    const auto dense = kkt.dense();
    la::fixed_matrix<double, 6, 6> fa{};
    for (std::size_t k = 0; k < 36; ++k)
        fa.elements[k] = dense[k];
    const la::fixed_vector<double, 6> fb{{1, -2, 3, -4, 5, -6}};
    const auto fx = la::solve(fa, fb);
    ASSERT_TRUE(fx.is_ok());
    factor f(kkt);
    ASSERT_TRUE(f.numeric(kkt).is_ok());
    const auto x = f.solve({1, -2, 3, -4, 5, -6});
    for (std::size_t i = 0; i < 6; ++i)
        EXPECT_NEAR(x[i], (*fx)[i], 1e-12);
}

TEST(SparseLdlRefine, DrivesTheResidualBelowTheTolerance)
{
    for (std::uint64_t seed = 1; seed <= 8; ++seed) {
        // σ = 0 and P zero on half the variables (zero pivots), ρ = 1e12 (a -1e-12 block): only static regularisation
        // lets it factor, and refinement against the matrix itself takes the perturbation out of the solution. Each
        // constraint row holds one of P's null variables, so A has full row rank and P is definite on A's null space
        const std::size_t nx = 30, m = 15;
        st::wyrand g(seed);
        triplets t;
        t.n = nx + m;
        for (std::size_t i = 0; i < nx; ++i)
            t.add(i, i, i < m ? 1 + st::uniform_real(g) : 0.0);
        for (std::size_t r = 0; r < m; ++r) {
            t.add(m + r, nx + r, 1);
            t.add(static_cast<std::size_t>(st::uniform_index(g, m)), nx + r, st::uniform_real(g) - 0.5);
            t.add(static_cast<std::size_t>(st::uniform_index(g, m)), nx + r, st::uniform_real(g) - 0.5);
            t.add(nx + r, nx + r, -1e-12);
        }
        const matrix kkt = t.assemble();
        for (const bool ordered : {false, true}) {
            std::vector<I> perm(kkt.n);
            for (std::size_t i = 0; i < kkt.n; ++i)
                perm[i] = static_cast<I>(i);
            if (ordered)
                perm = amd_of(kkt);
            const matrix b = permuted(kkt, perm);
            std::vector<std::int8_t> signs(b.n);
            for (std::size_t k = 0; k < b.n; ++k)
                signs[k] = static_cast<std::size_t>(perm[k]) < nx ? 1 : -1;
            factor f(b);
            const auto r = f.numeric(b, signs, 1e-8);
            ASSERT_TRUE(r.is_ok());
            EXPECT_GT(*r, 0u) << seed;
            const auto rhs = random_vector(b.n, seed + 5);
            auto x = f.solve(rhs);
            std::vector<double> res(b.n);
            const auto none = sp::ldl_refine(b.view(), f.l.as_const(), std::span<const double>(f.dinv), std::span<const double>(rhs), std::span<double>(x),
                                             std::span<double>(res), 0, 0.0);
            EXPECT_EQ(none.iterations, 0u);
            const auto refined = sp::ldl_refine(b.view(), f.l.as_const(), std::span<const double>(f.dinv), std::span<const double>(rhs), std::span<double>(x),
                                                std::span<double>(res), 30, 1e-10);
            EXPECT_LE(refined.residual, 1e-10) << seed;
            EXPECT_LT(refined.residual, none.residual) << seed;
            EXPECT_LE(refined.iterations, 30u);
            // The reported residual is the true one
            std::vector<double> ax(b.n);
            sp::spmv_symmetric_upper(b.view(), std::span<const double>(x), std::span<double>(ax));
            double worst = 0;
            for (std::size_t i = 0; i < b.n; ++i)
                worst = std::max(worst, std::abs(rhs[i] - ax[i]));
            EXPECT_DOUBLE_EQ(worst, refined.residual);
            // Already converged: no corrections
            const auto again = sp::ldl_refine(b.view(), f.l.as_const(), std::span<const double>(f.dinv), std::span<const double>(rhs), std::span<double>(x),
                                              std::span<double>(res), 30, 1e-10);
            EXPECT_EQ(again.iterations, 0u);
            std::printf("  seed %llu, %s order: %zu regularised, residual %.1e, after %zu corrections %.1e\n", static_cast<unsigned long long>(seed),
                        ordered ? "AMD" : "natural", *r, none.residual, refined.iterations, refined.residual);
        }
    }
}

TEST(SparseLdlRefactor, NewValuesOnTheSamePatternReuseTheAnalysis)
{
    const std::size_t nx = 40, m = 15;
    const matrix kkt = random_kkt(nx, m, 7, 1e-4, 1.0).assemble();
    const auto perm = amd_of(kkt);
    matrix b = permuted(kkt, perm);
    std::vector<std::int8_t> signs(b.n);
    for (std::size_t k = 0; k < b.n; ++k)
        signs[k] = static_cast<std::size_t>(perm[k]) < nx ? 1 : -1;
    factor f(b);
    ASSERT_TRUE(f.numeric(b, signs).is_ok());
    const auto lp_before = f.lp;
    const auto rhs = random_vector(b.n, 3);
    for (const double rho : {0.1, 10.0, 1000.0}) {
        // New ρ: every negative diagonal entry becomes -1/ρ, the pattern is unchanged
        for (std::size_t j = 0; j < b.n; ++j)
            for (std::size_t q = static_cast<std::size_t>(b.cp[j]); q < static_cast<std::size_t>(b.cp[j + 1]); ++q)
                if (static_cast<std::size_t>(b.ri[q]) == j && signs[j] < 0)
                    b.vx[q] = -1.0 / rho;
        const auto r = sp::ldl_refactor(b.view(), f.sym, f.l, std::span<double>(f.d), std::span<double>(f.dinv), std::span<I>(f.iwork), std::span<double>(f.xwork),
                                        std::span<const std::int8_t>(signs));
        ASSERT_TRUE(r.is_ok());
        EXPECT_EQ(*r, 0u);
        EXPECT_EQ(f.lp, lp_before);
        EXPECT_LT(max_abs_diff(f.solve(rhs), dense_solve(b, rhs)), 1e-8) << rho;
        // The same as a fresh analysis and factorisation
        factor g(b);
        ASSERT_TRUE(g.numeric(b, signs).is_ok());
        EXPECT_EQ(g.lx, f.lx);
        EXPECT_EQ(g.d, f.d);
    }
}

TEST(SparseLdlNumeric, RegularisesPivotsOfTheWrongSignOrTooSmall)
{
    // diag(2, -3, 1e-12, 0, -1e-12): expected signs + + + - -
    triplets t;
    t.n = 5;
    const double v[5] = {2, -3, 1e-12, 0, -1e-12};
    for (std::size_t i = 0; i < 5; ++i)
        t.add(i, i, v[i]);
    const matrix a = t.assemble();
    const std::vector<std::int8_t> signs{1, 1, 1, -1, -1};
    factor f(a);
    const auto r = f.numeric(a, signs, 1e-6);
    ASSERT_TRUE(r.is_ok());
    EXPECT_EQ(*r, 4u);   // all but the first
    EXPECT_EQ(f.d, (std::vector<double>{2, 1e-6, 1e-6, -1e-6, -1e-6}));
    // A zero sign means no expectation: that pivot is left alone
    const std::vector<std::int8_t> partial{0, -1, 0, 0, 0};
    factor g(a);
    EXPECT_EQ(g.numeric(a, partial, 1e-6).unwrap_err(), sp::ldl_error::zero_pivot);   // pivot 3 is 0 with no sign
    triplets t2 = t;
    t2.v[3] = 5;
    const matrix a2 = t2.assemble();
    factor h(a2);
    const auto r2 = h.numeric(a2, partial, 1e-6);
    ASSERT_TRUE(r2.is_ok());
    EXPECT_EQ(*r2, 0u);   // pivot 1 is -3 where - is expected: kept; the others have no sign
    EXPECT_EQ(h.d[1], -3.0);
    EXPECT_EQ(h.d[2], 1e-12);   // tiny, but no sign: kept
    EXPECT_EQ(h.d[4], -1e-12);
    // A KKT system with a zero (2, 2) block and a singular P: unfactorable without signs, solvable with them
    triplets k;
    k.n = 3;
    k.add(0, 0, 1);
    k.add(1, 1, 0);
    k.add(0, 2, 1);
    k.add(1, 2, 1);
    k.add(2, 2, 0);
    const matrix kk = k.assemble();
    factor u(kk);
    EXPECT_EQ(u.numeric(kk).unwrap_err(), sp::ldl_error::zero_pivot);
    factor s(kk);
    const std::vector<std::int8_t> ks{1, 1, -1};
    const auto rs = s.numeric(kk, ks, 1e-7);
    ASSERT_TRUE(rs.is_ok());
    EXPECT_EQ(*rs, 1u);
    const std::vector<double> rhs{1, 2, 3};
    auto x = s.solve(rhs);
    std::vector<double> res(3);
    const auto refined = sp::ldl_refine(kk.view(), s.l.as_const(), std::span<const double>(s.dinv), std::span<const double>(rhs), std::span<double>(x),
                                        std::span<double>(res), 20, 1e-12);
    EXPECT_LE(refined.residual, 1e-12);
    // [1 0 1; 0 0 1; 1 1 0] x = (1, 2, 3): x = (-1, 4, 2)
    EXPECT_NEAR(x[0], -1, 1e-9);
    EXPECT_NEAR(x[1], 4, 1e-9);
    EXPECT_NEAR(x[2], 2, 1e-9);
}

TEST(SparseLdlNumeric, ReportsZeroAndNonFinitePivots)
{
    // [0 1; 1 0]: the first pivot is zero
    triplets t;
    t.n = 2;
    t.add(0, 0, 0);
    t.add(0, 1, 1);
    t.add(1, 1, 0);
    const matrix a = t.assemble();
    factor f(a);
    EXPECT_EQ(f.numeric(a).unwrap_err(), sp::ldl_error::zero_pivot);
    // A structurally missing diagonal is a zero pivot
    triplets m;
    m.n = 2;
    m.add(1, 1, 1);
    m.add(0, 1, 1);
    const matrix am = m.assemble();
    factor fm(am);
    EXPECT_EQ(fm.numeric(am).unwrap_err(), sp::ldl_error::zero_pivot);
    // [1 1; 1 1]: the second pivot cancels to exactly zero
    triplets c;
    c.n = 2;
    c.add(0, 0, 1);
    c.add(0, 1, 1);
    c.add(1, 1, 1);
    const matrix ac = c.assemble();
    factor fc(ac);
    EXPECT_EQ(fc.numeric(ac).unwrap_err(), sp::ldl_error::zero_pivot);
    // NaN and infinity: an error with or without signs
    for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        triplets n;
        n.n = 2;
        n.add(0, 0, 1);
        n.add(1, 1, bad);
        const matrix an = n.assemble();
        factor fn(an);
        EXPECT_EQ(fn.numeric(an).unwrap_err(), sp::ldl_error::zero_pivot);
        const std::vector<std::int8_t> s{1, 1};
        EXPECT_EQ(fn.numeric(an, s).unwrap_err(), sp::ldl_error::zero_pivot);
    }
}

TEST(SparseLdlNumeric, ReportsShortSpansAndAMismatchedOrder)
{
    const matrix a = random_spd(6, 2, 4).assemble();
    factor f(a);
    auto call = [&](std::span<double> d, std::span<double> dinv, std::span<I> iw, std::span<double> xw, std::span<const std::int8_t> s = {}) {
        return sp::ldl_numeric(a.view(), f.sym, f.l, d, dinv, iw, xw, s);
    };
    const auto d = std::span<double>(f.d), di = std::span<double>(f.dinv), xw = std::span<double>(f.xwork);
    const auto iw = std::span<I>(f.iwork);
    EXPECT_TRUE(call(d, di, iw, xw).is_ok());
    EXPECT_EQ(call(d.first(5), di, iw, xw).unwrap_err(), sp::ldl_error::insufficient_capacity);
    EXPECT_EQ(call(d, di.first(5), iw, xw).unwrap_err(), sp::ldl_error::insufficient_capacity);
    EXPECT_EQ(call(d, di, iw.first(17), xw).unwrap_err(), sp::ldl_error::insufficient_capacity);
    EXPECT_EQ(call(d, di, iw, xw.first(5)).unwrap_err(), sp::ldl_error::insufficient_capacity);
    const std::vector<std::int8_t> short_signs{1, 1};
    EXPECT_EQ(call(d, di, iw, xw, short_signs).unwrap_err(), sp::ldl_error::insufficient_capacity);
    sp::csc_view<double, I> small{0, 0, f.lp, std::span<I>(f.li).first(f.sym.nnz - 1), f.lx};
    EXPECT_EQ(sp::ldl_numeric(a.view(), f.sym, small, d, di, iw, xw).unwrap_err(), sp::ldl_error::insufficient_capacity);
    const matrix other = random_spd(7, 2, 4).assemble();
    EXPECT_EQ(sp::ldl_numeric(other.view(), f.sym, f.l, d, di, iw, xw).unwrap_err(), sp::ldl_error::dimension_mismatch);
    // L's col_ptr in storage of its own: copied from the analysis
    std::vector<I> own(a.n + 1, -5);
    sp::csc_view<double, I> separate{0, 0, own, f.li, f.lx};
    ASSERT_TRUE(sp::ldl_numeric(a.view(), f.sym, separate, d, di, iw, xw).is_ok());
    EXPECT_EQ(own, f.lp);
    EXPECT_EQ(separate.rows, a.n);
    static_assert(noexcept(sp::ldl_numeric(std::declval<sp::csc_view<const double, const I>>(), f.sym, f.l, d, di, iw, xw)));
    static_assert(noexcept(sp::ldl_solve(f.l, std::declval<std::span<const double>>(), xw)));
    static_assert(noexcept(sp::ldl_symbolic(std::declval<sp::csc_view<const double, const I>>(), iw, iw, iw, iw)));
}

TEST(SparseLdlSolve, OfTheIdentityAndOfAnEmptySystem)
{
    triplets t;
    t.n = 3;
    for (std::size_t i = 0; i < 3; ++i)
        t.add(i, i, 1);
    const matrix a = t.assemble();
    factor f(a);
    ASSERT_TRUE(f.numeric(a).is_ok());
    EXPECT_EQ(f.sym.nnz, 0u);
    EXPECT_EQ(f.solve({4, 5, 6}), (std::vector<double>{4, 5, 6}));
    const matrix e = triplets{}.assemble();
    factor g(e);
    ASSERT_TRUE(g.numeric(e).is_ok());
    EXPECT_TRUE(g.solve({}).empty());
}

TEST(SparseLdl, WorksWithSixtyFourBitIndices)
{
    const matrix a = random_spd(30, 3, 9).assemble();
    const std::size_t n = a.n;
    std::vector<std::int64_t> cp(a.cp.begin(), a.cp.end()), ri(a.ri.begin(), a.ri.end()), parent(n), counts(n), lp(n + 1), iwork(3 * n);
    const sp::csc_view<const double, const std::int64_t> v{n, n, cp, ri, a.vx};
    const auto s = sp::ldl_symbolic(v, std::span<std::int64_t>(parent), std::span<std::int64_t>(counts), std::span<std::int64_t>(lp), std::span<std::int64_t>(iwork).first(n));
    ASSERT_TRUE(s.is_ok());
    std::vector<std::int64_t> li(s->nnz);
    std::vector<double> lx(s->nnz), d(n), dinv(n), xw(n);
    sp::csc_view<double, std::int64_t> l{n, n, lp, li, lx};
    ASSERT_TRUE(sp::ldl_numeric(v, *s, l, d, dinv, std::span<std::int64_t>(iwork), xw).is_ok());
    auto rhs = random_vector(n, 1);
    auto x = rhs;
    sp::ldl_solve(l.as_const(), std::span<const double>(dinv), std::span<double>(x));
    EXPECT_LT(max_abs_diff(x, dense_solve(a, rhs)), 1e-12);
}

// The KKT system of min ½ xᵀPx + qᵀx s.t. x₁ + x₂ + x₃ = 1, x₁ - x₃ = 0, with P = diag(4, 2, 1), ρ = 1e6
constexpr std::size_t kn = 5, kmax = 12;
constexpr sp::csc_matrix<double, kn, kn, kmax> kkt_matrix(double rho)
{
    const std::array<I, 11> ti{0, 1, 2, 0, 1, 2, 0, 2, 3, 4, 0}, tj{0, 1, 2, 3, 3, 3, 4, 4, 3, 4, 0};
    const std::array<double, 11> tv{3, 2, 1, 1, 1, 1, 1, -1, -1 / rho, -1 / rho, 1};   // P(0, 0) = 3 + 1, given as two triplets
    sp::csc_matrix<double, kn, kn, kmax> a{};
    auto av = a.view();
    std::array<I, kn> work{};
    (void)sp::from_triplets(kn, kn, std::span<const I>(ti), std::span<const I>(tj), std::span<const double>(tv), av, std::span<I>(work));
    return a;
}

// Order, permute, analyse, factor, solve, refine: the residual, in a constant expression
consteval double kkt_residual_at_compile_time()
{
    const auto a = kkt_matrix(1e6);
    std::array<I, kn> perm{};
    std::array<I, sp::amd_workspace_size(kn, kmax)> work{};
    if (!sp::amd(a.view(), std::span<I>(perm), std::span<I>(work)))
        return -1;
    sp::csc_matrix<double, kn, kn, kmax> b{};
    auto bv = b.view();
    std::array<I, 2 * kn> pw{};
    if (!sp::symmetric_permute_upper(a.view(), std::span<const I>(perm), bv, std::span<I>(pw)))
        return -1;
    constexpr std::size_t lnz = 8;
    sp::ldl_fixed<double, kn, lnz> f{};
    if (!f.analyse(std::as_const(b).view()))
        return -1;
    std::array<std::int8_t, kn> signs{};
    for (std::size_t k = 0; k < kn; ++k)
        signs[k] = perm[k] < 3 ? 1 : -1;
    if (!f.factor(std::as_const(b).view(), std::span<const std::int8_t>(signs)))
        return -1;
    const std::array<double, kn> q{-1, -1, -1, 1, 0};   // (−q, b) in the original numbering
    std::array<double, kn> rhs{}, x{}, res{};
    sp::permute(std::span<const I>(perm), std::span<const double>(q), std::span<double>(rhs));
    x = rhs;
    f.solve(std::span<double>(x));
    const auto r = sp::ldl_refine(std::as_const(b).view(), f.l(), std::span<const double>(f.dinv), std::span<const double>(rhs), std::span<double>(x),
                                  std::span<double>(res), 5, 1e-13);
    return r.residual;
}

// The analysis alone at compile time, sized by ldl_nnz; the numbers come at run time
constexpr std::size_t kkt_lnz = *sp::ldl_nnz<kn>(kkt_matrix(1.0).view());
constexpr auto kkt_analysis = [] {
    sp::ldl_fixed<double, kn, kkt_lnz> f{};
    (void)f.analyse(kkt_matrix(1.0).view());
    return f;
}();

TEST(SparseLdl, OrdersAnalysesAndFactorsASmallKktSystemInAConstantExpression)
{
    static_assert(kkt_residual_at_compile_time() >= 0 && kkt_residual_at_compile_time() <= 1e-13);
    static_assert(kkt_lnz == kkt_analysis.nnz && kkt_lnz == 6);
    static_assert(kkt_analysis.col_ptr[kn] == static_cast<I>(kkt_lnz));
    EXPECT_LE(kkt_residual_at_compile_time(), 1e-13);
    // The compile-time analysis, factored at run time with ρ = 1e6, solves as the dense system does
    auto f = kkt_analysis;
    const auto a = kkt_matrix(1e6);
    const std::array<std::int8_t, kn> signs{1, 1, 1, -1, -1};
    const auto r = f.factor(a.view(), std::span<const std::int8_t>(signs));
    ASSERT_TRUE(r.is_ok());
    EXPECT_EQ(*r, 0u);   // -1e-6 on the diagonal, but the constraint pivots are -1e-6 - Σ aᵢⱼ² / pⱼ after elimination
    std::array<double, kn> x{-1, -1, -1, 1, 0};
    f.solve(std::span<double>(x));
    matrix m;
    m.n = kn;
    m.cp.assign(a.col_ptr.begin(), a.col_ptr.end());
    m.ri.assign(a.row_idx.begin(), a.row_idx.begin() + static_cast<std::ptrdiff_t>(a.nnz()));
    m.vx.assign(a.values.begin(), a.values.begin() + static_cast<std::ptrdiff_t>(a.nnz()));
    const auto expect = dense_solve(m, {-1, -1, -1, 1, 0});
    for (std::size_t i = 0; i < kn; ++i)
        EXPECT_NEAR(x[i], expect[i], 1e-6);
    // Capacity: an ldl_fixed too small for L
    sp::ldl_fixed<double, kn, 3> tiny{};
    EXPECT_EQ(tiny.analyse(a.view()).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(sp::ldl_nnz<4>(a.view()).unwrap_err(), sp::sparse_error::dimension_mismatch);
    EXPECT_EQ(tiny.symbolic().n, kn);
}

} // namespace
