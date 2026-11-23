//
//  scaling_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/optim/scaling.hpp>: after Ruiz equilibration the KKT matrix's column norms (and, for the
//  rectangular form, a matrix's row and column norms) are within tolerance of 1; the cost scaling's measure is 1;
//  zero rows and columns stay unscaled; with power-of-two factors unscaling inverts scaling to the bit, and without
//  them to an ulp or so; a scaled equality-constrained QP, solved densely and unscaled, gives the original's primal
//  and dual solution; errors; constant evaluation.
//

#include <snicholls/core/optim/scaling.hpp>

#include <snicholls/core/sparse/csc.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace op = snicholls::optim;
namespace sp = snicholls::sparse;
namespace st = snicholls::stats;

struct csc_owner {
    std::size_t rows = 0, cols = 0;
    std::vector<std::int32_t> col_ptr, row_idx;
    std::vector<double> values;
    sp::csc_view<double> view() { return {rows, cols, col_ptr, row_idx, values}; }
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

// A badly scaled QP: P = S (BᵀB + I) S with S spanning 1e-3 to 1e3, A (m x n) with rows and columns scaled over
// 1e-2 to 1e4, q spanning 1e-1 to 1e3; a zero row and a variable that appears nowhere when asked
struct qp {
    std::size_t n, m;
    std::vector<double> p, a, q, b;
};

qp make_qp(std::size_t n, std::size_t m, std::uint64_t seed)
{
    st::wyrand g(seed);
    auto u = [&] { return st::uniform_real(g) - 0.5; };
    qp out{n, m, std::vector<double>(n * n, 0.0), std::vector<double>(m * n), std::vector<double>(n), std::vector<double>(m)};
    std::vector<double> bm(n * n), s(n);
    for (auto& e : bm)
        e = u();
    for (std::size_t j = 0; j < n; ++j)
        s[j] = std::pow(10.0, -3.0 + 6.0 * st::uniform_real(g));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double v = i == j ? 1.0 : 0.0;
            for (std::size_t k = 0; k < n; ++k)
                v += bm[k * n + i] * bm[k * n + j];
            out.p[i * n + j] = s[i] * v * s[j];
        }
    for (std::size_t i = 0; i < m; ++i) {
        const double ri = std::pow(10.0, -2.0 + 6.0 * st::uniform_real(g));
        for (std::size_t j = 0; j < n; ++j)
            out.a[i * n + j] = ri * u() * std::pow(10.0, 2.0 * static_cast<double>(j % 3));
    }
    for (auto& e : out.q)
        e = u() * std::pow(10.0, -1.0 + 4.0 * st::uniform_real(g));
    for (auto& e : out.b)
        e = u();
    return out;
}

// The KKT column norms of (upper P, A): max(‖P(:, j)‖∞, ‖A(:, j)‖∞) then ‖A(i, :)‖∞
std::vector<double> kkt_norms(csc_owner& p, csc_owner& a)
{
    const std::size_t n = p.cols, m = a.rows;
    std::vector<double> pn(n), an(n), rn(m), out;
    sp::symmetric_upper_inf_norms(p.view(), std::span<double>(pn));
    sp::column_inf_norms(a.view(), std::span<double>(an));
    sp::row_inf_norms(a.view(), std::span<double>(rn));
    for (std::size_t j = 0; j < n; ++j)
        out.push_back(std::max(pn[j], an[j]));
    out.insert(out.end(), rn.begin(), rn.end());
    return out;
}

// Solves the equality-constrained QP min ½xᵀPx + qᵀx s.t. Ax = b through its KKT system [P Aᵀ; A 0][x; y] = [-q; b]
std::vector<double> solve_kkt(const std::vector<double>& p, const std::vector<double>& a, const std::vector<double>& q, const std::vector<double>& b, std::size_t n, std::size_t m)
{
    const std::size_t nk = n + m;
    std::vector<double> k(nk * nk, 0.0), rhs(nk);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            k[i * nk + j] = p[i * n + j];
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            k[(n + i) * nk + j] = k[j * nk + n + i] = a[i * n + j];
    for (std::size_t j = 0; j < n; ++j)
        rhs[j] = -q[j];
    for (std::size_t i = 0; i < m; ++i)
        rhs[n + i] = b[i];
    return dense_solve(k, rhs);
}

// The dense symmetric matrix of an upper triangle
std::vector<double> symmetric_dense(csc_owner& upper)
{
    const std::size_t n = upper.cols;
    std::vector<double> d(n * n);
    sp::to_dense(upper.view(), std::span<double>(d));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            d[i * n + j] = d[j * n + i];
    return d;
}

static_assert(op::ruiz_workspace(5, 3) == 13 && op::ruiz_rectangular_workspace(3, 5) == 8);
static_assert(noexcept(op::ruiz_equilibrate(std::declval<const sp::csc_view<double>&>(), std::declval<const sp::csc_view<double>&>(), {}, {}, {}, std::declval<double&>(), 1, {})));
static_assert(noexcept(op::ruiz_equilibrate_rectangular(std::declval<const sp::csc_view<double>&>(), {}, {}, 1, {})));

// Constant evaluation: P = diag(100, 1e-2), A = [1e3 1e-1]; after equilibration (no cost scaling) every KKT norm is 1
constexpr bool equilibrates_at_compile_time()
{
    sp::csc_matrix<double, 2, 2, 2> p{{0, 1, 2}, {0, 1}, {100, 1e-2}};
    sp::csc_matrix<double, 1, 2, 2> a{{0, 1, 2}, {0, 0}, {1e3, 1e-1}};
    std::array<double, 2> q{1, 1}, d{}, w{};
    std::array<double, 1> e{};
    std::array<double, 5> work{};
    double c = 0;
    const auto pv = p.view();
    const auto av = a.view();
    if (!op::ruiz_equilibrate(pv, av, q, d, e, c, 40, work, {1e-4, 1e4, false, false}))
        return false;
    std::array<double, 2> pn{}, an{};
    std::array<double, 1> rn{};
    sp::symmetric_upper_inf_norms(pv, std::span<double>(pn));
    sp::column_inf_norms(av, std::span<double>(an));
    sp::row_inf_norms(av, std::span<double>(rn));
    (void)w;
    for (std::size_t j = 0; j < 2; ++j) {
        const double v = pn[j] > an[j] ? pn[j] : an[j];
        if (v < 0.999 || v > 1.001)
            return false;
    }
    return rn[0] > 0.999 && rn[0] < 1.001 && c == 1;
}
static_assert(equilibrates_at_compile_time());

TEST(OptimScalingRuiz, ScaledKktColumnNormsAreWithinToleranceOfOne)
{
    for (std::uint64_t seed : {1u, 2u, 3u}) {
        const auto pr = make_qp(8, 5, seed);
        auto p = from_dense(pr.p, 8, 8, true);
        auto a = from_dense(pr.a, 5, 8);
        auto q = pr.q;
        std::vector<double> d(8), e(5), work(op::ruiz_workspace(8, 5));
        double c = 0;
        // before: norms spread over many orders of magnitude
        const auto before = kkt_norms(p, a);
        EXPECT_GT(*std::max_element(before.begin(), before.end()) / *std::min_element(before.begin(), before.end()), 1e3);
        ASSERT_TRUE(op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 25, work, {1e-4, 1e4, false, false}));
        for (const double v : kkt_norms(p, a))
            EXPECT_NEAR(v, 1.0, 1e-3) << seed;
        EXPECT_EQ(c, 1.0);
        for (std::size_t j = 0; j < 8; ++j)
            EXPECT_NEAR(q[j], pr.q[j] * d[j], 1e-13 * std::abs(q[j]));
    }
}

TEST(OptimScalingRuiz, TenIterationsAsOsqpDoesBringTheNormsCloseToOne)
{
    const auto pr = make_qp(10, 6, 9);
    auto p = from_dense(pr.p, 10, 10, true);
    auto a = from_dense(pr.a, 6, 10);
    auto q = pr.q;
    std::vector<double> d(10), e(6), work(op::ruiz_workspace(10, 6));
    double c = 0;
    ASSERT_TRUE(op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 10, work, {1e-4, 1e4, false, false}));
    for (const double v : kkt_norms(p, a)) {
        EXPECT_GT(v, 0.9);
        EXPECT_LT(v, 1.1);
    }
}

TEST(OptimScalingRuiz, CostScalingMakesTheCostMeasureOne)
{
    const auto pr = make_qp(6, 3, 4);
    auto p = from_dense(pr.p, 6, 6, true);
    auto a = from_dense(pr.a, 3, 6);
    auto q = pr.q;
    std::vector<double> d(6), e(3), work(op::ruiz_workspace(6, 3)), pn(6);
    double c = 0;
    ASSERT_TRUE(op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 10, work));
    EXPECT_GT(c, 0.0);
    EXPECT_NE(c, 1.0);
    // after the last iteration's cost step, max(mean of P's column norms, ‖q‖∞) is 1
    sp::symmetric_upper_inf_norms(p.view(), std::span<double>(pn));
    double mean = 0, qn = 0;
    for (std::size_t j = 0; j < 6; ++j) {
        mean += pn[j] / 6;
        qn = std::max(qn, std::abs(q[j]));
    }
    EXPECT_NEAR(std::max(mean, qn), 1.0, 1e-12);
    for (std::size_t j = 0; j < 6; ++j)
        EXPECT_NEAR(q[j], c * d[j] * pr.q[j], 1e-12 * std::abs(q[j]));
}

TEST(OptimScalingRuiz, AZeroRowAndAnAbsentVariableAreLeftUnscaled)
{
    // P = diag(4, 0, 9) (variable 1 has no quadratic term and no constraint), A = [2 0 8; 0 0 0]
    const std::vector<double> pd{4, 0, 0, 0, 0, 0, 0, 0, 9}, ad{2, 0, 8, 0, 0, 0};
    auto p = from_dense(pd, 3, 3, true);
    auto a = from_dense(ad, 2, 3);
    std::vector<double> q{1, 1, 1}, d(3), e(2), work(op::ruiz_workspace(3, 2));
    double c = 0;
    ASSERT_TRUE(op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 10, work, {1e-4, 1e4, false, false}));
    EXPECT_EQ(d[1], 1.0);
    EXPECT_EQ(e[1], 1.0);
    EXPECT_NE(d[0], 1.0);
    EXPECT_NE(e[0], 1.0);
}

TEST(OptimScalingRuiz, ZeroIterationsGiveUnitScalingsAndUnchangedData)
{
    const auto pr = make_qp(4, 2, 5);
    auto p = from_dense(pr.p, 4, 4, true);
    auto a = from_dense(pr.a, 2, 4);
    const auto pv = p.values, av = a.values;
    auto q = pr.q;
    std::vector<double> d(4, 7), e(2, 7), work(op::ruiz_workspace(4, 2));
    double c = 7;
    ASSERT_TRUE(op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 0, work));
    EXPECT_EQ(d, std::vector<double>(4, 1.0));
    EXPECT_EQ(e, std::vector<double>(2, 1.0));
    EXPECT_EQ(c, 1.0);
    EXPECT_EQ(p.values, pv);
    EXPECT_EQ(a.values, av);
    EXPECT_EQ(q, pr.q);
}

TEST(OptimScalingUnscale, PowerOfTwoScalingIsInvertedToTheBit)
{
    const auto pr = make_qp(7, 4, 6);
    auto p = from_dense(pr.p, 7, 7, true);
    auto a = from_dense(pr.a, 4, 7);
    const auto pv = p.values, av = a.values;
    auto q = pr.q;
    std::vector<double> d(7), e(4), work(op::ruiz_workspace(7, 4));
    double c = 0;
    ASSERT_TRUE(op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 10, work, {1e-4, 1e4, true, true}));
    for (const double v : d)
        EXPECT_EQ(std::exp2(std::round(std::log2(v))), v);
    for (const double v : e)
        EXPECT_EQ(std::exp2(std::round(std::log2(v))), v);
    EXPECT_EQ(std::exp2(std::round(std::log2(c))), c);
    // the norms land within a factor of about √2 of 1
    for (const double v : kkt_norms(p, a)) {
        EXPECT_GT(v, 0.4);
        EXPECT_LT(v, 2.5);
    }
    op::unscale_data(p.view(), a.view(), q, d, e, c);
    EXPECT_EQ(p.values, pv);
    EXPECT_EQ(a.values, av);
    EXPECT_EQ(q, pr.q);
    // solutions too: scale then unscale is the identity, to the bit
    std::vector<double> x{1.25, -3.5, 1e-7, 2e5, 0.1, -0.3, 7}, y{0.5, -2.25, 3e-3, 11};
    const auto x0 = x, y0 = y;
    op::scale_primal(std::span<const double>(d), x);
    op::scale_dual(std::span<const double>(e), c, y);
    op::unscale_solution(std::span<const double>(d), e, c, x, y);
    EXPECT_EQ(x, x0);
    EXPECT_EQ(y, y0);
    std::vector<double> z = y0;
    std::vector<double> l = y0, u = y0;
    op::scale_bounds(std::span<const double>(e), l, u);
    op::unscale_constraint(std::span<const double>(e), l);
    EXPECT_EQ(l, z);
    EXPECT_EQ(op::unscale_objective(c, 3.0 * c), 3.0);
}

TEST(OptimScalingUnscale, OrdinaryScalingIsInvertedToAnUlpOrSo)
{
    const auto pr = make_qp(7, 4, 7);
    auto p = from_dense(pr.p, 7, 7, true);
    auto a = from_dense(pr.a, 4, 7);
    const auto pv = p.values, av = a.values;
    auto q = pr.q;
    std::vector<double> d(7), e(4), work(op::ruiz_workspace(7, 4));
    double c = 0;
    ASSERT_TRUE(op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 10, work));
    op::unscale_data(p.view(), a.view(), q, d, e, c);
    for (std::size_t k = 0; k < pv.size(); ++k)
        EXPECT_NEAR(p.values[k], pv[k], 1e-14 * std::abs(pv[k]));
    for (std::size_t k = 0; k < av.size(); ++k)
        EXPECT_NEAR(a.values[k], av[k], 1e-14 * std::abs(av[k]));
    for (std::size_t j = 0; j < 7; ++j)
        EXPECT_NEAR(q[j], pr.q[j], 1e-14 * std::abs(pr.q[j]));
}

TEST(OptimScalingUnscale, TheScaledQpsSolutionUnscaledIsTheOriginals)
{
    for (const bool pow2 : {false, true})
        for (std::uint64_t seed : {11u, 12u, 13u}) {
            constexpr std::size_t n = 6, m = 3;
            const auto pr = make_qp(n, m, seed);
            const auto original = solve_kkt(pr.p, pr.a, pr.q, pr.b, n, m);
            auto p = from_dense(pr.p, n, n, true);
            auto a = from_dense(pr.a, m, n);
            auto q = pr.q;
            std::vector<double> d(n), e(m), work(op::ruiz_workspace(n, m));
            double c = 0;
            ASSERT_TRUE(op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 10, work, {1e-4, 1e4, true, pow2}));
            auto l = pr.b, u = pr.b;
            op::scale_bounds(std::span<const double>(e), l, u);
            std::vector<double> ad(m * n);
            sp::to_dense(a.view(), std::span<double>(ad));
            const auto scaled = solve_kkt(symmetric_dense(p), ad, q, l, n, m);
            std::vector<double> x(scaled.begin(), scaled.begin() + n), y(scaled.begin() + n, scaled.end());
            op::unscale_solution(std::span<const double>(d), e, c, x, y);
            for (std::size_t j = 0; j < n; ++j)
                EXPECT_NEAR(x[j], original[j], 1e-8 * (1 + std::abs(original[j]))) << seed;
            for (std::size_t i = 0; i < m; ++i)
                EXPECT_NEAR(y[i], original[n + i], 1e-8 * (1 + std::abs(original[n + i]))) << seed;
            // and the constraint values: E⁻¹ Ā x̄ = A x
            std::vector<double> xb(scaled.begin(), scaled.begin() + n), zb(m), z(m);
            sp::spmv(a.view(), std::span<const double>(xb), std::span<double>(zb));
            op::unscale_constraint(std::span<const double>(e), zb);
            for (std::size_t i = 0; i < m; ++i)
                EXPECT_NEAR(zb[i], pr.b[i], 1e-9) << seed;
        }
}

TEST(OptimScalingRectangular, RowAndColumnNormsApproachOne)
{
    const auto pr = make_qp(9, 5, 21);
    auto a = from_dense(pr.a, 5, 9);
    const auto av = a.values;
    std::vector<double> r(5), s(9), work(op::ruiz_rectangular_workspace(5, 9)), rn(5), cn(9);
    ASSERT_TRUE(op::ruiz_equilibrate_rectangular(a.view(), r, s, 30, work));
    sp::row_inf_norms(a.view(), std::span<double>(rn));
    sp::column_inf_norms(a.view(), std::span<double>(cn));
    for (const double v : rn)
        EXPECT_NEAR(v, 1.0, 1e-3);
    for (const double v : cn)
        EXPECT_NEAR(v, 1.0, 1e-3);
    // Ā = R A S entry by entry
    for (std::size_t j = 0; j < 9; ++j)
        for (std::int32_t k = a.col_ptr[j]; k < a.col_ptr[j + 1]; ++k)
            EXPECT_NEAR(a.values[static_cast<std::size_t>(k)], r[static_cast<std::size_t>(a.row_idx[static_cast<std::size_t>(k)])] * av[static_cast<std::size_t>(k)] * s[j],
                        1e-13 * std::abs(a.values[static_cast<std::size_t>(k)]));
}

TEST(OptimScalingRectangular, ASquareSystemSolvedScaledGivesTheSameSolution)
{
    // A x = b with A badly scaled: Ā x̄ = R b, x = S x̄
    constexpr std::size_t n = 5;
    const auto pr = make_qp(n, n, 31);
    std::vector<double> am = pr.a;
    for (std::size_t i = 0; i < n; ++i)
        am[i * n + i] += 3 * std::abs(am[i * n + i]) + 1e-3;
    const auto x0 = dense_solve(am, pr.b);
    auto a = from_dense(am, n, n);
    std::vector<double> r(n), s(n), work(2 * n), ad(n * n), rb(n);
    ASSERT_TRUE(op::ruiz_equilibrate_rectangular(a.view(), r, s, 10, work));
    sp::to_dense(a.view(), std::span<double>(ad));
    for (std::size_t i = 0; i < n; ++i)
        rb[i] = r[i] * pr.b[i];
    auto x = dense_solve(ad, rb);
    op::unscale_primal(std::span<const double>(s), x);
    for (std::size_t j = 0; j < n; ++j)
        EXPECT_NEAR(x[j], x0[j], 1e-9 * (1 + std::abs(x0[j])));
}

TEST(OptimScalingErrors, MismatchedShapesAndShortWorkspaceWriteNothing)
{
    const auto pr = make_qp(4, 2, 8);
    auto p = from_dense(pr.p, 4, 4, true);
    auto a = from_dense(pr.a, 2, 4);
    const auto pv = p.values;
    auto q = pr.q;
    std::vector<double> d(4, 5), e(2, 5), short_work(9), work(10), d3(3);
    double c = 5;
    auto r = op::ruiz_equilibrate(p.view(), a.view(), q, d, e, c, 10, short_work);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.unwrap_err(), sp::sparse_error::insufficient_capacity);
    r = op::ruiz_equilibrate(p.view(), a.view(), q, d3, e, c, 10, work);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.unwrap_err(), sp::sparse_error::dimension_mismatch);
    auto a3 = from_dense(std::vector<double>(6, 1.0), 2, 3);
    r = op::ruiz_equilibrate(p.view(), a3.view(), q, d, e, c, 10, work);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.unwrap_err(), sp::sparse_error::dimension_mismatch);
    EXPECT_EQ(p.values, pv);
    EXPECT_EQ(d, std::vector<double>(4, 5.0));
    EXPECT_EQ(c, 5.0);
    std::vector<double> rr(2), ss(4), w5(5), s3(3);
    auto rr_res = op::ruiz_equilibrate_rectangular(a.view(), rr, ss, 3, w5);
    ASSERT_FALSE(rr_res);
    EXPECT_EQ(rr_res.unwrap_err(), sp::sparse_error::insufficient_capacity);
    rr_res = op::ruiz_equilibrate_rectangular(a.view(), rr, s3, 3, work);
    ASSERT_FALSE(rr_res);
    EXPECT_EQ(rr_res.unwrap_err(), sp::sparse_error::dimension_mismatch);
}

TEST(OptimScalingRuiz, AnEmptyProblemIsFine)
{
    sp::csc_view<double> p{0, 0, {}, {}, {}}, a{0, 0, {}, {}, {}};
    std::array<std::int32_t, 1> cp{0}, ca{0};
    p.col_ptr = cp;
    a.col_ptr = ca;
    double c = 0;
    EXPECT_TRUE(op::ruiz_equilibrate(p, a, {}, {}, {}, c, 10, {}));
    EXPECT_EQ(c, 1.0);
}

} // namespace
