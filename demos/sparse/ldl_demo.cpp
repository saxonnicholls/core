//
//  ldl_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/sparse/ldl.hpp>: the KKT system of a small equality-constrained QP (a four-asset
//  minimum-variance portfolio, fully invested, with a target return), end to end: assembled from triplets, ordered by
//  AMD, permuted (keeping the map of where each entry went), analysed once, factored with expected signs, solved,
//  refined; then ρ changed and the same pattern refactored through the map. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/sparse/ldl.hpp>

#include <snicholls/core/sparse/csc.hpp>
#include <snicholls/core/sparse/ordering.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace sp = snicholls::sparse;
using I = std::int32_t;

} // namespace

int main()
{
    std::printf("snicholls-core ldl demo\n");
    bool ok = true;

    //! [ldl]
    // min ½ xᵀ P x  subject to  Σ xᵢ = 1,  μᵀ x = 0.08:  P a covariance, x the weights. Its KKT system is
    //   [ P      Aᵀ    ] [x]   [0]
    //   [ A   -ρ⁻¹ I   ] [y] = [b]     (the -ρ⁻¹ I makes it quasi-definite; ρ → ∞ is the exact problem)
    constexpr std::size_t nx = 4, m = 2, n = nx + m;
    const double cov[nx][nx] = {{0.040, 0.006, 0.000, 0.002}, {0.006, 0.090, 0.010, 0.000}, {0.000, 0.010, 0.020, 0.000}, {0.002, 0.000, 0.000, 0.060}};
    const double mu[nx] = {0.06, 0.12, 0.04, 0.09};
    double rho = 1e6;
    std::vector<I> ti, tj;
    std::vector<double> tv;
    auto add = [&](std::size_t i, std::size_t j, double v) {   // upper triangle only
        ti.push_back(static_cast<I>(i));
        tj.push_back(static_cast<I>(j));
        tv.push_back(v);
    };
    for (std::size_t j = 0; j < nx; ++j)
        for (std::size_t i = 0; i <= j; ++i)
            if (cov[i][j] != 0)
                add(i, j, cov[i][j]);
    for (std::size_t i = 0; i < nx; ++i) {
        add(i, nx, 1.0);         // Σ xᵢ = 1
        add(i, nx + 1, mu[i]);   // μᵀ x = 0.08
    }
    add(nx, nx, -1 / rho);
    add(nx + 1, nx + 1, -1 / rho);
    std::vector<I> cp(n + 1), ri(ti.size()), work(n);
    std::vector<double> vx(ti.size());
    sp::csc_view<double, I> kkt{0, 0, cp, ri, vx};
    ok &= sp::from_triplets(n, n, std::span<const I>(ti), std::span<const I>(tj), std::span<const double>(tv), kkt, std::span<I>(work)).is_ok();

    // Once: the order, the permuted matrix (and where each entry went), the analysis
    std::vector<I> perm(n), amd_work(sp::amd_workspace_size(n, kkt.nnz()));
    ok &= sp::amd(kkt.as_const(), std::span<I>(perm), std::span<I>(amd_work)).is_ok();
    std::vector<I> pcp(n + 1), pri(kkt.nnz()), map(kkt.nnz()), pwork(2 * n);
    std::vector<double> pvx(kkt.nnz());
    sp::csc_view<double, I> k{0, 0, pcp, pri, pvx};
    ok &= sp::symmetric_permute_upper(kkt.as_const(), std::span<const I>(perm), k, std::span<I>(pwork), std::span<I>(map)).is_ok();
    std::vector<I> parent(n), counts(n), lp(n + 1), iwork(3 * n);
    const auto symbolic = sp::ldl_symbolic(k.as_const(), std::span<I>(parent), std::span<I>(counts), std::span<I>(lp), std::span<I>(iwork).first(n));
    ok &= symbolic.is_ok();
    std::vector<I> li(symbolic ? symbolic->nnz : 0);
    std::vector<double> lx(li.size()), d(n), dinv(n), xwork(n);
    sp::csc_view<double, I> l{n, n, lp, li, lx};

    // The expected sign of each pivot, in the new numbering: + for a weight, - for a multiplier
    std::vector<std::int8_t> signs(n);
    for (std::size_t i = 0; i < n; ++i)
        signs[i] = static_cast<std::size_t>(perm[i]) < nx ? 1 : -1;

    // Each iteration: factor, solve in the new numbering, refine, map back
    auto solve = [&](std::array<double, n>& solution, std::size_t& regularised, sp::ldl_refine_result<double>& refined) {
        const auto r = sp::ldl_numeric(k.as_const(), *symbolic, l, d, dinv, std::span<I>(iwork), xwork, signs);
        if (!r)
            return false;
        regularised = *r;
        const std::array<double, n> rhs{0, 0, 0, 0, 1, 0.08};
        std::array<double, n> prhs{}, px{}, residual{};
        sp::permute(std::span<const I>(perm), std::span<const double>(rhs), std::span<double>(prhs));
        px = prhs;
        sp::ldl_solve(l.as_const(), std::span<const double>(dinv), std::span<double>(px));
        refined = sp::ldl_refine(k.as_const(), l.as_const(), std::span<const double>(dinv), std::span<const double>(prhs), std::span<double>(px),
                                 std::span<double>(residual), 5, 1e-14);
        sp::inverse_permute(std::span<const I>(perm), std::span<const double>(px), std::span<double>(solution));
        return true;
    };
    std::array<double, n> first{}, second{};
    std::size_t reg1 = 0, reg2 = 0;
    sp::ldl_refine_result<double> ref1{}, ref2{};
    ok &= solve(first, reg1, ref1);

    // A new ρ: change the two diagonal values in the original matrix, carry them through the map, refactor
    rho = 1e10;
    for (std::size_t j = nx; j < n; ++j)
        vx[static_cast<std::size_t>(cp[j + 1]) - 1] = -1 / rho;   // the diagonal is the last entry of its column
    for (std::size_t e = 0; e < kkt.nnz(); ++e)
        pvx[static_cast<std::size_t>(map[e])] = vx[e];
    ok &= solve(second, reg2, ref2);
    //! [ldl]

    auto report = [&](const char* what, const std::array<double, n>& s, std::size_t reg, const sp::ldl_refine_result<double>& r) {
        double budget = 0, ret = 0;
        for (std::size_t i = 0; i < nx; ++i) {
            budget += s[i];
            ret += mu[i] * s[i];
        }
        std::printf("  %s: x = (%.4f, %.4f, %.4f, %.4f), Σx - 1 = %.1e, μᵀx - 0.08 = %.1e; %zu regularised, %zu corrections, residual %.1e\n", what, s[0], s[1], s[2],
                    s[3], budget - 1, ret - 0.08, reg, r.iterations, r.residual);
        return std::abs(budget - 1) < 1e-4 && std::abs(ret - 0.08) < 1e-4 && reg == 0 && r.residual <= 1e-14;
    };
    std::printf("  KKT order %zu, nnz(upper) %zu, nnz(L) %zu, AMD order (%d %d %d %d %d %d)\n", n, kkt.nnz(), symbolic ? symbolic->nnz : 0, perm[0], perm[1], perm[2],
                perm[3], perm[4], perm[5]);
    ok &= report("ρ = 1e6 ", first, reg1, ref1);
    ok &= report("ρ = 1e10", second, reg2, ref2);
    // With ρ larger the constraints hold more exactly, and the weights barely move
    double gap1 = 0, gap2 = 0, moved = 0;
    for (std::size_t i = 0; i < nx; ++i) {
        gap1 += first[i];
        gap2 += second[i];
        moved = std::max(moved, std::abs(first[i] - second[i]));
    }
    ok &= std::abs(gap2 - 1) < std::abs(gap1 - 1) && moved < 1e-3;
    std::printf(ok ? "ldl demo: ok\n" : "ldl demo: FAILED\n");
    return ok ? 0 : 1;
}
