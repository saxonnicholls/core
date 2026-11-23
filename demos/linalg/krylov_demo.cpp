//
//  krylov_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/linalg/krylov.hpp>: the 1-D Poisson system solved by CG through a CSC matrix, with and
//  without a Jacobi preconditioner on a badly scaled copy; the same system through a matrix-free callable; a
//  symmetric indefinite KKT system by MINRES over its upper triangle; and CG's breakdown on an indefinite matrix.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/krylov.hpp>

#include <snicholls/core/sparse/csc.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace sp = snicholls::sparse;

//! [krylov]
// The n x n second-difference matrix tridiag(-1, 2, -1), assembled as CSC from triplets
struct poisson {
    std::vector<std::int32_t> ti, tj, col_ptr, row_idx, work;
    std::vector<double> tv, values;
    sp::csc_view<double> a;
    explicit poisson(std::size_t n) : col_ptr(n + 1), row_idx(3 * n), work(n), values(3 * n)
    {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i > 0 ? i - 1 : 0; j <= i + 1 && j < n; ++j) {
                ti.push_back(static_cast<std::int32_t>(i));
                tj.push_back(static_cast<std::int32_t>(j));
                tv.push_back(i == j ? 2.0 : -1.0);
            }
        a = {n, n, col_ptr, row_idx, values};
        (void)sp::from_triplets(n, n, std::span<const std::int32_t>(ti), std::span<const std::int32_t>(tj), std::span<const double>(tv), a, std::span<std::int32_t>(work));
    }
};

bool krylov_example()
{
    constexpr std::size_t n = 100;
    poisson p(n);
    std::vector<double> b(n, 1.0), x(n, 0.0), work(la::conjugate_gradient_workspace(n));

    // 1. CG through the CSC matrix: at most n iterations in exact arithmetic
    const auto r = la::conjugate_gradient(la::make_csc_operator(p.a), b, x, work, {1e-10});
    std::printf("  CG, CSC operator:        %-9s %3zu iterations, residual %.2e\n", r.converged() ? "converged" : "failed", r.iterations, r.residual_norm);

    // 2. The same product as a callable (a GPU kernel would sit here), warm-started from the answer: no iterations
    const auto matrix_free = la::make_function_operator<double>(n, n, [](std::span<const double> v, std::span<double> y) {
        for (std::size_t i = 0; i < v.size(); ++i)
            y[i] = 2 * v[i] - (i > 0 ? v[i - 1] : 0.0) - (i + 1 < v.size() ? v[i + 1] : 0.0);
    });
    const auto warm = la::conjugate_gradient(matrix_free, b, x, work, {1e-8});
    std::printf("  CG, matrix-free, warm:   %-9s %3zu iterations\n", warm.converged() ? "converged" : "failed", warm.iterations);

    // 3. Rows and columns scaled by 1 to 1e3: Jacobi (the diagonal) undoes most of the damage
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i)
        d[i] = std::pow(10.0, 3.0 * static_cast<double>((i * 37) % 100) / 99);   // a scrambled 1 to 1e3
    sp::scale(p.a, std::span<const double>(d), std::span<const double>(d));
    std::vector<double> diag(n), x0(n, 0.0), x1(n, 0.0);
    la::csc_diagonal(p.a, std::span<double>(diag));
    const la::krylov_options<double> tight{1e-8, 0, 20 * n};
    const auto plain = la::conjugate_gradient(la::make_csc_operator(p.a), b, x0, work, tight);
    const auto jacobi = la::conjugate_gradient(la::make_csc_operator(p.a), b, x1, work, tight, la::jacobi_preconditioner<double>{diag});
    std::printf("  scaled, unpreconditioned: %3zu iterations; Jacobi: %3zu\n", plain.iterations, jacobi.iterations);

    // 4. A KKT system [P Aᵀ; A 0], symmetric indefinite, by MINRES over its upper triangle:
    //    P = diag(2, 2), A = [1 1]: upper triangle columns {2}, {2}, {1, 1, 0}
    const std::array<std::int32_t, 4> kp{0, 1, 2, 4};
    const std::array<std::int32_t, 4> ki{0, 1, 0, 1};
    const std::array<double, 4> kv{2, 2, 1, 1};
    const sp::csc_view<const double, const std::int32_t> kkt{3, 3, kp, ki, kv};
    const std::array<double, 3> rhs{1, -1, 1};   // [-q; b]: minimise x₁² + x₂² - x₁ + x₂ subject to x₁ + x₂ = 1
    std::array<double, 3> sol{};
    std::array<double, la::minres_workspace(3)> mw{};
    const auto m = la::minres(la::make_symmetric_csc_operator(kkt), rhs, sol, mw, {1e-12});
    std::printf("  MINRES on the KKT system: x = (%.4f, %.4f), multiplier %.4f, %zu iterations\n", sol[0], sol[1], sol[2], m.iterations);

    // 5. CG on an indefinite matrix stops with breakdown rather than diverging
    const std::array<double, 4> indefinite{1, 0, 0, -1};
    const std::array<double, 2> b2{1, 1};
    std::array<double, 2> x2{};
    std::array<double, 8> w2{};
    const auto bad = la::conjugate_gradient(la::make_dense_operator(std::span<const double>(indefinite), 2, 2), b2, x2, w2);
    std::printf("  CG on diag(1, -1):       %s\n", bad.status == la::krylov_status::breakdown ? "breakdown" : "unexpected");

    return r.converged() && r.iterations <= n && warm.converged() && warm.iterations == 0 && jacobi.converged() && jacobi.iterations * 2 < plain.iterations && m.converged() &&
           std::abs(sol[0] - 1) < 1e-10 && std::abs(sol[1]) < 1e-10 && std::abs(sol[2] + 1) < 1e-10 && bad.status == la::krylov_status::breakdown;
}
//! [krylov]

} // namespace

int main()
{
    std::printf("snicholls-core krylov demo\n");
    const bool ok = krylov_example();
    std::printf(ok ? "krylov demo: ok\n" : "krylov demo: FAILED\n");
    return ok ? 0 : 1;
}
