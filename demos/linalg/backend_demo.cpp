//
//  backend_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/linalg/backend.hpp>: generic code written once against the free functions,
//  run with the reference backend and with a second backend chosen at run time by branching once between
//  two instantiations; every free function, with matrices in any layout and right-hand sides as vectors
//  or matrices; and the extent check. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace la = snicholls::linalg;

namespace {

//! [backend]
// Generic code: solve A x = b with whichever backend it is given. Every call is resolved at compile time.
template <typename Backend>
double solve_and_check(const Backend& backend)
{
    std::array<double, 9> a{4, -2, 1, -2, 4, -2, 1, -2, 4};   // symmetric positive definite
    std::array<double, 3> b{11, -16, 17};                      // A (1, -2, 3)
    const la::matrix_view<double, la::layout_right> A(a.data(), {3, 3});
    const auto x = la::as_vector(std::span<double>(b));
    if (!la::cholesky_factor(backend, A) || !la::cholesky_solve(backend, A, x))
        return -1.0;
    return std::abs(b[0] - 1) + std::abs(b[1] + 2) + std::abs(b[2] - 3);   // the error
}
//! [backend]

// A second backend: the reference backend's operations under another type, as a companion library's would be.
struct counting_backend : la::reference_backend {};
static_assert(la::backend<counting_backend, double>);

//! [run_time_choice]
// Choosing a backend at run time: branch once, at the top, between two instantiations of the generic code.
// Inside each branch every call is still resolved at compile time; nothing is erased or virtual.
double run(bool use_reference)
{
    if (use_reference)
        return solve_and_check(la::reference);
    return solve_and_check(counting_backend{});
}
//! [run_time_choice]

bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol; }

} // namespace

int main(int argc, char**)
{
    std::printf("snicholls-core linalg backend demo\n");
    const double error_reference = run(true);
    const double error_other = run(argc > 99);
    std::printf("  Cholesky solve, error with the reference backend %.1e, with the other %.1e\n", error_reference, error_other);
    bool ok = error_reference >= 0 && error_reference < 1e-12 && error_other == error_reference;

    // BLAS: vectors and matrices in any layout.
    std::array<double, 3> xs{1, 2, 3}, ys{4, 5, 6};
    const auto x = la::as_vector(std::span<double>(xs));
    const auto y = la::as_vector(std::span<double>(ys));
    const double d = la::dot(la::reference, x, y).unwrap();
    ok = ok && la::axpy(la::reference, 2.0, x, y) && ys[2] == 12;   // y = (6, 9, 12)
    std::array<double, 6> as{1, 2, 3, 4, 5, 6};
    const la::matrix_view<double, la::layout_right> A(as.data(), {2, 3});   // [[1 2 3] [4 5 6]]
    std::array<double, 2> gv{};
    ok = ok && la::gemv(la::reference, 1.0, A, x, 0.0, la::as_vector(std::span<double>(gv))) && gv[0] == 14 && gv[1] == 32;
    std::array<double, 4> cs{1, 1, 1, 1};
    const la::matrix_view<double, la::layout_left> C(cs.data(), {2, 2});
    ok = ok && la::gemm(la::reference, 1.0, A, la::transposed(A), 1.0, C) && C(0, 0) == 15 && C(1, 0) == 33;   // A A^T + 1
    std::array<double, 4> ks{};
    const la::matrix_view<double, la::layout_right> K(ks.data(), {2, 2});
    ok = ok && la::syrk(la::reference, 1.0, A, 0.0, K, la::triangle::lower) && K(1, 0) == 32 && K(0, 1) == 0;
    std::array<double, 4> ts{2, 0, 1, 4};
    std::array<double, 2> tb{2, 9};
    ok = ok && la::triangular_solve(la::reference, la::matrix_view<const double, la::layout_right>(ts.data(), {2, 2}),
                                    la::as_vector(std::span<double>(tb)), la::triangle::lower) &&
         near(tb[0], 1) && near(tb[1], 2);
    std::printf("  dot %.0f, gemv (%.0f, %.0f), gemm %.0f, syrk %.0f, triangular solve (%.0f, %.0f)\n", d, gv[0], gv[1], C(1, 1), K(1, 1), tb[0],
                tb[1]);
    ok = ok && d == 32;

    // LU, its inverse and determinant; a matrix right-hand side.
    std::array<double, 4> lus{4, 3, 6, 3};
    std::array<la::pivot_index, 2> p{};
    const la::matrix_view<double, la::layout_right> LU(lus.data(), {2, 2});
    ok = ok && la::lu_factor(la::reference, LU, std::span<la::pivot_index>(p));
    std::array<double, 4> rhs{10, 7, 12, 9};   // two right-hand sides, row-major: columns (10, 12) and (7, 9)
    ok = ok && la::lu_solve(la::reference, LU, std::span<const la::pivot_index>(p), la::matrix_view<double, la::layout_right>(rhs.data(), {2, 2}));
    std::array<double, 4> inv{};
    ok = ok && la::lu_inverse(la::reference, LU, std::span<const la::pivot_index>(p), la::matrix_view<double, la::layout_right>(inv.data(), {2, 2}));
    const auto det = la::lu_log_det(LU, std::span<const la::pivot_index>(p)).unwrap();
    std::printf("  LU: x = (%.0f, %.0f) and (%.0f, %.0f), det %.1f, inverse [[%.2f %.2f] [%.2f %.2f]]\n", rhs[0], rhs[2], rhs[1], rhs[3],
                det.value(), inv[0], inv[1], inv[2], inv[3]);
    ok = ok && near(rhs[0], 1) && near(rhs[2], 2) && near(rhs[1], 1) && near(rhs[3], 1) && near(det.value(), -6) && near(inv[0], -0.5) &&
         near(inv[1], 0.5) && near(inv[2], 1) && near(inv[3], -2.0 / 3.0);

    // Cholesky and LDLT determinants; LDLT of an indefinite matrix.
    std::array<double, 4> ch{4, 2, 2, 3};
    ok = ok && la::cholesky_factor(la::reference, la::matrix_view<double, la::layout_right>(ch.data(), {2, 2}));
    const double det_ch = la::cholesky_log_det(la::matrix_view<const double, la::layout_right>(ch.data(), {2, 2})).unwrap().value();
    std::array<double, 4> ld{1, 2, 2, 1};   // eigenvalues 3 and -1
    ok = ok && la::ldlt_factor(la::reference, la::matrix_view<double, la::layout_right>(ld.data(), {2, 2}));
    std::array<double, 2> lb{5, 4};   // x = (1, 2)
    ok = ok && la::ldlt_solve(la::reference, la::matrix_view<const double, la::layout_right>(ld.data(), {2, 2}), la::as_vector(std::span<double>(lb)));
    const auto det_ld = la::ldlt_log_det(la::matrix_view<const double, la::layout_right>(ld.data(), {2, 2})).unwrap();
    std::printf("  det by Cholesky %.1f, by LDLT %.1f (sign %d), LDLT solve (%.0f, %.0f)\n", det_ch, det_ld.value(), det_ld.sign, lb[0], lb[1]);
    ok = ok && near(det_ch, 8) && near(det_ld.value(), -3) && near(lb[0], 1) && near(lb[1], 2);

    // Least squares: the line through (0, 1), (1, 3), (2, 5) is 1 + 2 t.
    std::array<double, 6> design{1, 0, 1, 1, 1, 2};
    std::array<double, 3> obs{1, 3, 5};
    std::array<double, 2> tau{};
    const la::matrix_view<double, la::layout_right> D(design.data(), {3, 2});
    ok = ok && la::least_squares(la::reference, D, std::span<double>(tau), la::as_vector(std::span<double>(obs)));
    std::array<double, 6> q{};
    ok = ok && la::qr_q(la::reference, D, std::span<const double>(tau), la::matrix_view<double, la::layout_right>(q.data(), {3, 2}));
    std::array<double, 3> obs2{1, 3, 5};
    ok = ok && la::qr_solve(la::reference, D, std::span<const double>(tau), la::as_vector(std::span<double>(obs2)));
    std::printf("  least squares: intercept %.3f, slope %.3f; Q's first column (%.3f, %.3f, %.3f)\n", obs[0], obs[1], q[0], q[2], q[4]);
    ok = ok && near(obs[0], 1) && near(obs[1], 2) && near(obs2[1], 2) && near(q[0] * q[0] + q[2] * q[2] + q[4] * q[4], 1);
    std::array<double, 4> qr2{1, 0, 0, 1};
    ok = ok && la::qr_factor(la::reference, la::matrix_view<double, la::layout_right>(qr2.data(), {2, 2}), std::span<double>(tau));

    // Eigen and SVD.
    std::array<double, 4> sym{2, 1, 1, 2}, vec{};
    std::array<double, 2> w{};
    ok = ok && la::symmetric_eigen(la::reference, la::matrix_view<double, la::layout_right>(sym.data(), {2, 2}), std::span<double>(w),
                                   la::matrix_view<double, la::layout_right>(vec.data(), {2, 2}));
    std::array<double, 4> sv{3, 0, 0, -4}, vv{};
    std::array<double, 2> s{};
    ok = ok && la::svd(la::reference, la::matrix_view<double, la::layout_right>(sv.data(), {2, 2}), std::span<double>(s),
                       la::matrix_view<double, la::layout_right>(vv.data(), {2, 2}));
    std::printf("  eigenvalues (%.3f, %.3f), singular values (%.3f, %.3f)\n", w[0], w[1], s[0], s[1]);
    ok = ok && near(w[0], 1) && near(w[1], 3) && near(s[0], 4) && near(s[1], 3);

    // Extents that do not fit are an error, before the backend runs.
    const auto bad = la::gemv(la::reference, 1.0, A, y, 0.0, x);
    std::printf("  gemv of a 2 x 3 matrix into a vector of 3: %s\n", bad ? "accepted" : "dimension_mismatch");
    ok = ok && !bad && bad.err() == la::error::dimension_mismatch;

    // An output that starts where an input does would read what it wrote: reported, nothing written.
    const auto alias = la::gemm(la::reference, 1.0, K, K, 0.0, K);
    std::printf("  gemm with C = A = B: %s\n", alias ? "accepted" : "aliased");
    ok = ok && !alias && alias.err() == la::error::aliased && K(1, 0) == 32;

    std::printf(ok ? "linalg backend demo: ok\n" : "linalg backend demo: FAILED\n");
    return ok ? 0 : 1;
}
