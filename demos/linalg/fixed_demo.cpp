//
//  fixed_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/linalg/fixed.hpp>: one Kalman update of a 4-state constant-velocity target from a 2-D
//  position fix, written with fixed-size matrices, then every other operation of the header (blocks, LU, the eigen
//  decomposition, tria, views) and each error. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/fixed.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

namespace {

namespace la = snicholls::linalg;

bool check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

template <std::size_t R, std::size_t C>
bool near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol)
{
    return la::norm(a - b) <= tol;
}

} // namespace

int main()
{
    std::printf("snicholls-core linalg fixed demo\n");
    bool ok = true;

    //! [fixed]
    // A target (x, vx, y, vy) after one second of constant velocity, measured in position only
    using m4 = la::fixed_matrix<double, 4, 4>;
    using m2 = la::fixed_matrix<double, 2, 2>;
    const la::fixed_matrix<double, 2, 2> cv{{1, 1, 0, 1}};
    const m4 f = la::block_diagonal(cv, cv);                         // the transition
    const la::fixed_matrix<double, 2, 4> h{{1, 0, 0, 0, 0, 0, 1, 0}}; // the measurement model
    const auto x0 = la::make_vector(0.0, 1.0, 0.0, 2.0);
    const m4 p0 = la::diagonal_matrix(la::make_vector(4.0, 1.0, 4.0, 1.0));
    const m4 q = m4::identity() * 0.01;
    const m2 r = m2::identity() * 0.25;

    const auto x = f * x0;                          // predict
    const m4 p = la::sandwich(f, p0) + q;           // F P Fᵀ + Q, exactly symmetric
    const auto z = la::make_vector(1.2, 1.9);
    const auto s = la::sandwich(h, p) + r;          // the innovation covariance
    const auto ls = la::cholesky(s);                // a result: S might not be positive definite
    if (!ls)
        return 1;
    // K = P Hᵀ S⁻¹, by two triangular solves rather than an inverse
    const auto k = la::transpose(la::cholesky_solve(*ls, h * p));
    const auto xu = x + k * (z - h * x);
    const m4 pu = la::symmetrise(p - k * s * la::transpose(k));
    const double log_det_s = la::cholesky_log_det(*ls);
    //! [fixed]
    std::printf("  predicted (%.2f, %.2f, %.2f, %.2f), updated (%.3f, %.3f, %.3f, %.3f), log det S %.4f\n", x[0], x[1], x[2], x[3], xu[0], xu[1], xu[2], xu[3], log_det_s);
    ok &= check(near(x, la::make_vector(1.0, 1.0, 2.0, 2.0), 0), "prediction moves by the velocity");
    ok &= check(near(*ls * la::transpose(*ls), s, 1e-12), "cholesky: L L^T = S");
    ok &= check(std::abs(log_det_s - std::log(la::determinant(s))) < 1e-12, "cholesky_log_det = log det");
    // The scalar Kalman gain per axis: P_xx / (P_xx + R) with P_xx = 4 + 1 + 0.01
    const double gain = 5.01 / (5.01 + 0.25);
    ok &= check(std::abs(k(0, 0) - gain) < 1e-12 && std::abs(xu[0] - (1.0 + gain * 0.2)) < 1e-12, "gain and update match the scalar filter");
    ok &= check(la::trace(pu) < la::trace(p), "the update shrinks the covariance");

    // Arithmetic and blocks
    const la::fixed_matrix<double, 2, 3> a{{1, 2, 3, 4, 5, 6}};
    ok &= check(la::transpose(a)(2, 1) == 6 && (a * 2.0)(1, 1) == 10 && (2.0 * a) == a * 2.0 && (a / 2.0)(0, 1) == 1, "transpose, scalar products, quotient");
    auto b = a;
    b += a;
    b -= a;
    ok &= check(b == a && (a + a - a) == a && (-a)(0, 0) == -1, "sum, difference, in place, negation");
    ok &= check(la::dot(la::make_vector(1.0, 2.0), la::make_vector(3.0, 4.0)) == 11 && la::squared_norm(a) == 91 && la::norm(la::make_vector(3.0, 4.0)) == 5,
                "dot, squared_norm, norm");
    ok &= check(la::outer(la::make_vector(1.0, 2.0), la::make_vector(3.0, 4.0)) == m2{{3, 4, 6, 8}}, "outer");
    ok &= check(la::diagonal_of(p0) == la::make_vector(4.0, 1.0, 4.0, 1.0) && (la::block<0, 0, 2, 2>(f)) == cv, "diagonal and block");
    m4 g = m4::zero();
    la::set_block<2, 2>(g, cv);
    ok &= check(g(2, 3) == 1 && g(0, 0) == 0 && la::stack(la::make_vector(1.0), la::make_vector(2.0, 3.0)) == la::make_vector(1.0, 2.0, 3.0), "set_block and stack");
    ok &= check(m2::zero() == m2{} && m2::identity()(1, 1) == 1 && m2::rows == 2 && m2::cols == 2 && m2::size == 4, "zero, identity, the sizes");

    // LU: solve, inverse and determinant, with a row exchange
    const la::fixed_matrix<double, 3, 3> m{{0, 2, 1, 1, 1, 0, 2, 0, 3}};
    const auto xs = la::solve(m, la::make_vector(5.0, 3.0, 7.0));
    const auto mi = la::inverse(m);
    std::printf("  det [[0,2,1],[1,1,0],[2,0,3]] = %.4f\n", la::determinant(m));
    ok &= check(xs.is_ok() && near(m * *xs, la::make_vector(5.0, 3.0, 7.0), 1e-14), "solve with pivoting");
    ok &= check(mi.is_ok() && near(m * *mi, la::fixed_matrix<double, 3, 3>::identity(), 1e-14), "inverse");
    ok &= check(std::abs(la::determinant(m) + 8) < 1e-14, "determinant -8 (by cofactors: 0*3 - 2*3 + 1*(-2))");
    const auto spdi = la::spd_inverse(s);
    ok &= check(spdi.is_ok() && near(s * *spdi, m2::identity(), 1e-12), "spd_inverse");
    const auto y = la::lower_solve(*ls, z);
    ok &= check(near(la::lower_transpose_solve(*ls, y), la::cholesky_solve(*ls, z), 1e-15), "lower_solve then lower_transpose_solve");

    // The symmetric eigen decomposition: [[2, 1], [1, 2]] has eigenvalues 1 and 3
    const auto e = la::symmetric_eigen(m2{{2, 1, 1, 2}});
    ok &= check(e.is_ok() && std::abs(e->values[0] - 1) < 1e-15 && std::abs(e->values[1] - 3) < 1e-15, "symmetric_eigen: 1 and 3, ascending");
    ok &= check(near(e->vectors * la::diagonal_matrix(e->values) * la::transpose(e->vectors), m2{{2, 1, 1, 2}}, 1e-14), "V diag(w) V^T = A");

    // tria: the square-root filter's triangularisation, S Sᵀ = A Aᵀ
    const la::fixed_matrix<double, 2, 4> sq{{1, 2, 0.5, 0, 0, 1, 0, 0.5}};
    const auto st = la::tria(sq);
    ok &= check(st(0, 1) == 0 && st(0, 0) >= 0 && st(1, 1) >= 0 && near(st * la::transpose(st), sq * la::transpose(sq), 1e-14), "tria: lower, S S^T = A A^T");

    // A view for the run-time-sized backend
    auto vm = a;
    const auto v = la::view(vm);
    const auto cvw = la::view(a);
    v(1, 2) = 60;
    ok &= check(vm(1, 2) == 60 && cvw(1, 2) == 6 && v.extent(0) == 2 && v.extent(1) == 3, "view shares the storage");

    // The errors
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ok &= check(la::cholesky(m2{{1, 2, 2, 1}}).unwrap_err_unchecked() == la::error::not_positive_definite, "cholesky of an indefinite matrix");
    ok &= check(la::spd_inverse(m2{{0, 0, 0, 0}}).unwrap_err_unchecked() == la::error::not_positive_definite, "spd_inverse of zero");
    ok &= check(la::solve(m2{{1, 2, 2, 4}}, z).unwrap_err_unchecked() == la::error::singular && la::inverse(m2{}).is_err() && la::determinant(m2{{1, 2, 2, 4}}) == 0,
                "solve, inverse and determinant of a singular matrix");
    ok &= check(la::symmetric_eigen(m2{{1, nan, nan, 1}}).unwrap_err_unchecked() == la::error::did_not_converge, "symmetric_eigen of NaN");

    // And at compile time
    constexpr la::fixed_matrix<double, 3, 3> c{{4, 12, -16, 12, 37, -43, -16, -43, 98}};
    constexpr auto cl = la::cholesky(c);
    static_assert(cl.is_ok() && (*cl)(2, 0) == -8 && (*cl)(2, 2) == 3);
    ok &= check((*cl)(1, 1) == 1, "cholesky in a constant expression");

    std::printf(ok ? "linalg fixed demo: ok\n" : "linalg fixed demo: FAILED\n");
    return ok ? 0 : 1;
}
