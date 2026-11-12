//
//  reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/linalg/reference.hpp>: a solve done by the compiler, the factor-model kind of
//  eigendecomposition, every outcome a factorisation reports (singular, not positive definite, rank
//  deficient), and the same solve through a row-major, a column-major and a strided matrix. Exits non-zero
//  if a result is not the one documented.
//

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <span>

namespace la = snicholls::linalg;

namespace {

//! [reference]
// The reference backend is constexpr: the compiler can solve a small system and check it.
constexpr std::array<double, 3> solve_at_compile_time()
{
    std::array<double, 9> a{2, 1, 0, 1, 3, 1, 0, 1, 4};
    std::array<double, 3> b{4, 10, 14};   // A (1, 2, 3)
    std::array<la::pivot_index, 3> p{};
    const la::matrix_view<double, la::layout_right> A(a.data(), {3, 3});
    if (!la::lu_factor(la::reference, A, std::span<la::pivot_index>(p)) ||
        !la::lu_solve(la::reference, A, std::span<const la::pivot_index>(p), la::as_vector(std::span<double>(b))))
        return {};
    return b;
}
static_assert(solve_at_compile_time()[2] > 2.999999 && solve_at_compile_time()[2] < 3.000001);

// At run time: the principal components of a 3 x 3 covariance matrix.
std::array<double, 3> principal_variances()
{
    std::array<double, 9> cov{4.0, 1.2, 0.4, 1.2, 2.0, 0.3, 0.4, 0.3, 1.0};
    std::array<double, 9> loadings{};
    std::array<double, 3> variances{};
    const auto r = la::symmetric_eigen(la::reference, la::matrix_view<double, la::layout_right>(cov.data(), {3, 3}),
                                       std::span<double>(variances), la::matrix_view<double, la::layout_right>(loadings.data(), {3, 3}));
    return r ? variances : std::array<double, 3>{};
}
//! [reference]

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

const char* name(la::error e)
{
    switch (e) {
    case la::error::singular: return "singular";
    case la::error::not_positive_definite: return "not_positive_definite";
    case la::error::did_not_converge: return "did_not_converge";
    case la::error::dimension_mismatch: return "dimension_mismatch";
    case la::error::aliased: return "aliased";
    }
    return "?";
}

} // namespace

int main()
{
    std::printf("snicholls-core linalg reference backend demo\n");
    const auto x = solve_at_compile_time();
    std::printf("  solved by the compiler: (%.6f, %.6f, %.6f)\n", x[0], x[1], x[2]);
    bool ok = near(x[0], 1, 1e-12) && near(x[1], 2, 1e-12) && near(x[2], 3, 1e-12);

    const auto v = principal_variances();
    std::printf("  principal variances (ascending): %.4f %.4f %.4f, total %.4f (the trace is 7)\n", v[0], v[1], v[2], v[0] + v[1] + v[2]);
    ok = ok && near(v[0] + v[1] + v[2], 7.0, 1e-12) && v[0] < v[1] && v[1] < v[2];

    // Every outcome a factorisation reports, as a value.
    std::array<double, 4> singular{1, 2, 2, 4};
    std::array<la::pivot_index, 2> p{};
    const auto lu = la::lu_factor(la::reference, la::matrix_view<double, la::layout_right>(singular.data(), {2, 2}), std::span<la::pivot_index>(p));
    std::array<double, 4> indefinite{1, 2, 2, 1};
    const auto ch = la::cholesky_factor(la::reference, la::matrix_view<double, la::layout_right>(indefinite.data(), {2, 2}));
    std::array<double, 4> zero{};
    const auto ld = la::ldlt_factor(la::reference, la::matrix_view<double, la::layout_right>(zero.data(), {2, 2}));
    std::array<double, 6> collinear{1, 2, 2, 4, 3, 6};
    std::array<double, 3> rhs{1, 2, 3};
    std::array<double, 2> tau{};
    const auto ls = la::least_squares(la::reference, la::matrix_view<double, la::layout_right>(collinear.data(), {3, 2}), std::span<double>(tau),
                                      la::as_vector(std::span<double>(rhs)));
    std::array<double, 4> zero_diagonal{1, 0, 1, 0};
    std::array<double, 2> b2{1, 1};
    const auto tr = la::triangular_solve(la::reference, la::matrix_view<const double, la::layout_right>(zero_diagonal.data(), {2, 2}),
                                         la::as_vector(std::span<double>(b2)), la::triangle::lower);
    std::printf("  LU of [[1 2] [2 4]]: %s; Cholesky of [[1 2] [2 1]]: %s; LDLT of 0: %s; least squares, collinear columns: %s; "
                "triangular, zero diagonal: %s\n",
                name(*lu.err()), name(*ch.err()), name(*ld.err()), name(*ls.err()), name(*tr.err()));
    ok = ok && lu.err() == la::error::singular && ch.err() == la::error::not_positive_definite && ld.err() == la::error::singular &&
         ls.err() == la::error::singular && tr.err() == la::error::singular;

    // The same 2 x 2 solve through a row-major, a column-major and a strided (no unit stride) matrix.
    std::array<double, 4> r{4, 3, 6, 3};
    std::array<double, 4> c{4, 6, 3, 3};
    std::array<double, 8> s{4, 0, 3, 0, 6, 0, 3, 0};
    std::array<double, 2> br{10, 12}, bc{10, 12}, bs{10, 12};
    std::array<la::pivot_index, 2> pr{}, pc{}, ps{};
    const la::matrix_view<double, la::layout_right> R(r.data(), {2, 2});
    const la::matrix_view<double, la::layout_left> C(c.data(), {2, 2});
    const la::matrix_view<double> S(s.data(), {2, 2}, {4, 2});
    ok = ok && la::lu_factor(la::reference, R, std::span<la::pivot_index>(pr)) && la::lu_factor(la::reference, C, std::span<la::pivot_index>(pc)) &&
         la::lu_factor(la::reference, S, std::span<la::pivot_index>(ps)) &&
         la::lu_solve(la::reference, R, std::span<const la::pivot_index>(pr), la::as_vector(std::span<double>(br))) &&
         la::lu_solve(la::reference, C, std::span<const la::pivot_index>(pc), la::as_vector(std::span<double>(bc))) &&
         la::lu_solve(la::reference, S, std::span<const la::pivot_index>(ps), la::as_vector(std::span<double>(bs)));
    std::printf("  row-major (%.0f, %.0f), column-major (%.0f, %.0f), strided (%.0f, %.0f)\n", br[0], br[1], bc[0], bc[1], bs[0], bs[1]);
    ok = ok && br == bc && br == bs && near(br[0], 1, 1e-15) && near(br[1], 2, 1e-15);

    // A badly scaled matrix: the default tolerance (relative to the largest element) calls diag(1e16, 1) singular;
    // LAPACK's rule, only an exact zero, solves it.
    std::array<double, 4> scaled{1e16, 0, 0, 1}, scaled2 = scaled;
    std::array<double, 2> sb{1e16, 1};
    const auto by_default = la::lu_factor(la::reference, la::matrix_view<double, la::layout_right>(scaled.data(), {2, 2}), std::span<la::pivot_index>(p));
    const la::reference_backend lapack_rule{.relative_tolerance = 0.0};
    ok = ok && !by_default && la::lu_factor(lapack_rule, la::matrix_view<double, la::layout_right>(scaled2.data(), {2, 2}), std::span<la::pivot_index>(p)) &&
         la::lu_solve(lapack_rule, la::matrix_view<double, la::layout_right>(scaled2.data(), {2, 2}), std::span<const la::pivot_index>(p),
                      la::as_vector(std::span<double>(sb))) &&
         sb[0] == 1 && sb[1] == 1;
    std::printf("  diag(1e16, 1): %s by default, solved with relative_tolerance = 0\n", by_default ? "solved" : "singular");

    // float works the same way, with float's tolerances.
    std::array<float, 4> f{2, 1, 1, 2};
    std::array<float, 2> fb{3, 3};
    ok = ok && la::cholesky_factor(la::reference, la::matrix_view<float, la::layout_right>(f.data(), {2, 2})) &&
         la::cholesky_solve(la::reference, la::matrix_view<const float, la::layout_right>(f.data(), {2, 2}), la::as_vector(std::span<float>(fb)));
    std::printf("  in float: (%.6f, %.6f); at most %d Jacobi sweeps\n", static_cast<double>(fb[0]), static_cast<double>(fb[1]),
                la::reference_backend::max_sweeps);
    ok = ok && std::abs(fb[0] - 1.0f) < 1e-6f && std::abs(fb[1] - 1.0f) < 1e-6f;

    std::printf(ok ? "linalg reference demo: ok\n" : "linalg reference demo: FAILED\n");
    return ok ? 0 : 1;
}
