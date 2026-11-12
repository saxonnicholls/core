//
//  types_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/linalg/types.hpp>: matrices and vectors as views of the caller's storage,
//  their transposes, the codes a batch reports and the pivot tolerance. Exits non-zero if a value is not
//  the one documented.
//

#include <snicholls/core/linalg/types.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <span>

int main()
{
    namespace la = snicholls::linalg;
    std::printf("snicholls-core linalg types demo\n");

    //! [views]
    // A 2 x 3 matrix in the caller's row-major storage; its transpose is the same storage read column-major.
    std::array<double, 6> storage{1, 2, 3, 4, 5, 6};
    const la::matrix_view<double, la::layout_right> a(storage.data(), {2, 3});
    const auto at = la::transposed(a);                      // matrix_view<double, layout_left>, 3 x 2
    const la::matrix_view<const double> strided = a;        // every layout converts to the canonical strided view
    const auto att = la::transposed(la::transposed(strided));
    // A std::span is a vector; a vector is a one-column matrix (a right-hand side).
    std::array<double, 3> x{1, 0, -1};
    const auto xv = la::as_vector(std::span<double>(x));
    const auto xc = la::as_column(xv);
    //! [views]
    std::printf("  a(1, 2) = %.0f, transposed(a)(2, 1) = %.0f, x as a %zu x %zu column\n", a(1, 2), at(2, 1), xc.extent(0), xc.extent(1));
    bool ok = at(2, 1) == 6 && la::transposed(at)(1, 2) == 6 && att(0, 1) == 2 && xc(2, 0) == -1;

    // A batch reports one status per system: ok, or the error with the same number.
    const la::status s = la::to_status(la::error::singular);
    std::printf("  error::singular reports status %d; ok is %d\n", static_cast<int>(s), static_cast<int>(la::status::ok));
    ok = ok && s == la::status::singular && la::to_status(la::error::dimension_mismatch) == la::status::dimension_mismatch;

    // A pivot below epsilon * n * scale counts as zero.
    const double tol = la::pivot_tolerance<double>(25, 4.0);
    std::printf("  pivot tolerance for n = 25, largest element 4: %.3g\n", tol);
    ok = ok && tol == 100.0 * std::numeric_limits<double>::epsilon();
    const double growth = la::ldlt_growth_limit<double>(25);
    std::printf("  unpivoted LDLT of order 25 accepts element growth up to %.0f\n", growth);
    ok = ok && growth > 25 * 8000.0 && growth < 25 * 8300.0;

    // A determinant as a sign and a logarithm: -6 without overflow.
    const la::signed_log_det<double> d{-1, std::log(6.0)};
    std::printf("  sign %d, log |det| %.4f, det %.4f\n", d.sign, d.log_abs, d.value());
    ok = ok && std::abs(d.value() + 6.0) < 1e-12;

    std::printf(ok ? "linalg types demo: ok\n" : "linalg types demo: FAILED\n");
    return ok ? 0 : 1;
}
