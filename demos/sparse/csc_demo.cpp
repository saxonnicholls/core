//
//  csc_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/sparse/csc.hpp>: a constraint matrix assembled from triplets in any order (duplicates
//  summed), checked, multiplied both ways, transposed, scaled by its norms (one step of equilibration), and a symmetric
//  matrix stored as its upper triangle, multiplied and permuted. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/sparse/csc.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace {

namespace sp = snicholls::sparse;
using I = std::int32_t;

bool near(double a, double b) { return std::abs(a - b) < 1e-12; }

} // namespace

int main()
{
    std::printf("snicholls-core csc demo\n");
    bool ok = true;

    //! [csc]
    // A 3 x 4 constraint matrix from triplets in any order; the two (0, 1) entries are summed to 2
    //   [ 1  2  .  . ]
    //   [ .  .  3  . ]
    //   [ 4  .  .  5 ]
    const std::array<I, 6> ti{2, 0, 1, 0, 2, 0}, tj{3, 1, 2, 0, 0, 1};
    const std::array<double, 6> tv{5, 1.5, 3, 1, 4, 0.5};
    sp::csc_matrix<double, 3, 4, 6> a{};
    auto av = a.view();
    std::array<I, 4> work{};
    const auto stored = sp::from_triplets(3, 4, std::span<const I>(ti), std::span<const I>(tj), std::span<const double>(tv), av, std::span<I>(work));
    // stored == 5 entries, is_valid(a.view())

    // y = A x and z = Aᵀ y
    const std::array<double, 4> x{1, 1, 1, 1};
    std::array<double, 3> y{};
    std::array<double, 4> z{};
    sp::spmv(std::as_const(a).view(), std::span<const double>(x), std::span<double>(y));             // (3, 3, 9)
    sp::spmv_transpose(std::as_const(a).view(), std::span<const double>(y), std::span<double>(z));   // (39, 6, 9, 45)

    // Aᵀ as a matrix of its own
    sp::csc_matrix<double, 4, 3, 6> t{};
    auto tview = t.view();
    std::array<I, 3> twork{};
    const bool transposed = sp::transpose(std::as_const(a).view(), tview, std::span<I>(twork)).is_ok();

    // One step of Ruiz equilibration: divide every row and column by the square root of its largest magnitude
    std::array<double, 3> rn{};
    std::array<double, 4> cn{};
    sp::row_inf_norms(std::as_const(a).view(), std::span<double>(rn));      // (2, 3, 5)
    sp::column_inf_norms(std::as_const(a).view(), std::span<double>(cn));   // (4, 2, 3, 5)
    std::array<double, 3> r{};
    std::array<double, 4> c{};
    for (std::size_t i = 0; i < 3; ++i)
        r[i] = 1 / std::sqrt(rn[i]);
    for (std::size_t j = 0; j < 4; ++j)
        c[j] = 1 / std::sqrt(cn[j]);
    sp::scale(a.view(), std::span<const double>(r), std::span<const double>(c));

    // A symmetric 3 x 3 matrix as its upper triangle, its product, and P S Pᵀ for the order (2, 0, 1)
    //   [ 4  1  . ]
    //   [ 1  3  2 ]
    //   [ .  2  5 ]
    const sp::csc_matrix<double, 3, 3, 5> s{{0, 1, 3, 5}, {0, 0, 1, 1, 2}, {4, 1, 3, 2, 5}};
    const std::array<double, 3> u{1, 2, 3};
    std::array<double, 3> su{};
    sp::spmv_symmetric_upper(s.view(), std::span<const double>(u), std::span<double>(su));   // (6, 13, 19)
    const std::array<I, 3> perm{2, 0, 1};
    sp::csc_matrix<double, 3, 3, 5> ps{};
    auto psv = ps.view();
    std::array<I, 6> pwork{};
    const bool permuted = sp::symmetric_permute_upper(s.view(), std::span<const I>(perm), psv, std::span<I>(pwork)).is_ok();
    std::array<double, 9> dense{};
    sp::to_dense(std::as_const(ps).view(), std::span<double>(dense));   // upper triangle of [5 . 2; . 4 1; 2 1 3]
    //! [csc]

    std::printf("  assembled %zu entries from 6 triplets, valid: %s\n", stored ? *stored : 0, sp::is_valid(std::as_const(a).view()) ? "yes" : "no");
    std::printf("  A x  = (%g, %g, %g)\n", y[0], y[1], y[2]);
    std::printf("  Aᵀ y = (%g, %g, %g, %g)\n", z[0], z[1], z[2], z[3]);
    std::printf("  row norms (%g, %g, %g), column norms (%g, %g, %g, %g)\n", rn[0], rn[1], rn[2], cn[0], cn[1], cn[2], cn[3]);
    std::printf("  S u  = (%g, %g, %g)\n", su[0], su[1], su[2]);
    std::printf("  P S Pᵀ, upper triangle: [%g %g %g; %g %g %g; %g %g %g]\n", dense[0], dense[1], dense[2], dense[3], dense[4], dense[5], dense[6], dense[7], dense[8]);
    ok &= stored && *stored == 5 && sp::is_valid(std::as_const(a).view());
    ok &= y == std::array<double, 3>{3, 3, 9} && z == std::array<double, 4>{39, 6, 9, 45};
    ok &= transposed && t.nnz() == 5 && t.col_ptr == std::array<I, 4>{0, 2, 3, 5};
    ok &= rn == std::array<double, 3>{2, 3, 5} && cn == std::array<double, 4>{4, 2, 3, 5};
    ok &= near(a.values[0], 1 / std::sqrt(2.0 * 4.0)) && near(a.values[4], 5 / std::sqrt(5.0 * 5.0));   // A(0,0) and A(2,3) after scaling
    ok &= su == std::array<double, 3>{6, 13, 19};
    ok &= permuted && dense == std::array<double, 9>{5, 0, 2, 0, 4, 1, 0, 0, 3};
    std::printf(ok ? "csc demo: ok\n" : "csc demo: FAILED\n");
    return ok ? 0 : 1;
}
