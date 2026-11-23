//
//  ordering_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/sparse/ordering.hpp>: the fill of L for a 30 x 30 grid Laplacian and for an arrow matrix
//  in natural order, reverse Cuthill-McKee order and approximate minimum degree order; then a vector carried into the
//  permuted numbering and back. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/sparse/ordering.hpp>

#include <snicholls/core/sparse/csc.hpp>
#include <snicholls/core/sparse/ldl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace sp = snicholls::sparse;
using I = std::int32_t;

// The upper triangle of a symmetric pattern, assembled from (i ≤ j) triplets
struct upper {
    std::size_t n = 0;
    std::vector<I> ti, tj, cp, ri, work;
    std::vector<double> tv, vx;
    void add(std::size_t i, std::size_t j, double v)
    {
        ti.push_back(static_cast<I>(i < j ? i : j));
        tj.push_back(static_cast<I>(i < j ? j : i));
        tv.push_back(v);
    }
    sp::csc_view<const double, const I> assemble()
    {
        cp.resize(n + 1);
        ri.resize(ti.size());
        vx.resize(ti.size());
        work.resize(n);
        sp::csc_view<double, I> out{0, 0, cp, ri, vx};
        (void)sp::from_triplets(n, n, std::span<const I>(ti), std::span<const I>(tj), std::span<const double>(tv), out, std::span<I>(work));
        return out;
    }
};

//! [ordering]
// nnz(L) of the matrix whose upper triangle is a, in the order perm (perm[new] = old)
std::size_t fill(sp::csc_view<const double, const I> a, std::span<const I> perm)
{
    const std::size_t n = a.cols, nnz = a.nnz();
    std::vector<I> cp(n + 1), ri(nnz), work(2 * n), parent(n), counts(n), lp(n + 1);
    std::vector<double> vx(nnz);
    sp::csc_view<double, I> b{0, 0, cp, ri, vx};
    if (!sp::symmetric_permute_upper(a, perm, b, std::span<I>(work)))
        return 0;
    const auto s = sp::ldl_symbolic(b, std::span<I>(parent), std::span<I>(counts), std::span<I>(lp), std::span<I>(work).first(n));
    return s ? s->nnz : 0;
}

struct fills {
    std::size_t natural, rcm, amd;
};

fills compare(sp::csc_view<const double, const I> a)
{
    const std::size_t n = a.cols;
    std::vector<I> natural(n), rcm(n), amd(n);
    for (std::size_t i = 0; i < n; ++i)
        natural[i] = static_cast<I>(i);
    std::vector<I> rcm_work(sp::rcm_workspace_size(n, a.nnz())), amd_work(sp::amd_workspace_size(n, a.nnz()));
    const bool ok = sp::reverse_cuthill_mckee(a, std::span<I>(rcm), std::span<I>(rcm_work)).is_ok() && sp::amd(a, std::span<I>(amd), std::span<I>(amd_work)).is_ok();
    if (!ok || !sp::is_permutation(std::span<const I>(amd)) || !sp::is_permutation(std::span<const I>(rcm)))
        return {};
    return {fill(a, natural), fill(a, rcm), fill(a, amd)};
}
//! [ordering]

} // namespace

int main()
{
    std::printf("snicholls-core ordering demo\n");
    bool ok = true;

    // The 5-point Laplacian on a 30 x 30 grid
    upper grid;
    const std::size_t k = 30;
    grid.n = k * k;
    for (std::size_t x = 0; x < k; ++x)
        for (std::size_t y = 0; y < k; ++y) {
            const std::size_t i = x * k + y;
            grid.add(i, i, 4);
            if (x + 1 < k)
                grid.add(i, i + k, -1);
            if (y + 1 < k)
                grid.add(i, i + 1, -1);
        }
    const fills g = compare(grid.assemble());
    std::printf("  grid 30 x 30:     nnz(L) natural %6zu, reverse Cuthill-McKee %6zu, AMD %6zu\n", g.natural, g.rcm, g.amd);
    ok &= g.amd > 0 && 2 * g.amd < g.natural && g.amd < g.rcm;

    // An arrow: row 0 joined to every other row. Natural order fills L completely; AMD orders the head last
    upper arrow;
    arrow.n = 200;
    for (std::size_t i = 0; i < arrow.n; ++i) {
        arrow.add(i, i, 300);
        if (i > 0)
            arrow.add(0, i, 1);
    }
    const auto av = arrow.assemble();
    const fills a = compare(av);
    std::printf("  arrow, order 200: nnz(L) natural %6zu, reverse Cuthill-McKee %6zu, AMD %6zu\n", a.natural, a.rcm, a.amd);
    ok &= a.natural == 200 * 199 / 2 && a.amd == 199;

    // A right-hand side into the AMD numbering, and a solution back
    std::vector<I> perm(arrow.n), work(sp::amd_workspace_size(arrow.n, av.nnz())), inv(arrow.n);
    ok &= sp::amd(av, std::span<I>(perm), std::span<I>(work)).is_ok();
    ok &= sp::inverse_permutation(std::span<const I>(perm), std::span<I>(inv)).is_ok();
    std::vector<double> b(arrow.n), pb(arrow.n), back(arrow.n);
    for (std::size_t i = 0; i < arrow.n; ++i)
        b[i] = static_cast<double>(i);
    sp::permute(std::span<const I>(perm), std::span<const double>(b), std::span<double>(pb));
    sp::inverse_permute(std::span<const I>(perm), std::span<const double>(pb), std::span<double>(back));
    std::printf("  the arrow head (row 0) is AMD's row %d; b[0] lands at pb[%d] = %g, and comes back: %s\n", inv[0], inv[0], pb[static_cast<std::size_t>(inv[0])],
                back == b ? "yes" : "no");
    ok &= inv[0] == static_cast<I>(arrow.n - 1) && pb[arrow.n - 1] == 0 && back == b;

    std::printf(ok ? "ordering demo: ok\n" : "ordering demo: FAILED\n");
    return ok ? 0 : 1;
}
