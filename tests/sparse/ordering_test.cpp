//
//  ordering_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/sparse/ordering.hpp>: amd and reverse_cuthill_mckee give permutations on every shape
//  (empty, one, diagonal, complete, disconnected, with entries below the diagonal); the fill of L (ldl_symbolic's
//  nnz) in AMD order against natural order, reverse Cuthill-McKee and an exact minimum degree written here, on grids,
//  arrows, random SPD and random KKT matrices; dense rows ordered last; the options; compaction of the quotient graph
//  (a workspace of exactly the documented size against a generous one); errors; the permutation helpers; and amd in a
//  constant expression.
//

#include <snicholls/core/sparse/ordering.hpp>

#include <snicholls/core/sparse/csc.hpp>
#include <snicholls/core/sparse/ldl.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace sp = snicholls::sparse;
namespace st = snicholls::stats;
using I = std::int32_t;

// A symmetric matrix as upper-triangle triplets, assembled into owned CSC storage
struct matrix {
    std::size_t n = 0;
    std::vector<I> cp, ri;
    std::vector<double> vx;
    [[nodiscard]] sp::csc_view<const double, const I> view() const { return {n, n, cp, ri, vx}; }
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
        const auto r = sp::from_triplets(n, n, std::span<const I>(i), std::span<const I>(j), std::span<const double>(v), out, std::span<I>(work));
        EXPECT_TRUE(r.is_ok());
        m.ri.resize(out.nnz());
        m.vx.resize(out.nnz());
        return m;
    }
};

triplets grid(std::size_t k)
{
    triplets t;
    t.n = k * k;
    for (std::size_t x = 0; x < k; ++x)
        for (std::size_t y = 0; y < k; ++y) {
            const std::size_t i = x * k + y;
            t.add(i, i, 4);
            if (x + 1 < k)
                t.add(i, i + k, -1);
            if (y + 1 < k)
                t.add(i, i + 1, -1);
        }
    return t;
}

// Node `head` is joined to every other node; the rest are a chain
triplets arrow(std::size_t n, std::size_t head)
{
    triplets t;
    t.n = n;
    for (std::size_t i = 0; i < n; ++i) {
        t.add(i, i, static_cast<double>(n) + 2);
        if (i != head)
            t.add(head, i, 1);
        if (i + 1 < n && i != head && i + 1 != head)
            t.add(i, i + 1, 0.5);
    }
    return t;
}

triplets random_spd(std::size_t n, std::size_t per_column, std::uint64_t seed)
{
    st::wyrand g(seed);
    triplets t;
    t.n = n;
    std::vector<double> diag(n, 1.0);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t e = 0; e < per_column; ++e) {
            const std::size_t i = static_cast<std::size_t>(st::uniform_index(g, n));
            if (i == j)
                continue;
            const double x = st::uniform_real(g) - 0.5;
            t.add(i, j, x);
            diag[i] += x < 0 ? -x : x;
            diag[j] += x < 0 ? -x : x;
        }
    for (std::size_t j = 0; j < n; ++j)
        t.add(j, j, diag[j]);
    return t;
}

// [P + σI  Aᵀ; A  -ρ⁻¹I]: P sparse positive semidefinite (diagonally dominant), A m x nx with a few entries a row
triplets random_kkt(std::size_t nx, std::size_t m, std::size_t per_row, std::uint64_t seed, double sigma = 1e-6, double rho = 0.1)
{
    triplets t = random_spd(nx, 2, seed);
    for (std::size_t k = 0; k < t.i.size(); ++k)
        if (t.i[k] == t.j[k])
            t.v[k] += sigma;
    t.n = nx + m;
    st::wyrand g(seed ^ 0x5eed);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t e = 0; e < per_row; ++e)
            t.add(static_cast<std::size_t>(st::uniform_index(g, nx)), nx + r, st::uniform_real(g) - 0.5);
        t.add(nx + r, nx + r, -1.0 / rho);
    }
    return t;
}

std::vector<I> natural(std::size_t n)
{
    std::vector<I> p(n);
    for (std::size_t i = 0; i < n; ++i)
        p[i] = static_cast<I>(i);
    return p;
}

// The entries strictly above the diagonal
std::size_t strict_upper(const matrix& a)
{
    std::size_t m = 0;
    for (std::size_t j = 0; j < a.n; ++j)
        for (std::size_t q = static_cast<std::size_t>(a.cp[j]); q < static_cast<std::size_t>(a.cp[j + 1]); ++q)
            m += static_cast<std::size_t>(a.ri[q]) < j ? 1 : 0;
    return m;
}

// AMD with the recommended workspace, a larger one, or (least = true) the least amd accepts
std::vector<I> amd_order(const matrix& a, sp::amd_options o = {}, std::size_t extra = 0, bool least = false)
{
    std::vector<I> perm(a.n, -7), work(least ? 13 * a.n + 2 + 2 * strict_upper(a) : sp::amd_workspace_size(a.n, a.view().nnz()) + extra);
    const auto r = sp::amd(a.view(), std::span<I>(perm), std::span<I>(work), o);
    EXPECT_TRUE(r.is_ok());
    return perm;
}

std::vector<I> rcm_order(const matrix& a)
{
    std::vector<I> perm(a.n, -7), work(sp::rcm_workspace_size(a.n, a.view().nnz()));
    const auto r = sp::reverse_cuthill_mckee(a.view(), std::span<I>(perm), std::span<I>(work));
    EXPECT_TRUE(r.is_ok());
    return perm;
}

// nnz(L) in the given order
std::size_t fill(const matrix& a, const std::vector<I>& perm)
{
    const std::size_t n = a.n, nnz = a.view().nnz();
    std::vector<I> cp(n + 1), ri(nnz), work(2 * n), parent(n), counts(n), lp(n + 1);
    std::vector<double> vx(nnz);
    sp::csc_view<double, I> b{0, 0, cp, ri, vx};
    EXPECT_TRUE(sp::symmetric_permute_upper(a.view(), std::span<const I>(perm), b, std::span<I>(work)).is_ok());
    const auto s = sp::ldl_symbolic(b, std::span<I>(parent), std::span<I>(counts), std::span<I>(lp), std::span<I>(work).first(n));
    EXPECT_TRUE(s.is_ok());
    return s ? s->nnz : 0;
}

// An exact minimum degree ordering, for reference: the explicit elimination graph, true degrees, lowest index on ties
std::vector<I> exact_minimum_degree(const matrix& a)
{
    const std::size_t n = a.n;
    std::vector<std::vector<char>> adj(n, std::vector<char>(n, 0));
    std::vector<std::size_t> deg(n, 0);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t q = static_cast<std::size_t>(a.cp[j]); q < static_cast<std::size_t>(a.cp[j + 1]); ++q) {
            const std::size_t i = static_cast<std::size_t>(a.ri[q]);
            if (i < j && !adj[i][j]) {
                adj[i][j] = adj[j][i] = 1;
                ++deg[i];
                ++deg[j];
            }
        }
    std::vector<char> done(n, 0);
    std::vector<I> perm;
    std::vector<std::size_t> nb;
    for (std::size_t step = 0; step < n; ++step) {
        std::size_t p = n;
        for (std::size_t i = 0; i < n; ++i)
            if (!done[i] && (p == n || deg[i] < deg[p]))
                p = i;
        perm.push_back(static_cast<I>(p));
        done[p] = 1;
        nb.clear();
        for (std::size_t i = 0; i < n; ++i)
            if (!done[i] && adj[p][i])
                nb.push_back(i);
        for (const std::size_t i : nb) {
            adj[p][i] = adj[i][p] = 0;
            --deg[i];
        }
        for (std::size_t x = 0; x < nb.size(); ++x)
            for (std::size_t y = x + 1; y < nb.size(); ++y)
                if (!adj[nb[x]][nb[y]]) {
                    adj[nb[x]][nb[y]] = adj[nb[y]][nb[x]] = 1;
                    ++deg[nb[x]];
                    ++deg[nb[y]];
                }
    }
    return perm;
}

bool is_perm(const std::vector<I>& p)
{
    std::vector<I> work(p.size());
    return sp::is_permutation(std::span<const I>(p), std::span<I>(work));
}

struct fills {
    std::size_t amd, natural, rcm, md;
};

fills measure(const char* name, const matrix& a, bool with_md)
{
    const auto pa = amd_order(a), pr = rcm_order(a);
    EXPECT_TRUE(is_perm(pa)) << name;
    EXPECT_TRUE(is_perm(pr)) << name;
    fills f{fill(a, pa), fill(a, natural(a.n)), fill(a, pr), with_md ? fill(a, exact_minimum_degree(a)) : 0};
    std::printf("  %-26s n %5zu  nnz(L): amd %8zu  natural %8zu  rcm %8zu  minimum degree %8zu\n", name, a.n, f.amd, f.natural, f.rcm, f.md);
    return f;
}

TEST(SparseOrderingAmd, GivesAPermutationOnEveryShape)
{
    // Empty
    {
        const matrix a = triplets{}.assemble();
        std::vector<I> perm, work(sp::amd_workspace_size(0, 0));
        EXPECT_TRUE(sp::amd(a.view(), std::span<I>(perm), std::span<I>(work)).is_ok());
    }
    // One, diagonal, complete, disconnected pieces
    triplets one;
    one.n = 1;
    one.add(0, 0, 2);
    EXPECT_EQ(amd_order(one.assemble()), std::vector<I>{0});
    triplets diag;
    diag.n = 9;
    for (std::size_t i = 0; i < 9; ++i)
        diag.add(i, i, 1);
    EXPECT_TRUE(is_perm(amd_order(diag.assemble())));
    for (const std::size_t n : {2u, 5u, 17u, 40u}) {
        triplets full;
        full.n = n;
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i <= j; ++i)
                full.add(i, j, i == j ? double(n) : 1.0);
        const matrix a = full.assemble();
        EXPECT_TRUE(is_perm(amd_order(a))) << n;
        EXPECT_EQ(fill(a, amd_order(a)), n * (n - 1) / 2) << n;
    }
    triplets pieces;
    pieces.n = 30;
    for (std::size_t i = 0; i < 30; ++i) {
        pieces.add(i, i, 3);
        if (i % 10 != 9)
            pieces.add(i, i + 1, -1);
    }
    const matrix pc = pieces.assemble();
    EXPECT_TRUE(is_perm(amd_order(pc)));
    EXPECT_EQ(fill(pc, amd_order(pc)), 27u);   // three paths: no fill
    // Entries below the diagonal are ignored
    matrix lower = grid(4).assemble();
    triplets both = grid(4);
    for (std::size_t k = 0, m = both.i.size(); k < m; ++k)
        if (both.i[k] != both.j[k]) {
            both.i.push_back(both.j[k]);
            both.j.push_back(both.i[k]);
            both.v.push_back(both.v[k]);
        }
    const matrix full_pattern = both.assemble();
    EXPECT_EQ(amd_order(full_pattern), amd_order(lower));
    // Many random shapes
    for (std::uint64_t seed = 1; seed <= 40; ++seed) {
        const matrix a = random_spd(5 + seed * 7, 1 + seed % 5, seed).assemble();
        EXPECT_TRUE(is_perm(amd_order(a))) << seed;
        EXPECT_TRUE(is_perm(amd_order(a, {-1.0, false}))) << seed;
    }
}

TEST(SparseOrderingAmd, FillOnGridsIsCloseToMinimumDegreeAndBelowNaturalAndRcm)
{
    for (const std::size_t k : {5u, 10u, 20u, 30u}) {
        char name[32];
        std::snprintf(name, sizeof name, "grid %zu x %zu", k, k);
        const fills f = measure(name, grid(k).assemble(), true);
        EXPECT_LE(static_cast<double>(f.amd), 1.25 * static_cast<double>(f.md)) << k;
        EXPECT_LT(f.amd, f.natural) << k;
        EXPECT_LT(f.amd, f.rcm) << k;
    }
    const fills big = measure("grid 100 x 100", grid(100).assemble(), false);
    EXPECT_LT(5 * big.amd, big.natural);
}

TEST(SparseOrderingAmd, PutsTheArrowHeadLastSoThereIsNoFill)
{
    for (const std::size_t n : {10u, 50u, 300u}) {
        const matrix a = arrow(n, 0).assemble();
        char name[32];
        std::snprintf(name, sizeof name, "arrow %zu, head first", n);
        const fills f = measure(name, a, n <= 50);
        // Head first fills everything; AMD's L has the chain and the head's row only
        EXPECT_EQ(f.natural, n * (n - 1) / 2);
        EXPECT_EQ(f.amd, 2 * n - 3);
        if (n <= 50) {
            EXPECT_LE(f.amd, f.md);
        }
    }
}

TEST(SparseOrderingAmd, FillOnRandomSpdAndKktMatricesIsCloseToMinimumDegree)
{
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        char name[48];
        std::snprintf(name, sizeof name, "random SPD, seed %llu", static_cast<unsigned long long>(seed));
        const fills f = measure(name, random_spd(300, 2, seed).assemble(), true);
        EXPECT_LE(static_cast<double>(f.amd), 1.25 * static_cast<double>(f.md)) << seed;
        EXPECT_LT(f.amd, f.natural) << seed;
        EXPECT_LT(f.amd, f.rcm) << seed;
    }
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        char name[48];
        std::snprintf(name, sizeof name, "random KKT, seed %llu", static_cast<unsigned long long>(seed));
        const fills f = measure(name, random_kkt(250, 150, 3, seed).assemble(), true);
        EXPECT_LE(static_cast<double>(f.amd), 1.25 * static_cast<double>(f.md)) << seed;
        EXPECT_LT(f.amd, f.natural) << seed;
        EXPECT_LT(f.amd, f.rcm) << seed;
    }
    const fills big = measure("random KKT, 3000 + 2000", random_kkt(3000, 2000, 4, 99).assemble(), false);
    EXPECT_LT(5 * big.amd, big.natural);
}

TEST(SparseOrderingAmd, OrdersADenseRowLast)
{
    // A grid plus one row joined to everything (a budget constraint): it is dense and goes last
    triplets t = grid(20);
    const std::size_t dense = t.n;
    t.n += 1;
    for (std::size_t i = 0; i < dense; ++i)
        t.add(i, dense, 1);
    t.add(dense, dense, 1000);
    const matrix a = t.assemble();
    const auto p = amd_order(a);
    EXPECT_EQ(p.back(), static_cast<I>(dense));
    // Dense handling off: still a permutation, and no better
    const auto q = amd_order(a, {-1.0, true});
    EXPECT_TRUE(is_perm(q));
    EXPECT_LE(fill(a, p), fill(a, q) + fill(a, q) / 10);
    // A threshold above every degree: nothing is dense
    const auto r = amd_order(a, {1e9, true});
    EXPECT_TRUE(is_perm(r));
}

TEST(SparseOrderingAmd, AggressiveAbsorptionIsAnOptionWithSimilarFill)
{
    for (const std::size_t k : {10u, 25u}) {
        const matrix a = grid(k).assemble();
        const std::size_t with = fill(a, amd_order(a)), without = fill(a, amd_order(a, {10.0, false}));
        EXPECT_LE(static_cast<double>(with), 1.2 * static_cast<double>(without)) << k;
        EXPECT_LE(static_cast<double>(without), 1.2 * static_cast<double>(with)) << k;
    }
}

TEST(SparseOrderingAmd, CompactingTheQuotientGraphDoesNotChangeTheOrder)
{
    // The least workspace amd accepts makes the graph compact itself many times on these; a generous one never does
    for (const matrix& a : {grid(10).assemble(), grid(40).assemble(), random_spd(800, 3, 5).assemble(), random_kkt(600, 400, 5, 3).assemble(),
                            arrow(100, 0).assemble()}) {
        const auto generous = amd_order(a, {}, 50 * a.view().nnz() + 10 * a.n);
        EXPECT_EQ(amd_order(a, {}, 0, true), generous);
        EXPECT_EQ(amd_order(a), generous);
    }
    for (std::uint64_t seed = 1; seed <= 30; ++seed) {
        const matrix a = random_spd(20 + seed * 11, 1 + seed % 6, seed).assemble();
        EXPECT_EQ(amd_order(a, {}, 0, true), amd_order(a, {}, 50 * a.view().nnz())) << seed;
    }
}

TEST(SparseOrderingAmd, ReportsErrors)
{
    const matrix a = grid(3).assemble();
    std::vector<I> perm(9), work(sp::amd_workspace_size(9, a.view().nnz()));
    sp::csc_view<const double, const I> rect{9, 8, std::span<const I>(a.cp).first(9), a.ri, a.vx};
    EXPECT_EQ(sp::amd(rect, std::span<I>(perm), std::span<I>(work)).unwrap_err(), sp::sparse_error::dimension_mismatch);
    std::vector<I> bad_rows = a.ri;
    bad_rows[1] = 50;
    sp::csc_view<const double, const I> bad{9, 9, a.cp, bad_rows, a.vx};
    EXPECT_EQ(sp::amd(bad, std::span<I>(perm), std::span<I>(work)).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(sp::amd(a.view(), std::span<I>(perm).first(8), std::span<I>(work)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(sp::amd(a.view(), std::span<I>(perm), std::span<I>(work).first(13 * 9 + 1 + 2 * strict_upper(a))).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_TRUE(sp::amd(a.view(), std::span<I>(perm), std::span<I>(work).first(13 * 9 + 2 + 2 * strict_upper(a))).is_ok());
    EXPECT_EQ(sp::amd(a.view(), std::span<I>(perm), std::span<I>(work).first(13 * 9)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    EXPECT_EQ(sp::reverse_cuthill_mckee(rect, std::span<I>(perm), std::span<I>(work)).unwrap_err(), sp::sparse_error::dimension_mismatch);
    EXPECT_EQ(sp::reverse_cuthill_mckee(bad, std::span<I>(perm), std::span<I>(work)).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(sp::reverse_cuthill_mckee(a.view(), std::span<I>(perm), std::span<I>(work).first(sp::rcm_workspace_size(9, a.view().nnz()) - 1)).unwrap_err(),
              sp::sparse_error::insufficient_capacity);
    static_assert(noexcept(sp::amd(std::declval<sp::csc_view<const double, const I>>(), std::declval<std::span<I>>(), std::declval<std::span<I>>())));
    static_assert(sp::amd_workspace_size(10, 20) == 14 * 10 + 1 + 40 + 8);
    static_assert(sp::rcm_workspace_size(10, 20) == 41 + 40);
}

TEST(SparseOrderingAmd, WorksWithSixtyFourBitIndices)
{
    const matrix a = grid(12).assemble();
    std::vector<std::int64_t> cp(a.cp.begin(), a.cp.end()), ri(a.ri.begin(), a.ri.end()), perm(a.n), work(sp::amd_workspace_size(a.n, a.view().nnz()));
    sp::csc_view<const double, const std::int64_t> v{a.n, a.n, cp, ri, a.vx};
    ASSERT_TRUE(sp::amd(v, std::span<std::int64_t>(perm), std::span<std::int64_t>(work)).is_ok());
    const auto p32 = amd_order(a);
    for (std::size_t k = 0; k < a.n; ++k)
        EXPECT_EQ(perm[k], p32[k]);
    EXPECT_TRUE(sp::is_permutation<std::int64_t>(perm));
}

// AMD of a 3 x 3 grid, then the fill it gives, in a constant expression
consteval std::size_t amd_fill_at_compile_time()
{
    constexpr std::size_t n = 9;
    std::array<I, 21> ti{}, tj{};
    std::array<double, 21> tv{};
    std::size_t m = 0;
    for (std::size_t x = 0; x < 3; ++x)
        for (std::size_t y = 0; y < 3; ++y) {
            const auto i = static_cast<I>(x * 3 + y);
            ti[m] = tj[m] = i;
            tv[m++] = 4;
            if (x < 2) {
                ti[m] = i;
                tj[m] = i + 3;
                tv[m++] = -1;
            }
            if (y < 2) {
                ti[m] = i;
                tj[m] = i + 1;
                tv[m++] = -1;
            }
        }
    sp::csc_matrix<double, n, n, 21> a{};
    auto av = a.view();
    std::array<I, n> w{};
    if (!sp::from_triplets(n, n, std::span<const I>(ti), std::span<const I>(tj), std::span<const double>(tv), av, std::span<I>(w)))
        return 0;
    std::array<I, n> perm{};
    std::array<I, sp::amd_workspace_size(n, 21)> work{};
    if (!sp::amd(a.view(), std::span<I>(perm), std::span<I>(work)) || !sp::is_permutation(std::span<const I>(perm)))
        return 0;
    sp::csc_matrix<double, n, n, 21> b{};
    auto bv = b.view();
    std::array<I, 2 * n> pw{};
    if (!sp::symmetric_permute_upper(std::as_const(a).view(), std::span<const I>(perm), bv, std::span<I>(pw)))
        return 0;
    const auto lnz = sp::ldl_nnz<n>(std::as_const(b).view());
    return lnz ? *lnz : 0;
}

TEST(SparseOrderingAmd, RunsInAConstantExpression)
{
    static_assert(amd_fill_at_compile_time() > 0 && amd_fill_at_compile_time() < 22);   // natural order gives 22
    EXPECT_EQ(amd_fill_at_compile_time(), fill(grid(3).assemble(), amd_order(grid(3).assemble())));
}

TEST(SparseOrderingRcm, ReducesTheBandwidthOfAShuffledBand)
{
    // A band of half-width 3, renumbered at random: RCM restores a band of about that width
    const std::size_t n = 200;
    std::vector<I> shuffle = natural(n);
    st::wyrand g(11);
    st::shuffle(std::span<I>(shuffle), g);
    triplets t;
    t.n = n;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t d = 0; d <= 3 && i + d < n; ++d)
            t.add(static_cast<std::size_t>(shuffle[i]), static_cast<std::size_t>(shuffle[i + d]), d == 0 ? 10 : -1);
    const matrix a = t.assemble();
    auto bandwidth = [&](const std::vector<I>& perm) {
        std::vector<I> inv(n);
        EXPECT_TRUE(sp::inverse_permutation(std::span<const I>(perm), std::span<I>(inv)).is_ok());
        std::size_t bw = 0;
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t q = static_cast<std::size_t>(a.cp[j]); q < static_cast<std::size_t>(a.cp[j + 1]); ++q) {
                const I d = inv[j] - inv[static_cast<std::size_t>(a.ri[q])];
                bw = std::max(bw, static_cast<std::size_t>(d < 0 ? -d : d));
            }
        return bw;
    };
    const auto p = rcm_order(a);
    EXPECT_TRUE(is_perm(p));
    EXPECT_LE(bandwidth(p), 6u);
    EXPECT_GT(bandwidth(natural(n)), 50u);
}

TEST(SparseOrderingRcm, OrdersAPathEndToEndAndEveryComponent)
{
    // A path numbered 0, 2, 4, ... 1, 3, 5: RCM walks it from one end
    triplets t;
    t.n = 10;
    std::vector<std::size_t> path{0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
    for (std::size_t k = 0; k < 10; ++k) {
        t.add(k, k, 2);
        if (k + 1 < 10)
            t.add(path[k], path[k + 1], -1);
    }
    const auto p = rcm_order(t.assemble());
    const bool forward = std::equal(p.begin(), p.end(), path.begin(), [](I a, std::size_t b) { return static_cast<std::size_t>(a) == b; });
    const bool backward = std::equal(p.begin(), p.end(), path.rbegin(), [](I a, std::size_t b) { return static_cast<std::size_t>(a) == b; });
    EXPECT_TRUE(forward || backward);
    // Isolated nodes and several components
    triplets d;
    d.n = 7;
    for (std::size_t i = 0; i < 7; ++i)
        d.add(i, i, 1);
    d.add(1, 3, 1);
    d.add(4, 6, 1);
    EXPECT_TRUE(is_perm(rcm_order(d.assemble())));
    const matrix none = triplets{}.assemble();
    std::vector<I> perm, work(1);
    EXPECT_TRUE(sp::reverse_cuthill_mckee(none.view(), std::span<I>(perm), std::span<I>(work)).is_ok());
    for (std::uint64_t seed = 1; seed <= 20; ++seed)
        EXPECT_TRUE(is_perm(rcm_order(random_spd(10 + seed * 13, 1 + seed % 4, seed).assemble())));
}

TEST(SparseOrderingPermutation, IsPermutationChecksRangeAndRepeatsWithAndWithoutWork)
{
    const std::vector<I> good{2, 0, 3, 1}, repeat{2, 0, 2, 1}, negative{0, -1, 2, 3}, large{0, 1, 2, 4};
    std::vector<I> work(4);
    for (const bool with : {false, true}) {
        const auto w = with ? std::span<I>(work) : std::span<I>();
        EXPECT_TRUE(sp::is_permutation(std::span<const I>(good), w));
        EXPECT_FALSE(sp::is_permutation(std::span<const I>(repeat), w));
        EXPECT_FALSE(sp::is_permutation(std::span<const I>(negative), w));
        EXPECT_FALSE(sp::is_permutation(std::span<const I>(large), w));
    }
    EXPECT_TRUE(sp::is_permutation(std::span<const I>()));
    static_assert(sp::is_permutation(std::span<const I>(std::array<I, 3>{1, 2, 0})));
}

TEST(SparseOrderingPermutation, InversePermutationAndPermuteRoundTrip)
{
    const std::vector<I> p{3, 0, 4, 1, 2};
    std::vector<I> inv(5);
    ASSERT_TRUE(sp::inverse_permutation(std::span<const I>(p), std::span<I>(inv)).is_ok());
    EXPECT_EQ(inv, (std::vector<I>{1, 3, 4, 0, 2}));
    for (std::size_t k = 0; k < 5; ++k)
        EXPECT_EQ(inv[static_cast<std::size_t>(p[k])], static_cast<I>(k));
    const std::vector<double> x{10, 11, 12, 13, 14};
    std::vector<double> y(5), z(5);
    sp::permute(std::span<const I>(p), std::span<const double>(x), std::span<double>(y));
    EXPECT_EQ(y, (std::vector<double>{13, 10, 14, 11, 12}));
    sp::inverse_permute(std::span<const I>(p), std::span<const double>(y), std::span<double>(z));
    EXPECT_EQ(z, x);
    // Errors
    EXPECT_EQ(sp::inverse_permutation(std::span<const I>(p), std::span<I>(inv).first(4)).unwrap_err(), sp::sparse_error::insufficient_capacity);
    const std::vector<I> repeat{0, 0, 1}, out{0, 3, 1};
    EXPECT_EQ(sp::inverse_permutation(std::span<const I>(repeat), std::span<I>(inv)).unwrap_err(), sp::sparse_error::invalid_index);
    EXPECT_EQ(sp::inverse_permutation(std::span<const I>(out), std::span<I>(inv)).unwrap_err(), sp::sparse_error::invalid_index);
    constexpr auto inverse_at_compile_time = [] {
        std::array<I, 3> q{2, 0, 1}, r{};
        return sp::inverse_permutation(std::span<const I>(q), std::span<I>(r)).is_ok() && r[0] == 1 && r[1] == 2 && r[2] == 0;
    };
    static_assert(inverse_at_compile_time());
}

} // namespace
