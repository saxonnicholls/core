//
//  ldl_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::sparse::ldl_symbolic, ldl_numeric and ldl_solve on two systems, each in natural order and in AMD order:
//  the 5-point Laplacian on a 100 x 100 grid (10,000 rows) and a random quasi-definite KKT system [P + σI Aᵀ;
//  A -ρ⁻¹I] of 5,000 rows (3,000 variables, 2,000 constraints, each constraint over a few neighbouring variables, as
//  a multi-period problem's are). nnz(L) is reported as a counter. main() refuses to benchmark unless every
//  factorisation solves its system to a residual of 1e-9 and AMD's L is smaller than natural order's.
//

#include <snicholls/core/sparse/ldl.hpp>

#include <snicholls/core/sparse/csc.hpp>
#include <snicholls/core/sparse/ordering.hpp>
#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
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

// A symmetric system as its upper triangle, permuted into the order given, with everything a factorisation needs
struct system {
    std::size_t n = 0;
    std::vector<I> cp, ri, perm, parent, counts, lp, li, iwork;
    std::vector<double> vx, lx, d, dinv, xwork, b, x;
    std::vector<std::int8_t> signs;
    sp::ldl_symbolic_result<I> sym{};
    sp::csc_view<double, I> l{};

    [[nodiscard]] sp::csc_view<const double, const I> a() const { return {n, n, cp, ri, vx}; }
    bool symbolic()
    {
        const auto s = sp::ldl_symbolic(a(), std::span<I>(parent), std::span<I>(counts), std::span<I>(lp), std::span<I>(iwork).first(n));
        if (!s)
            return false;
        sym = *s;
        li.resize(sym.nnz);
        lx.resize(sym.nnz);
        l = {n, n, lp, li, lx};
        return true;
    }
    bool numeric() { return sp::ldl_numeric(a(), sym, l, d, dinv, std::span<I>(iwork), xwork, signs).is_ok(); }
    void solve()
    {
        x = b;
        sp::ldl_solve(l.as_const(), std::span<const double>(dinv), std::span<double>(x));
    }
    [[nodiscard]] double residual() const
    {
        std::vector<double> ax(n);
        sp::spmv_symmetric_upper(a(), std::span<const double>(x), std::span<double>(ax));
        double worst = 0;
        for (std::size_t i = 0; i < n; ++i)
            worst = std::max(worst, std::abs(b[i] - ax[i]));
        return worst;
    }
};

struct triplets {
    std::size_t n = 0;
    std::vector<I> i, j;
    std::vector<double> v;
    std::vector<std::int8_t> signs;
    void add(std::size_t r, std::size_t c, double x)
    {
        i.push_back(static_cast<I>(r < c ? r : c));
        j.push_back(static_cast<I>(r < c ? c : r));
        v.push_back(x);
    }
};

triplets grid(std::size_t k)
{
    triplets t;
    t.n = k * k;
    t.signs.assign(t.n, 1);
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

triplets kkt(std::size_t nx, std::size_t m, std::uint64_t seed)
{
    st::wyrand g(seed);
    triplets t;
    t.n = nx + m;
    t.signs.assign(t.n, 1);
    for (std::size_t i = 0; i < nx; ++i) {
        t.add(i, i, 0.5 + st::uniform_real(g) + 1e-6);   // P + σI
        if (i + 1 < nx)
            t.add(i, i + 1, 0.2 * (st::uniform_real(g) - 0.5));
    }
    for (std::size_t r = 0; r < m; ++r) {
        const std::size_t base = r * nx / m;
        for (std::size_t e = 0; e < 4; ++e)
            t.add((base + static_cast<std::size_t>(st::uniform_index(g, 12))) % nx, nx + r, st::uniform_real(g) - 0.5);
        t.add(nx + r, nx + r, -1.0 / 0.1);   // -ρ⁻¹, ρ = 0.1
        t.signs[nx + r] = -1;
    }
    return t;
}

// The system in natural order or AMD order, assembled and analysed, with a right-hand side
system make(const triplets& t, bool amd)
{
    const std::size_t n = t.n;
    std::vector<I> cp(n + 1), ri(t.i.size()), work(n);
    std::vector<double> vx(t.i.size());
    sp::csc_view<double, I> a{0, 0, cp, ri, vx};
    (void)sp::from_triplets(n, n, std::span<const I>(t.i), std::span<const I>(t.j), std::span<const double>(t.v), a, std::span<I>(work));
    system s;
    s.n = n;
    s.perm.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        s.perm[i] = static_cast<I>(i);
    if (amd) {
        std::vector<I> amd_work(sp::amd_workspace_size(n, a.nnz()));
        (void)sp::amd(a.as_const(), std::span<I>(s.perm), std::span<I>(amd_work));
    }
    s.cp.resize(n + 1);
    s.ri.resize(a.nnz());
    s.vx.resize(a.nnz());
    std::vector<I> pwork(2 * n);
    sp::csc_view<double, I> p{0, 0, s.cp, s.ri, s.vx};
    (void)sp::symmetric_permute_upper(a.as_const(), std::span<const I>(s.perm), p, std::span<I>(pwork));
    s.parent.resize(n);
    s.counts.resize(n);
    s.lp.resize(n + 1);
    s.iwork.resize(3 * n);
    s.d.resize(n);
    s.dinv.resize(n);
    s.xwork.resize(n);
    s.signs.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        s.signs[i] = t.signs[static_cast<std::size_t>(s.perm[i])];
    s.b.resize(n);
    st::wyrand g(n);
    for (double& v : s.b)
        v = st::uniform_real(g) - 0.5;
    (void)s.symbolic();
    return s;
}

const triplets& problem(int which)
{
    static const triplets g = grid(100), k = kkt(3000, 2000, 7);
    return which == 0 ? g : k;
}

void BM_Symbolic(benchmark::State& state)
{
    system s = make(problem(static_cast<int>(state.range(0))), state.range(1) != 0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(s.symbolic());
        benchmark::ClobberMemory();
    }
    state.counters["nnz_L"] = static_cast<double>(s.sym.nnz);
}

void BM_Numeric(benchmark::State& state)
{
    system s = make(problem(static_cast<int>(state.range(0))), state.range(1) != 0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(s.numeric());
        benchmark::ClobberMemory();
    }
    state.counters["nnz_L"] = static_cast<double>(s.sym.nnz);
}

void BM_Solve(benchmark::State& state)
{
    system s = make(problem(static_cast<int>(state.range(0))), state.range(1) != 0);
    (void)s.numeric();
    for (auto _ : state) {
        s.solve();
        benchmark::DoNotOptimize(s.x.data());
        benchmark::ClobberMemory();
    }
    state.counters["nnz_L"] = static_cast<double>(s.sym.nnz);
}

// range(0): 0 the grid, 1 the KKT system; range(1): 0 natural order, 1 AMD
void arguments(benchmark::Benchmark* b)
{
    b->ArgNames({"kkt", "amd"});
    for (const int which : {0, 1})
        for (const int amd : {0, 1})
            b->Args({which, amd});
    b->Unit(benchmark::kMillisecond);
}
BENCHMARK(BM_Symbolic)->Apply(arguments);
BENCHMARK(BM_Numeric)->Apply(arguments);
BENCHMARK(BM_Solve)->Apply(arguments);

bool valid()
{
    for (const int which : {0, 1}) {
        std::size_t nnz[2] = {0, 0};
        for (const int amd : {0, 1}) {
            system s = make(problem(which), amd != 0);
            if (!s.numeric())
                return false;
            s.solve();
            const double r = s.residual();
            std::fprintf(stderr, "ldl benchmark: %s, %s order: n %zu, nnz(L) %zu, residual %.1e\n", which == 0 ? "grid 100 x 100" : "KKT 3000 + 2000",
                         amd ? "AMD" : "natural", s.n, s.sym.nnz, r);
            if (!(r < 1e-9))
                return false;
            nnz[amd] = s.sym.nnz;
        }
        if (!(nnz[1] < nnz[0]))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!valid()) {
        std::fprintf(stderr, "ldl benchmark: validity check FAILED - a factorisation does not solve its system, or AMD does not reduce fill; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
