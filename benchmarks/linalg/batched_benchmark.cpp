//
//  batched_benchmark.cpp
//  snicholls-core
//
//  Batched factorise-and-solve of 10^6 dense systems of order 25 and of order 65, double, one right-hand
//  side each: snicholls::linalg::lu_solve_batched, cholesky_solve_batched and ldlt_solve_batched with the reference backend (the
//  concept path: free function, backend member, per-system kernels) against the same algorithms written by
//  hand over raw pointers. Identical arithmetic in the same order: LU with partial pivoting and the
//  relative singularity test, unpivoted LDLT, then the substitutions.
//
//  Ten to the sixth systems of order 65 do not fit in memory at once (34 GB), and a real caller builds each
//  system just before solving it, so both sides stream: 1024 systems at a time are restored from a pristine
//  copy (one memcpy, the same for both) and solved. Set SNICHOLLS_CORE_BENCHMARK_SMOKE=1 (the CTest smoke run
//  does) for 4096 systems instead of 10^6.
//
//  The fairness check solves the same batch both ways first and refuses to benchmark unless the solutions,
//  pivots and statuses are bit-for-bit equal.
//

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace la = snicholls::linalg;
constexpr std::size_t chunk = 1024;

std::size_t systems()
{
    const char* smoke = std::getenv("SNICHOLLS_CORE_BENCHMARK_SMOKE");
    return smoke != nullptr && *smoke != '\0' && *smoke != '0' ? 4096 : 1000000;
}

// 1024 pristine systems of order m with their right-hand sides: bordered exponential covariances (symmetric,
// indefinite, as in batched kriging-style solves) with one singular (a repeated point); or, with spd set, the
// covariances alone plus a small nugget (positive definite, for Cholesky) with one made indefinite.
struct workload {
    std::size_t m;
    std::vector<double> a0, b0, a, b;
    std::vector<la::pivot_index> p;
    std::vector<la::status> s;
    explicit workload(std::size_t m_, bool spd = false)
        : m(m_), a0(chunk * m_ * m_), b0(chunk * m_), a(a0.size()), b(b0.size()), p(chunk * m_), s(chunk)
    {
        const std::size_t k = spd ? m : m - 1;
        std::vector<double> x(k);
        for (std::size_t sys = 0; sys < chunk; ++sys) {
            for (std::size_t i = 0; i < k; ++i)
                x[i] = static_cast<double>(i) + 0.45 * std::sin(static_cast<double>(sys * 131 + i * 17));
            if (sys == 333)
                x[2] = x[1];
            double* as = a0.data() + sys * m * m;
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j)
                    as[i * m + j] = i < k && j < k ? std::exp(-std::fabs(x[i] - x[j]) / 4.0) : (i == k && j == k ? 0.0 : 1.0);
                b0[sys * m + i] = i < k ? std::exp(-std::fabs(x[i] - 0.5 * static_cast<double>(k)) / 4.0) : 1.0;
            }
            if (spd) {
                for (std::size_t i = 0; i < m; ++i)
                    as[i * m + i] += 1e-3;
                if (sys == 333)
                    as[0] = -1.0;
            }
        }
    }
    void restore()
    {
        std::memcpy(a.data(), a0.data(), a.size() * sizeof(double));
        std::memcpy(b.data(), b0.data(), b.size() * sizeof(double));
    }
};

// ---- Through the interface ----------------------------------------------------------------------------------

std::size_t core_lu(workload& w)
{
    const auto r = la::lu_solve_batched(la::reference, la::batch_view<double, la::layout_right>(w.a.data(), {chunk, w.m, w.m}),
                                        std::span<la::pivot_index>(w.p), la::md_span<double, 2, la::layout_right>(w.b.data(), {chunk, w.m}),
                                        std::span<la::status>(w.s));
    return r ? *r : chunk;
}

std::size_t core_cholesky(workload& w)
{
    const auto r = la::cholesky_solve_batched(la::reference, la::batch_view<double, la::layout_right>(w.a.data(), {chunk, w.m, w.m}),
                                              la::md_span<double, 2, la::layout_right>(w.b.data(), {chunk, w.m}), std::span<la::status>(w.s));
    return r ? *r : chunk;
}

std::size_t core_ldlt(workload& w)
{
    const auto r = la::ldlt_solve_batched(la::reference, la::batch_view<double, la::layout_right>(w.a.data(), {chunk, w.m, w.m}),
                                          la::md_span<double, 2, la::layout_right>(w.b.data(), {chunk, w.m}), std::span<la::status>(w.s));
    return r ? *r : chunk;
}

// ---- By hand ------------------------------------------------------------------------------------------------

bool lu_by_hand(double* a, la::pivot_index* piv, double* b, std::size_t n)
{
    double scale = 0;
    for (std::size_t i = 0; i < n * n; ++i) {
        const double v = std::fabs(a[i]);
        if (v > scale || v != v)
            scale = v;
    }
    const double tolerance = std::numeric_limits<double>::epsilon() * static_cast<double>(n) * scale;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        double largest = std::fabs(a[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double v = std::fabs(a[i * n + k]);
            if (v > largest) {
                largest = v;
                p = i;
            }
        }
        piv[k] = static_cast<la::pivot_index>(p);
        if (!(largest > tolerance))
            return false;
        if (p != k)
            for (std::size_t j = 0; j < n; ++j) {
                const double t = a[k * n + j];
                a[k * n + j] = a[p * n + j];
                a[p * n + j] = t;
            }
        const double pivot = a[k * n + k];
        if (std::fabs(pivot) >= std::numeric_limits<double>::min()) {
            const double inverse = 1.0 / pivot;
            for (std::size_t i = k + 1; i < n; ++i)
                a[i * n + k] *= inverse;
        } else {
            for (std::size_t i = k + 1; i < n; ++i)
                a[i * n + k] = a[i * n + k] / pivot;
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double l = a[i * n + k];
            for (std::size_t j = k + 1; j < n; ++j)
                a[i * n + j] -= l * a[k * n + j];
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        const auto p = static_cast<std::size_t>(piv[k]);
        if (p != k) {
            const double t = b[k];
            b[k] = b[p];
            b[p] = t;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= a[i * n + j] * b[j];
        b[i] = s;
    }
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= a[i * n + j] * b[j];
        b[i] = s / a[i * n + i];
    }
    return true;
}

bool ldlt_by_hand(double* a, double* b, std::size_t n)
{
    double scale = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            const double v = std::fabs(a[i * n + j]);
            if (v > scale || v != v)
                scale = v;
        }
    const double tolerance = std::numeric_limits<double>::epsilon() * static_cast<double>(n) * scale;
    for (std::size_t j = 0; j < n; ++j) {
        double d = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k)
            d -= a[j * n + k] * (a[j * n + k] * a[k * n + k]);
        if (!(std::fabs(d) > tolerance))
            return false;
        a[j * n + j] = d;
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k)
                s -= a[i * n + k] * (a[j * n + k] * a[k * n + k]);
            a[i * n + j] = s / d;
        }
    }
    double growth = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double g = std::fabs(a[i * n + i]);
        for (std::size_t k = 0; k < i; ++k)
            g += a[i * n + k] * a[i * n + k] * std::fabs(a[k * n + k]);
        if (g > growth || g != g)
            growth = g;
    }
    if (!(growth <= static_cast<double>(n) / std::sqrt(std::sqrt(std::numeric_limits<double>::epsilon())) * scale))
        return false;
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= a[i * n + j] * b[j];
        b[i] = s;
    }
    for (std::size_t i = 0; i < n; ++i)
        b[i] = b[i] / a[i * n + i];
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= a[j * n + i] * b[j];
        b[i] = s;
    }
    return true;
}

bool cholesky_by_hand(double* a, double* b, std::size_t n)
{
    double scale = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = a[i * n + i];
        if (d > scale || d != d)
            scale = d;
    }
    const double tolerance = std::numeric_limits<double>::epsilon() * static_cast<double>(n) * scale * 1.0;
    for (std::size_t j = 0; j < n; ++j) {
        double d = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k)
            d -= a[j * n + k] * a[j * n + k];
        if (!(d > tolerance))
            return false;
        const double root = std::sqrt(d);
        a[j * n + j] = root;
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k)
                s -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = s / root;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= a[i * n + j] * b[j];
        b[i] = s / a[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= a[j * n + i] * b[j];
        b[i] = s / a[i * n + i];
    }
    return true;
}

std::size_t hand_cholesky(workload& w)
{
    std::size_t failures = 0;
    for (std::size_t s = 0; s < chunk; ++s) {
        const bool ok = cholesky_by_hand(w.a.data() + s * w.m * w.m, w.b.data() + s * w.m, w.m);
        w.s[s] = ok ? la::status::ok : la::status::not_positive_definite;
        failures += ok ? 0 : 1;
    }
    return failures;
}

std::size_t hand_lu(workload& w)
{
    std::size_t failures = 0;
    for (std::size_t s = 0; s < chunk; ++s) {
        const bool ok = lu_by_hand(w.a.data() + s * w.m * w.m, w.p.data() + s * w.m, w.b.data() + s * w.m, w.m);
        w.s[s] = ok ? la::status::ok : la::status::singular;
        failures += ok ? 0 : 1;
    }
    return failures;
}

std::size_t hand_ldlt(workload& w)
{
    std::size_t failures = 0;
    for (std::size_t s = 0; s < chunk; ++s) {
        const bool ok = ldlt_by_hand(w.a.data() + s * w.m * w.m, w.b.data() + s * w.m, w.m);
        w.s[s] = ok ? la::status::ok : la::status::singular;
        failures += ok ? 0 : 1;
    }
    return failures;
}

// ---- The benchmarks -----------------------------------------------------------------------------------------

template <std::size_t (*Solve)(workload&), bool Spd = false>
void run(benchmark::State& state)
{
    workload w(static_cast<std::size_t>(state.range(0)), Spd);
    const std::size_t n = systems();
    for (auto _ : state) {
        std::size_t failures = 0;
        for (std::size_t done = 0; done < n; done += chunk) {
            w.restore();
            benchmark::ClobberMemory();
            failures += Solve(w);
        }
        benchmark::DoNotOptimize(failures);
        benchmark::DoNotOptimize(w.b.data());
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(n));
    state.counters["systems"] = static_cast<double>(n);
}

void BM_LinalgBatched_LuCore(benchmark::State& s) { run<core_lu>(s); }
void BM_LinalgBatched_LuByHand(benchmark::State& s) { run<hand_lu>(s); }
void BM_LinalgBatched_CholeskyCore(benchmark::State& s) { run<core_cholesky, true>(s); }
void BM_LinalgBatched_CholeskyByHand(benchmark::State& s) { run<hand_cholesky, true>(s); }
void BM_LinalgBatched_LdltCore(benchmark::State& s) { run<core_ldlt>(s); }
void BM_LinalgBatched_LdltByHand(benchmark::State& s) { run<hand_ldlt>(s); }
BENCHMARK(BM_LinalgBatched_LuCore)->Arg(25)->Arg(65)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_LinalgBatched_LuByHand)->Arg(25)->Arg(65)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_LinalgBatched_CholeskyCore)->Arg(25)->Arg(65)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_LinalgBatched_CholeskyByHand)->Arg(25)->Arg(65)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_LinalgBatched_LdltCore)->Arg(25)->Arg(65)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_LinalgBatched_LdltByHand)->Arg(25)->Arg(65)->Unit(benchmark::kMillisecond);

bool same(const workload& x, const workload& y)
{
    return std::memcmp(x.b.data(), y.b.data(), x.b.size() * sizeof(double)) == 0 &&
           std::memcmp(x.a.data(), y.a.data(), x.a.size() * sizeof(double)) == 0 && x.s == y.s;
}

bool fair()
{
    for (const std::size_t m : {25u, 65u}) {
        workload c(m), h(m);
        c.restore();
        h.restore();
        if (core_lu(c) != 1 || hand_lu(h) != 1 || !same(c, h) || c.p != h.p)
            return false;
        c.restore();
        h.restore();
        if (core_ldlt(c) != 1 || hand_ldlt(h) != 1 || !same(c, h))
            return false;
        workload cs(m, true), hs(m, true);
        cs.restore();
        hs.restore();
        if (core_cholesky(cs) != 1 || hand_cholesky(hs) != 1 || !same(cs, hs))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "linalg batched benchmark: fairness check FAILED\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
