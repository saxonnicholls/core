//
//  fixed_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::linalg::fixed_matrix against the same loops written by hand over raw double arrays: the 4 x 4 product
//  (k inner to i, j innermost, as the header does) and the 4 x 4 Cholesky factorisation (the same column order, the
//  same tolerance test, the same report of failure), over 4096 matrices each. main() runs a fairness check first and refuses to benchmark unless
//  every result agrees to the bit.
//

#include <snicholls/core/linalg/fixed.hpp>

#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace st = snicholls::stats;
using m4 = la::fixed_matrix<double, 4, 4>;
constexpr std::size_t n = 4096;

struct inputs {
    std::vector<m4> a, b, spd;
};

const inputs& data()
{
    static const inputs in = [] {
        inputs r{std::vector<m4>(n), std::vector<m4>(n), std::vector<m4>(n)};
        st::wyrand g(5);
        for (std::size_t k = 0; k < n; ++k) {
            m4 x{};
            for (auto& e : r.a[k].elements)
                e = st::uniform_real(g) - 0.5;
            for (auto& e : r.b[k].elements)
                e = st::uniform_real(g) - 0.5;
            for (auto& e : x.elements)
                e = st::uniform_real(g) - 0.5;
            r.spd[k] = x * la::transpose(x) + m4::identity();
        }
        return r;
    }();
    return in;
}

void product_core(std::vector<m4>& out)
{
    const auto& d = data();
    for (std::size_t k = 0; k < n; ++k)
        out[k] = d.a[k] * d.b[k];
}

void product_by_hand(std::vector<m4>& out)
{
    const auto& d = data();
    for (std::size_t k = 0; k < n; ++k) {
        const double* a = d.a[k].elements.data();
        const double* b = d.b[k].elements.data();
        double c[16] = {};
        for (int i = 0; i < 4; ++i)
            for (int q = 0; q < 4; ++q) {
                const double aiq = a[i * 4 + q];
                for (int j = 0; j < 4; ++j)
                    c[i * 4 + j] += aiq * b[q * 4 + j];
            }
        std::memcpy(out[k].elements.data(), c, sizeof c);
    }
}

void cholesky_core(std::vector<m4>& out)
{
    const auto& d = data();
    for (std::size_t k = 0; k < n; ++k)
        if (const auto l = la::cholesky(d.spd[k]))
            out[k] = *l;
}

// The same factorisation with the same failure semantics: a matrix that is not positive definite leaves its slot
// untouched, as cholesky's error result does
struct factor {
    double l[16];
    bool ok;
};

factor cholesky_4x4(const double* a)
{
    factor f{{}, true};
    double scale = 0;
    for (int i = 0; i < 4; ++i)
        scale = std::fabs(a[i * 5]) > scale ? std::fabs(a[i * 5]) : scale;
    const double tol = std::numeric_limits<double>::epsilon() * 4.0 * scale;
    for (int j = 0; j < 4; ++j) {
        double dj = a[j * 5];
        for (int q = 0; q < j; ++q)
            dj -= f.l[j * 4 + q] * f.l[j * 4 + q];
        if (!(dj > tol)) {
            f.ok = false;
            return f;
        }
        const double ljj = std::sqrt(dj);
        f.l[j * 5] = ljj;
        for (int i = j + 1; i < 4; ++i) {
            double s = a[i * 4 + j];
            for (int q = 0; q < j; ++q)
                s -= f.l[i * 4 + q] * f.l[j * 4 + q];
            f.l[i * 4 + j] = s / ljj;
        }
    }
    return f;
}

void cholesky_by_hand(std::vector<m4>& out)
{
    const auto& d = data();
    for (std::size_t k = 0; k < n; ++k) {
        const factor f = cholesky_4x4(d.spd[k].elements.data());
        if (f.ok)
            std::memcpy(out[k].elements.data(), f.l, sizeof f.l);
    }
}

template <void (*Run)(std::vector<m4>&)>
void BM_Fixed(benchmark::State& state)
{
    std::vector<m4> out(n);
    data();
    for (auto _ : state) {
        Run(out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(n));
}
BENCHMARK(BM_Fixed<product_core>)->Name("BM_FixedProduct4x4_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Fixed<product_by_hand>)->Name("BM_FixedProduct4x4_HandWritten")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Fixed<cholesky_core>)->Name("BM_FixedCholesky4x4_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Fixed<cholesky_by_hand>)->Name("BM_FixedCholesky4x4_HandWritten")->Unit(benchmark::kMicrosecond);

bool fair()
{
    std::vector<m4> a(n), b(n), c(n), d(n);
    product_core(a);
    product_by_hand(b);
    cholesky_core(c);
    cholesky_by_hand(d);
    return std::memcmp(a.data(), b.data(), n * sizeof(m4)) == 0 && std::memcmp(c.data(), d.data(), n * sizeof(m4)) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "fixed benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
