//
//  krylov_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::linalg::conjugate_gradient through csc_operator, on the 2-D Laplacian of a 64 x 64 grid (n = 4096,
//  five-point stencil, CSC), against the same CG written by hand over the same three CSC arrays, in the same order:
//  r = b - A x (the product column by column, y[row] += a * x[col]), ‖r‖₂ as sqrt of the dot product in index order,
//  rᵀr, then per iteration q = A p, pᵀq, α, x += α p and r -= α q together, ‖r‖₂, rᵀr, β and p = r + β p. Both run
//  exactly 200 iterations from x = 0 (no tolerance can be met). Neither copies r into a preconditioned z (core's
//  CG uses r itself under identity_preconditioner). main() runs a fairness check first and refuses to benchmark
//  unless the two solutions agree to the bit.
//

#include <snicholls/core/linalg/krylov.hpp>

#include <snicholls/core/sparse/csc.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace sp = snicholls::sparse;
constexpr std::size_t grid = 64, n = grid * grid, iterations = 200;

struct laplacian {
    std::vector<std::int32_t> col_ptr, row_idx;
    std::vector<double> values, b;
};

const laplacian& data()
{
    static const laplacian l = [] {
        laplacian out;
        out.col_ptr.push_back(0);
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t gx = j % grid, gy = j / grid;
            auto add = [&](std::size_t i, double v) {
                out.row_idx.push_back(static_cast<std::int32_t>(i));
                out.values.push_back(v);
            };
            if (gy > 0)
                add(j - grid, -1);
            if (gx > 0)
                add(j - 1, -1);
            add(j, 4);
            if (gx + 1 < grid)
                add(j + 1, -1);
            if (gy + 1 < grid)
                add(j + grid, -1);
            out.col_ptr.push_back(static_cast<std::int32_t>(out.values.size()));
        }
        out.b.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            out.b[i] = 1.0 + static_cast<double>(i % 7) / 7;
        return out;
    }();
    return l;
}

void core(std::vector<double>& x, std::vector<double>& work)
{
    const auto& d = data();
    const sp::csc_view<const double, const std::int32_t> a{n, n, d.col_ptr, d.row_idx, d.values};
    for (auto& e : x)
        e = 0;
    auto r = la::conjugate_gradient(la::make_csc_operator(a), std::span<const double>(d.b), std::span<double>(x), std::span<double>(work), {0, 0, iterations});
    benchmark::DoNotOptimize(r);
}

void by_hand(std::vector<double>& xv, std::vector<double>& work)
{
    const auto& d = data();
    const std::int32_t* cp = d.col_ptr.data();
    const std::int32_t* ri = d.row_idx.data();
    const double* av = d.values.data();
    const double* b = d.b.data();
    double* x = xv.data();
    double* r = work.data();
    double* p = r + n;
    double* q = p + n;
    auto product = [&](const double* v, double* y) {
        for (std::size_t i = 0; i < n; ++i)
            y[i] = 0;
        for (std::size_t j = 0; j < n; ++j) {
            const double vj = v[j];
            for (std::int32_t k = cp[j]; k < cp[j + 1]; ++k)
                y[ri[k]] += av[k] * vj;
        }
    };
    auto dot = [](const double* u, const double* v) {
        double s = 0;
        for (std::size_t i = 0; i < n; ++i)
            s += u[i] * v[i];
        return s;
    };
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 0;
    product(x, q);
    for (std::size_t i = 0; i < n; ++i)
        r[i] = b[i] - q[i];
    double rnorm = std::sqrt(dot(r, r));
    double rr = dot(r, r);
    for (std::size_t i = 0; i < n; ++i)
        p[i] = r[i];
    for (std::size_t k = 0; k < iterations; ++k) {
        product(p, q);
        const double alpha = rr / dot(p, q);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        rnorm = std::sqrt(dot(r, r));
        const double rr_new = dot(r, r);
        const double beta = rr_new / rr;
        rr = rr_new;
        for (std::size_t i = 0; i < n; ++i)
            p[i] = r[i] + beta * p[i];
    }
    benchmark::DoNotOptimize(rnorm);
}

template <void (*Run)(std::vector<double>&, std::vector<double>&)>
void BM_Cg(benchmark::State& state)
{
    std::vector<double> x(n), work(la::conjugate_gradient_workspace(n));
    data();
    for (auto _ : state) {
        Run(x, work);
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(iterations));
}
BENCHMARK(BM_Cg<core>)->Name("BM_ConjugateGradientLaplacian64x64_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Cg<by_hand>)->Name("BM_ConjugateGradientLaplacian64x64_HandWritten")->Unit(benchmark::kMicrosecond);

bool fair()
{
    std::vector<double> a(n), b(n), wa(la::conjugate_gradient_workspace(n)), wb(la::conjugate_gradient_workspace(n));
    core(a, wa);
    by_hand(b, wb);
    return std::memcmp(a.data(), b.data(), n * sizeof(double)) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "krylov benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
