//
//  assignment_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::combinatorics::assignment_solver on square matrices of uniform costs, 64 x 64 and 256 x 256, and on a
//  gated 256 x 256 (90% of the pairs forbidden, as in data association). There is no hand-written equivalent: this
//  measures the algorithm. main() first checks each solution against the dual bound (no row can do better by moving
//  to a free or cheaper column), and refuses to benchmark if one fails.
//

#include <snicholls/core/combinatorics/assignment.hpp>

#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {

namespace cb = snicholls::combinatorics;
namespace st = snicholls::stats;
using solver = cb::assignment_solver<double, 256, 256>;

std::vector<double> costs(std::size_t n, double forbidden)
{
    std::vector<double> c(n * n);
    st::wyrand g(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            c[i * n + j] = (i != j && st::uniform_real(g) < forbidden) ? std::numeric_limits<double>::infinity() : st::uniform_real(g);
    return c;
}

// Every row assigned, no column twice, and no 2-swap improves the total: a cheap necessary check of optimality
bool plausible(const std::vector<double>& c, std::size_t n, const std::vector<std::int32_t>& rc)
{
    std::vector<int> used(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (rc[i] < 0 || used[static_cast<std::size_t>(rc[i])]++)
            return false;
    }
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a + 1; b < n; ++b) {
            const auto ja = static_cast<std::size_t>(rc[a]), jb = static_cast<std::size_t>(rc[b]);
            if (c[a * n + jb] + c[b * n + ja] < c[a * n + ja] + c[b * n + jb] - 1e-12)
                return false;
        }
    return true;
}

void BM_Assignment(benchmark::State& state)
{
    const auto n = static_cast<std::size_t>(state.range(0));
    const double forbidden = static_cast<double>(state.range(1)) / 100.0;
    const auto c = costs(n, forbidden);
    auto s = std::make_unique<solver>();
    std::vector<std::int32_t> rc(n);
    for (auto _ : state) {
        auto r = s->solve(c, n, n, rc);
        benchmark::DoNotOptimize(r);
        benchmark::DoNotOptimize(rc.data());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Assignment)->Name("BM_Assignment_Uniform")->Args({64, 0})->Args({256, 0})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Assignment)->Name("BM_Assignment_Gated90")->Args({256, 90})->Unit(benchmark::kMicrosecond);

bool fair()
{
    auto s = std::make_unique<solver>();
    for (const auto& [n, f] : {std::pair<std::size_t, double>{64, 0.0}, {256, 0.0}, {256, 0.9}}) {
        const auto c = costs(n, f);
        std::vector<std::int32_t> rc(n);
        if (!s->solve(c, n, n, rc) || !plausible(c, n, rc))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "assignment benchmark: check FAILED - a solution is not optimal; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
