//
//  resample_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::stats::systematic_resample against the same walk written by hand: the weights' sum, one uniform from
//  the same wyrand stream, then (i + u) / M walked against the running cumulative weight, 2^16 particles. main() runs
//  a fairness check first and refuses to benchmark unless every index agrees.
//

#include <snicholls/core/stats/resample.hpp>

#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace st = snicholls::stats;
constexpr std::size_t n = 1 << 16;

const std::vector<double>& weights()
{
    static const std::vector<double> w = [] {
        std::vector<double> v(n);
        st::wyrand g(3);
        for (double& x : v)
            x = st::uniform_real(g) * st::uniform_real(g);
        return v;
    }();
    return w;
}

void core(std::vector<std::size_t>& out)
{
    st::wyrand g(17);
    st::systematic_resample<double>(weights(), out, g);
}

void by_hand(std::vector<std::size_t>& out)
{
    const auto& w = weights();
    st::wyrand g(17);
    double sum = 0;
    for (const double x : w)
        sum += x;
    const double m = static_cast<double>(out.size());
    const double u0 = st::uniform_real(g);
    std::size_t j = 0;
    double cum = w[0] / sum;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double ui = (static_cast<double>(i) + u0) / m;
        while (ui >= cum && j < w.size() - 1)
            cum += w[++j] / sum;
        while (w[j] == 0.0 && j > 0)
            --j;
        out[i] = j;
    }
}

template <void (*Run)(std::vector<std::size_t>&)>
void BM_Resample(benchmark::State& state)
{
    std::vector<std::size_t> out(n);
    weights();
    for (auto _ : state) {
        Run(out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(n));
}
BENCHMARK(BM_Resample<core>)->Name("BM_SystematicResample_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Resample<by_hand>)->Name("BM_SystematicResample_HandWritten")->Unit(benchmark::kMicrosecond);

bool fair()
{
    std::vector<std::size_t> a(n), b(n);
    core(a);
    by_hand(b);
    return a == b;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "resample benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
