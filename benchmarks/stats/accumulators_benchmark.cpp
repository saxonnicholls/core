//
//  accumulators_benchmark.cpp
//  snicholls-core
//
//  1,000,000 returns into snicholls::stats::variance and covariance (mean, variance, covariance,
//  correlation), against the same Welford updates written by hand, in the same order. The fairness
//  check requires identical results.
//

#include <snicholls/core/stats/accumulators.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

std::vector<double> make(std::uint64_t x)
{
    std::vector<double> v(1'000'000);
    for (double& r : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        r = static_cast<double>(x % 20001) / 1e6 - 0.01;
    }
    return v;
}

const std::vector<double> xs = make(0x243f6a8885a308d3ULL);
const std::vector<double> ys = make(0x13198a2e03707344ULL);

struct moments {
    double mean, variance, covariance, correlation;
    bool operator==(const moments&) const = default;
};

moments core()
{
    snicholls::stats::variance<double> v;
    snicholls::stats::covariance<double> c;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        v.add(xs[i]);
        c.add(xs[i], ys[i]);
    }
    return {v.mean(), v.value(), c.value(), c.correlation()};
}

moments by_hand()
{
    std::uint64_t n = 0;
    double m = 0, m2 = 0;
    std::uint64_t cn = 0;
    double mx = 0, my = 0, cxy = 0, m2x = 0, m2y = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        ++n;
        const double d = xs[i] - m;
        m += d / static_cast<double>(n);
        m2 += d * (xs[i] - m);
        ++cn;
        const double dx = xs[i] - mx;
        mx += dx / static_cast<double>(cn);
        const double dy = ys[i] - my;
        my += dy / static_cast<double>(cn);
        cxy += dx * (ys[i] - my);
        m2x += dx * (xs[i] - mx);
        m2y += dy * (ys[i] - my);
    }
    return {m, m2 / static_cast<double>(n - 1), cxy / static_cast<double>(cn - 1), cxy / std::sqrt(m2x * m2y)};
}

void BM_Accumulators_Core(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(core());
    s.SetItemsProcessed(s.iterations() * static_cast<std::int64_t>(xs.size()));
}
BENCHMARK(BM_Accumulators_Core)->Unit(benchmark::kMicrosecond);

void BM_Accumulators_WelfordByHand(benchmark::State& s)
{
    for (auto _ : s)
        benchmark::DoNotOptimize(by_hand());
    s.SetItemsProcessed(s.iterations() * static_cast<std::int64_t>(xs.size()));
}
BENCHMARK(BM_Accumulators_WelfordByHand)->Unit(benchmark::kMicrosecond);

} // namespace

int main(int argc, char** argv)
{
    if (!(core() == by_hand())) {
        std::fprintf(stderr, "accumulators benchmark: fairness check FAILED - the moments differ\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
