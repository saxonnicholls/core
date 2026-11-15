//
//  normal_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::stats::normal_inverse_cdf and normal_box_muller against the same draws written by hand over the same
//  wyrand stream (the same uniform, the same normal_quantile; the same logarithm, square root, sine and cosine), a
//  million draws each. main() runs a fairness check first and refuses to benchmark unless every draw agrees to the bit.
//

#include <snicholls/core/stats/normal.hpp>

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace st = snicholls::stats;
constexpr std::size_t n = 1 << 20;

void inverse_core(std::vector<double>& out)
{
    st::wyrand g(1);
    st::normal_inverse_cdf z;
    for (double& x : out)
        x = z(g);
}

void inverse_by_hand(std::vector<double>& out)
{
    st::wyrand g(1);
    for (double& x : out)
        x = snicholls::math::normal_quantile(static_cast<double>((g() >> 12) * 2 + 1) * 0x1.0p-53);
}

void box_muller_core(std::vector<double>& out)
{
    st::wyrand g(1);
    st::normal_box_muller z;
    for (double& x : out)
        x = z(g);
}

void box_muller_by_hand(std::vector<double>& out)
{
    st::wyrand g(1);
    for (std::size_t i = 0; i + 1 < out.size(); i += 2) {
        const double u1 = static_cast<double>((g() >> 12) * 2 + 1) * 0x1.0p-53, u2 = static_cast<double>(g() >> 11) * 0x1.0p-53;
        const double r = std::sqrt(-2.0 * std::log(u1)), t = 2.0 * snicholls::math::pi * u2;
        out[i] = r * std::cos(t);
        out[i + 1] = r * std::sin(t);
    }
}

template <void (*Draw)(std::vector<double>&)>
void BM_Draws(benchmark::State& state)
{
    std::vector<double> out(n);
    for (auto _ : state) {
        Draw(out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(n));
}
BENCHMARK(BM_Draws<inverse_core>)->Name("BM_NormalInverseCdf_Core")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Draws<inverse_by_hand>)->Name("BM_NormalInverseCdf_HandWritten")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Draws<box_muller_core>)->Name("BM_NormalBoxMuller_Core")->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Draws<box_muller_by_hand>)->Name("BM_NormalBoxMuller_HandWritten")->Unit(benchmark::kMillisecond);

bool fair()
{
    std::vector<double> a(n), b(n), c(n), d(n);
    inverse_core(a);
    inverse_by_hand(b);
    box_muller_core(c);
    box_muller_by_hand(d);
    return std::memcmp(a.data(), b.data(), n * sizeof(double)) == 0 && std::memcmp(c.data(), d.data(), n * sizeof(double)) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "normal benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
