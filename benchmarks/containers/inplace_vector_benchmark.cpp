//
//  inplace_vector_benchmark.cpp
//  snicholls-core
//
//  Building a vector of 16 doubles per message, 4,096 messages: snicholls::containers::inplace_vector
//  against std::vector (reserved, then reused) and a plain array written by hand. The fairness check
//  compares the sums.
//

#include <snicholls/core/containers/inplace_vector.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstdio>
#include <vector>

namespace {

double with_inplace()
{
    double total = 0;
    for (int m = 0; m < 4096; ++m) {
        snicholls::containers::inplace_vector<double, 16> v;
        for (int i = 0; i < 16; ++i)
            v.push_back(m * 0.5 + i);
        for (const double x : v)
            total += x;
    }
    return total;
}

double with_std_vector()
{
    double total = 0;
    std::vector<double> v;
    v.reserve(16);
    for (int m = 0; m < 4096; ++m) {
        v.clear();
        for (int i = 0; i < 16; ++i)
            v.push_back(m * 0.5 + i);
        for (const double x : v)
            total += x;
    }
    return total;
}

double by_hand()
{
    double total = 0;
    for (int m = 0; m < 4096; ++m) {
        std::array<double, 16> v{};
        std::size_t n = 0;
        for (int i = 0; i < 16; ++i)
            v[n++] = m * 0.5 + i;
        for (std::size_t i = 0; i < n; ++i)
            total += v[i];
    }
    return total;
}

void BM_InplaceVector_Core(benchmark::State& s) { for (auto _ : s) benchmark::DoNotOptimize(with_inplace()); }
void BM_InplaceVector_StdVectorReused(benchmark::State& s) { for (auto _ : s) benchmark::DoNotOptimize(with_std_vector()); }
void BM_InplaceVector_ArrayByHand(benchmark::State& s) { for (auto _ : s) benchmark::DoNotOptimize(by_hand()); }
BENCHMARK(BM_InplaceVector_Core);
BENCHMARK(BM_InplaceVector_StdVectorReused);
BENCHMARK(BM_InplaceVector_ArrayByHand);

} // namespace

int main(int argc, char** argv)
{
    if (with_inplace() != by_hand() || with_std_vector() != by_hand()) {
        std::fprintf(stderr, "inplace_vector benchmark: fairness check FAILED\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
