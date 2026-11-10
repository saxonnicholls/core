//
//  zip_benchmark.cpp
//  snicholls-core
//
//  snicholls::zip against the index loop: the dot product of two std::vector<double> of 4,096
//  elements, the same arithmetic in the same order. main() runs a fairness check first and
//  refuses to benchmark if the sums differ (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/zip.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

std::vector<double> make_values(std::uint64_t x)
{
    std::vector<double> out(4096);
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<double>(x % 100000) / 100.0;
    }
    return out;
}

const std::vector<double> left = make_values(0x243f6a8885a308d3ULL);
const std::vector<double> right = make_values(0x13198a2e03707344ULL);

double dot_zip()
{
    double sum = 0;
    for (auto [a, b] : snicholls::zip(left, right))
        sum += a * b;
    return sum;
}

double dot_by_hand()
{
    double sum = 0;
    for (std::size_t i = 0; i < left.size() && i < right.size(); ++i)
        sum += left[i] * right[i];
    return sum;
}

void BM_Zip_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_zip());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(left.size()));
}
BENCHMARK(BM_Zip_Core);

void BM_Zip_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_by_hand());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(left.size()));
}
BENCHMARK(BM_Zip_HandWritten);

} // namespace

int main(int argc, char** argv)
{
    if (dot_zip() != dot_by_hand()) {
        std::fprintf(stderr, "zip benchmark: fairness check FAILED - the sums differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
