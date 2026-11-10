//
//  md_array_benchmark.cpp
//  snicholls-core
//
//  Summing a 16 x 16 x 16 x 16 grid in row-major order: snicholls::containers::md_array's a(i, j, k, l)
//  against the same offsets written by hand. The fairness check compares the sums.
//

#include <snicholls/core/containers/md_array.hpp>

#include <benchmark/benchmark.h>

#include <cstdio>

namespace {

using grid = snicholls::containers::md_array<double, 16, 16, 16, 16>;

grid data = [] {   // not const: the optimiser must not fold the sums
    grid g{};
    for (std::size_t i = 0; i < grid::size; ++i)
        g.elements[i] = static_cast<double>(i % 97) * 0.5;
    return g;
}();

double through_md_array()
{
    double total = 0;
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j)
            for (std::size_t k = 0; k < 16; ++k)
                for (std::size_t l = 0; l < 16; ++l)
                    total += data(i, j, k, l);
    return total;
}

double by_hand()
{
    double total = 0;
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j)
            for (std::size_t k = 0; k < 16; ++k)
                for (std::size_t l = 0; l < 16; ++l)
                    total += data.elements[((i * 16 + j) * 16 + k) * 16 + l];
    return total;
}

void BM_MdArray_Core(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(data.elements.data());
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(through_md_array());
    }
}
void BM_MdArray_OffsetsByHand(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(data.elements.data());
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(by_hand());
    }
}
BENCHMARK(BM_MdArray_Core);
BENCHMARK(BM_MdArray_OffsetsByHand);

} // namespace

int main(int argc, char** argv)
{
    if (through_md_array() != by_hand()) {
        std::fprintf(stderr, "md_array benchmark: fairness check FAILED\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
