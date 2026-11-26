//
//  algorithm_benchmark.cpp
//  snicholls-core
//
//  snicholls::fill and snicholls::copy against std::fill, std::copy and the hand-written loop: 4,096 doubles, the
//  same values written in the same order. main() runs a fairness check first and refuses to benchmark if any pair of
//  results differs (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/algorithm.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t n = 4096;

std::vector<double> make_values()
{
    std::vector<double> out(n);
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<double>(x % 100000) / 100.0;
    }
    return out;
}

const std::vector<double> source = make_values();

void fill_core(std::vector<double>& out, double v) { snicholls::fill(out.begin(), out.end(), v); }
void fill_std(std::vector<double>& out, double v) { std::fill(out.begin(), out.end(), v); }
void fill_by_hand(std::vector<double>& out, double v)
{
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = v;
}

void copy_core(std::vector<double>& out) { snicholls::copy(source.begin(), source.end(), out.begin()); }
void copy_std(std::vector<double>& out) { std::copy(source.begin(), source.end(), out.begin()); }
void copy_by_hand(std::vector<double>& out)
{
    for (std::size_t i = 0; i < source.size(); ++i)
        out[i] = source[i];
}

template <void (*Fill)(std::vector<double>&, double)>
void BM_Fill(benchmark::State& state)
{
    std::vector<double> out(n);
    double v = 1.0;
    for (auto _ : state) {
        Fill(out, v);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
        v += 1.0;
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_Fill<fill_core>)->Name("BM_Fill_Core");
BENCHMARK(BM_Fill<fill_std>)->Name("BM_Fill_Std");
BENCHMARK(BM_Fill<fill_by_hand>)->Name("BM_Fill_HandWritten");

template <void (*Copy)(std::vector<double>&)>
void BM_Copy(benchmark::State& state)
{
    std::vector<double> out(n);
    for (auto _ : state) {
        Copy(out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_Copy<copy_core>)->Name("BM_Copy_Core");
BENCHMARK(BM_Copy<copy_std>)->Name("BM_Copy_Std");
BENCHMARK(BM_Copy<copy_by_hand>)->Name("BM_Copy_HandWritten");

bool fair()
{
    std::vector<double> a(n), b(n), c(n);
    fill_core(a, 3.5);
    fill_std(b, 3.5);
    fill_by_hand(c, 3.5);
    if (a != b || a != c)
        return false;
    copy_core(a);
    copy_std(b);
    copy_by_hand(c);
    return a == b && a == c && a == source;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "algorithm benchmark: fairness check FAILED - the results differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
