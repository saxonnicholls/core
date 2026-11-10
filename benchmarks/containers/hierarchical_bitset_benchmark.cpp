//
//  hierarchical_bitset_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::hierarchical_bitset against std::vector<bool>, like for like: set 1,024 random bits
//  of 1,048,576, then walk every set bit with find_next (std::vector<bool>:
//  a linear scan for the next set bit, which is what a plain bitset offers).
//  Both sides get identical data (one fixed xorshift stream) and do the same
//  operations in the same order, and each returns a checksum of what it saw.
//  main() runs a fairness check first and refuses to benchmark if the
//  checksums disagree (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/containers/hierarchical_bitset.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

std::vector<std::uint64_t> random_keys(std::size_t n, std::uint64_t mask)
{
    std::vector<std::uint64_t> out(n);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    for (auto& k : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        k = x & mask;
    }
    return out;
}

constexpr std::size_t bits = 1u << 20;
constexpr std::size_t items = 1024;
const std::vector<std::uint64_t> positions = random_keys(items, bits - 1);

std::uint64_t run_core()
{
    snicholls::containers::hierarchical_bitset b(bits);
    for (const auto p : positions) b.set(static_cast<std::size_t>(p));
    std::uint64_t sum = 0;
    for (std::size_t i = b.find_next(0); i != snicholls::containers::hierarchical_bitset::npos; i = b.find_next(i + 1))
        sum += i;
    return sum;
}

std::uint64_t run_std()
{
    std::vector<bool> b(bits);
    for (const auto p : positions) b[static_cast<std::size_t>(p)] = true;
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < bits; ++i)
        if (b[i]) sum += i;
    return sum;
}

void BM_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_core());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Core)->Name("hierarchical_bitset");

void BM_Std(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_std());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Std)->Name("std_vector_bool");

} // namespace

int main(int argc, char** argv)
{
    if (run_core() != run_std()) {
        std::fprintf(stderr, "hierarchical_bitset benchmark: fairness check FAILED - the checksums disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
