//
//  flat_id_map_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::flat_id_map against std::unordered_map<std::uint64_t, int>, like for like: insert 4,096 random
//  ids, find every one, erase half, find again.
//  Both sides get identical data (one fixed xorshift stream) and do the same
//  operations in the same order, and each returns a checksum of what it saw.
//  main() runs a fairness check first and refuses to benchmark if the
//  checksums disagree (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/containers/flat_id_map.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
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

constexpr std::size_t items = 4096;
const std::vector<std::uint64_t> keys = random_keys(items, (1ULL << 40) - 1);

std::uint64_t run_core()
{
    snicholls::containers::flat_id_map<int> m(items);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i)
        m.insert(keys[i], static_cast<int>(i));
    for (std::size_t i = 0; i < items; ++i)
        if (const int* v = m.find(keys[i])) sum += static_cast<std::uint64_t>(*v);
    for (std::size_t i = 0; i < items; i += 2)
        m.erase(keys[i]);
    for (std::size_t i = 0; i < items; ++i)
        if (const int* v = m.find(keys[i])) sum += static_cast<std::uint64_t>(*v);
    return sum + m.size();
}

std::uint64_t run_std()
{
    std::unordered_map<std::uint64_t, int> m;
    m.reserve(items);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i)
        m.emplace(keys[i], static_cast<int>(i));
    for (std::size_t i = 0; i < items; ++i)
        if (const auto it = m.find(keys[i]); it != m.end()) sum += static_cast<std::uint64_t>(it->second);
    for (std::size_t i = 0; i < items; i += 2)
        m.erase(keys[i]);
    for (std::size_t i = 0; i < items; ++i)
        if (const auto it = m.find(keys[i]); it != m.end()) sum += static_cast<std::uint64_t>(it->second);
    return sum + m.size();
}

void BM_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_core());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Core)->Name("flat_id_map");

void BM_Std(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_std());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Std)->Name("std_unordered_map");

} // namespace

int main(int argc, char** argv)
{
    if (run_core() != run_std()) {
        std::fprintf(stderr, "flat_id_map benchmark: fairness check FAILED - the checksums disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
