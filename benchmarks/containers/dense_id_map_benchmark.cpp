//
//  dense_id_map_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::dense_id_map against std::unordered_map<std::uint64_t, int*>, like for like: a sliding window of
//  sequential exchange ids: insert the next id, erase the one 512 behind,
//  find the one 256 behind, for 65,536 ids.
//  Both sides get identical data and do the same
//  operations in the same order, and each returns a checksum of what it saw.
//  main() runs a fairness check first and refuses to benchmark if the
//  checksums disagree (CLAUDE.md "Zero overhead").
//
//  The container on its own, outside any order book: a container-only group.
//

#include <snicholls/core/containers/dense_id_map.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t items = 65536;
constexpr std::uint64_t first_id = 5'000'000;
int slots[1024];

std::uint64_t run_core()
{
    snicholls::containers::dense_id_map<int*> m(1024);
    std::uint64_t found = 0;
    for (std::uint64_t id = first_id; id < first_id + items; ++id) {
        m.insert(id, &slots[id % 1024]);
        if (id >= first_id + 512) m.erase(id - 512);
        if (id >= first_id + 256) found += m.find(id - 256) != nullptr ? 1 : 0;
    }
    return found * 1'000'000 + m.size();
}

std::uint64_t run_std()
{
    std::unordered_map<std::uint64_t, int*> m;
    m.reserve(1024);
    std::uint64_t found = 0;
    for (std::uint64_t id = first_id; id < first_id + items; ++id) {
        m.emplace(id, &slots[id % 1024]);
        if (id >= first_id + 512) m.erase(id - 512);
        if (id >= first_id + 256) found += m.find(id - 256) != m.end() ? 1 : 0;
    }
    return found * 1'000'000 + m.size();
}

void BM_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_core());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Core)->Name("dense_id_map");

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
        std::fprintf(stderr, "dense_id_map benchmark: fairness check FAILED - the checksums disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
