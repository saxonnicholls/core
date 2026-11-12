//
//  object_pool_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::object_pool against new and delete, like for like: create 4,096 orders,
//  destroy every other one, create 2,048 more, destroy everything. A third row
//  runs the same pool with its chunks from a std::pmr::unsynchronized_pool_resource,
//  to show what taking chunks through a memory_resource costs.
//  Both sides get identical data and do the same
//  operations in the same order, and each returns a checksum of what it saw.
//  main() runs a fairness check first and refuses to benchmark if the
//  checksums disagree (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/containers/object_pool.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory_resource>
#include <vector>

namespace {

constexpr std::size_t items = 4096;

struct order {
    long id;
    long quantity;
    order(long i, long q) noexcept : id(i), quantity(q) {}
};

std::uint64_t run_core(std::pmr::memory_resource* resource = std::pmr::new_delete_resource())
{
    snicholls::containers::object_pool<order, 1024> pool(resource);
    std::vector<order*> live;
    live.reserve(items);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i) live.push_back(pool.create(static_cast<long>(i), 1));
    for (std::size_t i = 0; i < items; i += 2) { sum += static_cast<std::uint64_t>(live[i]->id); pool.destroy(live[i]); }
    for (std::size_t i = 0; i < items; i += 2) live[i] = pool.create(static_cast<long>(i + items), 2);
    for (order* o : live) { sum += static_cast<std::uint64_t>(o->id + o->quantity); pool.destroy(o); }
    return sum;
}

std::uint64_t run_std()
{
    std::vector<order*> live;
    live.reserve(items);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i) live.push_back(new order(static_cast<long>(i), 1));
    for (std::size_t i = 0; i < items; i += 2) { sum += static_cast<std::uint64_t>(live[i]->id); delete live[i]; }
    for (std::size_t i = 0; i < items; i += 2) live[i] = new order(static_cast<long>(i + items), 2);
    for (order* o : live) { sum += static_cast<std::uint64_t>(o->id + o->quantity); delete o; }
    return sum;
}

void BM_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_core());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Core)->Name("object_pool");

void BM_CorePmr(benchmark::State& state)
{
    std::pmr::unsynchronized_pool_resource upstream;
    for (auto _ : state)
        benchmark::DoNotOptimize(run_core(&upstream));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_CorePmr)->Name("object_pool_pool_resource");

void BM_Std(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_std());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Std)->Name("new_delete");

} // namespace

int main(int argc, char** argv)
{
    std::pmr::unsynchronized_pool_resource check;
    if (run_core() != run_std() || run_core(&check) != run_std()) {
        std::fprintf(stderr, "object_pool benchmark: fairness check FAILED - the checksums disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
