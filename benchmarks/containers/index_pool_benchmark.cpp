//
//  index_pool_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::index_pool against a hand-written index free list
//  (a std::vector of records and a std::vector of free indices) and against
//  object_pool, like for like: create 4,096 orders, destroy every other one,
//  create 2,048 more, read every live order, destroy everything. Each side
//  gets identical data, does the same operations in the same order and returns
//  a checksum of what it saw; main() refuses to benchmark if they disagree
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/containers/index_pool.hpp>
#include <snicholls/core/containers/object_pool.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

constexpr std::size_t items = 4096;

struct order
{
    std::int64_t id;
    std::int64_t quantity;
};

using pool_t = snicholls::containers::index_pool<order>;

std::vector<std::uint64_t> region_words(pool_t::bytes_for(items) / 8 + 1);

std::uint64_t run_index_pool()
{
    auto pool = *pool_t::format(std::as_writable_bytes(std::span(region_words)));
    std::vector<std::uint32_t> live;
    live.reserve(items);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i) live.push_back(*pool.create(order{static_cast<std::int64_t>(i), 1}));
    for (std::size_t i = 0; i < items; i += 2) { sum += static_cast<std::uint64_t>(pool[live[i]].id); pool.destroy(live[i]); }
    for (std::size_t i = 0; i < items; i += 2) live[i] = *pool.create(order{static_cast<std::int64_t>(i + items), 2});
    for (std::uint32_t ix : live) { sum += static_cast<std::uint64_t>(pool[ix].id + pool[ix].quantity); pool.destroy(ix); }
    return sum;
}

std::uint64_t run_hand_written()
{
    std::vector<order> slots;
    slots.reserve(items);
    std::vector<std::uint32_t> free_list;
    free_list.reserve(items);
    auto create = [&](order o) {
        if (!free_list.empty()) {
            const std::uint32_t i = free_list.back();
            free_list.pop_back();
            slots[i] = o;
            return i;
        }
        slots.push_back(o);
        return static_cast<std::uint32_t>(slots.size() - 1);
    };
    std::vector<std::uint32_t> live;
    live.reserve(items);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i) live.push_back(create(order{static_cast<std::int64_t>(i), 1}));
    for (std::size_t i = 0; i < items; i += 2) { sum += static_cast<std::uint64_t>(slots[live[i]].id); free_list.push_back(live[i]); }
    for (std::size_t i = 0; i < items; i += 2) live[i] = create(order{static_cast<std::int64_t>(i + items), 2});
    for (std::uint32_t ix : live) { sum += static_cast<std::uint64_t>(slots[ix].id + slots[ix].quantity); free_list.push_back(ix); }
    return sum;
}

std::uint64_t run_object_pool()
{
    snicholls::containers::object_pool<order, items> pool;
    std::vector<order*> live;
    live.reserve(items);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i) live.push_back(pool.create(order{static_cast<std::int64_t>(i), 1}));
    for (std::size_t i = 0; i < items; i += 2) { sum += static_cast<std::uint64_t>(live[i]->id); pool.destroy(live[i]); }
    for (std::size_t i = 0; i < items; i += 2) live[i] = pool.create(order{static_cast<std::int64_t>(i + items), 2});
    for (order* o : live) { sum += static_cast<std::uint64_t>(o->id + o->quantity); pool.destroy(o); }
    return sum;
}

void BM_IndexPool(benchmark::State& state)
{
    for (auto _ : state) benchmark::DoNotOptimize(run_index_pool());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_IndexPool)->Name("index_pool");

void BM_HandWritten(benchmark::State& state)
{
    for (auto _ : state) benchmark::DoNotOptimize(run_hand_written());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_HandWritten)->Name("vector_free_list");

void BM_ObjectPool(benchmark::State& state)
{
    for (auto _ : state) benchmark::DoNotOptimize(run_object_pool());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_ObjectPool)->Name("object_pool");

} // namespace

int main(int argc, char** argv)
{
    const std::uint64_t want = run_hand_written();
    if (run_index_pool() != want || run_object_pool() != want) {
        std::fprintf(stderr, "index_pool benchmark: fairness check FAILED - the checksums disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
