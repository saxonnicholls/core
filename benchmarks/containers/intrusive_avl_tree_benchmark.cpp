//
//  intrusive_avl_tree_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::intrusive_avl_tree against std::set<long>, like for like: link 4,096 price levels
//  with random prices, find each, walk in order, unlink every one. The tree's
//  nodes belong to the caller (a vector, allocated outside the timed loop as a
//  pool would hold them); std::set allocates a node per insert, which is the
//  cost an intrusive tree exists to avoid.
//  Both sides get identical data (one fixed xorshift stream) and do the same
//  operations in the same order, and each returns a checksum of what it saw.
//  main() runs a fairness check first and refuses to benchmark if the
//  checksums disagree (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/containers/intrusive_avl_tree.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <set>
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
const std::vector<std::uint64_t> prices = random_keys(items, (1ULL << 30) - 1);

struct level {
    long price = 0;
    snicholls::containers::avl_hook<level> hook;
};
using tree = snicholls::containers::intrusive_avl_tree<
    level, snicholls::containers::member_avl_traits<level, &level::hook, &level::price, std::less<>>>;

std::vector<level> nodes(items);

std::uint64_t run_core()
{
    tree t;
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i) {
        nodes[i].price = static_cast<long>(prices[i]);
        if (!t.insert(nodes[i]).second) nodes[i].price = -1;   // a duplicate price: not linked
    }
    for (std::size_t i = 0; i < items; ++i)
        if (t.find(static_cast<long>(prices[i])) != nullptr) sum += 1;
    for (const level& l : t) sum += static_cast<std::uint64_t>(l.price);
    for (std::size_t i = 0; i < items; ++i)
        if (nodes[i].price >= 0) t.erase(nodes[i]);
    return sum + t.size();
}

std::uint64_t run_std()
{
    std::set<long> t;
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < items; ++i) t.insert(static_cast<long>(prices[i]));
    for (std::size_t i = 0; i < items; ++i)
        if (t.find(static_cast<long>(prices[i])) != t.end()) sum += 1;
    for (const long p : t) sum += static_cast<std::uint64_t>(p);
    for (std::size_t i = 0; i < items; ++i) t.erase(static_cast<long>(prices[i]));
    return sum + t.size();
}

void BM_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_core());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Core)->Name("intrusive_avl_tree");

void BM_Std(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(run_std());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items));
}
BENCHMARK(BM_Std)->Name("std_set");

} // namespace

int main(int argc, char** argv)
{
    if (run_core() != run_std()) {
        std::fprintf(stderr, "intrusive_avl_tree benchmark: fairness check FAILED - the checksums disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
