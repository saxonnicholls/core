//
//  offset_ptr_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::offset_ptr against a raw pointer: the same 4,096
//  nodes in one array, linked in the same shuffled order, walked end to end
//  summing the values. Both lists are built from one permutation and each walk
//  returns its sum; main() refuses to benchmark if they disagree (CLAUDE.md
//  "Zero overhead").
//

#include <snicholls/core/containers/offset_ptr.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

constexpr std::size_t nodes = 4096;

struct off_node
{
    std::uint64_t value;
    snicholls::containers::offset_ptr<off_node> next;
};
struct raw_node
{
    std::uint64_t value;
    raw_node* next;
};

std::vector<std::size_t> permutation()
{
    std::vector<std::size_t> p(nodes);
    std::iota(p.begin(), p.end(), std::size_t{0});
    std::uint64_t s = 0x9e3779b97f4a7c15ull;
    for (std::size_t i = nodes - 1; i > 0; --i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        std::swap(p[i], p[s % (i + 1)]);
    }
    return p;
}
const std::vector<std::size_t> order = permutation();

std::vector<off_node> off_list = [] {
    std::vector<off_node> v(nodes);
    for (std::size_t i = 0; i < nodes; ++i) v[i].value = i * 3 + 1;
    for (std::size_t k = 0; k + 1 < nodes; ++k) v[order[k]].next = &v[order[k + 1]];
    return v;
}();
std::vector<raw_node> raw_list = [] {
    std::vector<raw_node> v(nodes);
    for (std::size_t i = 0; i < nodes; ++i) v[i] = raw_node{i * 3 + 1, nullptr};
    for (std::size_t k = 0; k + 1 < nodes; ++k) v[order[k]].next = &v[order[k + 1]];
    return v;
}();

std::uint64_t walk_offset()
{
    std::uint64_t sum = 0;
    for (const off_node* p = &off_list[order[0]]; p; p = p->next.get()) sum += p->value;
    return sum;
}
std::uint64_t walk_raw()
{
    std::uint64_t sum = 0;
    for (const raw_node* p = &raw_list[order[0]]; p; p = p->next) sum += p->value;
    return sum;
}

void BM_Offset(benchmark::State& state)
{
    for (auto _ : state) benchmark::DoNotOptimize(walk_offset());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nodes));
}
BENCHMARK(BM_Offset)->Name("offset_ptr_walk");

void BM_Raw(benchmark::State& state)
{
    for (auto _ : state) benchmark::DoNotOptimize(walk_raw());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nodes));
}
BENCHMARK(BM_Raw)->Name("raw_pointer_walk");

} // namespace

int main(int argc, char** argv)
{
    if (walk_offset() != walk_raw()) {
        std::fprintf(stderr, "offset_ptr benchmark: fairness check FAILED - the sums disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
