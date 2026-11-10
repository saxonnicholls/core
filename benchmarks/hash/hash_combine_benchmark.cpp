//
//  hash_combine_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::hash_combine against the same combine written out by hand, with
//  the same constants and the same operations in the same order. Both hash the
//  same 4,096 three-field records, each as a chain of three combines from seed
//  0. main() runs a fairness check first and refuses to benchmark if the two
//  disagree on any record (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/hash/hash_combine.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct record {
    std::size_t a;
    std::size_t b;
    std::size_t c;
};

std::vector<record> make_records()
{
    std::vector<record> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for both
    for (auto& r : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; r.a = static_cast<std::size_t>(x);
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; r.b = static_cast<std::size_t>(x);
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; r.c = static_cast<std::size_t>(x);
    }
    return out;
}

const std::vector<record> records = make_records();

// The library
std::size_t hash_core(const record& r) noexcept
{
    std::size_t seed = 0;
    seed = snicholls::hash_combine(seed, r.a);
    seed = snicholls::hash_combine(seed, r.b);
    seed = snicholls::hash_combine(seed, r.c);
    return seed;
}

// By hand: Boost.ContainerHash 1.81+ hash_combine, written out
inline std::size_t combine_by_hand(std::size_t seed, std::size_t h) noexcept
{
    std::uint64_t x = static_cast<std::uint64_t>(seed) + 0x9e3779b9U + static_cast<std::uint64_t>(h);
    x ^= x >> 32;
    x *= 0xe9846af9b1a615dULL;
    x ^= x >> 32;
    x *= 0xe9846af9b1a615dULL;
    x ^= x >> 28;
    return static_cast<std::size_t>(x);
}

std::size_t hash_by_hand(const record& r) noexcept
{
    std::size_t seed = 0;
    seed = combine_by_hand(seed, r.a);
    seed = combine_by_hand(seed, r.b);
    seed = combine_by_hand(seed, r.c);
    return seed;
}

void BM_HashCombine_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& r : records)
            benchmark::DoNotOptimize(hash_core(r));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(records.size()));
}
BENCHMARK(BM_HashCombine_Core);

void BM_HashCombine_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& r : records)
            benchmark::DoNotOptimize(hash_by_hand(r));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(records.size()));
}
BENCHMARK(BM_HashCombine_HandWritten);

bool fair()
{
    for (const auto& r : records)
        if (hash_core(r) != hash_by_hand(r))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "hash_combine benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
