//
//  bloom_filter_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::bloom_filter against the same filter written out by
//  hand over a std::vector<std::uint64_t>: the same wyhash, the same double
//  hashing, the same bit positions in the same order. Both are filled with the
//  same 65,536 keys and queried with the same 4,096 keys, half members and half
//  strangers, and a std::unordered_set lookup runs beside them for scale.
//  main() runs a fairness check first and refuses to benchmark if the two
//  filters disagree on any query (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/containers/bloom_filter.hpp>
#include <snicholls/core/hash/wyhash.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

std::vector<std::string> make_keys(std::size_t n, std::uint64_t seed)
{
    std::vector<std::string> out;
    out.reserve(n);
    std::uint64_t x = seed;
    for (std::size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        out.push_back("0x" + std::to_string(x));
    }
    return out;
}

const std::vector<std::string> members = make_keys(65'536, 0x243f6a8885a308d3ULL);
const std::vector<std::string> strangers = make_keys(2048, 0x13198a2e03707344ULL);

std::vector<std::string> make_queries()
{
    std::vector<std::string> q;
    for (std::size_t i = 0; i < 2048; ++i) {
        q.push_back(members[i * 31 % members.size()]);
        q.push_back(strangers[i]);
    }
    return q;
}
const std::vector<std::string> queries = make_queries();

// By hand: the same arithmetic over a plain vector of words
struct hand_filter {
    std::size_t bits;
    std::uint32_t hashes;
    std::vector<std::uint64_t> words;

    explicit hand_filter(snicholls::containers::bloom_parameters p) : bits(p.bits), hashes(p.hashes), words(p.bits / 64, 0) {}

    void add(std::string_view key) noexcept
    {
        const std::uint64_t h1 = snicholls::wyhash(key);
        const std::uint64_t h2 = snicholls::splitmix64(h1) | 1u;
        for (std::uint32_t i = 0; i < hashes; ++i) {
            const std::uint64_t bit = (h1 + i * h2) % bits;
            words[bit >> 6] |= std::uint64_t{1} << (bit & 63);
        }
    }

    bool maybe_contains(std::string_view key) const noexcept
    {
        const std::uint64_t h1 = snicholls::wyhash(key);
        const std::uint64_t h2 = snicholls::splitmix64(h1) | 1u;
        for (std::uint32_t i = 0; i < hashes; ++i) {
            const std::uint64_t bit = (h1 + i * h2) % bits;
            if ((words[bit >> 6] & (std::uint64_t{1} << (bit & 63))) == 0)
                return false;
        }
        return true;
    }
};

const snicholls::containers::bloom_parameters sizing = snicholls::containers::bloom_sizing(members.size(), 0.01);

snicholls::containers::bloom_filter make_core()
{
    snicholls::containers::bloom_filter f(sizing);
    for (const auto& k : members)
        f.add(k);
    return f;
}

hand_filter make_hand()
{
    hand_filter f(sizing);
    for (const auto& k : members)
        f.add(k);
    return f;
}

std::unordered_set<std::string> make_set() { return {members.begin(), members.end()}; }

const snicholls::containers::bloom_filter core_filter = make_core();
const hand_filter hand = make_hand();
const std::unordered_set<std::string> exact = make_set();

void BM_BloomFilter_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& q : queries)
            benchmark::DoNotOptimize(core_filter.maybe_contains(q));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(queries.size()));
}
BENCHMARK(BM_BloomFilter_Core);

void BM_BloomFilter_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& q : queries)
            benchmark::DoNotOptimize(hand.maybe_contains(q));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(queries.size()));
}
BENCHMARK(BM_BloomFilter_HandWritten);

void BM_BloomFilter_UnorderedSetForScale(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& q : queries)
            benchmark::DoNotOptimize(exact.contains(q));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(queries.size()));
}
BENCHMARK(BM_BloomFilter_UnorderedSetForScale);

bool fair()
{
    for (const auto& q : queries)
        if (core_filter.maybe_contains(q) != hand.maybe_contains(q))
            return false;
    for (const auto& m : members)
        if (!core_filter.maybe_contains(m))
            return false;   // a false negative would be a bug, not a benchmark
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "bloom_filter benchmark: fairness check FAILED - the filters disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
