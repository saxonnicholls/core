//
//  perfect_map_benchmark.cpp
//  snicholls-core
//
//  Looking up 512 instrument codes (text keys), 64 times each in a fixed shuffled order: the
//  perfect hash (snicholls::containers::perfect_map), the sorted table (static_map), a
//  std::unordered_map, and a binary search over a sorted std::array written by hand. All four
//  hold the same 512 entries; the fairness check compares the sums of the values found.
//

#include <snicholls/core/containers/perfect_map.hpp>
#include <snicholls/core/containers/static_map.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t n = 512;

// 512 four-letter codes "AAAA", "AAAB", ... spaced through the alphabet, each with a value
constexpr std::array<char, n * 4> code_text = [] {
    std::array<char, n * 4> t{};
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t x = i * 97 + 13;
        for (std::size_t j = 0; j < 4; ++j) {
            t[i * 4 + 3 - j] = static_cast<char>('A' + x % 26);
            x /= 26;
        }
    }
    return t;
}();

constexpr std::array<std::pair<std::string_view, std::uint32_t>, n> entries = [] {
    std::array<std::pair<std::string_view, std::uint32_t>, n> e{};
    for (std::size_t i = 0; i < n; ++i)
        e[i] = {std::string_view(code_text.data() + i * 4, 4), static_cast<std::uint32_t>(i)};
    return e;
}();

constexpr snicholls::containers::perfect_map<std::string_view, std::uint32_t, n> perfect{entries};
constexpr snicholls::containers::static_map<std::string_view, std::uint32_t, n> sorted{entries};
const std::unordered_map<std::string_view, std::uint32_t> hashed(entries.begin(), entries.end());
const std::array<std::pair<std::string_view, std::uint32_t>, n> by_hand = [] {
    auto a = entries;
    std::sort(a.begin(), a.end());
    return a;
}();

const std::vector<std::string_view> queries = [] {
    std::vector<std::string_view> q;
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    for (std::size_t i = 0; i < n * 64; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        q.push_back(entries[x % n].first);
    }
    return q;
}();

std::uint64_t via_perfect()
{
    std::uint64_t s = 0;
    for (const auto q : queries)
        s += *perfect.find(q);
    return s;
}

std::uint64_t via_static()
{
    std::uint64_t s = 0;
    for (const auto q : queries)
        s += *sorted.find(q);
    return s;
}

std::uint64_t via_unordered()
{
    std::uint64_t s = 0;
    for (const auto q : queries)
        s += hashed.find(q)->second;
    return s;
}

std::uint64_t via_hand_binary_search()
{
    std::uint64_t s = 0;
    for (const auto q : queries) {
        const auto it = std::lower_bound(by_hand.begin(), by_hand.end(), q, [](const auto& e, std::string_view k) { return e.first < k; });
        s += it->second;
    }
    return s;
}

template <std::uint64_t (*Run)()>
void run(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(Run());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(queries.size()));
}

BENCHMARK(run<via_perfect>)->Name("BM_Lookup_PerfectMap");
BENCHMARK(run<via_static>)->Name("BM_Lookup_StaticMap");
BENCHMARK(run<via_hand_binary_search>)->Name("BM_Lookup_BinarySearchByHand");
BENCHMARK(run<via_unordered>)->Name("BM_Lookup_StdUnorderedMap");

} // namespace

int main(int argc, char** argv)
{
    const std::uint64_t expected = via_unordered();
    if (via_perfect() != expected || via_static() != expected || via_hand_binary_search() != expected) {
        std::fprintf(stderr, "perfect_map benchmark: fairness check FAILED - the lookups disagree\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
