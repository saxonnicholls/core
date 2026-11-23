//
//  key_value_store_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The key-value contract's dispatch cost on the reference store: a hand-written std::map (same comparator, no
//  result, no visitor), the concept path (a template over reference_key_value_store) and the virtual path (the
//  same template over key_value_store_interface, through key_value_store_adapter). Identical data - 10,000 keys
//  k/00000 .. k/09999 with 16-byte values - and identical arithmetic in the same order: 1,000 point reads summing
//  each value's first byte, a 100-key prefix scan summing value sizes, and 1,000 overwrites. main() runs a
//  fairness check first and refuses to benchmark if any pair disagrees (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/interfaces/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace snicholls;

namespace {

constexpr std::size_t keys = 10'000;
constexpr std::size_t probes = 1'000;

std::string key_of(std::size_t i)
{
    std::string k = std::to_string(i);
    return "k/" + std::string(5 - k.size(), '0') + k;
}

std::vector<std::byte> value_of(std::size_t i)
{
    std::vector<std::byte> v(16);
    for (std::size_t j = 0; j < v.size(); ++j)
        v[j] = static_cast<std::byte>((i * 31 + j) & 0xFF);
    return v;
}

const std::vector<std::string> probe_keys = [] {
    std::vector<std::string> out;
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    for (std::size_t i = 0; i < probes; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        out.push_back(key_of(x % keys));
    }
    return out;
}();

// --- hand-written --------------------------------------------------------------------------------------------------

struct by_hand_order {
    using is_transparent = void;
    bool operator()(kv_bytes a, kv_bytes b) const noexcept { return kv_less(a, b); }
};
using hand_map = std::map<std::vector<std::byte>, std::vector<std::byte>, by_hand_order>;

hand_map make_hand()
{
    hand_map m;
    for (std::size_t i = 0; i < keys; ++i) {
        const std::string k = key_of(i);
        const kv_bytes kb = kv_bytes_of(k);
        m.emplace(std::vector<std::byte>(kb.begin(), kb.end()), value_of(i));
    }
    return m;
}

std::uint64_t reads_by_hand(const hand_map& m)
{
    std::uint64_t sum = 0;
    for (const std::string& k : probe_keys)
        if (const auto it = m.find(kv_bytes_of(k)); it != m.end())
            sum += static_cast<std::uint64_t>(it->second[0]);
    return sum;
}

std::uint64_t scan_by_hand(const hand_map& m)
{
    std::uint64_t sum = 0;
    const kv_bytes prefix = kv_bytes_of(std::string_view("k/001"));
    for (auto it = m.lower_bound(prefix); it != m.end() && kv_starts_with(it->first, prefix); ++it)
        sum += it->second.size();
    return sum;
}

std::uint64_t puts_by_hand(hand_map& m)
{
    const std::vector<std::byte> v = value_of(7);
    for (const std::string& k : probe_keys) {
        const kv_bytes kb = kv_bytes_of(k);
        if (const auto it = m.find(kb); it != m.end()) {
            std::vector<std::byte> copy(v.begin(), v.end());
            it->second.swap(copy);
        } else {
            m.emplace(std::vector<std::byte>(kb.begin(), kb.end()), v);
        }
    }
    return m.size();
}

// --- through the contract: one template, instantiated for the model and for the interface -------------------------

template <key_value_store S>
void fill(S& s)
{
    for (std::size_t i = 0; i < keys; ++i)
        static_cast<void>(kv_put(s, key_of(i), value_of(i)));
}

template <key_value_store S>
std::uint64_t reads(const S& s)
{
    std::uint64_t sum = 0;
    for (const std::string& k : probe_keys)
        static_cast<void>(s.read(kv_bytes_of(k), [&](kv_bytes v) { sum += static_cast<std::uint64_t>(v[0]); }));
    return sum;
}

template <key_value_store S>
std::uint64_t scan(const S& s)
{
    std::uint64_t sum = 0;
    static_cast<void>(s.scan(kv_range::prefixed(kv_bytes_of(std::string_view("k/001"))), [&](kv_bytes, kv_bytes v) {
        sum += v.size();
        return true;
    }));
    return sum;
}

template <key_value_store S>
std::uint64_t puts(S& s)
{
    const std::vector<std::byte> v = value_of(7);
    for (const std::string& k : probe_keys)
        static_cast<void>(s.put(kv_bytes_of(k), v));
    return *kv_count(s);
}

struct stores {
    hand_map hand = make_hand();
    reference_key_value_store model;
    key_value_store_adapter<reference_key_value_store> adapted{reference_key_value_store{}};
    key_value_store_interface& virtual_form = adapted;

    stores()
    {
        fill(model);
        fill(virtual_form);
    }
};

stores& the_stores()
{
    static stores s;
    return s;
}

void BM_KvRead_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(reads_by_hand(the_stores().hand));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(probes));
}
BENCHMARK(BM_KvRead_HandWritten);

void BM_KvRead_Concept(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(reads(the_stores().model));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(probes));
}
BENCHMARK(BM_KvRead_Concept);

void BM_KvRead_Virtual(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(reads(the_stores().virtual_form));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(probes));
}
BENCHMARK(BM_KvRead_Virtual);

void BM_KvScanPrefix100_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(scan_by_hand(the_stores().hand));
    state.SetItemsProcessed(state.iterations() * 100);
}
BENCHMARK(BM_KvScanPrefix100_HandWritten);

void BM_KvScanPrefix100_Concept(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(scan(the_stores().model));
    state.SetItemsProcessed(state.iterations() * 100);
}
BENCHMARK(BM_KvScanPrefix100_Concept);

void BM_KvScanPrefix100_Virtual(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(scan(the_stores().virtual_form));
    state.SetItemsProcessed(state.iterations() * 100);
}
BENCHMARK(BM_KvScanPrefix100_Virtual);

void BM_KvPut_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(puts_by_hand(the_stores().hand));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(probes));
}
BENCHMARK(BM_KvPut_HandWritten);

void BM_KvPut_Concept(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(puts(the_stores().model));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(probes));
}
BENCHMARK(BM_KvPut_Concept);

void BM_KvPut_Virtual(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(puts(the_stores().virtual_form));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(probes));
}
BENCHMARK(BM_KvPut_Virtual);

} // namespace

int main(int argc, char** argv)
{
    // Fairness: the three paths read the same sums and leave the same number of keys.
    stores& s = the_stores();
    const std::uint64_t r = reads_by_hand(s.hand), sc = scan_by_hand(s.hand);
    if (reads(s.model) != r || reads(s.virtual_form) != r || scan(s.model) != sc || scan(s.virtual_form) != sc || sc != 1600) {
        std::fprintf(stderr, "key_value_store benchmark: fairness check FAILED - the reads differ; not benchmarking\n");
        return 1;
    }
    stores fresh;
    const std::uint64_t n = puts_by_hand(fresh.hand);
    if (puts(fresh.model) != n || puts(fresh.virtual_form) != n || reads(fresh.model) != reads_by_hand(fresh.hand) ||
        reads(fresh.virtual_form) != reads_by_hand(fresh.hand)) {
        std::fprintf(stderr, "key_value_store benchmark: fairness check FAILED - the writes differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
