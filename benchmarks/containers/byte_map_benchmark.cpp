//
//  byte_map_benchmark.cpp
//  snicholls-core
//
//  Looking up byte strings, half hits and half misses, in a fixed pseudo-random order, as a BPE merge does: the
//  BPE fixture's 512-token vocabulary (a table that fits in L1) and 100,000 synthetic keys skewed short (4 MB of
//  slots, past L2). Four lookups over the same keys and the same queries: snicholls::containers::byte_map_view one
//  at a time, the same view probing a batch of 32 and then finding them, std::unordered_map<std::string_view,
//  std::uint32_t>, and an open-addressing table written by hand ({string_view, value} slots, linear probing at the
//  same load, the whole key hashed with wyhash). The fairness check refuses to run unless all four give the same
//  sum of values found and the same number of misses.
//

#include <snicholls/core/containers/byte_map.hpp>
#include <snicholls/core/hash/wyhash.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace containers = snicholls::containers;

std::string unbase64(std::string_view in)
{
    if (in == "-")
        return {};
    static constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::uint32_t acc = 0;
    int bits = 0;
    for (const char c : in) {
        if (c == '=')
            break;
        acc = (acc << 6) | static_cast<std::uint32_t>(alphabet.find(c));
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((acc >> bits) & 0xff));
        }
    }
    return out;
}

std::uint64_t next(std::uint64_t& x)
{
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

// The hand-written equivalent: what one writes without the head word or the split lookup
class by_hand {
public:
    explicit by_hand(const std::vector<std::pair<std::string, std::uint32_t>>& keys)
        : slots_(containers::byte_map_view::required_slots(keys.size()), {std::string_view{}, containers::byte_map_no_value}),
          mask_(slots_.size() - 1)
    {
        for (const auto& [k, v] : keys) {
            std::size_t i = snicholls::wyhash(k) & mask_;
            while (slots_[i].second != containers::byte_map_no_value)
                i = (i + 1) & mask_;
            slots_[i] = {k, v};
        }
    }
    std::uint32_t operator()(std::string_view key) const noexcept
    {
        for (std::size_t i = snicholls::wyhash(key) & mask_;; i = (i + 1) & mask_) {
            if (slots_[i].second == containers::byte_map_no_value)
                return containers::byte_map_no_value;
            if (slots_[i].first == key)
                return slots_[i].second;
        }
    }

private:
    std::vector<std::pair<std::string_view, std::uint32_t>> slots_{};   // views into the caller's keys
    std::size_t mask_;
};

struct key_set {
    std::vector<std::pair<std::string, std::uint32_t>> keys;
    std::vector<std::string> misses;   // owned here, so the queries can view them
    std::vector<std::string_view> queries;

    std::vector<containers::byte_map_slot> slots;
    std::vector<char> arena;
    containers::byte_map_view view;
    std::unordered_map<std::string_view, std::uint32_t> hashed;
    by_hand hand;

    explicit key_set(std::vector<std::pair<std::string, std::uint32_t>> k, std::uint64_t seed)
        : keys(std::move(k)), slots(containers::byte_map_view::required_slots(keys.size())),
          arena(containers::byte_map_view::required_arena(keys)), view(slots, arena), hand(keys)
    {
        if (view.build(keys).is_err())
            std::fprintf(stderr, "byte_map benchmark: build failed\n");
        for (const auto& [s, v] : keys)
            hashed.emplace(s, v);
        // Misses as a merge sees them: a key with a byte added, or two keys glued
        std::uint64_t x = seed;
        for (std::size_t i = 0; i < 2048; ++i) {
            std::string m = keys[next(x) % keys.size()].first;
            if (i % 2 == 0)
                m.push_back(static_cast<char>(next(x) % 256));
            else
                m += keys[next(x) % keys.size()].first;
            if (hashed.find(m) == hashed.end())
                misses.push_back(std::move(m));
        }
        for (std::size_t i = 0; i < 65'536; ++i)
            queries.push_back(i % 2 == 0 ? std::string_view(keys[next(x) % keys.size()].first) : std::string_view(misses[next(x) % misses.size()]));
    }
};

struct tally {
    std::uint64_t sum = 0, misses = 0;
    void add(std::uint32_t v)
    {
        if (v == containers::byte_map_no_value)
            ++misses;
        else
            sum += v;
    }
    bool operator==(const tally&) const = default;
};

const key_set& fixture()
{
    static const key_set s = [] {
        std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/bpe_fixture.txt");
        std::vector<std::pair<std::string, std::uint32_t>> vocab;
        for (std::string line; std::getline(in, line);) {
            std::istringstream ls(line);
            std::string kind, b;
            std::uint32_t r = 0;
            if (ls >> kind >> b >> r && kind == "vocab")
                vocab.emplace_back(unbase64(b), r);
        }
        return key_set(std::move(vocab), 0x243f6a8885a308d3ULL);
    }();
    return s;
}

const key_set& synthetic()
{
    static const key_set s = [] {
        std::vector<std::pair<std::string, std::uint32_t>> keys;
        std::unordered_map<std::string, std::uint32_t> seen;
        std::uint64_t x = 0x13198a2e03707344ULL;
        while (keys.size() < 100'000) {   // lengths 1 to 10, a quarter up to 40: a vocabulary's shape
            std::string k(1 + next(x) % (next(x) % 4 == 0 ? 40 : 10), ' ');
            for (char& c : k)
                c = static_cast<char>('a' + next(x) % 26);
            if (seen.emplace(k, 0).second)
                keys.emplace_back(std::move(k), static_cast<std::uint32_t>(keys.size()));
        }
        return key_set(std::move(keys), 0xa4093822299f31d0ULL);
    }();
    return s;
}

tally via_byte_map(const key_set& s)
{
    tally t;
    for (const auto q : s.queries)
        t.add(s.view(q));
    return t;
}

tally via_byte_map_batched(const key_set& s)
{
    tally t;
    std::array<containers::byte_map_probe, 32> probes{};
    for (std::size_t b = 0; b < s.queries.size(); b += 32) {   // 65,536 queries: whole batches
        for (std::size_t i = 0; i < 32; ++i)
            probes[i] = s.view.probe(s.queries[b + i]);
        for (std::size_t i = 0; i < 32; ++i)
            t.add(s.view.find(s.queries[b + i], probes[i]));
    }
    return t;
}

tally via_unordered(const key_set& s)
{
    tally t;
    for (const auto q : s.queries) {
        const auto it = s.hashed.find(q);
        t.add(it == s.hashed.end() ? containers::byte_map_no_value : it->second);
    }
    return t;
}

tally via_hand(const key_set& s)
{
    tally t;
    for (const auto q : s.queries)
        t.add(s.hand(q));
    return t;
}

template <const key_set& (*Set)(), tally (*Run)(const key_set&)>
void run(benchmark::State& state)
{
    const key_set& s = Set();
    for (auto _ : state)
        benchmark::DoNotOptimize(Run(s));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(s.queries.size()));
}

BENCHMARK(run<fixture, via_byte_map>)->Name("BM_Lookup512_ByteMap");
BENCHMARK(run<fixture, via_byte_map_batched>)->Name("BM_Lookup512_ByteMapBatched");
BENCHMARK(run<fixture, via_hand>)->Name("BM_Lookup512_OpenAddressingByHand");
BENCHMARK(run<fixture, via_unordered>)->Name("BM_Lookup512_StdUnorderedMap");
BENCHMARK(run<synthetic, via_byte_map>)->Name("BM_Lookup100K_ByteMap");
BENCHMARK(run<synthetic, via_byte_map_batched>)->Name("BM_Lookup100K_ByteMapBatched");
BENCHMARK(run<synthetic, via_hand>)->Name("BM_Lookup100K_OpenAddressingByHand");
BENCHMARK(run<synthetic, via_unordered>)->Name("BM_Lookup100K_StdUnorderedMap");

} // namespace

int main(int argc, char** argv)
{
    for (const key_set* s : {&fixture(), &synthetic()}) {
        const tally want = via_unordered(*s);
        if (s->keys.size() < 512 || s->view.size() != s->keys.size() || want.misses != s->queries.size() / 2 ||
            !(via_byte_map(*s) == want) || !(via_byte_map_batched(*s) == want) || !(via_hand(*s) == want)) {
            std::fprintf(stderr, "byte_map benchmark: fairness check FAILED - the lookups disagree\n");
            return 1;
        }
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
