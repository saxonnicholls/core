//
//  wyhash_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::wyhash (constexpr, byte-explicit reads) against the reference
//  written out by hand: memcpy reads and the compiler's 128-bit multiply, the
//  same arithmetic in the same order. Both hash the same 4,096 keys of 1 to 96
//  bytes; fnv1a_64 runs beside them for scale. main() runs a fairness check
//  first and refuses to benchmark if the two wyhash paths disagree on any key
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/hash/fnv.hpp>
#include <snicholls/core/hash/wyhash.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::string> make_keys()
{
    std::vector<std::string> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for both
    for (auto& k : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        k.resize(1 + x % 96);
        for (auto& c : k) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            c = static_cast<char>(x);
        }
    }
    return out;
}

const std::vector<std::string> keys = make_keys();

// The library
std::uint64_t hash_core(const std::string& k) noexcept { return snicholls::wyhash(std::string_view{k}); }

// By hand: the reference's reads (memcpy, host byte order) and its multiply
inline void mum_by_hand(std::uint64_t* a, std::uint64_t* b) noexcept
{
    const snicholls::detail::wide_product p = snicholls::detail::umul128(*a, *b);
    *a = p.low;
    *b = p.high;
}
inline std::uint64_t mix_by_hand(std::uint64_t a, std::uint64_t b) noexcept
{
    mum_by_hand(&a, &b);
    return a ^ b;
}
inline std::uint64_t r8(const unsigned char* p) noexcept { std::uint64_t v; std::memcpy(&v, p, 8); return v; }
inline std::uint64_t r4(const unsigned char* p) noexcept { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
inline std::uint64_t r3(const unsigned char* p, std::size_t k) noexcept
{
    return (static_cast<std::uint64_t>(p[0]) << 16) | (static_cast<std::uint64_t>(p[k >> 1]) << 8) | p[k - 1];
}

std::uint64_t hash_by_hand(const std::string& key) noexcept
{
    const auto* p = reinterpret_cast<const unsigned char*>(key.data());
    std::size_t len = key.size();
    const std::uint64_t* secret = snicholls::wyhash_secret;
    std::uint64_t seed = 0;
    seed ^= mix_by_hand(seed ^ secret[0], secret[1]);
    std::uint64_t a, b;
    if (len <= 16) {
        if (len >= 4) {
            a = (r4(p) << 32) | r4(p + ((len >> 3) << 2));
            b = (r4(p + len - 4) << 32) | r4(p + len - 4 - ((len >> 3) << 2));
        } else if (len > 0) {
            a = r3(p, len);
            b = 0;
        } else {
            a = b = 0;
        }
    } else {
        std::size_t i = len;
        if (i > 48) {
            std::uint64_t see1 = seed, see2 = seed;
            do {
                seed = mix_by_hand(r8(p) ^ secret[1], r8(p + 8) ^ seed);
                see1 = mix_by_hand(r8(p + 16) ^ secret[2], r8(p + 24) ^ see1);
                see2 = mix_by_hand(r8(p + 32) ^ secret[3], r8(p + 40) ^ see2);
                p += 48;
                i -= 48;
            } while (i > 48);
            seed ^= see1 ^ see2;
        }
        while (i > 16) {
            seed = mix_by_hand(r8(p) ^ secret[1], r8(p + 8) ^ seed);
            i -= 16;
            p += 16;
        }
        a = r8(p + i - 16);
        b = r8(p + i - 8);
    }
    a ^= secret[1];
    b ^= seed;
    mum_by_hand(&a, &b);
    return mix_by_hand(a ^ secret[0] ^ len, b ^ secret[1]);
}

void BM_Wyhash_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& k : keys)
            benchmark::DoNotOptimize(hash_core(k));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(keys.size()));
}
BENCHMARK(BM_Wyhash_Core);

void BM_Wyhash_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& k : keys)
            benchmark::DoNotOptimize(hash_by_hand(k));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(keys.size()));
}
BENCHMARK(BM_Wyhash_HandWritten);

void BM_Wyhash_Fnv1a64ForScale(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& k : keys)
            benchmark::DoNotOptimize(snicholls::fnv1a_64(std::string_view{k}));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(keys.size()));
}
BENCHMARK(BM_Wyhash_Fnv1a64ForScale);

bool fair()
{
    for (const auto& k : keys)
        if (hash_core(k) != hash_by_hand(k))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "wyhash benchmark: fairness check FAILED - the two paths disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
