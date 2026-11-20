//
//  blake2_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::blake2b_512 and blake2s_256 against BLAKE2 written by hand for run time from RFC 7693 (memcpy
//  loads, the sigma table), on 256 KiB of 64-byte and 16 KiB messages; OpenSSL's BLAKE2b512 and BLAKE2s256 beside
//  them when CMake found it (informational). main() runs a fairness check first and refuses to benchmark if any
//  digests disagree.
//

#include <snicholls/core/hash/blake2.hpp>

#include "by_hand.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>

namespace {

template <typename F>
void run(benchmark::State& state, F digest)
{
    const auto messages = by_hand::corpus(static_cast<std::size_t>(state.range(0)));
    for (auto _ : state)
        for (const auto& m : messages)
            benchmark::DoNotOptimize(digest(m));
    state.SetBytesProcessed(state.iterations() * by_hand::corpus_bytes(messages));
}

#if SNICHOLLS_CORE_BENCH_OPENSSL
std::array<unsigned char, 64> openssl(const EVP_MD* md, std::span<const unsigned char> m)
{
    std::array<unsigned char, 64> out{};
    if (md != nullptr)
        by_hand::openssl_digest(md, m.data(), m.size(), out.data());
    return out;
}
#endif

void BM_Blake2b_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::blake2b_512::hash(m); }); }
void BM_Blake2b_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::blake2b_512(m.data(), m.size()); }); }
void BM_Blake2s_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::blake2s_256::hash(m); }); }
void BM_Blake2s_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::blake2s_256(m.data(), m.size()); }); }
BENCHMARK(BM_Blake2b_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Blake2b_HandWritten)->Arg(64)->Arg(16384);
BENCHMARK(BM_Blake2s_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Blake2s_HandWritten)->Arg(64)->Arg(16384);
#if SNICHOLLS_CORE_BENCH_OPENSSL
void BM_Blake2b_OpenSSL(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return openssl(EVP_blake2b512(), m); }); }
void BM_Blake2s_OpenSSL(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return openssl(EVP_blake2s256(), m); }); }
BENCHMARK(BM_Blake2b_OpenSSL)->Arg(64)->Arg(16384);
BENCHMARK(BM_Blake2s_OpenSSL)->Arg(64)->Arg(16384);
#endif

bool fair()
{
    const auto big = by_hand::corpus(16384).front();
    for (const std::size_t size : {0u, 1u, 63u, 64u, 65u, 128u, 129u, 1000u, 16384u}) {
        const auto msg = std::span<const unsigned char>(big).first(size);
        const auto a = snicholls::blake2b_512::hash(msg);
        const auto b = by_hand::blake2b_512(msg.data(), msg.size());
        const auto c = snicholls::blake2s_256::hash(msg);
        const auto d = by_hand::blake2s_256(msg.data(), msg.size());
        if (std::memcmp(a.data(), b.data(), 64) != 0 || std::memcmp(c.data(), d.data(), 32) != 0)
            return false;
#if SNICHOLLS_CORE_BENCH_OPENSSL
        if (std::memcmp(a.data(), openssl(EVP_blake2b512(), msg).data(), 64) != 0 ||
            std::memcmp(c.data(), openssl(EVP_blake2s256(), msg).data(), 32) != 0)
            return false;
#endif
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "blake2 benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
