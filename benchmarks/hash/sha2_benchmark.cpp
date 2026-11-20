//
//  sha2_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::sha256 and sha512 (constexpr, byte-assembled loads) against the same compression written by hand
//  for run time (memcpy loads, a byte swap, an 64- or 80-word schedule), on 256 KiB of messages of 64 bytes and of
//  16 KiB; OpenSSL's EVP digests beside them when CMake found OpenSSL (informational: core never uses it). main()
//  runs a fairness check first and refuses to benchmark if any two digests disagree (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/hash/sha2.hpp>

#include "by_hand.hpp"

#include <benchmark/benchmark.h>

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

void BM_Sha256_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::sha256::hash(m); }); }
void BM_Sha256_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::sha256(m.data(), m.size()); }); }
void BM_Sha512_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::sha512::hash(m); }); }
void BM_Sha512_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::sha512(m.data(), m.size()); }); }
BENCHMARK(BM_Sha256_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Sha256_HandWritten)->Arg(64)->Arg(16384);
BENCHMARK(BM_Sha512_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Sha512_HandWritten)->Arg(64)->Arg(16384);

#if SNICHOLLS_CORE_BENCH_OPENSSL
std::array<unsigned char, 64> openssl(const EVP_MD* md, const by_hand::bytes& m)
{
    std::array<unsigned char, 64> out{};
    by_hand::openssl_digest(md, m.data(), m.size(), out.data());
    return out;
}
void BM_Sha256_OpenSSL(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return openssl(EVP_sha256(), m); }); }
void BM_Sha512_OpenSSL(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return openssl(EVP_sha512(), m); }); }
BENCHMARK(BM_Sha256_OpenSSL)->Arg(64)->Arg(16384);
BENCHMARK(BM_Sha512_OpenSSL)->Arg(64)->Arg(16384);
#endif

bool fair()
{
    const auto big = by_hand::corpus(16384).front();
    for (const std::size_t size : {0u, 1u, 55u, 56u, 64u, 111u, 112u, 128u, 1000u, 16384u}) {
        const auto msg = std::span<const unsigned char>(big).first(size);
        const auto a = snicholls::sha256::hash(msg);
        const auto b = by_hand::sha256(msg.data(), msg.size());
        const auto c = snicholls::sha512::hash(msg);
        const auto d = by_hand::sha512(msg.data(), msg.size());
        if (std::memcmp(a.data(), b.data(), 32) != 0 || std::memcmp(c.data(), d.data(), 64) != 0)
            return false;
#if SNICHOLLS_CORE_BENCH_OPENSSL
        const auto e = openssl(EVP_sha256(), by_hand::bytes(msg.begin(), msg.end()));
        const auto f = openssl(EVP_sha512(), by_hand::bytes(msg.begin(), msg.end()));
        if (std::memcmp(a.data(), e.data(), 32) != 0 || std::memcmp(c.data(), f.data(), 64) != 0)
            return false;
#endif
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "sha2 benchmark: fairness check FAILED - the digests disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
