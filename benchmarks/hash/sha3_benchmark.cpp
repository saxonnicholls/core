//
//  sha3_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::keccak256 and sha3_256 against the Keccak sponge written by hand for run time (memcpy lane loads,
//  the same rho-pi walk), on 256 KiB of 64-byte and 16 KiB messages; OpenSSL's SHA3-256, and its KECCAK-256 where
//  the installed OpenSSL has it, beside them (informational). main() runs a fairness check first and refuses to
//  benchmark if any digests disagree.
//

#include <snicholls/core/hash/sha3.hpp>

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

void BM_Keccak256_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::keccak256::hash(m); }); }
void BM_Keccak256_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::keccak_256_family(m.data(), m.size(), 0x01); }); }
void BM_Sha3_256_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::sha3_256::hash(m); }); }
void BM_Sha3_256_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::keccak_256_family(m.data(), m.size(), 0x06); }); }
BENCHMARK(BM_Keccak256_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Keccak256_HandWritten)->Arg(64)->Arg(16384);
BENCHMARK(BM_Sha3_256_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Sha3_256_HandWritten)->Arg(64)->Arg(16384);
#if SNICHOLLS_CORE_BENCH_OPENSSL
const EVP_MD* openssl_keccak() { static const EVP_MD* md = EVP_get_digestbyname("KECCAK-256"); return md; }
void BM_Sha3_256_OpenSSL(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return openssl(EVP_sha3_256(), m); }); }
void BM_Keccak256_OpenSSL(benchmark::State& s)
{
    if (openssl_keccak() == nullptr) {
        s.SkipWithMessage("this OpenSSL has no KECCAK-256");
        return;
    }
    run(s, [](const by_hand::bytes& m) { return openssl(openssl_keccak(), m); });
}
BENCHMARK(BM_Keccak256_OpenSSL)->Arg(64)->Arg(16384);
BENCHMARK(BM_Sha3_256_OpenSSL)->Arg(64)->Arg(16384);
#endif

bool fair()
{
    const auto big = by_hand::corpus(16384).front();
    for (const std::size_t size : {0u, 1u, 135u, 136u, 137u, 1000u, 16384u}) {
        const auto msg = std::span<const unsigned char>(big).first(size);
        const auto a = snicholls::keccak256::hash(msg);
        const auto b = by_hand::keccak_256_family(msg.data(), msg.size(), 0x01);
        const auto c = snicholls::sha3_256::hash(msg);
        const auto d = by_hand::keccak_256_family(msg.data(), msg.size(), 0x06);
        if (std::memcmp(a.data(), b.data(), 32) != 0 || std::memcmp(c.data(), d.data(), 32) != 0)
            return false;
#if SNICHOLLS_CORE_BENCH_OPENSSL
        if (std::memcmp(c.data(), openssl(EVP_sha3_256(), msg).data(), 32) != 0)
            return false;
        if (openssl_keccak() != nullptr && std::memcmp(a.data(), openssl(openssl_keccak(), msg).data(), 32) != 0)
            return false;
#endif
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "sha3 benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
