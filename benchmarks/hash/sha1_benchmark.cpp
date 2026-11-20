//
//  sha1_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::sha1 against SHA-1 written by hand for run time (memcpy loads, an 80-word schedule, the function
//  chosen per round), on 256 KiB of 64-byte and 16 KiB messages; OpenSSL beside them when CMake found it
//  (informational). main() runs a fairness check first and refuses to benchmark if any digests disagree.
//

#include <snicholls/core/hash/sha1.hpp>

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

void BM_Sha1_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::sha1::hash(m); }); }
void BM_Sha1_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::sha1(m.data(), m.size()); }); }
BENCHMARK(BM_Sha1_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Sha1_HandWritten)->Arg(64)->Arg(16384);
#if SNICHOLLS_CORE_BENCH_OPENSSL
void BM_Sha1_OpenSSL(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return openssl(EVP_sha1(), m); }); }
BENCHMARK(BM_Sha1_OpenSSL)->Arg(64)->Arg(16384);
#endif

bool fair()
{
    const auto big = by_hand::corpus(16384).front();
    for (const std::size_t size : {0u, 1u, 55u, 56u, 64u, 1000u, 16384u}) {
        const auto msg = std::span<const unsigned char>(big).first(size);
        const auto a = snicholls::sha1::hash(msg);
        const auto b = by_hand::sha1(msg.data(), msg.size());
        if (std::memcmp(a.data(), b.data(), 20) != 0)
            return false;
#if SNICHOLLS_CORE_BENCH_OPENSSL
        if (std::memcmp(a.data(), openssl(EVP_sha1(), msg).data(), 20) != 0)
            return false;
#endif
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "sha1 benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
