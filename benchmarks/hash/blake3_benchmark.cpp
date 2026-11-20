//
//  blake3_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::blake3 against BLAKE3 written by hand for run time the way the official reference implementation
//  writes it (memcpy loads, the message permuted between rounds, the chaining-value stack), on 256 KiB of 64-byte
//  and 16 KiB messages. Both are portable single-threaded code; the official SIMD builds are faster and are not
//  compared. main() runs a fairness check first and refuses to benchmark if any digests disagree.
//

#include <snicholls/core/hash/blake3.hpp>

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

void BM_Blake3_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::blake3::hash(m); }); }
void BM_Blake3_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::b3::hash(m.data(), m.size()); }); }
BENCHMARK(BM_Blake3_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Blake3_HandWritten)->Arg(64)->Arg(16384);

bool fair()
{
    const auto big = by_hand::corpus(16384).front();
    for (const std::size_t size : {0u, 1u, 64u, 65u, 1023u, 1024u, 1025u, 2048u, 3073u, 16384u}) {
        const auto msg = std::span<const unsigned char>(big).first(size);
        const auto a = snicholls::blake3::hash(msg);
        const auto b = by_hand::b3::hash(msg.data(), msg.size());
        if (std::memcmp(a.data(), b.data(), 32) != 0)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "blake3 benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
