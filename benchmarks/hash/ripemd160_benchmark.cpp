//
//  ripemd160_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ripemd160 against RIPEMD-160 written by hand for run time (memcpy loads, table-driven word order
//  and rotations), on 256 KiB of 64-byte and 16 KiB messages. main() runs a fairness check first and refuses to
//  benchmark if any digests disagree. (OpenSSL 3 keeps RIPEMD-160 in its legacy provider, so it is not compared.)
//

#include <snicholls/core/hash/ripemd160.hpp>

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

void BM_Ripemd160_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::ripemd160::hash(m); }); }
void BM_Ripemd160_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::ripemd160(m.data(), m.size()); }); }
BENCHMARK(BM_Ripemd160_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Ripemd160_HandWritten)->Arg(64)->Arg(16384);

bool fair()
{
    const auto big = by_hand::corpus(16384).front();
    for (const std::size_t size : {0u, 1u, 55u, 56u, 64u, 1000u, 16384u}) {
        const auto msg = std::span<const unsigned char>(big).first(size);
        const auto a = snicholls::ripemd160::hash(msg);
        const auto b = by_hand::ripemd160(msg.data(), msg.size());
        if (std::memcmp(a.data(), b.data(), 20) != 0)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "ripemd160 benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
