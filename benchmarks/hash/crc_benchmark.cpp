//
//  crc_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::crc32, crc32c and crc64_xz (constexpr tables, slicing-by-8) against the same slicing-by-8 written by
//  hand (tables built at start-up, memcpy loads), and the byte-at-a-time table loop for scale, on 256 KiB of 64-byte
//  and 16 KiB messages. main() runs a fairness check first and refuses to benchmark if any CRCs disagree.
//

#include <snicholls/core/hash/crc.hpp>

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

void BM_Crc32_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::crc32::hash(m); }); }
void BM_Crc32_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::crc_sliced<std::uint32_t, 0xEDB88320u>(m.data(), m.size()); }); }
void BM_Crc32_BytewiseForScale(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::crc_bytewise<std::uint32_t, 0xEDB88320u>(m.data(), m.size()); }); }
void BM_Crc32c_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::crc32c::hash(m); }); }
void BM_Crc32c_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::crc_sliced<std::uint32_t, 0x82F63B78u>(m.data(), m.size()); }); }
void BM_Crc64Xz_Core(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return snicholls::crc64_xz::hash(m); }); }
void BM_Crc64Xz_HandWritten(benchmark::State& s) { run(s, [](const by_hand::bytes& m) { return by_hand::crc_sliced<std::uint64_t, 0xC96C5795D7870F42ull>(m.data(), m.size()); }); }
BENCHMARK(BM_Crc32_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Crc32_HandWritten)->Arg(64)->Arg(16384);
BENCHMARK(BM_Crc32_BytewiseForScale)->Arg(64)->Arg(16384);
BENCHMARK(BM_Crc32c_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Crc32c_HandWritten)->Arg(64)->Arg(16384);
BENCHMARK(BM_Crc64Xz_Core)->Arg(64)->Arg(16384);
BENCHMARK(BM_Crc64Xz_HandWritten)->Arg(64)->Arg(16384);

bool fair()
{
    const auto big = by_hand::corpus(16384).front();
    for (const std::size_t size : {0u, 1u, 7u, 8u, 9u, 64u, 1000u, 16384u}) {
        const auto msg = std::span<const unsigned char>(big).first(size);
        if (snicholls::crc32::hash(msg) != by_hand::crc_sliced<std::uint32_t, 0xEDB88320u>(msg.data(), msg.size()) ||
            snicholls::crc32::hash(msg) != by_hand::crc_bytewise<std::uint32_t, 0xEDB88320u>(msg.data(), msg.size()) ||
            snicholls::crc32c::hash(msg) != by_hand::crc_sliced<std::uint32_t, 0x82F63B78u>(msg.data(), msg.size()) ||
            snicholls::crc64_xz::hash(msg) != by_hand::crc_sliced<std::uint64_t, 0xC96C5795D7870F42ull>(msg.data(), msg.size()))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "crc benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
