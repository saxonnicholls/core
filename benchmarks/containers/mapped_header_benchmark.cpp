//
//  mapped_header_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::mapped_header encode then decode against the
//  hand-written equivalent: the same fields copied into 64 bytes with memcpy
//  and the same wyhash over the first 56, then read back and the hash checked.
//  Both run over the same 1,024 headers and return the sum of the counts they
//  read back; main() refuses to benchmark if the sums disagree (CLAUDE.md
//  "Zero overhead"). A compiler barrier between the encode and the decode, on
//  both sides, stops the compiler proving the checksum unchanged and dropping
//  the second hash. The difference left is the explicit little-endian byte
//  order.
//

#include <snicholls/core/containers/mapped_header.hpp>
#include <snicholls/core/hash/wyhash.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

using namespace snicholls::containers;

std::vector<mapped_header> make_headers()
{
    std::vector<mapped_header> hs;
    for (std::uint64_t i = 0; i < 1024; ++i) hs.push_back(mapped_header{i * 7, 24, 8, i, i * 2 + 1, i ^ 5});
    return hs;
}
const std::vector<mapped_header> headers = make_headers();

std::uint64_t run_core()
{
    std::array<std::byte, 64> buf{};
    std::uint64_t sum = 0;
    for (const mapped_header& h : headers) {
        (void)encode_header(h, buf);
        benchmark::DoNotOptimize(buf.data());
        benchmark::ClobberMemory();   // the header goes to memory and comes back, as it would through a file
        auto back = decode_header(buf);
        if (back.is_ok()) sum += back->count;
    }
    return sum;
}

std::uint64_t run_hand_written()
{
    std::array<std::byte, 64> buf{};
    std::uint64_t sum = 0;
    for (const mapped_header& h : headers) {
        const std::uint32_t version = mapped_header_version, size = 64;
        std::memcpy(buf.data(), &mapped_header_magic, 8);
        std::memcpy(buf.data() + 8, &version, 4);
        std::memcpy(buf.data() + 12, &size, 4);
        std::memcpy(buf.data() + 16, &h.type_tag, 8);
        std::memcpy(buf.data() + 24, &h.element_size, 4);
        std::memcpy(buf.data() + 28, &h.element_align, 4);
        std::memcpy(buf.data() + 32, &h.count, 8);
        std::memcpy(buf.data() + 40, &h.capacity, 8);
        std::memcpy(buf.data() + 48, &h.user, 8);
        const std::uint64_t sum_w = snicholls::wyhash(std::span<const std::byte>(buf).first(56), mapped_header_magic);
        std::memcpy(buf.data() + 56, &sum_w, 8);
        benchmark::DoNotOptimize(buf.data());
        benchmark::ClobberMemory();

        std::uint64_t magic, stored, count, capacity;
        std::memcpy(&magic, buf.data(), 8);
        std::memcpy(&stored, buf.data() + 56, 8);
        std::memcpy(&count, buf.data() + 32, 8);
        std::memcpy(&capacity, buf.data() + 40, 8);
        if (magic == mapped_header_magic
            && stored == snicholls::wyhash(std::span<const std::byte>(buf).first(56), mapped_header_magic)
            && count <= capacity)
            sum += count;
    }
    return sum;
}

void BM_Core(benchmark::State& state)
{
    for (auto _ : state) benchmark::DoNotOptimize(run_core());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(headers.size()));
}
BENCHMARK(BM_Core)->Name("mapped_header_round_trip");

void BM_Hand(benchmark::State& state)
{
    for (auto _ : state) benchmark::DoNotOptimize(run_hand_written());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(headers.size()));
}
BENCHMARK(BM_Hand)->Name("memcpy_wyhash_round_trip");

} // namespace

int main(int argc, char** argv)
{
    if (run_core() != run_hand_written()) {
        std::fprintf(stderr, "mapped_header benchmark: fairness check FAILED - the sums disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
