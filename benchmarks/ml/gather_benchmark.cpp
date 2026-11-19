//
//  gather_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::gather_rows and scatter_add_rows against the same loops written by hand: a batch of 64 sequences of
//  196 token rows of 768 floats, 49 context rows gathered from each (I-JEPA's ViT-B/16 at a quarter visible), and
//  their gradients scattered back. main() runs a fairness check first and refuses to benchmark if any element differs
//  in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/gather.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
const auto& backend = snicholls::linalg::reference;

constexpr std::size_t seqs = 64, n = 196, k = 49, width = 768;

struct state_t {
    std::vector<float> src = std::vector<float>(seqs * n * width), rows = std::vector<float>(seqs * k * width),
                       dsrc = std::vector<float>(seqs * n * width);
    std::vector<std::uint32_t> indices = std::vector<std::uint32_t>(seqs * k);
    state_t()
    {
        std::uint64_t x = 0x9e3779b97f4a7c15ULL;   // a fixed xorshift stream: identical data for both
        for (auto& e : src) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            e = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
        }
        for (std::size_t s = 0; s < seqs; ++s)
            for (std::size_t j = 0; j < k; ++j)
                indices[s * k + j] = std::uint32_t((j * 4 + s) % n);   // every fourth token, shifted per sequence
    }
};

void gather_by_hand(const state_t& s, float* out)
{
    for (std::size_t q = 0; q < seqs; ++q)
        for (std::size_t j = 0; j < k; ++j)
            std::memcpy(out + (q * k + j) * width, s.src.data() + (q * n + s.indices[q * k + j]) * width, width * sizeof(float));
}

void scatter_add_by_hand(const state_t& s, float* dsrc)
{
    for (std::size_t q = 0; q < seqs; ++q)
        for (std::size_t j = 0; j < k; ++j) {
            float* d = dsrc + (q * n + s.indices[q * k + j]) * width;
            const float* g = s.rows.data() + (q * k + j) * width;
            for (std::size_t c = 0; c < width; ++c)
                d[c] += g[c];
        }
}

void BM_GatherRows_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::gather_rows(backend, std::span<const float>(s.src), s.indices, std::span<float>(s.rows), width, seqs);
        benchmark::DoNotOptimize(s.rows.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * std::int64_t(seqs * k * width * sizeof(float)));
}
BENCHMARK(BM_GatherRows_Core);

void BM_GatherRows_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        gather_by_hand(s, s.rows.data());
        benchmark::DoNotOptimize(s.rows.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * std::int64_t(seqs * k * width * sizeof(float)));
}
BENCHMARK(BM_GatherRows_HandWritten);

void BM_ScatterAddRows_Core(benchmark::State& state)
{
    state_t s;
    gather_by_hand(s, s.rows.data());
    for (auto _ : state) {
        ml::scatter_add_rows(backend, std::span<const float>(s.rows), s.indices, std::span<float>(s.dsrc), width, seqs);
        benchmark::DoNotOptimize(s.dsrc.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * std::int64_t(seqs * k * width * sizeof(float)));
}
BENCHMARK(BM_ScatterAddRows_Core);

void BM_ScatterAddRows_HandWritten(benchmark::State& state)
{
    state_t s;
    gather_by_hand(s, s.rows.data());
    for (auto _ : state) {
        scatter_add_by_hand(s, s.dsrc.data());
        benchmark::DoNotOptimize(s.dsrc.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * std::int64_t(seqs * k * width * sizeof(float)));
}
BENCHMARK(BM_ScatterAddRows_HandWritten);

bool same(const std::vector<float>& a, const std::vector<float>& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

bool fair()
{
    state_t a, b;
    ml::gather_rows(backend, std::span<const float>(a.src), a.indices, std::span<float>(a.rows), width, seqs);
    gather_by_hand(b, b.rows.data());
    for (int r = 0; r < 3; ++r) {
        ml::scatter_add_rows(backend, std::span<const float>(a.rows), a.indices, std::span<float>(a.dsrc), width, seqs);
        scatter_add_by_hand(b, b.dsrc.data());
    }
    return same(a.rows, b.rows) && same(a.dsrc, b.dsrc);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "gather benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
