//
//  hadamard_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's orthonormal Walsh-Hadamard transform against ggml's CPU row loop written out by hand
//  (ggml_compute_forward_fwht_impl, scalar path: scale, then butterflies of length 1, 2, 4, ...), on the same rows:
//  one rotation of 4,096 and blocks of 128 over 4,096. main() runs a fairness check first and refuses to benchmark if
//  any result differs in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/hadamard.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr std::size_t n = 4096;

std::vector<float> make(std::uint64_t x)
{
    std::vector<float> out(n);
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return out;
}

const std::vector<float>& row()
{
    static const auto r = make(21);
    return r;
}

// By hand: ggml's loop on each block of `block`, in place on a copy of the row, as core's is timed
void fwht_by_hand(std::vector<float>& x, std::size_t block)
{
    x = row();
    const float scale = 1.0f / std::sqrt(static_cast<float>(block));
    for (std::size_t b = 0; b < x.size(); b += block) {
        float* d = x.data() + b;
        for (std::size_t j = 0; j < block; ++j)
            d[j] = d[j] * scale;
        for (std::size_t len = 1; len < block; len <<= 1)
            for (std::size_t i = 0; i < block; i += 2 * len)
                for (std::size_t j = 0; j < len; ++j) {
                    const float u = d[i + j];
                    const float v = d[i + len + j];
                    d[i + j] = u + v;
                    d[i + len + j] = u - v;
                }
    }
}

void core_transform(std::vector<float>& x, std::size_t block)
{
    x = row();
    if (block == n)
        ml::hadamard(x);
    else
        ml::block_hadamard(x, block);
}

void BM_Hadamard4096_Core(benchmark::State& state)
{
    std::vector<float> x(n);
    for (auto _ : state) {
        core_transform(x, n);
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_Hadamard4096_Core);

void BM_Hadamard4096_HandWritten(benchmark::State& state)
{
    std::vector<float> x(n);
    for (auto _ : state) {
        fwht_by_hand(x, n);
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_Hadamard4096_HandWritten);

void BM_BlockHadamard128_Core(benchmark::State& state)
{
    std::vector<float> x(n);
    for (auto _ : state) {
        core_transform(x, 128);
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_BlockHadamard128_Core);

void BM_BlockHadamard128_HandWritten(benchmark::State& state)
{
    std::vector<float> x(n);
    for (auto _ : state) {
        fwht_by_hand(x, 128);
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_BlockHadamard128_HandWritten);

bool fair()
{
    for (const std::size_t block : {n, std::size_t{128}}) {
        std::vector<float> core(n), hand(n);
        core_transform(core, block);
        fwht_by_hand(hand, block);
        if (std::memcmp(core.data(), hand.data(), n * sizeof(float)) != 0)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "hadamard benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
