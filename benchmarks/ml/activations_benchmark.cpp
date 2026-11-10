//
//  activations_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::sigmoid and gelu_tanh against the same formulas written out by hand, with the same
//  constants and the same operations in the same order, over the same 65,536 floats in [-8, 8]. main()
//  runs a fairness check first and refuses to benchmark if any pair disagrees in a single bit
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/activations.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

std::vector<float> make_inputs()
{
    std::vector<float> out(65536);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 16.0 - 8.0);
    }
    return out;
}

const std::vector<float> inputs = make_inputs();

// By hand: the stable sigmoid, and GPT-2's GELU as x * sigmoid(2u), with the library's float constants
inline float sigmoid_by_hand(float v) noexcept
{
    if (v >= 0.0f)
        return 1.0f / (1.0f + std::exp(-v));
    const float e = std::exp(v);
    return e / (1.0f + e);
}

inline float gelu_tanh_by_hand(float v) noexcept
{
    const float sqrt_2_over_pi = 2.0f / static_cast<float>(2.50662827463100050242);
    const float two_u = 2.0f * sqrt_2_over_pi * (v + 0.044715f * v * v * v);
    return v * sigmoid_by_hand(two_u);
}

void BM_Sigmoid_Core(benchmark::State& state)
{
    std::vector<float> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < inputs.size(); ++i)
            out[i] = snicholls::ml::sigmoid(inputs[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_Sigmoid_Core);

void BM_Sigmoid_HandWritten(benchmark::State& state)
{
    std::vector<float> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < inputs.size(); ++i)
            out[i] = sigmoid_by_hand(inputs[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_Sigmoid_HandWritten);

void BM_GeluTanh_Core(benchmark::State& state)
{
    std::vector<float> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < inputs.size(); ++i)
            out[i] = snicholls::ml::gelu_tanh(inputs[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_GeluTanh_Core);

void BM_GeluTanh_HandWritten(benchmark::State& state)
{
    std::vector<float> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < inputs.size(); ++i)
            out[i] = gelu_tanh_by_hand(inputs[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_GeluTanh_HandWritten);

bool fair()
{
    for (const float v : inputs)
        if (std::bit_cast<std::uint32_t>(snicholls::ml::sigmoid(v)) != std::bit_cast<std::uint32_t>(sigmoid_by_hand(v))
            || std::bit_cast<std::uint32_t>(snicholls::ml::gelu_tanh(v)) != std::bit_cast<std::uint32_t>(gelu_tanh_by_hand(v)))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "activations benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
