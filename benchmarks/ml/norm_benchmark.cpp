//
//  norm_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::layer_norm and rms_norm against the same loops written out by hand, over the same 512
//  rows of 768 floats (GPT-2 small's channels), with the same weight and bias: the same sums in the same
//  order and the same expression per element. main() runs a fairness check first and refuses to benchmark
//  if any pair disagrees in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/norm.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

constexpr std::size_t rows = 512, channels = 768;

std::vector<float> make(std::size_t n, double lo, double hi, std::uint64_t seed)
{
    std::vector<float> out(n);
    std::uint64_t x = seed;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(lo + (hi - lo) * static_cast<double>(x >> 11) * 0x1.0p-53);
    }
    return out;
}

const std::vector<float> input = make(rows * channels, -3.0, 3.0, 0x243f6a8885a308d3ULL);
const std::vector<float> weight = make(channels, 0.5, 1.5, 0x13198a2e03707344ULL);
const std::vector<float> bias = make(channels, -0.1, 0.1, 0xa4093822299f31d0ULL);

void layer_norm_by_hand(const float* x, float* y)
{
    float mean = 0.0f;
    for (std::size_t i = 0; i < channels; ++i)
        mean += x[i];
    mean = mean / static_cast<float>(channels);
    float var = 0.0f;
    for (std::size_t i = 0; i < channels; ++i)
        var += (x[i] - mean) * (x[i] - mean);
    var = var / static_cast<float>(channels);
    const float rstd = 1.0f / std::sqrt(var + 1e-5f);
    for (std::size_t i = 0; i < channels; ++i)
        y[i] = (x[i] - mean) * rstd * weight[i] + bias[i];
}

void rms_norm_by_hand(const float* x, float* y)
{
    float ms = 0.0f;
    for (std::size_t i = 0; i < channels; ++i)
        ms += x[i] * x[i];
    ms = ms / static_cast<float>(channels);
    const float rstd = 1.0f / std::sqrt(ms + 1e-6f);
    for (std::size_t i = 0; i < channels; ++i)
        y[i] = x[i] * rstd * weight[i];
}

void layer_norm_core(const float* x, float* y)
{
    snicholls::ml::layer_norm(std::span<const float>(x, channels), weight, bias, std::span<float>(y, channels));
}

void rms_norm_core(const float* x, float* y)
{
    snicholls::ml::rms_norm(std::span<const float>(x, channels), weight, std::span<float>(y, channels), 1e-6f);
}

template <void (*F)(const float*, float*)>
void run(benchmark::State& state)
{
    std::vector<float> out(input.size());
    for (auto _ : state) {
        for (std::size_t r = 0; r < rows; ++r)
            F(input.data() + r * channels, out.data() + r * channels);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(input.size()));
}

BENCHMARK(run<layer_norm_core>)->Name("BM_LayerNorm_Core");
BENCHMARK(run<layer_norm_by_hand>)->Name("BM_LayerNorm_HandWritten");
BENCHMARK(run<rms_norm_core>)->Name("BM_RmsNorm_Core");
BENCHMARK(run<rms_norm_by_hand>)->Name("BM_RmsNorm_HandWritten");

template <void (*A)(const float*, float*), void (*B)(const float*, float*)>
bool same()
{
    std::vector<float> a(input.size()), b(input.size());
    for (std::size_t r = 0; r < rows; ++r) {
        A(input.data() + r * channels, a.data() + r * channels);
        B(input.data() + r * channels, b.data() + r * channels);
    }
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!same<layer_norm_core, layer_norm_by_hand>() || !same<rms_norm_core, rms_norm_by_hand>()) {
        std::fprintf(stderr, "norm benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
