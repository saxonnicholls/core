//
//  glu_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::glu_rows against the same loops written out by hand, over the same gate and up rows of 14,336
//  floats (Llama 3 8B's d_ff), for SwiGLU and gpt-oss's clamped SwiGLU: the same expression per element in the same
//  order (core's sigmoid, as the hand loop calls it). main() runs a fairness check first and refuses to benchmark if
//  any pair disagrees in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/glu.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

constexpr std::size_t width = 14336;

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

const std::vector<float> gate = make(width, -9.0, 9.0, 0x243f6a8885a308d3ULL);
const std::vector<float> up = make(width, -9.0, 9.0, 0x13198a2e03707344ULL);

void swiglu_by_hand(float* y)
{
    for (std::size_t i = 0; i < width; ++i)
        y[i] = gate[i] * snicholls::ml::sigmoid(gate[i]) * up[i];
}

void swiglu_oai_by_hand(float* y)
{
    for (std::size_t i = 0; i < width; ++i) {
        const float x = gate[i] < 7.0f ? gate[i] : 7.0f;
        const float u = up[i] < -7.0f ? -7.0f : up[i] > 7.0f ? 7.0f : up[i];
        const float out = x * snicholls::ml::sigmoid(1.702f * x);
        y[i] = out * (u + 1.0f);
    }
}

void swiglu_core(float* y) { snicholls::ml::glu_rows(snicholls::ml::glu_kind::swiglu, gate, up, std::span<float>(y, width)); }
void swiglu_oai_core(float* y) { snicholls::ml::glu_rows(snicholls::ml::glu_kind::swiglu_oai, gate, up, std::span<float>(y, width)); }

template <void (*F)(float*)>
void run(benchmark::State& state)
{
    std::vector<float> out(width);
    for (auto _ : state) {
        F(out.data());
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(width));
}

BENCHMARK(run<swiglu_core>)->Name("BM_SwiGLU_Core");
BENCHMARK(run<swiglu_by_hand>)->Name("BM_SwiGLU_HandWritten");
BENCHMARK(run<swiglu_oai_core>)->Name("BM_SwiGLU_OAI_Core");
BENCHMARK(run<swiglu_oai_by_hand>)->Name("BM_SwiGLU_OAI_HandWritten");

template <void (*A)(float*), void (*B)(float*)>
bool same()
{
    std::vector<float> a(width), b(width);
    A(a.data());
    B(b.data());
    for (std::size_t i = 0; i < width; ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!same<swiglu_core, swiglu_by_hand>() || !same<swiglu_oai_core, swiglu_oai_by_hand>()) {
        std::fprintf(stderr, "glu benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
