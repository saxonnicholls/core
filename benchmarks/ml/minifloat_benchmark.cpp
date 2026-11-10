//
//  minifloat_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::to_bfloat16 and to_float against the usual bfloat16 bit manipulation written out by hand,
//  over the same 65,536 floats; and, for scale, the generic to_minifloat<float8_e4m3>. main() runs a fairness check first and refuses to benchmark if any pair disagrees
//  in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/minifloat.hpp>

#include <benchmark/benchmark.h>

#include <bit>
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
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 200.0 - 100.0);
    }
    return out;
}

const std::vector<float> inputs = make_inputs();

// By hand: round to nearest even on the float's bits, NaN kept quiet; and the shift back
inline std::uint16_t bf16_by_hand(float x) noexcept
{
    const std::uint32_t u = std::bit_cast<std::uint32_t>(x);
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
    return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float bf16_to_float_by_hand(std::uint16_t b) noexcept { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

void BM_ToBfloat16_Core(benchmark::State& state)
{
    std::vector<snicholls::ml::bfloat16> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < inputs.size(); ++i)
            out[i] = snicholls::ml::to_bfloat16(inputs[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_ToBfloat16_Core);

void BM_ToBfloat16_HandWritten(benchmark::State& state)
{
    std::vector<std::uint16_t> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < inputs.size(); ++i)
            out[i] = bf16_by_hand(inputs[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_ToBfloat16_HandWritten);

void BM_Bfloat16ToFloat_Core(benchmark::State& state)
{
    std::vector<snicholls::ml::bfloat16> in(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        in[i] = snicholls::ml::to_bfloat16(inputs[i]);
    std::vector<float> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < in.size(); ++i)
            out[i] = snicholls::ml::to_float(in[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_Bfloat16ToFloat_Core);

void BM_Bfloat16ToFloat_HandWritten(benchmark::State& state)
{
    std::vector<std::uint16_t> in(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        in[i] = bf16_by_hand(inputs[i]);
    std::vector<float> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < in.size(); ++i)
            out[i] = bf16_to_float_by_hand(in[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_Bfloat16ToFloat_HandWritten);

// For scale, not a pair: the generic conversion to FP8 E4M3 (no hand-written equivalent is shorter)
void BM_ToFloat8E4M3_Generic(benchmark::State& state)
{
    std::vector<snicholls::ml::float8_e4m3> out(inputs.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < inputs.size(); ++i)
            out[i] = snicholls::ml::to_minifloat<snicholls::ml::float8_e4m3>(inputs[i]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(inputs.size()));
}
BENCHMARK(BM_ToFloat8E4M3_Generic);

bool fair()
{
    for (const float v : inputs) {
        const auto core = snicholls::ml::to_bfloat16(v);
        const std::uint16_t hand = bf16_by_hand(v);
        if (core.bits != hand
            || std::bit_cast<std::uint32_t>(snicholls::ml::to_float(core)) != std::bit_cast<std::uint32_t>(bf16_to_float_by_hand(hand)))
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "minifloat benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
