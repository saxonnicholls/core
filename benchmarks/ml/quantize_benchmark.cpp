//
//  quantize_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's quantised dot products and Q8_0 quantisation against ggml's scalar reference loops written
//  out by hand (ggml_vec_dot_{q8_0,q4_0,mxfp4}_q8_0_generic, quantize_row_q8_0_ref), on the same rows of 4,096
//  weights and activations. main() runs a fairness check first and refuses to benchmark if any result differs in
//  a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/quantize.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr std::size_t n = 4096, blocks = n / 32;
constexpr std::int8_t kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

std::vector<float> make(std::uint64_t x)
{
    std::vector<float> out(n);
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return out;
}

struct rows {
    std::vector<float> w = make(11), a = make(12);
    std::vector<ml::block_q8_0> w8 = std::vector<ml::block_q8_0>(blocks), a8 = std::vector<ml::block_q8_0>(blocks);
    std::vector<ml::block_q4_0> w4 = std::vector<ml::block_q4_0>(blocks);
    std::vector<ml::block_mxfp4> wm = std::vector<ml::block_mxfp4>(blocks);
    rows()
    {
        ml::quantize(w, std::span(w8));
        ml::quantize(a, std::span(a8));
        ml::quantize(w, std::span(w4));
        ml::quantize(w, std::span(wm));
    }
};

const rows& data()
{
    static const rows r;
    return r;
}

float half_to_float(ml::float16 h) { return ml::to_float(h); }

float e8m0_half(std::uint8_t e)
{
    const std::uint32_t bits = e < 2 ? std::uint32_t{0x00200000} << e : std::uint32_t(e - 1) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// By hand: ggml's generic loops, one statement per multiplication and addition as core writes them
float dot_q8_by_hand(const ml::block_q8_0* x, const ml::block_q8_0* y, std::size_t nb)
{
    float sumf = 0;
    for (std::size_t ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < 32; j++)
            sumi += x[ib].qs[std::size_t(j)] * y[ib].qs[std::size_t(j)];
        const float scale = half_to_float(x[ib].d) * half_to_float(y[ib].d);
        const float term = float(sumi) * scale;
        sumf += term;
    }
    return sumf;
}

float dot_q4_by_hand(const ml::block_q4_0* x, const ml::block_q8_0* y, std::size_t nb)
{
    float sumf = 0;
    for (std::size_t ib = 0; ib < nb; ++ib) {
        int sumi0 = 0, sumi1 = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            sumi0 += ((x[ib].qs[j] & 0x0F) - 8) * y[ib].qs[j];
            sumi1 += ((x[ib].qs[j] >> 4) - 8) * y[ib].qs[j + 16];
        }
        const float term = float(sumi0 + sumi1) * half_to_float(x[ib].d) * half_to_float(y[ib].d);
        sumf += term;
    }
    return sumf;
}

float dot_mx_by_hand(const ml::block_mxfp4* x, const ml::block_q8_0* y, std::size_t nb)
{
    float sumf = 0;
    for (std::size_t ib = 0; ib < nb; ++ib) {
        const float d = half_to_float(y[ib].d) * e8m0_half(x[ib].e);
        int sumi1 = 0, sumi2 = 0;
        for (std::size_t j = 0; j < 16; ++j) {
            sumi1 += y[ib].qs[j] * kvalues[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + 16] * kvalues[x[ib].qs[j] >> 4];
        }
        const float term = d * float(sumi1 + sumi2);
        sumf += term;
    }
    return sumf;
}

void quantize_q8_by_hand(const float* x, ml::block_q8_0* y, std::size_t nb)
{
    for (std::size_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (std::size_t j = 0; j < 32; j++)
            amax = std::fmax(amax, std::fabs(x[i * 32 + j]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = ml::to_minifloat<ml::float16>(d);
        for (std::size_t j = 0; j < 32; ++j) {
            const float x0 = x[i * 32 + j] * id;
            y[i].qs[j] = static_cast<std::int8_t>(std::round(x0));
        }
    }
}

void BM_DotQ8_0_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(ml::dot(std::span<const ml::block_q8_0>(data().w8), data().a8));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotQ8_0_Core);

void BM_DotQ8_0_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_q8_by_hand(data().w8.data(), data().a8.data(), blocks));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotQ8_0_HandWritten);

void BM_DotQ4_0_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(ml::dot(std::span<const ml::block_q4_0>(data().w4), data().a8));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotQ4_0_Core);

void BM_DotQ4_0_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_q4_by_hand(data().w4.data(), data().a8.data(), blocks));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotQ4_0_HandWritten);

void BM_DotMxfp4_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(ml::dot(std::span<const ml::block_mxfp4>(data().wm), data().a8));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotMxfp4_Core);

void BM_DotMxfp4_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_mx_by_hand(data().wm.data(), data().a8.data(), blocks));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotMxfp4_HandWritten);

void BM_QuantizeQ8_0_Core(benchmark::State& state)
{
    std::vector<ml::block_q8_0> out(blocks);
    for (auto _ : state) {
        ml::quantize(data().w, std::span(out));
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_QuantizeQ8_0_Core);

void BM_QuantizeQ8_0_HandWritten(benchmark::State& state)
{
    std::vector<ml::block_q8_0> out(blocks);
    for (auto _ : state) {
        quantize_q8_by_hand(data().w.data(), out.data(), blocks);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_QuantizeQ8_0_HandWritten);

bool fair()
{
    const auto& r = data();
    const auto bits = [](float f) { return std::bit_cast<std::uint32_t>(f); };
    if (bits(ml::dot(std::span<const ml::block_q8_0>(r.w8), r.a8)) != bits(dot_q8_by_hand(r.w8.data(), r.a8.data(), blocks))
        || bits(ml::dot(std::span<const ml::block_q4_0>(r.w4), r.a8)) != bits(dot_q4_by_hand(r.w4.data(), r.a8.data(), blocks))
        || bits(ml::dot(std::span<const ml::block_mxfp4>(r.wm), r.a8)) != bits(dot_mx_by_hand(r.wm.data(), r.a8.data(), blocks)))
        return false;
    std::vector<ml::block_q8_0> hand(blocks);
    quantize_q8_by_hand(r.w.data(), hand.data(), blocks);
    return std::memcmp(hand.data(), r.w8.data(), blocks * sizeof(ml::block_q8_0)) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "quantize benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
