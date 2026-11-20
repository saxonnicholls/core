//
//  quant_k_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's K-quant dot products and dequantisation against ggml's scalar reference loops written out by
//  hand (ggml_vec_dot_q4_K_q8_K_generic, ggml_vec_dot_q6_K_q8_K_generic, dequantize_row_q4_K), on the same rows of
//  4,096 weights and activations. main() runs a fairness check first and refuses to benchmark if any result
//  differs in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/quant_k.hpp>

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

constexpr std::size_t n = 4096, blocks = n / 256;

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
    std::vector<ml::block_q4_k> w4 = std::vector<ml::block_q4_k>(blocks);
    std::vector<ml::block_q6_k> w6 = std::vector<ml::block_q6_k>(blocks);
    std::vector<ml::block_q8_k> a8 = std::vector<ml::block_q8_k>(blocks);
    rows()
    {
        ml::quantize(w, std::span(w4));
        ml::quantize(w, std::span(w6));
        ml::quantize(a, std::span(a8));
    }
};

const rows& data()
{
    static const rows r;
    return r;
}

float half(ml::float16 h) { return ml::to_float(h); }

void scales_k4(const std::uint8_t* s, std::uint8_t* sc, std::uint8_t* mn)
{
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = s[j] & 63;
            mn[j] = s[j + 4] & 63;
        } else {
            sc[j] = static_cast<std::uint8_t>((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
            mn[j] = static_cast<std::uint8_t>((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
        }
    }
}

// By hand: ggml's generic loops, one statement per multiplication and addition as core writes them
float dot_q4k_by_hand(const ml::block_q4_k* x, const ml::block_q8_k* y, std::size_t nb)
{
    float sums[8] = {};
    float sumf = 0;
    for (std::size_t i = 0; i < nb; ++i) {
        std::int8_t aux8[256];
        const std::uint8_t* q4 = x[i].qs.data();
        std::int8_t* a = aux8;
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = static_cast<std::int8_t>(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = static_cast<std::int8_t>(q4[l] >> 4);
            a += 32;
            q4 += 32;
        }
        std::uint8_t sc[8], mn[8];
        scales_k4(x[i].scales.data(), sc, mn);
        int sumi = 0;
        for (int j = 0; j < 16; ++j) sumi += y[i].bsums[std::size_t(j)] * mn[j / 2];
        std::int32_t aux32[8] = {};
        const std::int8_t* q8 = y[i].qs.data();
        a = aux8;
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 8; ++l) aux32[l] += sc[j] * (q8[l] * a[l]);
                q8 += 8;
                a += 8;
            }
        const float d = half(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) {
            const float term = d * float(aux32[l]);
            sums[l] += term;
        }
        const float dmin = half(x[i].dmin) * y[i].d;
        const float term = dmin * float(sumi);
        sumf -= term;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

float dot_q6k_by_hand(const ml::block_q6_k* x, const ml::block_q8_k* y, std::size_t nb)
{
    float sums[8] = {};
    for (std::size_t i = 0; i < nb; ++i) {
        std::int8_t aux8[256];
        const std::uint8_t* q4 = x[i].ql.data();
        const std::uint8_t* qh = x[i].qh.data();
        std::int8_t* a = aux8;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = static_cast<std::int8_t>(((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                a[l + 32] = static_cast<std::int8_t>(((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                a[l + 64] = static_cast<std::int8_t>(((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = static_cast<std::int8_t>(((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            a += 128;
            q4 += 64;
            qh += 32;
        }
        std::int32_t aux32[8] = {};
        const std::int8_t* q8 = y[i].qs.data();
        a = aux8;
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 8; ++l) aux32[l] += x[i].scales[std::size_t(j)] * (q8[l] * a[l]);
                q8 += 8;
                a += 8;
            }
        const float d = half(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) {
            const float term = d * float(aux32[l]);
            sums[l] += term;
        }
    }
    float sumf = 0;
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

void dequantize_q4k_by_hand(const ml::block_q4_k* x, float* y, std::size_t nb)
{
    for (std::size_t i = 0; i < nb; ++i) {
        const std::uint8_t* q = x[i].qs.data();
        const float d = half(x[i].d), min = half(x[i].dmin);
        std::uint8_t sc[8], mn[8];
        scales_k4(x[i].scales.data(), sc, mn);
        for (int j = 0, is = 0; j < 256; j += 64, is += 2) {
            const float d1 = d * float(sc[is]), m1 = min * float(mn[is]);
            const float d2 = d * float(sc[is + 1]), m2 = min * float(mn[is + 1]);
            for (int l = 0; l < 32; ++l) {
                const float v = d1 * float(q[l] & 0xF);
                *y++ = v - m1;
            }
            for (int l = 0; l < 32; ++l) {
                const float v = d2 * float(q[l] >> 4);
                *y++ = v - m2;
            }
            q += 32;
        }
    }
}

void BM_DotQ4K_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(ml::dot(std::span<const ml::block_q4_k>(data().w4), std::span<const ml::block_q8_k>(data().a8)));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotQ4K_Core);

void BM_DotQ4K_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_q4k_by_hand(data().w4.data(), data().a8.data(), blocks));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotQ4K_HandWritten);

void BM_DotQ6K_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(ml::dot(std::span<const ml::block_q6_k>(data().w6), std::span<const ml::block_q8_k>(data().a8)));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotQ6K_Core);

void BM_DotQ6K_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_q6k_by_hand(data().w6.data(), data().a8.data(), blocks));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotQ6K_HandWritten);

void BM_DequantizeQ4K_Core(benchmark::State& state)
{
    std::vector<float> out(n);
    for (auto _ : state) {
        ml::dequantize(std::span<const ml::block_q4_k>(data().w4), out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DequantizeQ4K_Core);

void BM_DequantizeQ4K_HandWritten(benchmark::State& state)
{
    std::vector<float> out(n);
    for (auto _ : state) {
        dequantize_q4k_by_hand(data().w4.data(), out.data(), blocks);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DequantizeQ4K_HandWritten);

bool fair()
{
    const auto& r = data();
    const auto bits = [](float f) { return std::bit_cast<std::uint32_t>(f); };
    if (bits(ml::dot(std::span<const ml::block_q4_k>(r.w4), std::span<const ml::block_q8_k>(r.a8))) != bits(dot_q4k_by_hand(r.w4.data(), r.a8.data(), blocks))
        || bits(ml::dot(std::span<const ml::block_q6_k>(r.w6), std::span<const ml::block_q8_k>(r.a8))) != bits(dot_q6k_by_hand(r.w6.data(), r.a8.data(), blocks)))
        return false;
    std::vector<float> core(n), hand(n);
    ml::dequantize(std::span<const ml::block_q4_k>(r.w4), core);
    dequantize_q4k_by_hand(r.w4.data(), hand.data(), blocks);
    return std::memcmp(core.data(), hand.data(), n * sizeof(float)) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "quant_k benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
