//
//  quant_iq_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's IQ dot products and dequantisation against ggml's scalar reference loops written out by hand
//  (ggml_vec_dot_{iq2_xxs,iq3_s,iq1_s,iq4_xs}_q8_K_generic, dequantize_row_iq4_xs), on the same rows of 4,096
//  weights and activations. main() runs a fairness check first and refuses to benchmark if any result differs in a
//  single bit (CLAUDE.md "Zero overhead"). Quantisation is a conversion step and is not benchmarked here.
//

#include <snicholls/core/ml/quant_iq.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr std::size_t n = 4096, superblocks = n / 256;

std::vector<float> make(std::uint64_t x)
{
    std::vector<float> out(n);
    for (auto& v : out) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return out;
}

struct rows {
    std::vector<float> w = make(21), a = make(22), importance = std::vector<float>(n, 1.0f);
    std::vector<ml::block_q8_k> a8 = std::vector<ml::block_q8_k>(superblocks);
    std::vector<ml::block_iq2_xxs> w2 = std::vector<ml::block_iq2_xxs>(superblocks);
    std::vector<ml::block_iq3_s> w3 = std::vector<ml::block_iq3_s>(superblocks);
    std::vector<ml::block_iq1_s> w1 = std::vector<ml::block_iq1_s>(superblocks);
    std::vector<ml::block_iq4_xs> w4 = std::vector<ml::block_iq4_xs>(superblocks);
    rows()
    {
        ml::quantize(a, std::span(a8));
        ml::quantize(w, importance, std::span(w2));
        ml::quantize(w, importance, std::span(w3));
        ml::quantize(w, importance, std::span(w1));
        ml::quantize(w, importance, std::span(w4));
    }
};

const rows& data()
{
    static const rows r;
    return r;
}

float half(ml::float16 h) { return ml::to_float(h); }

std::uint8_t byte_of(std::uint64_t v, int j) { return static_cast<std::uint8_t>(v >> (8 * j)); }

// By hand: ggml's generic loops over its own byte layout (memcpy of the words), one statement per float operation
float dot_iq2_xxs_by_hand(const ml::block_iq2_xxs* x, const ml::block_q8_k* y, std::size_t nb)
{
    float sumf = 0.f;
    for (std::size_t i = 0; i < nb; ++i) {
        const float d = half(x[i].d) * y[i].d;
        const std::uint16_t* q2 = x[i].qs.data();
        const std::int8_t* q8 = y[i].qs.data();
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            std::uint32_t aux32[2];
            std::memcpy(aux32, q2, 2 * sizeof(std::uint32_t));
            q2 += 4;
            const auto* aux8 = reinterpret_cast<const std::uint8_t*>(aux32);
            const std::uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = ml::iq2xxs_grid[aux8[l]];
                const std::uint8_t signs = ml::iq2_signs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j)
                    sumi += byte_of(grid, j) * q8[j] * (signs & (1 << j) ? -1 : 1);
                q8 += 8;
            }
            bsum += sumi * std::int32_t(ls);
        }
        const float term = d * float(bsum);
        sumf += term;
    }
    return 0.125f * sumf;
}

float dot_iq3_s_by_hand(const ml::block_iq3_s* x, const ml::block_q8_k* y, std::size_t nb)
{
    float sumf = 0.f;
    for (std::size_t i = 0; i < nb; ++i) {
        const float d = half(x[i].d) * y[i].d;
        const std::uint8_t* qs = x[i].qs.data();
        const std::uint8_t* qh = x[i].qh.data();
        const std::uint8_t* signs = x[i].signs.data();
        const std::int8_t* q8 = y[i].qs.data();
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            for (int h = 0; h < 2; ++h) {
                const std::uint32_t ls = 2 * ((x[i].scales[std::size_t(ib32 / 2)] >> (4 * h)) & 0xf) + 1;
                std::int32_t sumi = 0;
                for (int l = 0; l < 4; ++l) {
                    const std::uint32_t g1 = ml::iq3s_grid[qs[2 * l + 0] | ((qh[ib32 + h] << (8 - 2 * l)) & 256)];
                    const std::uint32_t g2 = ml::iq3s_grid[qs[2 * l + 1] | ((qh[ib32 + h] << (7 - 2 * l)) & 256)];
                    for (int j = 0; j < 4; ++j) {
                        sumi += byte_of(g1, j) * q8[j + 0] * (signs[l] & (1 << j) ? -1 : 1);
                        sumi += byte_of(g2, j) * q8[j + 4] * (signs[l] & (1 << (j + 4)) ? -1 : 1);
                    }
                    q8 += 8;
                }
                qs += 8;
                signs += 4;
                bsum += sumi * std::int32_t(ls);
            }
        }
        const float term = d * float(bsum);
        sumf += term;
    }
    return sumf;
}

float dot_iq1_s_by_hand(const ml::block_iq1_s* x, const ml::block_q8_k* y, std::size_t nb)
{
    float sumf = 0;
    for (std::size_t i = 0; i < nb; i++) {
        const std::int8_t* q8 = y[i].qs.data();
        const std::uint8_t* qs = x[i].qs.data();
        const std::uint16_t* qh = x[i].qh.data();
        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = 2 * ((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = ml::iq1s_grid[qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)];
                for (int j = 0; j < 8; ++j)
                    lsum += q8[j] * static_cast<std::int8_t>(byte_of(grid, j));
                q8 += 8;
            }
            sumi += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[std::size_t(2 * ib)] + y[i].bsums[std::size_t(2 * ib + 1)]);
            qs += 4;
        }
        const float dd = half(x[i].d) * y[i].d;
        const float shift = 0.125f * float(sumi1);
        const float s = float(sumi) + shift;
        const float term = dd * s;
        sumf += term;
    }
    return sumf;
}

float dot_iq4_xs_by_hand(const ml::block_iq4_xs* x, const ml::block_q8_k* y, std::size_t nb)
{
    float sumf = 0;
    for (std::size_t ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = half(x[ibl].d) * y[ibl].d;
        std::uint16_t h = x[ibl].scales_h;
        const std::uint8_t* qs = x[ibl].qs.data();
        const std::int8_t* q8 = y[ibl].qs.data();
        for (int ib = 0; ib < 8; ib += 2) {
            const std::uint8_t ls1 = static_cast<std::uint8_t>((x[ibl].scales_l[std::size_t(ib / 2)] & 0xf) | ((h << 4) & 0x30));
            const std::uint8_t ls2 = static_cast<std::uint8_t>((x[ibl].scales_l[std::size_t(ib / 2)] >> 4) | ((h << 2) & 0x30));
            h = static_cast<std::uint16_t>(h >> 4);
            for (const std::uint8_t ls : {ls1, ls2}) {
                const float dl = d4d8 * float(ls - 32);
                int sumi1 = 0, sumi2 = 0;
                for (int j = 0; j < 16; ++j) {
                    sumi1 += q8[j + 0] * ml::iq4nl_values[qs[j] & 0xf];
                    sumi2 += q8[j + 16] * ml::iq4nl_values[qs[j] >> 4];
                }
                const float term = dl * float(sumi1 + sumi2);
                sumf += term;
                qs += 16;
                q8 += 32;
            }
        }
    }
    return sumf;
}

void dequantize_iq4_xs_by_hand(const ml::block_iq4_xs* x, float* y, std::size_t nb)
{
    for (std::size_t i = 0; i < nb; i++) {
        const std::uint8_t* qs = x[i].qs.data();
        const float d = half(x[i].d);
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((x[i].scales_l[std::size_t(ib / 2)] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * float(ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * float(ml::iq4nl_values[qs[j] & 0xf]);
                y[j + 16] = dl * float(ml::iq4nl_values[qs[j] >> 4]);
            }
            y += 32;
            qs += 16;
        }
    }
}

#define SNICHOLLS_IQ_DOT_PAIR(Name, member, by_hand)                                                                     \
    void BM_Dot##Name##_Core(benchmark::State& state)                                                                    \
    {                                                                                                                    \
        for (auto _ : state)                                                                                             \
            benchmark::DoNotOptimize(ml::dot(std::span(std::as_const(data().member)), std::span(std::as_const(data().a8)))); \
        state.SetItemsProcessed(state.iterations() * std::int64_t{n});                                                   \
    }                                                                                                                    \
    BENCHMARK(BM_Dot##Name##_Core);                                                                                      \
    void BM_Dot##Name##_HandWritten(benchmark::State& state)                                                             \
    {                                                                                                                    \
        for (auto _ : state)                                                                                             \
            benchmark::DoNotOptimize(by_hand(data().member.data(), data().a8.data(), superblocks));                      \
        state.SetItemsProcessed(state.iterations() * std::int64_t{n});                                                   \
    }                                                                                                                    \
    BENCHMARK(BM_Dot##Name##_HandWritten);

SNICHOLLS_IQ_DOT_PAIR(Iq2Xxs, w2, dot_iq2_xxs_by_hand)
SNICHOLLS_IQ_DOT_PAIR(Iq3S, w3, dot_iq3_s_by_hand)
SNICHOLLS_IQ_DOT_PAIR(Iq1S, w1, dot_iq1_s_by_hand)
SNICHOLLS_IQ_DOT_PAIR(Iq4Xs, w4, dot_iq4_xs_by_hand)

void BM_DequantizeIq4Xs_Core(benchmark::State& state)
{
    std::vector<float> out(n);
    for (auto _ : state) {
        ml::dequantize(std::span(std::as_const(data().w4)), out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DequantizeIq4Xs_Core);

void BM_DequantizeIq4Xs_HandWritten(benchmark::State& state)
{
    std::vector<float> out(n);
    for (auto _ : state) {
        dequantize_iq4_xs_by_hand(data().w4.data(), out.data(), superblocks);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DequantizeIq4Xs_HandWritten);

bool fair()
{
    const auto& r = data();
    const auto bits = [](float f) { return std::bit_cast<std::uint32_t>(f); };
    const auto a = std::span(r.a8);
    if (bits(ml::dot(std::span(r.w2), a)) != bits(dot_iq2_xxs_by_hand(r.w2.data(), r.a8.data(), superblocks))
        || bits(ml::dot(std::span(r.w3), a)) != bits(dot_iq3_s_by_hand(r.w3.data(), r.a8.data(), superblocks))
        || bits(ml::dot(std::span(r.w1), a)) != bits(dot_iq1_s_by_hand(r.w1.data(), r.a8.data(), superblocks))
        || bits(ml::dot(std::span(r.w4), a)) != bits(dot_iq4_xs_by_hand(r.w4.data(), r.a8.data(), superblocks)))
        return false;
    std::vector<float> core(n), hand(n);
    ml::dequantize(std::span(r.w4), core);
    dequantize_iq4_xs_by_hand(r.w4.data(), hand.data(), superblocks);
    return std::memcmp(core.data(), hand.data(), n * sizeof(float)) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "quant_iq benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
