//
//  quant_ternary_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's TQ1_0 and TQ2_0 dot products with Q8_K against ggml's scalar loops written out by hand
//  (ggml_vec_dot_tq1_0_q8_K_generic, ggml_vec_dot_tq2_0_q8_K_generic), on the same rows of 4,096 weights.
//  main() runs a fairness check first and refuses to benchmark if any result differs in a single bit.
//

#include <snicholls/core/ml/quant_ternary.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
    std::vector<ml::block_tq1_0> t1 = std::vector<ml::block_tq1_0>(blocks);
    std::vector<ml::block_tq2_0> t2 = std::vector<ml::block_tq2_0>(blocks);
    std::vector<ml::block_q8_k> a = std::vector<ml::block_q8_k>(blocks);
    rows()
    {
        const auto w = make(11), x = make(12);
        ml::quantize(w, std::span(t1));
        ml::quantize(w, std::span(t2));
        ml::quantize(x, std::span(a));
    }
};

const rows& data()
{
    static const rows r;
    return r;
}

// By hand: ggml's generic loops
float dot_tq1_by_hand(const ml::block_tq1_0* x, const ml::block_q8_k* y, std::size_t nb)
{
    const std::uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        int sum = 0;
        for (std::size_t l = 0; l < 5; ++l)
            for (std::size_t m = 0; m < 32; ++m) {
                const std::uint8_t q = static_cast<std::uint8_t>(x[i].qs[m] * pow3[l]);
                const std::uint16_t xi = static_cast<std::uint16_t>((std::uint16_t(q) * 3) >> 8);
                sum += (xi - 1) * y[i].qs[l * 32 + m];
            }
        for (std::size_t l = 0; l < 5; ++l)
            for (std::size_t m = 0; m < 16; ++m) {
                const std::uint8_t q = static_cast<std::uint8_t>(x[i].qs[32 + m] * pow3[l]);
                const std::uint16_t xi = static_cast<std::uint16_t>((std::uint16_t(q) * 3) >> 8);
                sum += (xi - 1) * y[i].qs[160 + l * 16 + m];
            }
        for (std::size_t l = 0; l < 4; ++l)
            for (std::size_t j = 0; j < 4; ++j) {
                const std::uint8_t q = static_cast<std::uint8_t>(x[i].qh[j] * pow3[l]);
                const std::uint16_t xi = static_cast<std::uint16_t>((std::uint16_t(q) * 3) >> 8);
                sum += (xi - 1) * y[i].qs[240 + l * 4 + j];
            }
        const float scale = ml::to_float(x[i].d) * y[i].d;
        const float term = float(sum) * scale;
        sumf += term;
    }
    return sumf;
}

float dot_tq2_by_hand(const ml::block_tq2_0* x, const ml::block_q8_k* y, std::size_t nb)
{
    float sumf = 0.0f;
    for (std::size_t i = 0; i < nb; ++i) {
        std::int32_t sumi = 0;
        for (std::size_t j = 0; j < 64; j += 32)
            for (std::size_t l = 0; l < 4; ++l)
                for (std::size_t k = 0; k < 32; ++k)
                    sumi += y[i].qs[j * 4 + l * 32 + k] * (((x[i].qs[j + k] >> (l * 2)) & 3) - 1);
        const float d = y[i].d * ml::to_float(x[i].d);
        const float term = float(sumi) * d;
        sumf += term;
    }
    return sumf;
}

void BM_DotTq1_0_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(ml::dot(std::span<const ml::block_tq1_0>(data().t1), std::span<const ml::block_q8_k>(data().a)));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotTq1_0_Core);

void BM_DotTq1_0_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_tq1_by_hand(data().t1.data(), data().a.data(), blocks));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotTq1_0_HandWritten);

void BM_DotTq2_0_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(ml::dot(std::span<const ml::block_tq2_0>(data().t2), std::span<const ml::block_q8_k>(data().a)));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotTq2_0_Core);

void BM_DotTq2_0_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(dot_tq2_by_hand(data().t2.data(), data().a.data(), blocks));
    state.SetItemsProcessed(state.iterations() * std::int64_t{n});
}
BENCHMARK(BM_DotTq2_0_HandWritten);

bool fair()
{
    const auto& r = data();
    const auto bits = [](float f) { return std::bit_cast<std::uint32_t>(f); };
    return bits(ml::dot(std::span<const ml::block_tq1_0>(r.t1), std::span<const ml::block_q8_k>(r.a))) == bits(dot_tq1_by_hand(r.t1.data(), r.a.data(), blocks))
           && bits(ml::dot(std::span<const ml::block_tq2_0>(r.t2), std::span<const ml::block_q8_k>(r.a))) == bits(dot_tq2_by_hand(r.t2.data(), r.a.data(), blocks));
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "quant_ternary benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
