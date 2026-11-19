//
//  diloco_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::ordered_sum (four workers' parts) and diloco_outer_step (the paper's Nesterov settings) against the
//  same loops written by hand, and the bfloat16 pair, compress_bf16 (error feedback) and ordered_sum_bf16 (four
//  workers' bfloat16 parts), against theirs (the same operations in the same order), over 1M floats. main() runs a fairness check
//  first and refuses to benchmark if any element differs in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/diloco.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr std::size_t n = 1 << 20, workers = 4;
const ml::diloco_config config{};

std::vector<float> filled(std::uint64_t seed)
{
    std::vector<float> v(n);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL ^ seed;   // a fixed xorshift stream: identical data for both
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        e = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

std::vector<std::uint16_t> encoded(const std::vector<float>& v)
{
    std::vector<std::uint16_t> b(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        b[i] = ml::to_bfloat16(v[i]).bits;
    return b;
}

// bfloat16 by hand: round to nearest even with a quiet NaN, decode by a shift
std::uint16_t bf16_by_hand(float x)
{
    const std::uint32_t u = std::bit_cast<std::uint32_t>(x);
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
    return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float decode_by_hand(std::uint16_t b)
{
    return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}
bool finite_by_hand(float x)
{
    return (std::bit_cast<std::uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}

void compress_by_hand(const float* delta, float* residual, std::uint16_t* out)
{
    for (std::size_t i = 0; i < n; ++i) {
        const float v = delta[i] + residual[i];
        out[i] = bf16_by_hand(v);
        const float back = decode_by_hand(out[i]);
        residual[i] = finite_by_hand(v) && finite_by_hand(back) ? v - back : 0.0f;
    }
}

template <class Bits>
void sum_bf16_by_hand(const Bits& bits, float* out)
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = decode_by_hand(bits[0][i]);
    for (std::size_t k = 1; k < workers; ++k) {
        const std::uint16_t* p = bits[k].data();
        for (std::size_t i = 0; i < n; ++i)
            out[i] = out[i] + decode_by_hand(p[i]);
    }
}

struct state_t {
    std::array<std::vector<float>, workers> parts{filled(1), filled(2), filled(3), filled(4)};
    std::array<std::span<const float>, workers> views{parts[0], parts[1], parts[2], parts[3]};
    std::vector<float> out = std::vector<float>(n), global = filled(5), momentum = std::vector<float>(n, 0.0f),
                       mean = filled(6);
    std::vector<float> residual = std::vector<float>(n, 0.0f);
    std::vector<std::uint16_t> wire = std::vector<std::uint16_t>(n);
    std::array<std::vector<std::uint16_t>, workers> bits{encoded(parts[0]), encoded(parts[1]), encoded(parts[2]), encoded(parts[3])};
    std::array<std::span<const std::uint16_t>, workers> bit_views{bits[0], bits[1], bits[2], bits[3]};
};

void sum_by_hand(const std::array<std::vector<float>, workers>& parts, float* out)
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = parts[0][i];
    for (std::size_t k = 1; k < workers; ++k) {
        const float* p = parts[k].data();
        for (std::size_t i = 0; i < n; ++i)
            out[i] = out[i] + p[i];
    }
}

// SGD with Nesterov momentum, each product and sum its own statement, as core writes it
void outer_by_hand(float* global, const float* mean, float* momentum, float mu, float lr)
{
    for (std::size_t i = 0; i < n; ++i) {
        const float decayed = mu * momentum[i];
        momentum[i] = decayed + mean[i];
        const float ahead = mu * momentum[i];
        const float step = mean[i] + ahead;
        const float change = lr * step;
        global[i] = global[i] - change;
    }
}

void BM_OrderedSum_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::ordered_sum<float>(s.views, s.out);
        benchmark::DoNotOptimize(s.out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_OrderedSum_Core);

void BM_OrderedSum_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        sum_by_hand(s.parts, s.out.data());
        benchmark::DoNotOptimize(s.out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_OrderedSum_HandWritten);

void BM_OuterStep_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::diloco_outer_step<float>(s.global, s.mean, s.momentum, config);
        benchmark::DoNotOptimize(s.global.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_OuterStep_Core);

void BM_OuterStep_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        outer_by_hand(s.global.data(), s.mean.data(), s.momentum.data(), static_cast<float>(config.momentum),
                      static_cast<float>(config.outer_lr));
        benchmark::DoNotOptimize(s.global.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_OuterStep_HandWritten);

void BM_CompressBf16_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::compress_bf16(s.mean, s.residual, s.wire);
        benchmark::DoNotOptimize(s.wire.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_CompressBf16_Core);

void BM_CompressBf16_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        compress_by_hand(s.mean.data(), s.residual.data(), s.wire.data());
        benchmark::DoNotOptimize(s.wire.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_CompressBf16_HandWritten);

void BM_OrderedSumBf16_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::ordered_sum_bf16(s.bit_views, s.out);
        benchmark::DoNotOptimize(s.out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_OrderedSumBf16_Core);

void BM_OrderedSumBf16_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        sum_bf16_by_hand(s.bits, s.out.data());
        benchmark::DoNotOptimize(s.out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_OrderedSumBf16_HandWritten);

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
    ml::ordered_sum<float>(a.views, a.out);
    sum_by_hand(b.parts, b.out.data());
    for (int r = 0; r < 10; ++r) {
        ml::diloco_outer_step<float>(a.global, a.mean, a.momentum, config);
        outer_by_hand(b.global.data(), b.mean.data(), b.momentum.data(), static_cast<float>(config.momentum),
                      static_cast<float>(config.outer_lr));
    }
    if (!(same(a.out, b.out) && same(a.global, b.global) && same(a.momentum, b.momentum)))
        return false;
    for (int r = 0; r < 3; ++r) {   // the residual carried over rounds
        ml::compress_bf16(a.mean, a.residual, a.wire);
        compress_by_hand(b.mean.data(), b.residual.data(), b.wire.data());
    }
    ml::ordered_sum_bf16(a.bit_views, a.out);
    sum_bf16_by_hand(b.bits, b.out.data());
    return a.wire == b.wire && same(a.residual, b.residual) && same(a.out, b.out);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "diloco benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
