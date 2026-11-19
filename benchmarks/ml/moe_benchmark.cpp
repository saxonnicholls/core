//
//  moe_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's mixture-of-experts steps that carry the work, against the same loops written by hand (the same
//  operations in the same order): the grouped product through linalg::reference (16 experts of 128 x 256 over 1024
//  permuted rows, 256 tokens routed to 4 each), the gather into expert order, and the weighted combine back. main()
//  runs a fairness check first and refuses to benchmark if any element differs in a single bit (CLAUDE.md "Zero
//  overhead").
//

#include <snicholls/core/ml/moe.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using u32 = std::uint32_t;

constexpr ml::transformer_config C{.d_model = 256, .n_heads = 4, .n_kv_heads = 4, .head_dim = 64, .d_ff = 128, .mlp = ml::mlp_kind::gelu,
                                   .experts = 16, .top_k = 4, .expert_d_ff = 128};
constexpr std::size_t rows = 256, n = rows * C.top_k, d = C.d_model, f = C.expert_d_ff, E = C.experts;

std::vector<float> filled(std::size_t size, std::uint64_t seed)
{
    std::vector<float> v(size);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL ^ seed;   // a fixed xorshift stream: identical data for both
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        e = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

struct state_t {
    std::vector<float> xs = filled(rows * d, 1), logits = filled(rows * E, 2), weights = std::vector<float>(n), w = filled(E * f * d, 3),
                       xp = std::vector<float>(n * d), yp = filled(n * d, 4), up = std::vector<float>(n * f), ys = std::vector<float>(rows * d);
    std::vector<u32> experts = std::vector<u32>(n), counts = std::vector<u32>(E), offsets = std::vector<u32>(E + 1), order = std::vector<u32>(n),
                     inverse = std::vector<u32>(n);
    state_t()
    {
        const auto& b = snicholls::linalg::reference;
        ml::moe_route_rows<C>(b, logits, {}, experts, std::span<float>(weights));
        ml::moe_count_rows<C>(b, experts, counts);
        ml::moe_offsets<C>(b, counts, offsets);
        ml::moe_permutation<C>(b, experts, offsets, order, inverse);
        ml::moe_gather_rows<C>(b, xs, order, std::span<float>(xp));
    }
    ml::grouped_weights<float> grouped() const { return {w, E, f, d}; }
};

// Expert by expert, each output a dot product from 0 in input order, as the reference gemm does
void grouped_by_hand(const float* w, const u32* offsets, const float* x, float* y)
{
    for (std::size_t e = 0; e < E; ++e)
        for (std::size_t r = offsets[e]; r < offsets[e + 1]; ++r)
            for (std::size_t o = 0; o < f; ++o) {
                const float* wr = w + (e * f + o) * d;
                float s = 0;
                for (std::size_t i = 0; i < d; ++i)
                    s += x[r * d + i] * wr[i];
                y[r * f + o] = s;
            }
}

void gather_by_hand(const float* xs, const u32* order, float* xp)
{
    for (std::size_t p = 0; p < n; ++p) {
        const float* from = xs + (order[p] / C.top_k) * d;
        for (std::size_t i = 0; i < d; ++i)
            xp[p * d + i] = from[i];
    }
}

void combine_by_hand(const float* yp, const float* w, const u32* inverse, float* ys)
{
    constexpr std::size_t K = C.top_k;
    for (std::size_t r = 0; r < rows; ++r) {
        float* y = ys + r * d;
        const float* p0 = yp + inverse[r * K] * d;
        for (std::size_t i = 0; i < d; ++i)
            y[i] = w[r * K] * p0[i];
        for (std::size_t s = 1; s < K; ++s) {
            const float ws = w[r * K + s];
            const float* p = yp + inverse[r * K + s] * d;
            for (std::size_t i = 0; i < d; ++i) {
                const float v = ws * p[i];
                y[i] = y[i] + v;
            }
        }
    }
}

void BM_GroupedLinear_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::grouped_linear(snicholls::linalg::reference, s.grouped(), s.offsets, s.xp, s.up);
        benchmark::DoNotOptimize(s.up.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n * f * d));
}
BENCHMARK(BM_GroupedLinear_Core)->Unit(benchmark::kMicrosecond);

void BM_GroupedLinear_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        grouped_by_hand(s.w.data(), s.offsets.data(), s.xp.data(), s.up.data());
        benchmark::DoNotOptimize(s.up.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n * f * d));
}
BENCHMARK(BM_GroupedLinear_HandWritten)->Unit(benchmark::kMicrosecond);

void BM_Gather_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::moe_gather_rows<C>(snicholls::linalg::reference, s.xs, s.order, std::span<float>(s.xp));
        benchmark::DoNotOptimize(s.xp.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n * d));
}
BENCHMARK(BM_Gather_Core)->Unit(benchmark::kMicrosecond);

void BM_Gather_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        gather_by_hand(s.xs.data(), s.order.data(), s.xp.data());
        benchmark::DoNotOptimize(s.xp.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n * d));
}
BENCHMARK(BM_Gather_HandWritten)->Unit(benchmark::kMicrosecond);

void BM_Combine_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::moe_combine_rows<C>(snicholls::linalg::reference, s.yp, s.weights, s.inverse, std::span<float>(s.ys));
        benchmark::DoNotOptimize(s.ys.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n * d));
}
BENCHMARK(BM_Combine_Core)->Unit(benchmark::kMicrosecond);

void BM_Combine_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        combine_by_hand(s.yp.data(), s.weights.data(), s.inverse.data(), s.ys.data());
        benchmark::DoNotOptimize(s.ys.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n * d));
}
BENCHMARK(BM_Combine_HandWritten)->Unit(benchmark::kMicrosecond);

bool same(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

bool fair()
{
    const auto& backend = snicholls::linalg::reference;
    state_t a, b;
    std::vector<float> xp(n * d);
    gather_by_hand(b.xs.data(), b.order.data(), xp.data());
    ml::grouped_linear(backend, a.grouped(), a.offsets, a.xp, a.up);
    grouped_by_hand(b.w.data(), b.offsets.data(), xp.data(), b.up.data());
    ml::moe_combine_rows<C>(backend, a.yp, a.weights, a.inverse, std::span<float>(a.ys));
    combine_by_hand(b.yp.data(), b.weights.data(), b.inverse.data(), b.ys.data());
    return same(a.xp, xp) && same(a.up, b.up) && same(a.ys, b.ys);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "moe benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
