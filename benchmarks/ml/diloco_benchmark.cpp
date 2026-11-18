//
//  diloco_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::ordered_sum (four workers' parts) and diloco_outer_step (the paper's Nesterov settings) against the
//  same loops written by hand (the same operations in the same order), over 1M floats. main() runs a fairness check
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

struct state_t {
    std::array<std::vector<float>, workers> parts{filled(1), filled(2), filled(3), filled(4)};
    std::array<std::span<const float>, workers> views{parts[0], parts[1], parts[2], parts[3]};
    std::vector<float> out = std::vector<float>(n), global = filled(5), momentum = std::vector<float>(n, 0.0f),
                       mean = filled(6);
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
    return same(a.out, b.out) && same(a.global, b.global) && same(a.momentum, b.momentum);
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
