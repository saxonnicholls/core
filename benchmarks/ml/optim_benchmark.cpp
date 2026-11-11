//
//  optim_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::adamw_step against llm.c's adamw_update loop written by hand (the same operations in the same
//  order, the bias corrections computed once per step), over 1M float parameters for ten steps. main() runs a
//  fairness check first and refuses to benchmark if any parameter or moment differs in a single bit (CLAUDE.md
//  "Zero overhead").
//

#include <snicholls/core/ml/optim.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr std::size_t n = 1 << 20;
const ml::adamw_config<float> config{.learning_rate = 3e-4f, .beta1 = 0.9f, .beta2 = 0.95f, .eps = 1e-8f, .weight_decay = 0.1f};

struct state_t {
    std::vector<float> p, g, m, v;
    state_t() : p(n), g(n), m(n, 0.0f), v(n, 0.0f)
    {
        std::uint64_t x = 0x9e3779b97f4a7c15ULL;   // a fixed xorshift stream: identical data for both
        for (std::size_t i = 0; i < n; ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            p[i] = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
            g[i] = static_cast<float>(static_cast<double>(x & 0xffff) / 65536.0 - 0.5);
        }
    }
};

// By hand: llm.c's adamw_update
void adamw_by_hand(float* p, const float* g, float* m, float* v, std::size_t count, std::uint64_t t, float lr, float b1, float b2,
                   float eps, float wd)
{
    const float c1 = 1.0f - snicholls::math::pow(b1, static_cast<float>(t));
    const float c2 = 1.0f - snicholls::math::pow(b2, static_cast<float>(t));
    for (std::size_t i = 0; i < count; ++i) {
        m[i] = b1 * m[i] + (1.0f - b1) * g[i];
        v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
        const float m_hat = m[i] / c1;
        const float v_hat = v[i] / c2;
        p[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * p[i]);
    }
}

void BM_AdamW_Core(benchmark::State& state)
{
    state_t s;
    std::uint64_t t = 0;
    for (auto _ : state) {
        ml::adamw_step(s.p, s.g, s.m, s.v, ++t, config);
        benchmark::DoNotOptimize(s.p.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_AdamW_Core);

void BM_AdamW_HandWritten(benchmark::State& state)
{
    state_t s;
    std::uint64_t t = 0;
    for (auto _ : state) {
        adamw_by_hand(s.p.data(), s.g.data(), s.m.data(), s.v.data(), n, ++t, config.learning_rate, config.beta1, config.beta2,
                      config.eps, config.weight_decay);
        benchmark::DoNotOptimize(s.p.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_AdamW_HandWritten);

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
    for (std::uint64_t t = 1; t <= 10; ++t) {
        ml::adamw_step(a.p, a.g, a.m, a.v, t, config);
        adamw_by_hand(b.p.data(), b.g.data(), b.m.data(), b.v.data(), n, t, config.learning_rate, config.beta1, config.beta2, config.eps,
                      config.weight_decay);
    }
    return same(a.p, b.p) && same(a.m, b.m) && same(a.v, b.v);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "optim benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
