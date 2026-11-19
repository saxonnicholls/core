//
//  jepa_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::jepa_loss_rows (L2 with its gradient) and ema_rows against the same loops written by hand (the same
//  operations in the same order): the loss over 64 samples' 49 target rows of 768 floats, and the EMA over 1M floats.
//  main() runs a fairness check first and refuses to benchmark if any value differs in a single bit (CLAUDE.md "Zero
//  overhead").
//

#include <snicholls/core/ml/jepa.hpp>

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
const auto& backend = snicholls::linalg::reference;

constexpr std::size_t rows = 64 * 49, width = 768, params = 1 << 20;
constexpr double momentum = 0.996;

std::vector<float> filled(std::uint64_t seed, std::size_t n)
{
    std::vector<float> v(n);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL ^ seed;   // a fixed xorshift stream: identical data for both
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        e = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

// The L2 loss as core sums it: each row's channels, then the rows, then one division; the gradient added
float loss_by_hand(const float* p, const float* t, float* dp)
{
    const float g = 1.0f / float(rows * width);
    float total = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        float row = 0;
        for (std::size_t c = 0; c < width; ++c) {
            const float d = p[r * width + c] - t[r * width + c];
            row += d * d;
            dp[r * width + c] += g * (2.0f * d);
        }
        total += row;
    }
    return 1.0f * (total / float(rows * width));
}

void ema_by_hand(float* target, const float* online)
{
    const float m = float(momentum), r = float(1.0 - momentum);
    for (std::size_t i = 0; i < params; ++i) {
        const float kept = target[i] * m;
        const float added = r * online[i];
        target[i] = kept + added;
    }
}

struct state_t {
    std::vector<float> pred = filled(1, rows * width), target = filled(2, rows * width), dpred = std::vector<float>(rows * width);
    std::vector<float> slow = filled(3, params), fast = filled(4, params);
};

void BM_JepaLossL2_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        benchmark::DoNotOptimize(ml::jepa_loss_rows(backend, ml::jepa_loss::l2, std::span<const float>(s.pred), std::span<const float>(s.target), width,
                                                    std::span<float>(s.dpred)));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * width));
}
BENCHMARK(BM_JepaLossL2_Core);

void BM_JepaLossL2_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        benchmark::DoNotOptimize(loss_by_hand(s.pred.data(), s.target.data(), s.dpred.data()));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * width));
}
BENCHMARK(BM_JepaLossL2_HandWritten);

void BM_Ema_Core(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ml::ema_rows(backend, std::span<float>(s.slow), std::span<const float>(s.fast), momentum);
        benchmark::DoNotOptimize(s.slow.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(params));
}
BENCHMARK(BM_Ema_Core);

void BM_Ema_HandWritten(benchmark::State& state)
{
    state_t s;
    for (auto _ : state) {
        ema_by_hand(s.slow.data(), s.fast.data());
        benchmark::DoNotOptimize(s.slow.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(params));
}
BENCHMARK(BM_Ema_HandWritten);

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
    const float la = ml::jepa_loss_rows(backend, ml::jepa_loss::l2, std::span<const float>(a.pred), std::span<const float>(a.target), width,
                                        std::span<float>(a.dpred));
    const float lb = loss_by_hand(b.pred.data(), b.target.data(), b.dpred.data());
    for (int r = 0; r < 5; ++r) {
        ml::ema_rows(backend, std::span<float>(a.slow), std::span<const float>(a.fast), momentum);
        ema_by_hand(b.slow.data(), b.fast.data());
    }
    return std::bit_cast<std::uint32_t>(la) == std::bit_cast<std::uint32_t>(lb) && same(a.dpred, b.dpred) && same(a.slow, b.slow);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "jepa benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
