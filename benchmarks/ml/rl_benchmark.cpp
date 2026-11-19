//
//  rl_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::policy_loss_rows (GRPO's settings: clip 0.2, k3 KL with beta 0.04, sequence mean, an entropy bonus)
//  against the same loop written by hand, over 64 sequences of 1024 tokens with three quarters of each a response.
//  main() runs a fairness check first and refuses to benchmark if the loss or any gradient differs in a single bit
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/rl.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr std::size_t rows = 64, len = 1024, n = rows * len;
const ml::policy_loss_config config{.kl_beta = 0.04, .entropy_coef = 0.01};

std::vector<float> filled(std::uint64_t seed, double lo, double hi)
{
    std::vector<float> v(n);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL ^ seed;   // a fixed xorshift stream: identical data for both
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        e = static_cast<float>(lo + (hi - lo) * (static_cast<double>(x >> 11) * 0x1.0p-53));
    }
    return v;
}

const std::vector<float> logps = filled(1, -3, -0.01), old = filled(2, -3, -0.01), ref = filled(3, -3, -0.01), adv = filled(4, -1, 1),
                         ent = filled(5, 0, 3);
const std::vector<std::uint8_t> mask = [] {
    std::vector<std::uint8_t> m(n);
    for (std::size_t i = 0; i < n; ++i)
        m[i] = (i % len) >= len / 4 ? 1 : 0;
    return m;
}();

// The metrics core keeps, kept here too
std::uint64_t tokens_seen = 0, clipped_seen = 0;
float kl_sum = 0, ratio_sum = 0, entropy_sum = 0;

float by_hand(float* dlogps, float* dents)
{
    std::size_t total = 0;
    for (const auto m : mask)
        total += m != 0 ? 1 : 0;
    (void)total;
    const float lo = 1.0f - 0.2f, hi = 1.0f + 0.2f, beta = 0.04f, coef = 0.01f;
    float loss = 0.0f;
    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t count = 0;
        for (std::size_t t = r * len; t < (r + 1) * len; ++t)
            count += mask[t] != 0 ? 1 : 0;
        const float w = count == 0 ? 0.0f : static_cast<float>(1.0 / (static_cast<double>(rows) * static_cast<double>(count)));
        for (std::size_t t = r * len; t < (r + 1) * len; ++t) {
            if (mask[t] == 0) {
                dlogps[t] = 0.0f;
                dents[t] = 0.0f;
                continue;
            }
            const float ratio = std::exp(logps[t] - old[t]);
            const float cr = ratio < lo ? lo : (ratio > hi ? hi : ratio);
            const float un = ratio * adv[t], cl = cr * adv[t];
            const bool clipped = cl < un;
            float l = clipped ? -cl : -un;
            float d = clipped ? 0.0f : -un;
            const float u = ref[t] - logps[t];
            const float kl = std::expm1(u) - u;
            l += beta * kl;
            d += beta * -std::expm1(u);
            ++tokens_seen;
            clipped_seen += clipped ? 1 : 0;
            kl_sum += kl;
            ratio_sum += ratio;
            l -= coef * ent[t];
            entropy_sum += ent[t];
            dents[t] = -coef * w;
            loss += w * l;
            dlogps[t] = w * d;
        }
    }
    return loss;
}

float core(float* dlogps, float* dents)
{
    ml::policy_loss_metrics<float> m;
    return ml::policy_loss_rows<float>(snicholls::linalg::reference, mask, logps, old, ref, adv, ent, {dlogps, n}, {dents, n}, rows, config, m);
}

void BM_PolicyLoss_Core(benchmark::State& state)
{
    std::vector<float> d(n), e(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(core(d.data(), e.data()));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_PolicyLoss_Core);

void BM_PolicyLoss_HandWritten(benchmark::State& state)
{
    std::vector<float> d(n), e(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(by_hand(d.data(), e.data()));
        benchmark::DoNotOptimize(kl_sum);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_PolicyLoss_HandWritten);

bool fair()
{
    std::vector<float> d1(n), e1(n), d2(n), e2(n);
    const float a = core(d1.data(), e1.data()), b = by_hand(d2.data(), e2.data());
    if (std::bit_cast<std::uint32_t>(a) != std::bit_cast<std::uint32_t>(b))
        return false;
    for (std::size_t i = 0; i < n; ++i)
        if (std::bit_cast<std::uint32_t>(d1[i]) != std::bit_cast<std::uint32_t>(d2[i]) || std::bit_cast<std::uint32_t>(e1[i]) != std::bit_cast<std::uint32_t>(e2[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "rl benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
