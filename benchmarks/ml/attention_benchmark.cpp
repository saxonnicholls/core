//
//  attention_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::attend against the same one-pass recurrence written by hand (the same operations in the same
//  order, std::exp), one query of 64 channels over caches of 256 and 4,096 float keys and values. main() runs a
//  fairness check first and refuses to benchmark if any output differs in a single bit (CLAUDE.md "Zero
//  overhead").
//

#include <snicholls/core/ml/attention.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t d = 64;

std::vector<float> make(std::size_t n, std::uint64_t seed)
{
    std::vector<float> out(n);
    std::uint64_t x = seed;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return out;
}

struct cache {
    std::vector<float> q, k, v;
    explicit cache(std::size_t n) : q(make(d, 1)), k(make(n * d, 2)), v(make(n * d, 3)) {}
};

const float scale = 1.0f / std::sqrt(static_cast<float>(d));

// By hand: flash attention's recurrence for one query, with the head's width known at run time as attend has it
float attend_by_hand(const float* q, const float* k, const float* v, std::size_t n, std::size_t d, float* out)
{
    for (std::size_t i = 0; i < d; ++i)
        out[i] = 0.0f;
    float max = -std::numeric_limits<float>::infinity(), sum = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        float s = 0.0f;
        for (std::size_t i = 0; i < d; ++i)
            s += q[i] * k[j * d + i];
        s = scale * s;
        if (s == -std::numeric_limits<float>::infinity())
            continue;
        if (s > max) {
            const float c = std::exp(max - s);
            for (std::size_t i = 0; i < d; ++i)
                out[i] = out[i] * c + v[j * d + i];
            sum = sum * c + 1.0f;
            max = s;
        } else {
            const float w = std::exp(s - max);
            for (std::size_t i = 0; i < d; ++i)
                out[i] += w * v[j * d + i];
            sum += w;
        }
    }
    for (std::size_t i = 0; i < d; ++i)
        out[i] = sum == 0.0f ? 0.0f : out[i] / sum;
    return sum == 0.0f ? -std::numeric_limits<float>::infinity() : max + std::log(sum);
}

void BM_Attend_Core(benchmark::State& state)
{
    const cache c(static_cast<std::size_t>(state.range(0)));
    std::vector<float> out(d);
    for (auto _ : state) {
        benchmark::DoNotOptimize(snicholls::ml::attend(c.q, c.k, c.v, out, scale));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_Attend_Core)->Arg(256)->Arg(4096);

void BM_Attend_HandWritten(benchmark::State& state)
{
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    const cache c(n);
    std::vector<float> out(d);
    std::size_t width = d;
    benchmark::DoNotOptimize(width);   // as opaque to the compiler as attend's q.size()
    for (auto _ : state) {
        benchmark::DoNotOptimize(attend_by_hand(c.q.data(), c.k.data(), c.v.data(), n, width, out.data()));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_Attend_HandWritten)->Arg(256)->Arg(4096);

bool fair()
{
    for (const std::size_t n : {std::size_t{256}, std::size_t{4096}}) {
        const cache c(n);
        std::vector<float> a(d), b(d);
        const float la = snicholls::ml::attend(c.q, c.k, c.v, a, scale);
        const float lb = attend_by_hand(c.q.data(), c.k.data(), c.v.data(), n, d, b.data());
        if (std::bit_cast<std::uint32_t>(la) != std::bit_cast<std::uint32_t>(lb))
            return false;
        for (std::size_t i = 0; i < d; ++i)
            if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
                return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "attention benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
