//
//  softmax_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::softmax and cross_entropy_with_gradient against the same loops written out by hand, over
//  the same row of 50,304 float logits (GPT-2's padded vocabulary): the maximum first, the exponentials and
//  their sum in the same order, then the same scaling. main() runs a fairness check first and refuses to
//  benchmark if any pair disagrees in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/softmax.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

std::vector<float> make_logits()
{
    std::vector<float> out(50304);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 20.0 - 10.0);
    }
    return out;
}

const std::vector<float> logits = make_logits();
constexpr std::size_t target = 1234;

float max_by_hand(const std::vector<float>& in)
{
    float m = -std::numeric_limits<float>::infinity();
    for (const float x : in) {
        if (x != x)
            return x;
        if (x > m)
            m = x;
    }
    return m;
}

void softmax_by_hand(const std::vector<float>& in, std::vector<float>& out)
{
    const float m = max_by_hand(in);
    float s = 0.0f;
    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = std::exp(in[i] - m);
        s += out[i];
    }
    const float inv = 1.0f / s;
    for (float& p : out)
        p *= inv;
}

float cross_entropy_by_hand(const std::vector<float>& in, std::size_t t, std::vector<float>& g, float scale)
{
    const float m = max_by_hand(in);
    float s = 0.0f;
    for (const float x : in)
        s += std::exp(x - m);
    const float lse = m + std::log(s);
    for (std::size_t i = 0; i < in.size(); ++i)
        g[i] = scale * (std::exp(in[i] - lse) - (i == t ? 1.0f : 0.0f));
    return lse - in[t];
}

void BM_Softmax_Core(benchmark::State& state)
{
    std::vector<float> out(logits.size());
    for (auto _ : state) {
        snicholls::ml::softmax(logits, out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(logits.size()));
}
BENCHMARK(BM_Softmax_Core);

void BM_Softmax_HandWritten(benchmark::State& state)
{
    std::vector<float> out(logits.size());
    for (auto _ : state) {
        softmax_by_hand(logits, out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(logits.size()));
}
BENCHMARK(BM_Softmax_HandWritten);

void BM_CrossEntropyWithGradient_Core(benchmark::State& state)
{
    std::vector<float> g(logits.size());
    for (auto _ : state) {
        benchmark::DoNotOptimize(snicholls::ml::cross_entropy_with_gradient(logits, target, g, 0.5f));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(logits.size()));
}
BENCHMARK(BM_CrossEntropyWithGradient_Core);

void BM_CrossEntropyWithGradient_HandWritten(benchmark::State& state)
{
    std::vector<float> g(logits.size());
    for (auto _ : state) {
        benchmark::DoNotOptimize(cross_entropy_by_hand(logits, target, g, 0.5f));
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(logits.size()));
}
BENCHMARK(BM_CrossEntropyWithGradient_HandWritten);

bool fair()
{
    std::vector<float> a(logits.size()), b(logits.size());
    snicholls::ml::softmax(logits, a);
    softmax_by_hand(logits, b);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    const float la = snicholls::ml::cross_entropy_with_gradient(logits, target, a, 0.5f);
    const float lb = cross_entropy_by_hand(logits, target, b, 0.5f);
    if (std::bit_cast<std::uint32_t>(la) != std::bit_cast<std::uint32_t>(lb))
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "softmax benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
