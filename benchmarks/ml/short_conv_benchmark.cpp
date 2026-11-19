//
//  short_conv_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::short_conv_forward (Mamba's causal depthwise convolution) against the same loop written out by hand,
//  over 512 tokens of 2,560 channels (Mamba-2 2.7B's d_inner plus its B and C) with 4 taps and a bias: the same sums
//  from zero in tap order, then the bias. main() runs a fairness check first and refuses to benchmark if the pair
//  disagrees in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/short_conv.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr std::size_t n = 512, ch = 2560, K = 4;

std::vector<float> make(std::size_t count, double lo, double hi, std::uint64_t seed)
{
    std::vector<float> out(count);
    std::uint64_t x = seed;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(lo + (hi - lo) * static_cast<double>(x >> 11) * 0x1.0p-53);
    }
    return out;
}

const std::vector<float> xs = make(n * ch, -1, 1, 0x243f6a8885a308d3ULL), taps = make(ch * K, -1, 1, 0x13198a2e03707344ULL),
                         bias = make(ch, -0.1, 0.1, 0xa4093822299f31d0ULL);

void core(std::vector<float>& y)
{
    ml::short_conv_forward(snicholls::linalg::reference, ml::short_conv_weights<float>{taps, ch, K, bias}, std::span<const float>(xs),
                           std::span<float>(y));
}

void by_hand(std::vector<float>& y)
{
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < ch; ++c) {
            float sum = 0.0f;
            for (std::size_t j = 0; j < K; ++j) {
                const std::size_t back = K - 1 - j;
                sum += taps[c * K + j] * (back <= i ? xs[(i - back) * ch + c] : 0.0f);
            }
            y[i * ch + c] = sum + bias[c];
        }
}

template <void (*F)(std::vector<float>&)>
void run(benchmark::State& state)
{
    std::vector<float> y(n * ch);
    for (auto _ : state) {
        F(y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n * ch));
}

BENCHMARK(run<core>)->Name("BM_ShortConv_Core");
BENCHMARK(run<by_hand>)->Name("BM_ShortConv_HandWritten");

bool same()
{
    std::vector<float> a(n * ch), b(n * ch);
    core(a);
    by_hand(b);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!same()) {
        std::fprintf(stderr, "short_conv benchmark: fairness check FAILED - the pair disagrees; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
