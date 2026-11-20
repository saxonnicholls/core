//
//  conv_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::conv_forward (the direct loop) and conv_forward_im2col (im2col and the matrix product) against the
//  7-deep loop written out by hand (output channel, row, column, input channel, kernel row, kernel column; the taps in
//  the padding skipped, the bias last), over the same 3x3 convolution with padding 1 of 64 channels of 32x32 floats
//  into 64: the same sums in the same order. main() runs a fairness check first and refuses to benchmark if any pair
//  disagrees in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/conv.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

constexpr std::size_t channels = 64, side = 32, k = 3;
constexpr snicholls::ml::conv_shape<2> shape{.in_channels = channels, .out_channels = channels, .input = {side, side}, .kernel = {k, k},
                                             .padding = {1, 1}};

std::vector<float> make(std::size_t n, std::uint64_t seed)
{
    std::vector<float> out(n);
    std::uint64_t x = seed;   // a fixed xorshift stream: identical data for every form
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return out;
}

const std::vector<float> input = make(shape.x_size(), 0x243f6a8885a308d3ULL);
const std::vector<float> weight = make(shape.weight_size(), 0x13198a2e03707344ULL);
const std::vector<float> bias = make(channels, 0xa4093822299f31d0ULL);

void by_hand(std::vector<float>& y, std::vector<float>&)
{
    const float* x = input.data();
    const float* w = weight.data();
    for (std::size_t oc = 0; oc < channels; ++oc)
        for (std::size_t oy = 0; oy < side; ++oy)
            for (std::size_t ox = 0; ox < side; ++ox) {
                float sum = 0.0f;
                for (std::size_t ic = 0; ic < channels; ++ic)
                    for (std::size_t ky = 0; ky < k; ++ky) {
                        if (oy + ky < 1 || oy + ky - 1 >= side)
                            continue;
                        for (std::size_t kx = 0; kx < k; ++kx) {
                            if (ox + kx < 1 || ox + kx - 1 >= side)
                                continue;
                            const float term = w[((oc * channels + ic) * k + ky) * k + kx] * x[(ic * side + oy + ky - 1) * side + ox + kx - 1];
                            sum += term;
                        }
                    }
                y[(oc * side + oy) * side + ox] = sum + bias[oc];
            }
}

void core_direct(std::vector<float>& y, std::vector<float>&)
{
    snicholls::ml::conv_forward(shape, input, weight, bias, y);
}

void core_im2col(std::vector<float>& y, std::vector<float>& cols)
{
    snicholls::ml::conv_forward_im2col(shape, input, weight, bias, y, cols);
}

template <void (*F)(std::vector<float>&, std::vector<float>&)>
void run(benchmark::State& state)
{
    std::vector<float> out(shape.y_size()), cols(shape.column_size());
    for (auto _ : state) {
        F(out, cols);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(shape.y_size() * channels * k * k));   // multiply-adds
}

BENCHMARK(run<core_direct>)->Name("BM_Conv2d_Core_Direct");
BENCHMARK(run<core_im2col>)->Name("BM_Conv2d_Core_Im2col");
BENCHMARK(run<by_hand>)->Name("BM_Conv2d_HandWritten");

template <void (*A)(std::vector<float>&, std::vector<float>&), void (*B)(std::vector<float>&, std::vector<float>&)>
bool same()
{
    std::vector<float> a(shape.y_size()), b(shape.y_size()), cols(shape.column_size());
    A(a, cols);
    B(b, cols);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!same<core_direct, by_hand>() || !same<core_im2col, by_hand>()) {
        std::fprintf(stderr, "conv benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
