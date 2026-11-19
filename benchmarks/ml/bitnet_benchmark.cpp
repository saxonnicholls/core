//
//  bitnet_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's BitLinear forward (per-token absmax int8 activations, a ternary x int8 sum per output, one
//  multiplication by the scales) against the same arithmetic written out by hand, on a 1,024 x 2,048 layer, and
//  absmean weight quantisation against its hand loop. main() runs a fairness check first and refuses to benchmark if
//  any result differs in a single bit.
//

#include <snicholls/core/ml/bitnet.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr std::size_t rows = 1024, cols = 2048;

std::vector<float> make(std::size_t count, std::uint64_t x)
{
    std::vector<float> out(count);
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return out;
}

struct layer {
    std::vector<float> w = make(rows * cols, 3), x = make(cols, 4);
    std::vector<std::int8_t> t = std::vector<std::int8_t>(rows * cols);
    float gamma = ml::absmean_quantize(w, t);
};

const layer& data()
{
    static const layer l;
    return l;
}

// By hand: quant_input, then the integer product and the epilogue acc / s · γ
void bitlinear_by_hand(const std::int8_t* t, float gamma, const float* x, float* y, std::int8_t* q)
{
    float amax = 0.0f;
    for (std::size_t j = 0; j < cols; ++j)
        amax = std::fmax(amax, std::fabs(x[j]));
    const float s = 127.0f / (amax > 1e-5f ? amax : 1e-5f);
    for (std::size_t j = 0; j < cols; ++j)
        q[j] = static_cast<std::int8_t>(std::fmin(127.0f, std::fmax(-128.0f, std::nearbyint(x[j] * s))));
    for (std::size_t i = 0; i < rows; ++i) {
        std::int32_t acc = 0;
        for (std::size_t j = 0; j < cols; ++j)
            acc += std::int32_t(t[i * cols + j]) * std::int32_t(q[j]);
        const float a = float(acc) / s;
        y[i] = a * gamma;
    }
}

float absmean_by_hand(const float* w, std::int8_t* q, std::size_t count)
{
    double sum = 0;
    for (std::size_t i = 0; i < count; ++i)
        sum += std::fabs(double(w[i]));
    const float mean = float(sum / double(count));
    const float gamma = mean > 1e-5f ? mean : 1e-5f;
    const float s = 1.0f / gamma;
    for (std::size_t i = 0; i < count; ++i)
        q[i] = static_cast<std::int8_t>(std::fmin(1.0f, std::fmax(-1.0f, std::nearbyint(w[i] * s))));
    return gamma;
}

void BM_BitLinear_Core(benchmark::State& state)
{
    const auto& d = data();
    const ml::bitlinear_weights w{d.t, rows, cols, d.gamma};
    std::vector<float> y(rows);
    std::vector<std::int8_t> q(cols);
    for (auto _ : state) {
        ml::bitlinear(w, d.x, y, q);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * cols));
}
BENCHMARK(BM_BitLinear_Core);

void BM_BitLinear_HandWritten(benchmark::State& state)
{
    const auto& d = data();
    std::vector<float> y(rows);
    std::vector<std::int8_t> q(cols);
    for (auto _ : state) {
        bitlinear_by_hand(d.t.data(), d.gamma, d.x.data(), y.data(), q.data());
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * cols));
}
BENCHMARK(BM_BitLinear_HandWritten);

void BM_AbsmeanQuantize_Core(benchmark::State& state)
{
    std::vector<std::int8_t> q(rows * cols);
    for (auto _ : state)
        benchmark::DoNotOptimize(ml::absmean_quantize(data().w, q));
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * cols));
}
BENCHMARK(BM_AbsmeanQuantize_Core);

void BM_AbsmeanQuantize_HandWritten(benchmark::State& state)
{
    std::vector<std::int8_t> q(rows * cols);
    for (auto _ : state)
        benchmark::DoNotOptimize(absmean_by_hand(data().w.data(), q.data(), rows * cols));
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * cols));
}
BENCHMARK(BM_AbsmeanQuantize_HandWritten);

bool fair()
{
    const auto& d = data();
    std::vector<float> y1(rows), y2(rows);
    std::vector<std::int8_t> q(cols), t(rows * cols);
    ml::bitlinear(ml::bitlinear_weights{d.t, rows, cols, d.gamma}, d.x, y1, q);
    bitlinear_by_hand(d.t.data(), d.gamma, d.x.data(), y2.data(), q.data());
    return std::memcmp(y1.data(), y2.data(), rows * sizeof(float)) == 0 && absmean_by_hand(d.w.data(), t.data(), rows * cols) == d.gamma
           && t == d.t;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "bitnet benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
