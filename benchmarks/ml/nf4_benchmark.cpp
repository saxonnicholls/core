//
//  nf4_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::linear through NF4 weights (dequantised on the fly) against the same loop written by hand over the
//  same bytes, and against the dense product of the dequantised matrix: a 1024 x 1024 matrix, one input row. main()
//  runs a fairness check first and refuses to benchmark if any output differs in a single bit (CLAUDE.md "Zero
//  overhead").
//

#include <snicholls/core/ml/nf4.hpp>

#include <snicholls/core/ml/linear.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr std::size_t rows = 1024, cols = 1024, n = rows * cols;

std::vector<float> filled(std::uint64_t seed, std::size_t count)
{
    std::vector<float> v(count);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL ^ seed;   // a fixed xorshift stream: identical data for both
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        e = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

struct state_t {
    std::vector<float> w = filled(1, n), x = filled(2, cols), deq = std::vector<float>(n);
    std::vector<std::uint8_t> codes = std::vector<std::uint8_t>(ml::nf4_code_bytes(n));
    std::vector<ml::float8_e4m3> scales = std::vector<ml::float8_e4m3>(ml::nf4_block_count(n));
    std::vector<float> groups = std::vector<float>(ml::nf4_group_count(n));
    ml::nf4_weights q{};
    state_t()
    {
        const float mean = ml::nf4_quantize(w, codes, scales, groups);
        q = ml::nf4_weights{codes, scales, groups, mean, rows, cols};
        ml::nf4_dequantize(q, deq);
    }
};

const state_t& data()
{
    static const state_t s;
    return s;
}

void by_hand(const state_t& s, float* y)
{
    const auto& table = ml::decode_table<ml::float8_e4m3>;
    for (std::size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        for (std::size_t j0 = 0; j0 < cols; j0 += 64) {
            const std::size_t base = i * cols + j0, block = base / 64;
            const float scale = table[s.scales[block].bits] * s.groups[block / 256] + s.q.scale_mean;
            for (std::size_t j = 0; j < 64; ++j) {
                const std::size_t idx = base + j;
                const std::uint8_t byte = s.codes[idx / 2];
                const std::uint8_t code = idx % 2 == 0 ? byte >> 4 : byte & 0x0f;
                const float p = (ml::nf4_values[code] * scale) * s.x[j0 + j];
                acc += p;
            }
        }
        y[i] = acc;
    }
}

void BM_Nf4Linear_Core(benchmark::State& state)
{
    const auto& s = data();
    std::vector<float> y(rows);
    for (auto _ : state) {
        ml::linear(s.q, s.x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_Nf4Linear_Core);

void BM_Nf4Linear_HandWritten(benchmark::State& state)
{
    const auto& s = data();
    std::vector<float> y(rows);
    for (auto _ : state) {
        by_hand(s, y.data());
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_Nf4Linear_HandWritten);

void BM_DequantisedDenseLinear(benchmark::State& state)
{
    const auto& s = data();
    std::vector<float> y(rows);
    const ml::dense_weights<float> dense{s.deq, rows, cols};
    for (auto _ : state) {
        ml::linear(dense, s.x, y);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}
BENCHMARK(BM_DequantisedDenseLinear);

bool fair()
{
    const auto& s = data();
    std::vector<float> a(rows), b(rows), c(rows);
    ml::linear(s.q, s.x, a);
    by_hand(s, b.data());
    ml::linear(ml::dense_weights<float>{s.deq, rows, cols}, s.x, c);
    for (std::size_t i = 0; i < rows; ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]) || std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(c[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "nf4 benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
