//
//  bitnet_kernels_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's bitnet.cpp reference kernels against hand-written loops on a 1,024 x 1,536 ternary matrix: the
//  I2_S dot product (ggml_vec_dot_i2_i8_s_1x1's sum) row by row against the plain int8 ternary product, and the TL1
//  and TL2 lookup-table GEMVs against the same plain product (the comparison the LUT method is about). main() runs
//  a fairness check first and refuses to benchmark if any result differs.
//

#include <snicholls/core/ml/bitnet_kernels.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr std::size_t rows = 1024, cols = 1536;

struct matrix {
    std::vector<std::int8_t> t = std::vector<std::int8_t>(rows * cols), a = std::vector<std::int8_t>(cols);
    std::vector<std::uint8_t> i2 = std::vector<std::uint8_t>(ml::i2_s_bytes(rows * cols));
    std::vector<std::uint8_t> c1 = std::vector<std::uint8_t>(rows * cols / 2);
    std::vector<ml::tl2_code_t> c2 = std::vector<ml::tl2_code_t>(rows * cols / 3);
    matrix()
    {
        std::uint64_t s = 5;
        for (auto& v : t) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            v = static_cast<std::int8_t>(int((s >> 33) % 3) - 1);
        }
        for (auto& v : a) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            v = static_cast<std::int8_t>(s >> 56);
        }
        ml::pack_i2_s(t, 1.0f, i2);
        ml::tl1_encode(t, c1);
        ml::tl2_encode(t, c2);
    }
};

const matrix& data()
{
    static const matrix m;
    return m;
}

// By hand: the plain ternary x int8 product
void ternary_by_hand(const std::int8_t* t, const std::int8_t* a, std::int32_t* y)
{
    for (std::size_t i = 0; i < rows; ++i) {
        std::int32_t s = 0;
        for (std::size_t j = 0; j < cols; ++j)
            s += std::int32_t(t[i * cols + j]) * std::int32_t(a[j]);
        y[i] = s;
    }
}

void i2_s_rows(const matrix& m, std::int32_t* y)
{
    const std::span<const std::uint8_t> p(m.i2);
    for (std::size_t i = 0; i < rows; ++i)
        y[i] = ml::i2_s_dot(p.subspan(i * cols / 4, cols / 4), m.a);
}

void BM_I2sDot_Core(benchmark::State& state)
{
    std::vector<std::int32_t> y(rows);
    for (auto _ : state) {
        i2_s_rows(data(), y.data());
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * cols));
}
BENCHMARK(BM_I2sDot_Core);

void BM_Tl1Gemv_Core(benchmark::State& state)
{
    std::vector<std::int32_t> y(rows);
    std::vector<std::array<std::int16_t, 16>> lut(cols / 2);
    for (auto _ : state) {
        ml::tl1_gemv(data().c1, rows, cols, data().a, lut, y);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * cols));
}
BENCHMARK(BM_Tl1Gemv_Core);

void BM_Tl2Gemv_Core(benchmark::State& state)
{
    std::vector<std::int32_t> y(rows);
    std::vector<std::array<std::int16_t, 16>> lut(cols / 3);
    for (auto _ : state) {
        ml::tl2_gemv(data().c2, rows, cols, data().a, lut, y);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * cols));
}
BENCHMARK(BM_Tl2Gemv_Core);

void BM_TernaryGemv_HandWritten(benchmark::State& state)
{
    std::vector<std::int32_t> y(rows);
    for (auto _ : state) {
        ternary_by_hand(data().t.data(), data().a.data(), y.data());
        benchmark::DoNotOptimize(y.data());
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(rows * cols));
}
BENCHMARK(BM_TernaryGemv_HandWritten);

bool fair()
{
    const auto& m = data();
    std::vector<std::int32_t> hand(rows), i2(rows), y1(rows), y2(rows);
    std::vector<std::array<std::int16_t, 16>> l1(cols / 2), l2(cols / 3);
    ternary_by_hand(m.t.data(), m.a.data(), hand.data());
    i2_s_rows(m, i2.data());
    ml::tl1_gemv(m.c1, rows, cols, m.a, l1, y1);
    ml::tl2_gemv(m.c2, rows, cols, m.a, l2, y2);
    return hand == i2 && hand == y1 && hand == y2;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "bitnet_kernels benchmark: fairness check FAILED - the results disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
