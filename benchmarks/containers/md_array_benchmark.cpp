//
//  md_array_benchmark.cpp
//  snicholls-core
//
//  Summing a 16 x 16 x 16 x 16 grid in row-major order: snicholls::containers::md_array's a(i, j, k, l)
//  against the same offsets written by hand. Then a 256 x 256 matrix through md_span in each layout
//  (row-major, column-major, strided with run-time strides) against the same offsets by hand, each walked
//  with its contiguous index innermost. The fairness check compares every pair's sums.
//

#include <snicholls/core/containers/md_array.hpp>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using grid = snicholls::containers::md_array<double, 16, 16, 16, 16>;

grid data = [] {   // not const: the optimiser must not fold the sums
    grid g{};
    for (std::size_t i = 0; i < grid::size; ++i)
        g.elements[i] = static_cast<double>(i % 97) * 0.5;
    return g;
}();

double through_md_array()
{
    double total = 0;
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j)
            for (std::size_t k = 0; k < 16; ++k)
                for (std::size_t l = 0; l < 16; ++l)
                    total += data(i, j, k, l);
    return total;
}

double by_hand()
{
    double total = 0;
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j)
            for (std::size_t k = 0; k < 16; ++k)
                for (std::size_t l = 0; l < 16; ++l)
                    total += data.elements[((i * 16 + j) * 16 + k) * 16 + l];
    return total;
}

void BM_MdArray_Core(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(data.elements.data());
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(through_md_array());
    }
}
void BM_MdArray_OffsetsByHand(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(data.elements.data());
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(by_hand());
    }
}
BENCHMARK(BM_MdArray_Core);
BENCHMARK(BM_MdArray_OffsetsByHand);

namespace sc = snicholls::containers;
constexpr std::size_t n = 256;
std::vector<double> matrix = [] {   // not const: the optimiser must not fold the sums
    std::vector<double> m(n * n);
    for (std::size_t i = 0; i < m.size(); ++i)
        m[i] = static_cast<double>(i % 89) * 0.25;
    return m;
}();
std::size_t row_stride = n;   // a run-time stride, as a strided view has

double span_right()
{
    const sc::md_span<const double, 2> a(matrix.data(), {n, n});
    double total = 0;
    for (std::size_t i = 0; i < a.extent(0); ++i)
        for (std::size_t j = 0; j < a.extent(1); ++j)
            total += a(i, j);
    return total;
}
double hand_right()
{
    const double* a = matrix.data();
    double total = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            total += a[i * n + j];
    return total;
}
double span_left()
{
    const sc::md_span<const double, 2, sc::layout_left> a(matrix.data(), {n, n});
    double total = 0;
    for (std::size_t j = 0; j < a.extent(1); ++j)
        for (std::size_t i = 0; i < a.extent(0); ++i)
            total += a(i, j);
    return total;
}
double hand_left()
{
    const double* a = matrix.data();
    double total = 0;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            total += a[i + j * n];
    return total;
}
double span_stride()
{
    const sc::md_span<const double, 2, sc::layout_stride> a(matrix.data(), {n, n}, {row_stride, 1});
    double total = 0;
    for (std::size_t i = 0; i < a.extent(0); ++i)
        for (std::size_t j = 0; j < a.extent(1); ++j)
            total += a(i, j);
    return total;
}
double hand_stride()
{
    const double* a = matrix.data();
    const std::size_t s0 = row_stride, s1 = 1;
    double total = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            total += a[i * s0 + j * s1];
    return total;
}

template <double (*F)()>
void run(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(matrix.data());
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(F());
    }
    s.SetItemsProcessed(static_cast<std::int64_t>(s.iterations() * n * n));
}
void BM_MdSpan_RowMajor(benchmark::State& s) { run<span_right>(s); }
void BM_MdSpan_RowMajorByHand(benchmark::State& s) { run<hand_right>(s); }
void BM_MdSpan_ColumnMajor(benchmark::State& s) { run<span_left>(s); }
void BM_MdSpan_ColumnMajorByHand(benchmark::State& s) { run<hand_left>(s); }
void BM_MdSpan_Strided(benchmark::State& s) { run<span_stride>(s); }
void BM_MdSpan_StridedByHand(benchmark::State& s) { run<hand_stride>(s); }
BENCHMARK(BM_MdSpan_RowMajor);
BENCHMARK(BM_MdSpan_RowMajorByHand);
BENCHMARK(BM_MdSpan_ColumnMajor);
BENCHMARK(BM_MdSpan_ColumnMajorByHand);
BENCHMARK(BM_MdSpan_Strided);
BENCHMARK(BM_MdSpan_StridedByHand);

} // namespace

int main(int argc, char** argv)
{
    if (through_md_array() != by_hand() || span_right() != hand_right() || span_left() != hand_left() ||
        span_stride() != hand_stride()) {
        std::fprintf(stderr, "md_array benchmark: fairness check FAILED\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
