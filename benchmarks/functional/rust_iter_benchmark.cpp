//
//  rust_iter_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Benchmark for <snicholls/core/functional/rust_iter.hpp>: a moving sum over fn::rust::slice::windows_of<8>, against
//  the hand-written index loop.
//
//  Like for like: the same 4096 prices in the same vector, the same window of 8, the eight elements of each window
//  added in the same order from 0.0, and the same single `sum +=` per window.
//

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include <snicholls/core/functional/rust_iter.hpp>

namespace {

namespace slice = snicholls::fn::rust::slice;

constexpr std::size_t count = 4096;
constexpr std::size_t width = 8;

std::vector<double> make_prices()
{
    std::vector<double> ps;
    for (std::size_t i = 0; i < count; ++i) ps.push_back(0.5 + static_cast<double>(i % 997) * 0.25);
    return ps;
}

double run_windows(const std::vector<double>& ps, double sum)
{
    for (std::span<const double, width> w : slice::windows_of<width>(ps)) {
        double s = 0.0;
        for (std::size_t j = 0; j < width; ++j) s += w[j];
        sum += s;
    }
    return sum;
}

double run_hand_written(const std::vector<double>& ps, double sum)
{
    for (std::size_t i = 0; i + width <= ps.size(); ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < width; ++j) s += ps[i + j];
        sum += s;
    }
    return sum;
}

void functional_windows(benchmark::State& state)
{
    std::vector<double> ps = make_prices();
    double sum = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(ps.data());
        sum = run_windows(ps, sum);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(count - width + 1));
}

void functional_hand_written(benchmark::State& state)
{
    std::vector<double> ps = make_prices();
    double sum = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(ps.data());
        sum = run_hand_written(ps, sum);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(count - width + 1));
}

BENCHMARK(functional_windows)->Name("functional/rust_windows_of_8");
BENCHMARK(functional_hand_written)->Name("functional/hand_written_index_loop");

// The two forms must do the same work: identical sums, bit for bit.
bool fairness_check()
{
    const std::vector<double> ps = make_prices();
    const double a = run_windows(ps, 0.0);
    const double b = run_hand_written(ps, 0.0);
    if (a != b) {
        std::fprintf(stderr, "fairness check failed: windows %.17g, hand-written %.17g\n", a, b);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fairness_check()) return 1;
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
