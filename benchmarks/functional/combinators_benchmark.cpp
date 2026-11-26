//
//  combinators_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Benchmark for <snicholls/core/functional/combinators.hpp> and <snicholls/core/functional/curry.hpp>: a
//  price-adjustment pipeline built with fn::pipe from curried pieces, against the hand-written lambda.
//
//  Like for like: the same 4096 prices in the same vector, the same three functions (convert, widen, clamp)
//  with the same constants, applied in the same order, and the same single `sum +=` per price.
//

#include <cstdint>
#include <cstdio>
#include <vector>

#include <benchmark/benchmark.h>

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/curry.hpp>

namespace {

namespace fn = snicholls::fn;

constexpr double convert(double rate, double price) noexcept { return price * rate; }
constexpr double widen(double spread_bp, double price) noexcept { return price * (1.0 + spread_bp / 10'000.0); }
constexpr double clamp(double lo, double hi, double price) noexcept { return price < lo ? lo : (hi < price ? hi : price); }

constexpr std::size_t count = 4096;

std::vector<double> make_prices()
{
    std::vector<double> ps;
    for (std::size_t i = 0; i < count; ++i) ps.push_back(0.5 + static_cast<double>(i % 997) * 0.25);
    return ps;
}

double run_composed(const std::vector<double>& ps, double rate, double sum)
{
    const auto quote = fn::pipe(fn::curry(convert)(rate), fn::curry(widen)(25.0), fn::curry(clamp)(1.0, 200.0));
    for (const double p : ps) sum += quote(p);
    return sum;
}

double run_hand_written(const std::vector<double>& ps, double rate, double sum)
{
    const auto quote = [rate](double p) { return clamp(1.0, 200.0, widen(25.0, convert(rate, p))); };
    for (const double p : ps) sum += quote(p);
    return sum;
}

void functional_composed(benchmark::State& state)
{
    const std::vector<double> ps = make_prices();
    double rate = 1.085;
    double sum = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(rate);
        sum = run_composed(ps, rate, sum);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(count));
}

void functional_hand_written(benchmark::State& state)
{
    const std::vector<double> ps = make_prices();
    double rate = 1.085;
    double sum = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(rate);
        sum = run_hand_written(ps, rate, sum);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(count));
}

BENCHMARK(functional_composed)->Name("functional/pipe_of_curried");
BENCHMARK(functional_hand_written)->Name("functional/hand_written");

// The two forms must do the same work: identical sums, bit for bit.
bool fairness_check()
{
    const std::vector<double> ps = make_prices();
    const double a = run_composed(ps, 1.085, 0.0);
    const double b = run_hand_written(ps, 1.085, 0.0);
    if (a != b) {
        std::fprintf(stderr, "fairness check failed: composed %.17g, hand-written %.17g\n", a, b);
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
