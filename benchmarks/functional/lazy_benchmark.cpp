//
//  lazy_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Benchmark for <snicholls/core/functional/lazy.hpp> and <snicholls/core/functional/state.hpp>: an infinite stream
//  (iterate, map, filter, take) and a State-monad program, each against the hand-written loop.
//
//  Like for like: the same start value, the same recurrence, the same test and the same additions in the same
//  order; for State, the same fills against the same balance in the same order.
//

#include <array>
#include <cstdint>
#include <cstdio>

#include <benchmark/benchmark.h>

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/lazy.hpp>
#include <snicholls/core/functional/state.hpp>

namespace {

namespace fn = snicholls::fn;
namespace st = snicholls::fn::stream;

constexpr std::size_t count = 256;

// A linear-congruential walk, its values scaled, the ones in the upper half kept: the first 256 summed.
std::uint64_t run_stream(std::uint64_t seed)
{
    const auto walk = st::iterate([](std::uint64_t x) { return x * 6364136223846793005ULL + 1442695040888963407ULL; }, seed);
    const auto kept = st::filter([](std::uint64_t x) { return (x >> 63) != 0; }, st::map([](std::uint64_t x) { return x ^ (x >> 29); }, walk));
    std::uint64_t sum = 0;
    for (const std::uint64_t v : st::take<count>(kept)) sum += v;
    return sum;
}

std::uint64_t run_stream_hand_written(std::uint64_t seed)
{
    std::array<std::uint64_t, count> out{};
    std::uint64_t x = seed;
    for (std::size_t n = 0; n < count;) {
        const std::uint64_t v = x ^ (x >> 29);
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        if ((v >> 63) != 0) out[n++] = v;
    }
    std::uint64_t sum = 0;
    for (const std::uint64_t v : out) sum += v;
    return sum;
}

// A day of fills against a cash balance, as a State program and as a loop.
double run_state(double cash)
{
    const auto fill = [](double cost) {
        return fn::mbind(fn::get_state, [cost](double c) { return fn::mthen(fn::put_state(c - cost), fn::pure<fn::state<>>(c >= cost)); });
    };
    const auto day = fn::mbind(fill(40.0), [fill](bool a) {
        return fn::mbind(fill(25.0), [a, fill](bool b) {
            return fn::fmap([a, b](bool c) { return a && b && c; }, fill(10.0));
        });
    });
    const auto [covered, left] = fn::run_state(day, cash);
    return covered ? left : -left;
}

double run_state_hand_written(double cash)
{
    bool covered = true;
    for (const double cost : {40.0, 25.0, 10.0}) {
        covered = covered && cash >= cost;
        cash -= cost;
    }
    return covered ? cash : -cash;
}

void stream_composed(benchmark::State& state)
{
    std::uint64_t seed = 42;
    for (auto _ : state) {
        benchmark::DoNotOptimize(seed);
        std::uint64_t r = run_stream(seed);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(count));
}

void stream_hand_written(benchmark::State& state)
{
    std::uint64_t seed = 42;
    for (auto _ : state) {
        benchmark::DoNotOptimize(seed);
        std::uint64_t r = run_stream_hand_written(seed);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(count));
}

void state_composed(benchmark::State& state)
{
    double cash = 100.0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(cash);
        double r = run_state(cash);
        benchmark::DoNotOptimize(r);
    }
}

void state_hand_written(benchmark::State& state)
{
    double cash = 100.0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(cash);
        double r = run_state_hand_written(cash);
        benchmark::DoNotOptimize(r);
    }
}

BENCHMARK(stream_composed)->Name("functional/stream_take_256");
BENCHMARK(stream_hand_written)->Name("functional/stream_hand_written");
BENCHMARK(state_composed)->Name("functional/state_day_of_fills");
BENCHMARK(state_hand_written)->Name("functional/state_hand_written");

// Each pair must compute the same thing: identical results, bit for bit.
bool fairness_check()
{
    for (std::uint64_t seed : {1ULL, 42ULL, 0xDEADBEEFULL})
        if (run_stream(seed) != run_stream_hand_written(seed)) {
            std::fprintf(stderr, "fairness check failed: stream, seed %llu\n", static_cast<unsigned long long>(seed));
            return false;
        }
    for (double cash : {100.0, 70.0, 10.0})
        if (run_state(cash) != run_state_hand_written(cash)) {
            std::fprintf(stderr, "fairness check failed: state, cash %g\n", cash);
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
