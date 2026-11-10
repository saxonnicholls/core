//
//  stages_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Benchmark for <snicholls/core/pipeline/stages.hpp>: a static chain (filter | transform | sink) against the
//  hand-written loop it replaces.
//
//  Like for like: the same 1024 quotes in the same vector, the same predicate,
//  the same subtraction, and the same single `sum +=` per kept quote, in the
//  same order.
//

#include <cstdint>
#include <cstdio>
#include <vector>

#include <benchmark/benchmark.h>

#include <snicholls/core/pipeline/stages.hpp>

namespace {

struct quote
{
    std::int64_t ts;
    double bid;
    double ask;
};

std::vector<quote> make_quotes()
{
    std::vector<quote> qs;
    for (int i = 0; i < 1024; ++i)
        qs.push_back({i, 1.0 + i * 0.001, 1.0002 + i * 0.001 + (i % 5) * 0.0001});
    return qs;
}

double run_chain(const std::vector<quote>& qs, double sum)
{
    using namespace snicholls::pipeline;
    auto chain = filter([](const quote& q) { return q.ts % 3 != 0; }) |
                 transform([](const quote& q) { return q.ask - q.bid; }) | [&sum](double spread) { sum += spread; };
    for (const quote& q : qs) chain(q);
    return sum;
}

double run_hand_written(const std::vector<quote>& qs, double sum)
{
    for (const quote& q : qs)
        if (q.ts % 3 != 0) sum += q.ask - q.bid;
    return sum;
}

void stage_chain(benchmark::State& state)
{
    const std::vector<quote> qs = make_quotes();
    double sum = 0;
    for (auto _ : state) {
        sum = run_chain(qs, sum);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * 1024);
}

void stage_hand_written(benchmark::State& state)
{
    const std::vector<quote> qs = make_quotes();
    double sum = 0;
    for (auto _ : state) {
        sum = run_hand_written(qs, sum);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * 1024);
}

BENCHMARK(stage_chain)->Name("stage/chain");
BENCHMARK(stage_hand_written)->Name("stage/hand_written");

// The two forms must do the same work: identical sums, bit for bit.
bool fairness_check()
{
    const std::vector<quote> qs = make_quotes();
    const double a = run_chain(qs, 0.0);
    const double b = run_hand_written(qs, 0.0);
    if (a != b) {
        std::fprintf(stderr, "fairness check failed: chain %.17g, hand-written %.17g\n", a, b);
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
