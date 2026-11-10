//
//  invoke_with_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::invoke_any against the same run-time dispatch written out by hand as a
//  chain of std::any_cast checks in the same order, and against the direct compile-time
//  call through the same overloaded, for scale. All three run the same 4,096 calls
//  (one of four shapes each, arguments prebuilt as std::any). main() runs a fairness
//  check first and refuses to benchmark if the three disagree on any call (CLAUDE.md
//  "Zero overhead").
//

#include <snicholls/core/invoke_with.hpp>
#include <snicholls/core/overloaded.hpp>

#include <benchmark/benchmark.h>

#include <any>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

// Four handlers, every result a long so that the three paths can be compared.
const auto handlers = snicholls::overloaded{
    [](int x) -> long { return 2L * x; },
    [](double d) -> long { return static_cast<long>(d * 3); },
    [](const std::string& s) -> long { return static_cast<long>(s.size()); },
    [](int a, int b) -> long { return a + b; },
};

struct call {
    int kind;          // 0: int, 1: double, 2: string, 3: int, int
    int i;
    double d;
    std::string s;
    int a;
    int b;
    std::vector<std::any> args;   // the same values, type-erased, built once
};

std::vector<call> make_calls()
{
    std::vector<call> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for all three
    for (auto& c : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        c.kind = static_cast<int>(x % 4);
        c.i = static_cast<int>(x % 1000);
        c.d = static_cast<double>(x % 1000) / 8.0;
        c.s = std::string(x % 16, 'x');
        c.a = static_cast<int>(x % 100);
        c.b = static_cast<int>((x >> 8) % 100);
        switch (c.kind) {
        case 0: c.args = {c.i}; break;
        case 1: c.args = {c.d}; break;
        case 2: c.args = {c.s}; break;
        default: c.args = {c.a, c.b}; break;
        }
    }
    return out;
}

const std::vector<call> calls = make_calls();

// The library's run-time path
long via_invoke_any(const call& c)
{
    return std::any_cast<long>(snicholls::invoke_any(handlers, c.args).unwrap());
}

// By hand: the same checks in the same order (arity, then each argument's dynamic type, per
// handler as written), the result through a std::any as the library returns it.
long via_any_cast_by_hand(const call& c)
{
    const std::span<const std::any> args = c.args;
    std::any out;
    if (args.size() == 1)
        if (const auto* p = std::any_cast<int>(&args[0])) {
            out = handlers(*p);
            return std::any_cast<long>(out);
        }
    if (args.size() == 1)
        if (const auto* p = std::any_cast<double>(&args[0])) {
            out = handlers(*p);
            return std::any_cast<long>(out);
        }
    if (args.size() == 1)
        if (const auto* p = std::any_cast<std::string>(&args[0])) {
            out = handlers(*p);
            return std::any_cast<long>(out);
        }
    if (args.size() == 2) {
        const auto* a = std::any_cast<int>(&args[0]);
        const auto* b = a ? std::any_cast<int>(&args[1]) : nullptr;
        if (a && b) {
            out = handlers(*a, *b);
            return std::any_cast<long>(out);
        }
    }
    return -1;
}

// The compile-time path, for scale: the static types are known, so the call is direct.
long via_direct_call(const call& c)
{
    switch (c.kind) {
    case 0: return handlers(c.i);
    case 1: return handlers(c.d);
    case 2: return handlers(c.s);
    default: return handlers(c.a, c.b);
    }
}

void BM_InvokeWith_Core(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& c : calls)
            benchmark::DoNotOptimize(via_invoke_any(c));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(calls.size()));
}
BENCHMARK(BM_InvokeWith_Core);

void BM_InvokeWith_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& c : calls)
            benchmark::DoNotOptimize(via_any_cast_by_hand(c));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(calls.size()));
}
BENCHMARK(BM_InvokeWith_HandWritten);

void BM_InvokeWith_DirectCall(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& c : calls)
            benchmark::DoNotOptimize(via_direct_call(c));
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(calls.size()));
}
BENCHMARK(BM_InvokeWith_DirectCall);

bool fair()
{
    for (const auto& c : calls) {
        const long expected = via_direct_call(c);
        if (via_invoke_any(c) != expected || via_any_cast_by_hand(c) != expected)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "invoke_with benchmark: fairness check FAILED - the paths disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
