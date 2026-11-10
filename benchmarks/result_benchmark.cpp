//
//  result_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Benchmark for <snicholls/core/result.hpp>.
//

// Like-for-like benchmark: snicholls::result<int, E> versus what a careful engineer writes by hand to return "a value or
// an error" from a function that is not inlined (RESULT_PLAN.md, Performance):
//
//   result_core      snicholls::result<int, parse_error>, returned by value
//   result_traced    snicholls::result<int, snicholls::traced<parse_error>> (success path only: an error allocates by design)
//   hand_tagged      struct { union { int value; parse_error error; }; bool ok; }, returned by value
//   error_code       parse_error f(int in, int& out), the C style out-parameter
//   exceptions       int f(int in) that throws on error, caught in the loop
//   *_propagate      the same through one extra layer that forwards the error (SNICHOLLS_TRY versus `if (!r.ok) return r`)
//
// Fairness: identical inputs (the same shuffled vector, generated from one seed per error rate), identical work per
// element (sum the values, count the errors), every callee [[gnu::noinline]] so the value really crosses a call
// boundary. main() checks that all variants agree on (sum, errors) for every error rate before benchmarking, and
// refuses to run otherwise. The argument of each benchmark is the error rate in percent. Only Release numbers mean
// anything; the CTest smoke run (benchmark.result_smoke) only proves it builds and agrees.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <snicholls/core/result.hpp>
#include <snicholls/core/traced.hpp>

#if defined(__GNUC__) || defined(__clang__)
#define SNICHOLLS_BENCH_NOINLINE __attribute__((noinline))
#else
#define SNICHOLLS_BENCH_NOINLINE
#endif

namespace {

enum class parse_error : std::uint8_t
{
    none = 0,
    negative = 1,
};

struct hand_tagged
{
    union
    {
        int value;
        parse_error error;
    };
    bool ok;
};

struct parse_exception : std::exception
{
    parse_error error;
    explicit parse_exception(parse_error e) noexcept : error(e) {}
    const char* what() const noexcept override { return "negative"; }
};

// ---- the callees: identical arithmetic, four ways of reporting the error ---------------------------------------

SNICHOLLS_BENCH_NOINLINE snicholls::result<int, parse_error> parse_core(int x) noexcept
{
    if (x < 0) [[unlikely]]
        return snicholls::err(parse_error::negative);
    return snicholls::ok(x * 3 + 1);
}

SNICHOLLS_BENCH_NOINLINE snicholls::result<int, snicholls::traced<parse_error>> parse_traced(int x)
{
    if (x < 0) [[unlikely]]
        return snicholls::traced_err(parse_error::negative);
    return snicholls::ok(x * 3 + 1);
}

SNICHOLLS_BENCH_NOINLINE hand_tagged parse_hand(int x) noexcept
{
    hand_tagged r;
    if (x < 0) [[unlikely]]
    {
        r.error = parse_error::negative;
        r.ok = false;
        return r;
    }
    r.value = x * 3 + 1;
    r.ok = true;
    return r;
}

SNICHOLLS_BENCH_NOINLINE parse_error parse_code(int x, int& out) noexcept
{
    if (x < 0) [[unlikely]]
        return parse_error::negative;
    out = x * 3 + 1;
    return parse_error::none;
}

SNICHOLLS_BENCH_NOINLINE int parse_throw(int x)
{
    if (x < 0) [[unlikely]]
        throw parse_exception(parse_error::negative);
    return x * 3 + 1;
}

// One extra layer that only forwards the error.
SNICHOLLS_BENCH_NOINLINE snicholls::result<int, parse_error> forward_core(int x) noexcept
{
    SNICHOLLS_TRY_ASSIGN(const int v, parse_core(x));
    return snicholls::ok(v + 1);
}

SNICHOLLS_BENCH_NOINLINE hand_tagged forward_hand(int x) noexcept
{
    hand_tagged r = parse_hand(x);
    if (!r.ok) [[unlikely]]
        return r;
    r.value += 1;
    return r;
}

// ---- inputs ----------------------------------------------------------------------------------------------------------

constexpr std::size_t input_count = 4096;

const std::vector<int>& inputs(int error_percent)
{
    static std::vector<std::vector<int>> cache(101);
    auto& v = cache[static_cast<std::size_t>(error_percent)];
    if (v.empty())
    {
        std::mt19937 rng(20261101u + static_cast<unsigned>(error_percent));   // fixed seed per rate: reproducible
        const std::size_t errors = input_count * static_cast<std::size_t>(error_percent) / 100;
        for (std::size_t i = 0; i < input_count; ++i)
            v.push_back(i < errors ? -static_cast<int>(i % 97) - 1 : static_cast<int>(i % 1000));
        std::shuffle(v.begin(), v.end(), rng);
    }
    return v;
}

struct tally
{
    std::int64_t sum = 0;
    std::int64_t errors = 0;
    friend bool operator==(const tally&, const tally&) = default;
};

// ---- the loops (identical work per element) --------------------------------------------------------------------------

tally run_core(const std::vector<int>& in)
{
    tally t;
    for (const int x : in)
    {
        const auto r = parse_core(x);
        if (r.is_ok()) [[likely]]
            t.sum += *r;
        else
            ++t.errors;
    }
    return t;
}

tally run_traced(const std::vector<int>& in)
{
    tally t;
    for (const int x : in)
    {
        const auto r = parse_traced(x);
        if (r.is_ok()) [[likely]]
            t.sum += *r;
        else
            ++t.errors;
    }
    return t;
}

tally run_hand(const std::vector<int>& in)
{
    tally t;
    for (const int x : in)
    {
        const hand_tagged r = parse_hand(x);
        if (r.ok) [[likely]]
            t.sum += r.value;
        else
            ++t.errors;
    }
    return t;
}

tally run_code(const std::vector<int>& in)
{
    tally t;
    for (const int x : in)
    {
        int v = 0;
        if (parse_code(x, v) == parse_error::none) [[likely]]
            t.sum += v;
        else
            ++t.errors;
    }
    return t;
}

tally run_throw(const std::vector<int>& in)
{
    tally t;
    for (const int x : in)
    {
        try
        {
            t.sum += parse_throw(x);
        }
        catch (const parse_exception&)
        {
            ++t.errors;
        }
    }
    return t;
}

tally run_core_propagate(const std::vector<int>& in)
{
    tally t;
    for (const int x : in)
    {
        const auto r = forward_core(x);
        if (r.is_ok()) [[likely]]
            t.sum += *r;
        else
            ++t.errors;
    }
    return t;
}

tally run_hand_propagate(const std::vector<int>& in)
{
    tally t;
    for (const int x : in)
    {
        const hand_tagged r = forward_hand(x);
        if (r.ok) [[likely]]
            t.sum += r.value;
        else
            ++t.errors;
    }
    return t;
}

template <tally (*Run)(const std::vector<int>&)>
void bench(benchmark::State& state)
{
    const auto& in = inputs(static_cast<int>(state.range(0)));
    for (auto _ : state)
    {
        tally t = Run(in);
        benchmark::DoNotOptimize(t);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(in.size()));
}

void error_rates(::benchmark::Benchmark* b)
{
    for (const int pct : {0, 1, 10, 50, 100})
        b->Arg(pct);
}

void success_only(::benchmark::Benchmark* b) { b->Arg(0); }

BENCHMARK(bench<run_core>)->Name("result_core")->Apply(error_rates);
BENCHMARK(bench<run_hand>)->Name("result_hand_tagged")->Apply(error_rates);
BENCHMARK(bench<run_code>)->Name("result_error_code")->Apply(error_rates);
BENCHMARK(bench<run_throw>)->Name("result_exceptions")->Apply(error_rates);
BENCHMARK(bench<run_traced>)->Name("result_traced_success")->Apply(success_only);
BENCHMARK(bench<run_core_propagate>)->Name("result_core_propagate")->Apply(error_rates);
BENCHMARK(bench<run_hand_propagate>)->Name("result_hand_propagate")->Apply(error_rates);

// Every variant must agree on every error rate before anything is timed.
bool fair()
{
    for (const int pct : {0, 1, 10, 50, 100})
    {
        const auto& in = inputs(pct);
        const tally reference = run_hand(in);
        if (run_core(in) != reference || run_code(in) != reference || run_throw(in) != reference ||
            run_traced(in) != reference)
            return false;
        // The propagating variants add 1 to every value.
        const tally forwarded = run_hand_propagate(in);
        if (run_core_propagate(in) != forwarded || forwarded.errors != reference.errors ||
            forwarded.sum != reference.sum + static_cast<std::int64_t>(in.size()) - reference.errors)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    static_assert(sizeof(snicholls::result<int, parse_error>) == sizeof(hand_tagged));
    if (!fair())
    {
        std::fprintf(stderr, "result_benchmark: variants disagree - refusing to benchmark\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
