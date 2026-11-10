//
//  struct_tuple_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The aggregate conversions of <snicholls/core/reflect/struct_tuple.hpp>
//  against hand-written member access: to_tuple vs std::tuple{s.a, s.b, s.c},
//  tie vs writing the members directly, from_tuple vs S{get<0>(t), ...}. Each
//  pair works on the same 4,096 records in the same container and does the
//  same arithmetic in the same order. main() runs a fairness check first and
//  refuses to benchmark if any pair disagrees (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/reflect/struct_tuple.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {

struct quote {
    long id;
    double px;
    int qty;
};

using quote_tuple = std::tuple<long, double, int>;

std::vector<quote> make_quotes()
{
    std::vector<quote> out(4096);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = quote{static_cast<long>(i), 100.0 + static_cast<double>(i) * 0.25, static_cast<int>(i % 97)};
    return out;
}

std::vector<quote_tuple> make_tuples()
{
    std::vector<quote_tuple> out;
    for (const auto& q : make_quotes())
        out.emplace_back(q.id, q.px, q.qty);
    return out;
}

const std::vector<quote> quotes = make_quotes();
const std::vector<quote_tuple> tuples = make_tuples();

// --- to_tuple ---------------------------------------------------------------------

void to_tuple_core(std::vector<quote_tuple>& out)
{
    for (std::size_t i = 0; i < quotes.size(); ++i)
        out[i] = snicholls::to_tuple(quotes[i]);
}

void to_tuple_by_hand(std::vector<quote_tuple>& out)
{
    for (std::size_t i = 0; i < quotes.size(); ++i)
        out[i] = quote_tuple{quotes[i].id, quotes[i].px, quotes[i].qty};
}

// --- tie: read two members, write the third -----------------------------------------

void tie_core(std::vector<quote>& qs)
{
    for (auto& q : qs) {
        auto [id, px, qty] = snicholls::tie(q);
        px = px + static_cast<double>(qty);
        qty = qty + static_cast<int>(id & 1);
    }
}

void tie_by_hand(std::vector<quote>& qs)
{
    for (auto& q : qs) {
        q.px = q.px + static_cast<double>(q.qty);
        q.qty = q.qty + static_cast<int>(q.id & 1);
    }
}

// --- from_tuple ---------------------------------------------------------------------

void from_tuple_core(std::vector<quote>& out)
{
    for (std::size_t i = 0; i < tuples.size(); ++i)
        out[i] = snicholls::from_tuple<quote>(tuples[i]);
}

void from_tuple_by_hand(std::vector<quote>& out)
{
    for (std::size_t i = 0; i < tuples.size(); ++i)
        out[i] = quote{std::get<0>(tuples[i]), std::get<1>(tuples[i]), std::get<2>(tuples[i])};
}

// --- benchmarks ---------------------------------------------------------------------

template <typename Out, void (*F)(Out&)>
void run(benchmark::State& state, Out out)
{
    for (auto _ : state) {
        F(out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(out.size()));
}

void BM_ToTuple_Core(benchmark::State& s) { run<std::vector<quote_tuple>, to_tuple_core>(s, tuples); }
void BM_ToTuple_HandWritten(benchmark::State& s) { run<std::vector<quote_tuple>, to_tuple_by_hand>(s, tuples); }
void BM_Tie_Core(benchmark::State& s) { run<std::vector<quote>, tie_core>(s, quotes); }
void BM_Tie_HandWritten(benchmark::State& s) { run<std::vector<quote>, tie_by_hand>(s, quotes); }
void BM_FromTuple_Core(benchmark::State& s) { run<std::vector<quote>, from_tuple_core>(s, quotes); }
void BM_FromTuple_HandWritten(benchmark::State& s) { run<std::vector<quote>, from_tuple_by_hand>(s, quotes); }
BENCHMARK(BM_ToTuple_Core);
BENCHMARK(BM_ToTuple_HandWritten);
BENCHMARK(BM_Tie_Core);
BENCHMARK(BM_Tie_HandWritten);
BENCHMARK(BM_FromTuple_Core);
BENCHMARK(BM_FromTuple_HandWritten);

// --- fairness -----------------------------------------------------------------------

bool same(const std::vector<quote>& a, const std::vector<quote>& b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].id != b[i].id || a[i].px != b[i].px || a[i].qty != b[i].qty)
            return false;
    return true;
}

bool fair()
{
    std::vector<quote_tuple> t1(quotes.size()), t2(quotes.size());
    to_tuple_core(t1);
    to_tuple_by_hand(t2);
    if (t1 != t2)
        return false;

    std::vector<quote> q1 = quotes, q2 = quotes;
    tie_core(q1);
    tie_by_hand(q2);
    if (!same(q1, q2))
        return false;

    std::vector<quote> f1(tuples.size()), f2(tuples.size());
    from_tuple_core(f1);
    from_tuple_by_hand(f2);
    return same(f1, f2) && same(f1, quotes);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "struct_tuple benchmark: fairness check FAILED - a pair disagrees; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
