//
//  pivot_benchmark.cpp
//  snicholls-core
//
//  snicholls::stats::pivot over 1,000,000 fills against the same aggregation written out:
//  - by currency (8) and index (4), both enumerations: P&L summed, return averaged (Welford);
//    by hand, two fixed arrays indexed by the enumerations' values.
//  - by desk name (32 strings, found at run time): P&L summed; by hand, a std::unordered_map.
//  The arithmetic is the same in the same order, so the fairness check asks for identical
//  results and refuses to benchmark otherwise.
//

#include <snicholls/core/stats/pivot.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

enum class currency : std::uint8_t { usd, eur, gbp, jpy, chf, aud, cad, nzd };
enum class equity_index : std::uint8_t { spx, sx5e, ukx, nky };
struct pnl {
    double value;
};
struct ret {
    double value;
};
struct desk {
    std::string name;
};
using fill = std::tuple<currency, equity_index, pnl, ret, desk>;

struct pnl_of {
    double operator()(const fill& f) const { return std::get<pnl>(f).value; }
};
struct ret_of {
    double operator()(const fill& f) const { return std::get<ret>(f).value; }
};
struct desk_of {
    const std::string& operator()(const fill& f) const { return std::get<desk>(f).name; }
};

const std::vector<fill> fills = [] {
    std::vector<fill> out;
    out.reserve(1'000'000);
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    for (int i = 0; i < 1'000'000; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        out.emplace_back(static_cast<currency>(x % 8), static_cast<equity_index>((x >> 8) % 4), pnl{static_cast<double>(x % 2001) - 1000.0},
                         ret{static_cast<double>((x >> 16) % 2001) / 100000.0 - 0.01}, desk{"desk" + std::to_string((x >> 24) % 32)});
    }
    return out;
}();

using enum_pivot = snicholls::stats::pivot<fill, snicholls::meta::type_list<currency>, snicholls::meta::type_list<equity_index>,
                                           snicholls::stats::measure<pnl_of, snicholls::stats::sum<double>>,
                                           snicholls::stats::measure<ret_of, snicholls::stats::mean<double>>>;
using desk_pivot = snicholls::stats::pivot<fill, snicholls::meta::type_list<desk_of>, snicholls::meta::type_list<>,
                                           snicholls::stats::measure<pnl_of, snicholls::stats::sum<double>>>;

struct by_hand {
    std::array<std::array<double, 4>, 8> sum{};
    std::array<std::array<std::uint64_t, 4>, 8> n{};
    std::array<std::array<double, 4>, 8> mean{};
};

by_hand enums_by_hand()
{
    by_hand h;
    for (const fill& f : fills) {
        const auto c = static_cast<std::size_t>(std::get<currency>(f));
        const auto i = static_cast<std::size_t>(std::get<equity_index>(f));
        h.sum[c][i] += std::get<pnl>(f).value;
        ++h.n[c][i];
        h.mean[c][i] += (std::get<ret>(f).value - h.mean[c][i]) / static_cast<double>(h.n[c][i]);
    }
    return h;
}

enum_pivot enums_by_pivot()
{
    enum_pivot p;
    p.add_all(fills);
    return p;
}

std::unordered_map<std::string, double> desks_by_hand()
{
    std::unordered_map<std::string, double> sums;
    for (const fill& f : fills)
        sums[std::get<desk>(f).name] += std::get<pnl>(f).value;
    return sums;
}

desk_pivot desks_by_pivot()
{
    desk_pivot p;
    p.add_all(fills);
    return p;
}

bool fair()
{
    const by_hand h = enums_by_hand();
    const enum_pivot p = enums_by_pivot();
    for (std::size_t c = 0; c < 8; ++c)
        for (std::size_t i = 0; i < 4; ++i) {
            const typename enum_pivot::keys_type keys{static_cast<currency>(c), static_cast<equity_index>(i)};
            if (p.value<0>(keys) != h.sum[c][i] || p.value<1>(keys) != h.mean[c][i])
                return false;
        }
    const auto sums = desks_by_hand();
    const desk_pivot d = desks_by_pivot();
    for (const auto& [name, total] : sums)
        if (d.value<0>({name}) != total)
            return false;
    return true;
}

void BM_Pivot_EnumKeysCore(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(enums_by_pivot());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(fills.size()));
}
BENCHMARK(BM_Pivot_EnumKeysCore)->Unit(benchmark::kMicrosecond);

void BM_Pivot_EnumKeysByHand(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(enums_by_hand());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(fills.size()));
}
BENCHMARK(BM_Pivot_EnumKeysByHand)->Unit(benchmark::kMicrosecond);

void BM_Pivot_StringKeysCore(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(desks_by_pivot());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(fills.size()));
}
BENCHMARK(BM_Pivot_StringKeysCore)->Unit(benchmark::kMicrosecond);

void BM_Pivot_StringKeysByHand(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(desks_by_hand());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(fills.size()));
}
BENCHMARK(BM_Pivot_StringKeysByHand)->Unit(benchmark::kMicrosecond);

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "pivot benchmark: fairness check FAILED - the aggregations differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
