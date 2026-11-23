//
//  day_count_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  finance::act_act_isda::year_fraction and finance::thirty_360::year_fraction against the same arithmetic written by
//  hand (the same civil-date conversions, the same sums, the same reduction by std::gcd), over 4,096 pairs of dates
//  up to 30 years apart. main() runs a fairness check first and refuses to benchmark if the two sides disagree on
//  any pair (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/time/civil.hpp>

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

namespace {

using namespace snicholls;
using std::chrono::sys_days;

std::vector<std::pair<sys_days, sys_days>> make_pairs()
{
    std::vector<std::pair<sys_days, sys_days>> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (auto& [a, b] : out) {
        const auto start = static_cast<int>(next() % 40000);   // 1970 to 2079
        a = sys_days{std::chrono::days{start}};
        b = a + std::chrono::days{static_cast<int>(next() % 11000)};
    }
    return out;
}

const std::vector<std::pair<sys_days, sys_days>>& pairs()
{
    static const auto p = make_pairs();
    return p;
}

// ACT/ACT ISDA by hand: numerator over 365 * 366, reduced
std::pair<std::int64_t, std::int64_t> act_act_by_hand(sys_days start, sys_days end) noexcept
{
    const std::int64_t a = start.time_since_epoch().count(), b = end.time_since_epoch().count();
    const std::int64_t y1 = civil_from_days(a).year, y2 = civil_from_days(b).year;
    const std::int64_t w1 = is_leap_year(y1) ? 365 : 366, w2 = is_leap_year(y2) ? 365 : 366;
    const std::int64_t n = y1 == y2 ? (b - a) * w1
                                    : (y2 - y1 - 1) * 133590 + (days_from_civil(y1 + 1, 1, 1) - a) * w1 + (b - days_from_civil(y2, 1, 1)) * w2;
    const std::int64_t g = std::gcd(n, std::int64_t{133590});
    return {n / g, 133590 / g};
}

// 30/360 Bond Basis by hand
std::pair<std::int64_t, std::int64_t> thirty_by_hand(sys_days start, sys_days end) noexcept
{
    const civil_date a = civil_from_days(start.time_since_epoch().count()), b = civil_from_days(end.time_since_epoch().count());
    const std::int64_t d1 = a.day == 31 ? 30 : a.day;
    const std::int64_t d2 = b.day == 31 && d1 == 30 ? 30 : b.day;
    const std::int64_t n = 360 * (b.year - a.year) + 30 * (static_cast<std::int64_t>(b.month) - static_cast<std::int64_t>(a.month)) + (d2 - d1);
    const std::int64_t g = std::gcd(n, std::int64_t{360});
    return {g == 0 ? 0 : n / g, g == 0 ? 1 : 360 / g};
}

void act_act_library(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& [a, b] : pairs()) {
            auto f = finance::act_act_isda::year_fraction(a, b);
            benchmark::DoNotOptimize(f);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

void act_act_hand(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& [a, b] : pairs()) {
            auto f = act_act_by_hand(a, b);
            benchmark::DoNotOptimize(f);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

void thirty_360_library(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& [a, b] : pairs()) {
            auto f = finance::thirty_360::year_fraction(a, b);
            benchmark::DoNotOptimize(f);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

void thirty_360_hand(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& [a, b] : pairs()) {
            auto f = thirty_by_hand(a, b);
            benchmark::DoNotOptimize(f);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

BENCHMARK(act_act_library);
BENCHMARK(act_act_hand);
BENCHMARK(thirty_360_library);
BENCHMARK(thirty_360_hand);

} // namespace

int main(int argc, char** argv)
{
    for (const auto& [a, b] : pairs()) {
        const auto f = finance::act_act_isda::year_fraction(a, b), t = finance::thirty_360::year_fraction(a, b);
        const auto [fn, fd] = act_act_by_hand(a, b);
        const auto [tn, td] = thirty_by_hand(a, b);
        if (f.numerator() != fn || f.denominator() != fd || t.numerator() != tn || t.denominator() != td) {
            std::fprintf(stderr, "day_count_benchmark: the fairness check failed: the two sides disagree\n");
            return 1;
        }
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
