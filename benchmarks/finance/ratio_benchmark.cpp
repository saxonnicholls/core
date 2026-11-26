//
//  ratio_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  finance::ratio's exact <=> (a continued-fraction comparison) against the 64-bit cross-multiplication year_fraction
//  used before 0.27.0, over 4,096 pairs of day-count fractions (numerators up to 2^32, denominators up to 133,590,
//  where cross-multiplication is still exact), and finance::multiply against the same cancel-then-multiply written by
//  hand. main() runs a fairness check first and refuses to benchmark if the two sides disagree on any pair
//  (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/finance/ratio.hpp>

#include <benchmark/benchmark.h>

#include <compare>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

namespace {

using snicholls::finance::ratio;

std::vector<std::pair<ratio, ratio>> make_pairs()
{
    std::vector<std::pair<ratio, ratio>> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    constexpr std::int64_t dens[] = {360, 365, 133590};
    for (auto& [a, b] : out) {
        a = ratio{static_cast<std::int64_t>(next() % (1ULL << 32)) - (1LL << 31), dens[next() % 3]};
        b = ratio{static_cast<std::int64_t>(next() % (1ULL << 32)) - (1LL << 31), dens[next() % 3]};
    }
    return out;
}

const std::vector<std::pair<ratio, ratio>>& pairs()
{
    static const auto p = make_pairs();
    return p;
}

// year_fraction's order before 0.27.0
std::strong_ordering cross_multiplied(const ratio& a, const ratio& b) noexcept
{
    return a.numerator() * b.denominator() <=> b.numerator() * a.denominator();
}

// multiply by hand: cancel across, multiply, no overflow check (these pairs fit)
std::pair<std::int64_t, std::int64_t> multiply_by_hand(const ratio& a, const ratio& b) noexcept
{
    const std::int64_t g1 = std::gcd(a.numerator(), b.denominator()), g2 = std::gcd(b.numerator(), a.denominator());
    const std::int64_t n = (a.numerator() / g1) * (b.numerator() / g2);
    return {n, n == 0 ? 1 : (a.denominator() / g2) * (b.denominator() / g1)};
}

void compare_library(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& [a, b] : pairs()) {
            auto o = a <=> b;
            benchmark::DoNotOptimize(o);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

void compare_cross_multiplied(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& [a, b] : pairs()) {
            auto o = cross_multiplied(a, b);
            benchmark::DoNotOptimize(o);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

void multiply_library(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& [a, b] : pairs()) {
            auto m = multiply(a, b);
            benchmark::DoNotOptimize(m);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

void multiply_hand(benchmark::State& state)
{
    for (auto _ : state)
        for (const auto& [a, b] : pairs()) {
            auto m = multiply_by_hand(a, b);
            benchmark::DoNotOptimize(m);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(pairs().size()));
}

BENCHMARK(compare_library);
BENCHMARK(compare_cross_multiplied);
BENCHMARK(multiply_library);
BENCHMARK(multiply_hand);

} // namespace

int main(int argc, char** argv)
{
    for (const auto& [a, b] : pairs()) {
        const auto m = multiply(a, b);
        const auto [n, d] = multiply_by_hand(a, b);
        if ((a <=> b) != cross_multiplied(a, b) || !m || m->numerator() != n || m->denominator() != d) {
            std::fprintf(stderr, "ratio_benchmark: the fairness check failed: the two sides disagree\n");
            return 1;
        }
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
