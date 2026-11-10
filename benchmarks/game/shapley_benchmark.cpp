//
//  shapley_benchmark.cpp
//  snicholls-core
//
//  snicholls::game::shapley of a 16-player game from its table, against formula (3) of Moehle,
//  Boyd and Ang written out: for each player, every coalition without it, its lift weighted by
//  |S|! (n - |S| - 1)! / n!. The two sum the same terms in different orders, so the fairness check
//  asks for agreement to 1e-12 of the total and refuses to benchmark otherwise.
//

#include <snicholls/core/game/shapley.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t players = 16;

std::vector<double> make_table()
{
    std::vector<double> out(std::size_t{1} << players);
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    for (double& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<double>(x % 100000) / 1000.0;
    }
    return out;
}

const std::vector<double> table = make_table();

std::array<double, players> by_formula_3()
{
    std::array<double, players + 1> factorial{};
    factorial[0] = 1;
    for (std::size_t k = 1; k <= players; ++k)
        factorial[k] = factorial[k - 1] * static_cast<double>(k);
    std::array<double, players> a{};
    for (std::size_t i = 0; i < players; ++i)
        for (std::uint64_t s = 0; s < table.size(); ++s) {
            if ((s >> i) & 1u)
                continue;
            const auto k = static_cast<std::size_t>(std::popcount(s));
            a[i] += factorial[k] * factorial[players - k - 1] / factorial[players] * (table[s | (std::uint64_t{1} << i)] - table[s]);
        }
    return a;
}

std::array<double, players> by_core()
{
    return snicholls::game::shapley<players>(snicholls::game::from_table(table)).values;
}

void BM_Shapley_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(by_core());
}
BENCHMARK(BM_Shapley_Core)->Unit(benchmark::kMicrosecond);

void BM_Shapley_Formula3ByHand(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(by_formula_3());
}
BENCHMARK(BM_Shapley_Formula3ByHand)->Unit(benchmark::kMicrosecond);

} // namespace

int main(int argc, char** argv)
{
    const auto a = by_core();
    const auto b = by_formula_3();
    for (std::size_t i = 0; i < players; ++i)
        if (std::abs(a[i] - b[i]) > 1e-12 * std::abs(table.back())) {
            std::fprintf(stderr, "shapley benchmark: fairness check FAILED - the attributions differ; not benchmarking\n");
            return 1;
        }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
