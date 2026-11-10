//
//  bootstrap_benchmark.cpp
//  snicholls-core
//
//  The bootstrap distribution of a mean: 1,024 observations, 2,000 resamples, five ways.
//
//  - Standard: std::mt19937_64, std::uniform_int_distribution, each resample gathered into a vector.
//  - PrecomputedTable: snicholls::stats::resample_table, every index drawn once in advance (what a
//    table built at compile time would give) and read: the cost of reuse.
//  - CoreIndices / HandWrittenIndices: snicholls::stats::for_each_resample against the same draws
//    written out by hand (wyrand, two 32-bit Lemire draws per call), summing through the indices.
//  - CoreCounts: multinomial counts, then one sequential weighted pass.
//  - CorePoisson: independent Poisson(1) counts, then one sequential weighted pass.
//
//  Fairness, checked before anything is timed: CoreIndices, HandWrittenIndices and PrecomputedTable
//  give identical replicates; CoreCounts gives the same replicates to 1e-12 (it sums in another
//  order); Standard and CorePoisson draw differently, so their standard errors must agree with
//  CoreIndices' to 10%.
//

#include <snicholls/core/stats/bootstrap.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace {

constexpr std::size_t n = 1024;
constexpr std::size_t resamples = 2000;
constexpr std::uint64_t seed = 2026;

std::vector<double> make_data()
{
    snicholls::stats::wyrand g(99);
    std::vector<double> x(n);
    for (double& v : x)
        v = snicholls::stats::uniform_real(g) - 0.45;
    return x;
}

const std::vector<double> data = make_data();

std::vector<double> standard()
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    std::vector<double> resample(n), out;
    out.reserve(resamples);
    for (std::size_t b = 0; b < resamples; ++b) {
        for (double& r : resample)
            r = data[pick(rng)];
        out.push_back(std::accumulate(resample.begin(), resample.end(), 0.0) / n);
    }
    return out;
}

std::vector<double> core_indices()
{
    snicholls::stats::wyrand g(seed);
    std::vector<double> out;
    out.reserve(resamples);
    snicholls::stats::for_each_resample(n, resamples, g, [&](std::span<const std::uint32_t> idx) {
        double sum = 0;
        for (const std::uint32_t i : idx)
            sum += data[i];
        out.push_back(sum / n);
    });
    return out;
}

std::vector<double> hand_written_indices()
{
    snicholls::stats::wyrand g(seed);
    std::uint64_t bits = 0;
    bool have = false;
    const auto bits32 = [&]() -> std::uint32_t {
        if (have) {
            have = false;
            return static_cast<std::uint32_t>(bits >> 32);
        }
        bits = g();
        have = true;
        return static_cast<std::uint32_t>(bits);
    };
    std::vector<double> out;
    out.reserve(resamples);
    std::vector<std::uint32_t> idx(n);
    for (std::size_t b = 0; b < resamples; ++b) {
        for (std::uint32_t& i : idx) {
            std::uint64_t m = static_cast<std::uint64_t>(bits32()) * n;
            if (static_cast<std::uint32_t>(m) < n) {
                const std::uint32_t threshold = (std::uint32_t{0} - static_cast<std::uint32_t>(n)) % static_cast<std::uint32_t>(n);
                while (static_cast<std::uint32_t>(m) < threshold)
                    m = static_cast<std::uint64_t>(bits32()) * n;
            }
            i = static_cast<std::uint32_t>(m >> 32);
        }
        double sum = 0;
        for (const std::uint32_t i : idx)
            sum += data[i];
        out.push_back(sum / n);
    }
    return out;
}

const snicholls::stats::resample_table table = [] {
    snicholls::stats::wyrand g(seed);
    return snicholls::stats::resample_table(n, resamples, g);
}();

std::vector<double> precomputed_table()
{
    std::vector<double> out;
    out.reserve(resamples);
    table.for_each([&](std::span<const std::uint32_t> idx) {
        double sum = 0;
        for (const std::uint32_t i : idx)
            sum += data[i];
        out.push_back(sum / n);
    });
    return out;
}

std::vector<double> core_counts()
{
    snicholls::stats::wyrand g(seed);
    std::vector<double> out;
    out.reserve(resamples);
    snicholls::stats::for_each_resample_counts(n, resamples, g, [&](std::span<const std::uint32_t> counts) {
        double sum = 0;
        for (std::size_t i = 0; i < n; ++i)
            sum += static_cast<double>(counts[i]) * data[i];
        out.push_back(sum / n);
    });
    return out;
}

std::vector<double> core_poisson()
{
    snicholls::stats::wyrand g(seed);
    std::vector<double> out;
    out.reserve(resamples);
    snicholls::stats::for_each_poisson_resample(n, resamples, g, [&](std::span<const std::uint32_t> counts) {
        double sum = 0, weight = 0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += static_cast<double>(counts[i]) * data[i];
            weight += static_cast<double>(counts[i]);
        }
        out.push_back(sum / weight);
    });
    return out;
}

double standard_error(const std::vector<double>& r)
{
    const double m = std::accumulate(r.begin(), r.end(), 0.0) / static_cast<double>(r.size());
    double s = 0;
    for (const double x : r)
        s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(r.size() - 1));
}

bool fair()
{
    const auto core = core_indices();
    if (core != hand_written_indices() || core != precomputed_table())
        return false;
    const auto counts = core_counts();
    for (std::size_t b = 0; b < resamples; ++b)
        if (std::abs(counts[b] - core[b]) > 1e-12)
            return false;
    const double se = standard_error(core);
    return std::abs(standard_error(standard()) / se - 1) < 0.1 && std::abs(standard_error(core_poisson()) / se - 1) < 0.1;
}

template <std::vector<double> (*Run)()>
void run(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(Run());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(resamples));
}

BENCHMARK(run<standard>)->Name("BM_Bootstrap_StandardMt19937Gather")->Unit(benchmark::kMicrosecond);
BENCHMARK(run<precomputed_table>)->Name("BM_Bootstrap_ResampleTableReused")->Unit(benchmark::kMicrosecond);
BENCHMARK(run<core_indices>)->Name("BM_Bootstrap_CoreIndices")->Unit(benchmark::kMicrosecond);
BENCHMARK(run<hand_written_indices>)->Name("BM_Bootstrap_HandWrittenIndices")->Unit(benchmark::kMicrosecond);
BENCHMARK(run<core_counts>)->Name("BM_Bootstrap_CoreMultinomialCounts")->Unit(benchmark::kMicrosecond);
BENCHMARK(run<core_poisson>)->Name("BM_Bootstrap_CorePoissonCounts")->Unit(benchmark::kMicrosecond);

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "bootstrap benchmark: fairness check FAILED - the methods disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
