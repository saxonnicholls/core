//
//  ann_forest_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::containers::ann_forest against brute force on the same data: 20,000 clustered points in 32 dimensions
//  (64 clusters, members normal around uniform centres), 200 queries drawn from the same mixture, the ten nearest of
//  each, under the Euclidean and the angular metric, with forests of 10 and 20 trees and budgets of trees x k (Annoy's
//  default search_k) and trees x k x 10 candidates. The brute-force scan keeps the ten best by the same insertion the forest re-ranks with, over the same
//  metric. Each query benchmark reports recall@10 against brute force as the counter `recall`; build is timed too.
//
//  main() runs a validity check first and refuses to benchmark unless brute force agrees with a full sort, every
//  distance the forests return is exact and nearest first, and every forest's recall@10 at the larger budget is at
//  least 0.8.
//
//  Smoke mode (SNICHOLLS_CORE_BENCHMARK_SMOKE=1, or a --benchmark_min_time below 0.1 s, as the CTest run passes):
//  2,000 points and 20 queries, so that setup stays quick; its numbers mean nothing.
//

#include <snicholls/core/containers/ann_forest.hpp>

#include <snicholls/core/metric/angular.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

namespace ct = snicholls::containers;
namespace mt = snicholls::metric;
namespace st = snicholls::stats;

constexpr std::size_t D = 32;
constexpr std::size_t k = 10;
constexpr std::size_t leaf_size = 16;
using point = std::array<float, D>;
using top_k = std::array<ct::ann_neighbour, k>;

std::size_t n_points = 20000;
std::size_t n_queries = 200;

struct dataset {
    std::vector<point> points, queries;
};

const dataset& data()
{
    static const dataset d = [] {
        st::wyrand g(2026);
        st::normal_box_muller z;
        std::vector<point> centres(64);
        for (auto& c : centres)
            for (auto& x : c)
                x = static_cast<float>(st::uniform_real(g) * 2 - 1);
        auto draw = [&](std::size_t n) {
            std::vector<point> out(n);
            for (auto& p : out) {
                const auto& c = centres[st::uniform_index(g, centres.size())];
                for (std::size_t i = 0; i < D; ++i)
                    p[i] = static_cast<float>(c[i] + 0.15 * z(g));
            }
            return out;
        };
        dataset s;
        s.points = draw(n_points);
        s.queries = draw(n_queries);
        return s;
    }();
    return d;
}

// Brute force: every point, the same metric, the ten best kept by insertion (as the forest re-ranks)
template <typename Metric>
std::size_t brute_force(const std::vector<point>& pts, const point& q, top_k& out, const Metric& m)
{
    std::size_t have = 0;
    for (std::uint32_t i = 0; i < pts.size(); ++i) {
        const double d = static_cast<double>(m(pts[i], q));
        if (have == k && !(d < out[k - 1].distance))
            continue;
        std::size_t j = have < k ? have++ : k - 1;
        for (; j > 0 && out[j - 1].distance > d; --j)
            out[j] = out[j - 1];
        out[j] = {i, d};
    }
    return have;
}

template <typename Metric>
const std::vector<top_k>& truth()
{
    static const std::vector<top_k> t = [] {
        std::vector<top_k> out(data().queries.size());
        for (std::size_t q = 0; q < out.size(); ++q)
            brute_force(data().points, data().queries[q], out[q], Metric{});
        return out;
    }();
    return t;
}

template <typename Metric>
struct forest_fixture {
    using forest = ct::ann_forest<point, D, Metric>;
    std::vector<ct::ann_node> nodes;
    std::vector<std::uint32_t> items;
    std::vector<float> normals;
    std::optional<forest> index;
    std::size_t trees = 0;

    explicit forest_fixture(std::size_t t) : trees(t)
    {
        const auto s = forest::suggested_storage(data().points.size(), trees, leaf_size);
        nodes.resize(s.nodes);
        items.resize(s.items);
        normals.resize(s.normals);
        st::wyrand g(7);
        auto r = forest::build(data().points, trees, leaf_size, nodes, items, normals, g);
        if (r)
            index = r.unwrap();
    }
};

template <typename Metric>
forest_fixture<Metric>& fixture(std::size_t trees)
{
    static std::map<std::size_t, std::unique_ptr<forest_fixture<Metric>>> cache;
    auto& slot = cache[trees];
    if (!slot)
        slot = std::make_unique<forest_fixture<Metric>>(trees);
    return *slot;
}

// Recall@10 of a forest against brute force, and whether every distance was exact and nearest first
template <typename Metric>
std::pair<double, bool> evaluate(std::size_t trees, std::size_t budget)
{
    const auto& fx = fixture<Metric>(trees);
    if (!fx.index)
        return {0.0, false};
    const Metric m{};
    std::vector<std::uint32_t> candidates(budget);
    std::vector<ct::ann_frontier> frontier(4096);
    std::size_t hit = 0;
    bool exact = true;
    for (std::size_t q = 0; q < data().queries.size(); ++q) {
        top_k out{};
        const point& query = data().queries[q];
        const std::size_t got = fx.index->query(query, out, candidates, frontier);
        exact = exact && got == k;
        for (std::size_t i = 0; i < got; ++i) {
            exact = exact && out[i].distance == static_cast<double>(m(data().points[out[i].index], query)) && (i == 0 || out[i - 1].distance <= out[i].distance);
            for (const auto& t : truth<Metric>()[q])
                hit += t.index == out[i].index;
        }
    }
    return {static_cast<double>(hit) / static_cast<double>(data().queries.size() * k), exact};
}

template <typename Metric>
void BM_Build(benchmark::State& state)
{
    using forest = ct::ann_forest<point, D, Metric>;
    const auto trees = static_cast<std::size_t>(state.range(0));
    const auto s = forest::suggested_storage(data().points.size(), trees, leaf_size);
    std::vector<ct::ann_node> nodes(s.nodes);
    std::vector<std::uint32_t> items(s.items);
    std::vector<float> normals(s.normals);
    for (auto _ : state) {
        st::wyrand g(7);
        auto r = forest::build(data().points, trees, leaf_size, nodes, items, normals, g);
        benchmark::DoNotOptimize(r);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(data().points.size()));
}

template <typename Metric>
void BM_Query(benchmark::State& state)
{
    const auto trees = static_cast<std::size_t>(state.range(0));
    const auto budget = trees * k * static_cast<std::size_t>(state.range(1));
    const auto& fx = fixture<Metric>(trees);
    std::vector<std::uint32_t> candidates(budget);
    std::vector<ct::ann_frontier> frontier(4096);
    top_k out{};
    std::size_t q = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(fx.index->query(data().queries[q], out, candidates, frontier));
        benchmark::DoNotOptimize(out.data());
        q = q + 1 == data().queries.size() ? 0 : q + 1;
    }
    state.SetItemsProcessed(state.iterations());
    state.counters["recall"] = evaluate<Metric>(trees, budget).first;
}

template <typename Metric>
void BM_BruteForce(benchmark::State& state)
{
    const Metric m{};
    top_k out{};
    std::size_t q = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(brute_force(data().points, data().queries[q], out, m));
        benchmark::DoNotOptimize(out.data());
        q = q + 1 == data().queries.size() ? 0 : q + 1;
    }
    state.SetItemsProcessed(state.iterations());
    state.counters["recall"] = 1.0;
}

BENCHMARK(BM_Build<mt::euclidean>)->Name("BM_AnnForest_Build_Euclidean")->Arg(10)->Arg(20)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Build<mt::angular>)->Name("BM_AnnForest_Build_Angular")->Arg(10)->Arg(20)->Unit(benchmark::kMillisecond);
// Arguments: trees, and the budget as a multiple of trees x k
BENCHMARK(BM_Query<mt::euclidean>)->Name("BM_AnnForest_Query_Euclidean")->ArgNames({"trees", "budget_x"})->ArgsProduct({{10, 20}, {1, 10}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_BruteForce<mt::euclidean>)->Name("BM_BruteForce_Query_Euclidean")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Query<mt::angular>)->Name("BM_AnnForest_Query_Angular")->ArgNames({"trees", "budget_x"})->ArgsProduct({{10, 20}, {1, 10}})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_BruteForce<mt::angular>)->Name("BM_BruteForce_Query_Angular")->Unit(benchmark::kMicrosecond);

// Brute force agrees with a full sort; every forest's distances are exact, nearest first, at a recall of 0.8 or more
template <typename Metric>
bool valid(const char* name)
{
    const Metric m{};
    for (std::size_t q = 0; q < data().queries.size(); ++q) {
        std::vector<double> all(data().points.size());
        for (std::size_t i = 0; i < all.size(); ++i)
            all[i] = static_cast<double>(m(data().points[i], data().queries[q]));
        std::sort(all.begin(), all.end());
        for (std::size_t i = 0; i < k; ++i)
            if (truth<Metric>()[q][i].distance != all[i])
                return false;
    }
    for (const std::size_t trees : {10u, 20u}) {
        const auto [small, small_exact] = evaluate<Metric>(trees, trees * k);
        const auto [r, exact] = evaluate<Metric>(trees, trees * k * 10);
        std::printf("ann_forest benchmark: %s, %zu points, %zu trees: recall@10 %.3f at budget %zu, %.3f at budget %zu%s\n", name, data().points.size(), trees, small,
                    trees * k, r, trees * k * 10, exact && small_exact ? "" : " (a distance was not exact)");
        if (!exact || !small_exact || r < 0.8)
            return false;
    }
    return true;
}

bool smoke(int argc, char** argv)
{
    const char* env = std::getenv("SNICHOLLS_CORE_BENCHMARK_SMOKE");
    if (env != nullptr && *env != '\0' && *env != '0')
        return true;
    constexpr const char flag[] = "--benchmark_min_time=";
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], flag, sizeof flag - 1) == 0 && std::strtod(argv[i] + sizeof flag - 1, nullptr) < 0.1)
            return true;
    return false;
}

} // namespace

int main(int argc, char** argv)
{
    if (smoke(argc, argv)) {
        n_points = 2000;
        n_queries = 20;
    }
    if (!valid<mt::euclidean>("euclidean") || !valid<mt::angular>("angular")) {
        std::fprintf(stderr, "ann_forest benchmark: validity check FAILED - brute force disagrees with a sort, a distance is inexact, or recall is below 0.8; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
