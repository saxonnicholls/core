//
//  snell_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  stochastic::lattice_envelope on a 64-step binomial American put, against the same backward induction written by
//  hand over one row (identical payoffs, the same products summed in the same order, the exercise boundary tracked
//  as it goes) and against the dense snell_envelope on the same lattice written as a 65 x 65 matrix, its boundary read
//  off its table. main() runs a fairness check first and refuses to benchmark if the three disagree on the value by a
//  single bit or on the boundary at any step (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/stochastic/snell.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

namespace st = snicholls::stochastic;

constexpr std::size_t steps = 64, nodes = steps + 1;

struct lattice {
    std::array<double, 2> stencil{};
    double discount = 0;
    std::array<double, 2 * steps + 1> level{};   // the spot at net k - steps up-moves
};

lattice make_lattice()
{
    lattice l;
    const double dt = 1.0 / steps, up = std::exp(0.2 * std::sqrt(dt)), p = (std::exp(0.05 * dt) - 1 / up) / (up - 1 / up);
    l.stencil = {1 - p, p};
    l.discount = std::exp(-0.05 * dt);
    for (std::size_t k = 0; k < l.level.size(); ++k)
        l.level[k] = 100 * std::pow(up, static_cast<double>(k) - static_cast<double>(steps));
    return l;
}

const lattice& data()
{
    static const lattice l = make_lattice();
    return l;
}

// The payoff at node j of step t: an American put struck at 100
double put(const lattice& l, std::size_t t, std::size_t j) noexcept
{
    const double s = l.level[steps + 2 * j - t];
    return s < 100 ? 100 - s : 0.0;
}

// What each side produces: the value and, per step, the lowest and highest node where exercising strictly beats waiting
struct answer {
    double value = 0;
    std::array<std::size_t, steps> first{}, last{};
    bool operator==(const answer&) const = default;
};
constexpr std::size_t none = static_cast<std::size_t>(-1);

answer by_lattice(const lattice& l)
{
    const auto r = st::lattice_envelope<steps>(l.stencil, [&](std::size_t t, std::size_t j) { return put(l, t, j); }, l.discount);
    return {r.value, r.first_exercise, r.last_exercise};
}

answer by_hand(const lattice& l)
{
    answer out;
    std::array<double, nodes> row{};
    for (std::size_t j = 0; j < nodes; ++j)
        row[j] = put(l, steps, j);
    for (std::size_t t = steps; t-- > 0;) {
        out.first[t] = out.last[t] = none;
        for (std::size_t j = 0; j <= t; ++j) {
            double expected = 0;
            expected += l.stencil[0] * row[j];
            expected += l.stencil[1] * row[j + 1];
            const double continuation = l.discount * expected, stop = put(l, t, j);
            if (stop > continuation) {
                out.first[t] = out.first[t] == none ? j : out.first[t];
                out.last[t] = j;
            }
            row[j] = stop > continuation ? stop : continuation;
        }
    }
    out.value = row[0];
    return out;
}

std::array<std::array<double, nodes>, nodes> dense_matrix(const lattice& l)
{
    std::array<std::array<double, nodes>, nodes> p{};
    for (std::size_t j = 0; j < nodes; ++j) {
        p[j][j] += l.stencil[0];
        p[j][j + 1 < nodes ? j + 1 : j] += l.stencil[1];
    }
    return p;
}

answer by_dense(const lattice& l, const std::array<std::array<double, nodes>, nodes>& matrix)
{
    // The full table: about 100 KB, so it lives in static storage rather than on the stack
    static st::fixed_snell<nodes, steps> table;
    table = st::snell_envelope<nodes, steps>(matrix, [&](std::size_t t, std::size_t j) { return j > t ? 0.0 : put(l, t, j); }, l.discount);
    answer out{table.value[0][0], {}, {}};
    for (std::size_t t = 0; t < steps; ++t) {
        out.first[t] = out.last[t] = none;
        for (std::size_t j = 0; j <= t; ++j)
            if (table.payoff[t][j] > table.continuation[t][j]) {
                out.first[t] = out.first[t] == none ? j : out.first[t];
                out.last[t] = j;
            }
    }
    return out;
}

void lattice_library(benchmark::State& state)
{
    lattice l = data();
    for (auto _ : state) {
        benchmark::DoNotOptimize(l);
        auto v = by_lattice(l);
        benchmark::DoNotOptimize(v);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nodes * (nodes + 1) / 2));
}

void lattice_hand(benchmark::State& state)
{
    lattice l = data();
    for (auto _ : state) {
        benchmark::DoNotOptimize(l);
        auto v = by_hand(l);
        benchmark::DoNotOptimize(v);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nodes * (nodes + 1) / 2));
}

void dense_library(benchmark::State& state)
{
    lattice l = data();
    static const auto matrix = dense_matrix(l);
    for (auto _ : state) {
        benchmark::DoNotOptimize(l);
        auto v = by_dense(l, matrix);
        benchmark::DoNotOptimize(v);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(nodes * (nodes + 1) / 2));
}

BENCHMARK(lattice_library);
BENCHMARK(lattice_hand);
BENCHMARK(dense_library);

} // namespace

int main(int argc, char** argv)
{
    const lattice& l = data();
    const answer a = by_lattice(l), b = by_hand(l), c = by_dense(l, dense_matrix(l));
    if (!(a == b) || !(a == c)) {
        std::fprintf(stderr, "snell_benchmark: the fairness check failed: %.17g, %.17g, %.17g\n", a.value, b.value, c.value);
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
