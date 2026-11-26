//
//  option_pricing_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  finance::black_scholes, black_scholes_greeks and binomial<64> (American: the value and the exercise boundary) against
//  the same formulas written by hand (the same <cmath> calls and normal distribution functions, the same arithmetic in
//  the same order), over 4,096 options with
//  spot, strike, expiry, rates and volatility drawn at random. main() runs a fairness check first and refuses to
//  benchmark if the two sides disagree on any option by a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/finance/option_pricing.hpp>
#include <snicholls/core/math/constexpr_math.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace finance = snicholls::finance;
namespace math = snicholls::math;
using option = finance::vanilla_option<>;

std::vector<option> make_options()
{
    std::vector<option> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;
    auto uniform = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return static_cast<double>(x >> 11) * 0x1.0p-53; };
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = {.kind = i % 2 ? finance::option_kind::put : finance::option_kind::call, .spot = 50 + 100 * uniform(), .strike = 100,
                  .expiry = 0.05 + 2 * uniform(), .rate = 0.06 * uniform(), .yield = 0.03 * uniform(), .vol = 0.05 + 0.6 * uniform()};
    return out;
}

const std::vector<option>& options()
{
    static const auto o = make_options();
    return o;
}

// ---- By hand ------------------------------------------------------------------------------------------------------
double price_by_hand(const option& o) noexcept
{
    const double sd = o.vol * std::sqrt(o.expiry);
    const double d1 = (std::log(o.spot / o.strike) + (o.rate - o.yield + o.vol * o.vol / 2) * o.expiry) / sd, d2 = d1 - sd;
    const double qdf = std::exp(-o.yield * o.expiry), rdf = std::exp(-o.rate * o.expiry), w = o.kind == finance::option_kind::call ? 1.0 : -1.0;
    return w * (o.spot * qdf * math::normal_cdf(w * d1) - o.strike * rdf * math::normal_cdf(w * d2));
}

finance::greeks<> greeks_by_hand(const option& o) noexcept
{
    const double sd = o.vol * std::sqrt(o.expiry);
    const double d1 = (std::log(o.spot / o.strike) + (o.rate - o.yield + o.vol * o.vol / 2) * o.expiry) / sd, d2 = d1 - sd;
    const double qdf = std::exp(-o.yield * o.expiry), rdf = std::exp(-o.rate * o.expiry), w = o.kind == finance::option_kind::call ? 1.0 : -1.0;
    const double density = math::normal_pdf(d1), nd1 = math::normal_cdf(w * d1), nd2 = math::normal_cdf(w * d2);
    const double vega = o.spot * qdf * density * std::sqrt(o.expiry);
    return {w * qdf * nd1,
            qdf * density / (o.spot * sd),
            vega,
            -qdf * density * d2 / o.vol,
            vega * d1 * d2 / o.vol,
            -o.spot * qdf * density * o.vol / (2 * std::sqrt(o.expiry)) - w * o.rate * o.strike * rdf * nd2 + w * o.yield * o.spot * qdf * nd1,
            w * o.strike * o.expiry * rdf * nd2};
}

constexpr std::size_t steps = 64;

// The value and the early-exercise boundary, as binomial<64> gives them
struct tree_answer {
    double value = 0;
    std::array<double, 64> boundary{};
    bool operator==(const tree_answer&) const = default;
};

tree_answer binomial_by_hand(const option& o) noexcept
{
    const double dt = o.expiry / steps, jump = o.vol * std::sqrt(dt);
    const double up = std::exp(jump), down = std::exp(-jump), p = (std::exp((o.rate - o.yield) * dt) - down) / (up - down);
    const double discount = std::exp(-o.rate * dt), w = o.kind == finance::option_kind::call ? 1.0 : -1.0;
    std::array<double, 2 * steps + 1> level{};
    for (std::size_t k = 0; k <= 2 * steps; ++k)
        level[k] = o.spot * std::exp(jump * (static_cast<double>(k) - static_cast<double>(steps)));
    const auto exercise = [&](std::size_t t, std::size_t j) {
        const double x = w * (level[steps + 2 * j - t] - o.strike);
        return x > 0 ? x : 0.0;
    };
    tree_answer out;
    std::array<double, steps + 1> row{};
    for (std::size_t j = 0; j <= steps; ++j)
        row[j] = exercise(steps, j);
    for (std::size_t t = steps; t-- > 0;) {
        out.boundary[t] = 0;
        for (std::size_t j = 0; j <= t; ++j) {
            double expected = 0;
            expected += (1 - p) * row[j];
            expected += p * row[j + 1];
            const double continuation = discount * expected, stop = exercise(t, j);
            const double spot = level[steps + 2 * j - t];
            if (stop > continuation && (w < 0 || out.boundary[t] == 0))   // a put's highest exercising node, a call's lowest
                out.boundary[t] = spot;
            row[j] = stop > continuation ? stop : continuation;
        }
    }
    out.value = row[0];
    return out;
}

// ---- The groups ----------------------------------------------------------------------------------------------------
template <typename F>
void run(benchmark::State& state, F f)
{
    for (auto _ : state)
        for (const option& o : options()) {
            auto v = f(o);
            benchmark::DoNotOptimize(v);
        }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(options().size()));
}

void black_scholes_library(benchmark::State& state) { run(state, [](const option& o) { return finance::black_scholes(o); }); }
void black_scholes_hand(benchmark::State& state) { run(state, price_by_hand); }
void greeks_library(benchmark::State& state) { run(state, [](const option& o) { return finance::black_scholes_greeks(o); }); }
void greeks_hand(benchmark::State& state) { run(state, greeks_by_hand); }
tree_answer binomial_by_library(const option& o) noexcept
{
    const auto b = finance::binomial<steps>(o);
    return {b.value, b.boundary};
}

void binomial_64_library(benchmark::State& state) { run(state, binomial_by_library); }
void binomial_64_hand(benchmark::State& state) { run(state, binomial_by_hand); }

BENCHMARK(black_scholes_library);
BENCHMARK(black_scholes_hand);
BENCHMARK(greeks_library);
BENCHMARK(greeks_hand);
BENCHMARK(binomial_64_library);
BENCHMARK(binomial_64_hand);

bool same(const finance::greeks<>& a, const finance::greeks<>& b)
{
    return a.delta == b.delta && a.gamma == b.gamma && a.vega == b.vega && a.vanna == b.vanna && a.volga == b.volga && a.theta == b.theta
           && a.rho == b.rho;
}

} // namespace

int main(int argc, char** argv)
{
    for (const option& o : options())
        if (finance::black_scholes(o) != price_by_hand(o) || !same(finance::black_scholes_greeks(o), greeks_by_hand(o))
            || !(binomial_by_library(o) == binomial_by_hand(o))) {
            std::fprintf(stderr, "option_pricing_benchmark: the fairness check failed: the two sides disagree\n");
            return 1;
        }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
