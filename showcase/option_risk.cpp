//
//  option_risk.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase: one put option priced four ways that must agree. Black-Scholes in closed form and a binomial lattice
//  solved by its Snell envelope (American and European, with the early-exercise boundary) are computed by the
//  compiler and checked with static_assert. At run time, Monte Carlo with a seeded wyrand stream gets its error bar
//  from a bootstrap of the payoffs, and quasi-Monte Carlo with randomly shifted Sobol copies gets a tighter one from
//  the spread of the copies. Every estimate is coloured by how far it lies from the closed form. Exits non-zero if
//  any figure is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/functional/tabulate.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stats/accumulators.hpp>
#include <snicholls/core/stats/bootstrap.hpp>
#include <snicholls/core/stats/low_discrepancy.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/stochastic/snell.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

namespace risk {

using namespace snicholls;
namespace math = snicholls::math;

// ---- The contract and the market --------------------------------------------------------------------------------
constexpr double spot = 100, strike = 100, rate = 0.05, vol = 0.20, expiry = 1.0;

struct price {   // a premium, in dollars
    double value;
};
constexpr auto attributes(const price&) { return attrs(colours::cyan, number_format{"{:.4f}"}); }
struct miss {   // distance from the closed form, in dollars: green within 2 cents, amber within 10, red beyond
    double value;
};
constexpr auto attributes(const miss&)
{
    return attrs(threshold_colours{colours::green, {{0.02, colours::yellow}, {0.10, colours::red}}}, number_format{"{:.4f}"});
}

// ---- 1. Black-Scholes, by the compiler ------------------------------------------------------------------------------
constexpr double black_scholes_put = [] {
    const double sd = vol * math::sqrt(expiry), d1 = (math::log(spot / strike) + (rate + vol * vol / 2) * expiry) / sd;
    return strike * math::exp(-rate * expiry) * math::normal_cdf(sd - d1) - spot * math::normal_cdf(-d1);
}();
static_assert(black_scholes_put > 5.5734 && black_scholes_put < 5.5736);   // the textbook 5.5735

// ---- 2. A binomial lattice as a Markov chain, solved by its Snell envelope, by the compiler -------------------------
constexpr std::size_t steps = 24, levels = 2 * steps + 1;   // level k is (k - steps) up-moves net
constexpr double dt = expiry / steps, up = math::exp(vol * math::sqrt(dt)), p_up = (math::exp(rate * dt) - 1 / up) / (up - 1 / up);
constexpr auto level_price = fn::tabulate<levels>([](std::size_t k) { return spot * math::pow(up, static_cast<double>(k) - steps); });

constexpr auto lattice = [] {
    std::array<std::array<double, levels>, levels> p{};
    for (std::size_t k = 0; k < levels; ++k) {
        p[k][k < levels - 1 ? k + 1 : k] += p_up;   // the edges are never reached before expiry
        p[k][k > 0 ? k - 1 : k] += 1 - p_up;
    }
    return p;
}();
constexpr auto exercise = [](std::size_t, std::size_t k) { return std::max(0.0, strike - level_price(k)); };
constexpr auto at_expiry_only = [](std::size_t t, std::size_t k) {
    return t == steps ? std::max(0.0, strike - level_price(k)) : std::numeric_limits<double>::lowest();
};
constexpr auto american = stochastic::snell_envelope<levels, steps>(lattice, exercise, math::exp(-rate * dt));
constexpr auto european = stochastic::snell_envelope<levels, steps>(lattice, at_expiry_only, math::exp(-rate * dt));
constexpr double american_put = american.value[0][steps], european_put = european.value[0][steps];

static_assert(math::abs(european_put - black_scholes_put) < 0.10, "24 steps agree with the closed form to 10 cents; the error falls as 1/steps");
static_assert(american_put > european_put, "the right to exercise early is worth something for a put");
static_assert(american.stops(steps - 1, steps - 10) && !american.stops(0, steps), "deep in the money, exercise; at the money, wait");

// The early-exercise boundary: at each date, the highest price at which exercising beats waiting.
constexpr auto boundary = fn::tabulate<steps>([](std::size_t t) {
    double b = 0;
    for (std::size_t k = 0; k < levels; ++k)
        b = american.stops(t, k) && american.payoff[t][k] > 0 ? std::max(b, level_price(k)) : b;
    return b;
});

} // namespace risk

int main()
{
    using namespace risk;
    namespace st = snicholls::stats;
    bool ok = true;
    const auto expect = [&](bool good, std::string_view what) {
        acout{} << "  " << styled(good ? colours::green : colours::red, good ? "ok  " : "FAIL") << ' ' << what << '\n';
        ok &= good;
    };
    println(colours::cyan, "A one-year at-the-money put: S = K = 100, r = 5%, vol = 20%");

    println("\n1-2. By the compiler");
    acout{} << "  Black-Scholes        " << price{black_scholes_put} << '\n';
    acout{} << "  lattice, European    " << price{european_put} << "  off by " << miss{std::abs(european_put - black_scholes_put)} << '\n';
    acout{} << "  lattice, American    " << price{american_put} << "  early exercise is worth " << price{american_put - european_put} << '\n';
    acout{} << "  exercise below:";
    for (std::size_t t = 0; t < steps; t += 4)
        acout{} << "  month " << t / 2 << ": " << price{boundary(t)};
    acout{} << '\n';
    expect(boundary(steps - 1) > boundary(0), "the boundary rises towards expiry, as theory says");

    // The payoff along one terminal draw, shared by both simulations
    const auto put_payoff = [](double z) {
        const double terminal = spot * std::exp((rate - vol * vol / 2) * expiry + vol * std::sqrt(expiry) * z);
        return std::exp(-rate * expiry) * std::max(0.0, strike - terminal);
    };

    println("\n3. Monte Carlo: 20,000 wyrand draws, the error bar by bootstrap");
    st::wyrand g(2026);
    st::normal_inverse_cdf normal;
    std::vector<double> payoffs(20'000);
    for (double& v : payoffs)
        v = put_payoff(normal(g));
    const auto mean = [](const st::weighted_moments& m) { return m.mean; };
    const auto boot = st::bootstrap_moments(payoffs, 2'000, g, mean);
    const auto [low, high] = boot.percentile_interval(0.99);
    acout{} << "  estimate " << price{boot.estimate} << " +/- " << price{boot.standard_error()} << ", 99% interval [" << price{low} << ", "
            << price{high} << "]  off by " << miss{std::abs(boot.estimate - black_scholes_put)} << '\n';
    expect(low < black_scholes_put && black_scholes_put < high, "the closed form lies inside the bootstrap's 99% interval");

    println("\n4. Quasi-Monte Carlo: 8 shifted Sobol copies of 4,096 points");
    constexpr int copies = 8, points = 1 << 12;
    st::variance<double> across;
    for (int c = 0; c < copies; ++c) {
        st::sobol<1> sobol(g);   // a random shift: each copy balanced, together an error bar
        double sum = 0;
        for (int i = 0; i < points; ++i)
            sum += put_payoff(math::normal_quantile(sobol.next()[0] + 0x1.0p-33));   // the centre of its 2^-32 cell
        across.add(sum / points);
    }
    const double qmc = across.mean(), qmc_error = std::sqrt(across.value() / copies);
    acout{} << "  estimate " << price{qmc} << " +/- " << price{qmc_error} << "  off by " << miss{std::abs(qmc - black_scholes_put)} << '\n';
    expect(std::abs(qmc - black_scholes_put) < 4 * qmc_error, "the closed form lies within four of its standard errors");
    expect(qmc_error < boot.standard_error() / 5, "with 1.6 times the points, an error bar over five times tighter");

    acout{} << '\n';
    println(ok ? colours::green : colours::red, "option risk: {}", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
