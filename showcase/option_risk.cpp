//
//  option_risk.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase: one put option priced four ways that must agree. Black-Scholes in closed form and a binomial lattice
//  solved by its Snell envelope (finance/option_pricing.hpp: American and European, with the early-exercise boundary)
//  are computed by the compiler and checked with static_assert. At run time, Monte Carlo with a seeded wyrand stream gets its error bar
//  from a bootstrap of the payoffs, and quasi-Monte Carlo with randomly shifted Sobol copies gets a tighter one from
//  the spread of the copies. Every estimate is coloured by how far it lies from the closed form. Exits non-zero if
//  any figure is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/finance/option_pricing.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stats/accumulators.hpp>
#include <snicholls/core/stats/bootstrap.hpp>
#include <snicholls/core/stats/low_discrepancy.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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
constexpr finance::vanilla_option<double> put{
    .kind = finance::option_kind::put, .spot = spot, .strike = strike, .expiry = expiry, .rate = rate, .vol = vol};
constexpr double black_scholes_put = finance::black_scholes(put);
static_assert(black_scholes_put > 5.5734 && black_scholes_put < 5.5736);   // the textbook 5.5735

// ---- 2. A binomial lattice solved by its Snell envelope, by the compiler ---------------------------------------------
// A 24-step Cox-Ross-Rubinstein tree, each step one stencil over one row (stochastic::lattice_envelope), with the
// early-exercise boundary: at each date, the highest spot at which exercising beats waiting.
constexpr std::size_t steps = 24;
constexpr auto american = finance::binomial<steps>(put);
constexpr auto european = finance::binomial<steps>(put, finance::exercise_style::european);
constexpr double american_put = american.value, european_put = european.value;

static_assert(math::abs(european_put - black_scholes_put) < 0.10, "24 steps agree with the closed form to 10 cents; the error falls as 1/steps");
static_assert(american_put > european_put, "the right to exercise early is worth something for a put");
static_assert(american.boundary[steps - 1] > 0 && american.boundary[0] == 0, "deep in the money near expiry, exercise; at the money today, wait");

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
    for (std::size_t t = 6; t < steps; t += 4)
        acout{} << "  month " << t / 2 << ": " << price{american.boundary[t]};
    acout{} << '\n';
    expect(american.boundary[steps - 1] > american.boundary[6], "the boundary rises towards expiry, as theory says");

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
