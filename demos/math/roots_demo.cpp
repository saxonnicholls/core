//
//  roots_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/math/roots.hpp>: Kepler's equation solved by the compiler, an implied
//  volatility found by Brent's method and by Newton's with the vega, bisection for comparison, a
//  bracket found outward from a guess, and the statuses a search can end with. Exits non-zero if a value
//  is not the one documented.
//

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/math/roots.hpp>

#include <cstdio>
#include <utility>

namespace {

namespace math = snicholls::math;

// Black-Scholes: a call on S at strike K, expiry t years, rate r, volatility sigma; and its vega.
constexpr double bs_call(double S, double K, double t, double r, double sigma)
{
    const double sd = sigma * math::sqrt(t);
    const double d1 = (math::log(S / K) + (r + sigma * sigma / 2) * t) / sd;
    return S * math::normal_cdf(d1) - K * math::exp(-r * t) * math::normal_cdf(d1 - sd);
}
constexpr double bs_vega(double S, double K, double t, double r, double sigma)
{
    const double d1 = (math::log(S / K) + (r + sigma * sigma / 2) * t) / (sigma * math::sqrt(t));
    return S * math::normal_pdf(d1) * math::sqrt(t);
}

} // namespace

int main()
{
    std::printf("snicholls-core roots demo\n");
    bool ok = true;

    //! [roots]
    // Kepler's equation M = E - e sin E: the eccentric anomaly of an orbit with e = 0.5 at mean anomaly 1,
    // solved by the compiler.
    constexpr auto kepler = math::brent([](double E) { return E - 0.5 * math::sin(E) - 1.0; }, 0.0, math::pi);
    static_assert(kepler.status == math::root_status::converged);
    static_assert(math::near(kepler.x, 1.4987011335178483, 1e-15));

    // The implied volatility of an at-the-money one-year call priced 10.450583572185565 (S = K = 100, r = 5%),
    // which is the Black-Scholes price at 20% volatility.
    constexpr double price = 10.450583572185565;
    const auto excess = [&](double sigma) { return bs_call(100, 100, 1, 0.05, sigma) - price; };
    const auto iv = math::brent(excess, 0.01, 3.0, 1e-14);

    // With the vega at hand, Newton's step, kept inside the bracket.
    const auto with_vega = [&](double sigma) { return std::pair{excess(sigma), bs_vega(100, 100, 1, 0.05, sigma)}; };
    const auto iv_newton = math::newton_safeguarded(with_vega, 0.01, 3.0, 0.5, 1e-14);
    //! [roots]

    std::printf("  Kepler E = %.16f in %d evaluations (at compile time)\n", kepler.x, kepler.iterations);
    std::printf("  implied vol: Brent %.15f in %d evaluations, Newton %.15f in %d\n", iv.x, iv.iterations, iv_newton.x,
                iv_newton.iterations);
    ok &= iv.status == math::root_status::converged && math::near(iv.x, 0.2, 1e-13);
    ok &= iv_newton.status == math::root_status::converged && math::near(iv_newton.x, 0.2, 1e-13);

    // Bisection: the same answer, one bit per evaluation.
    const auto iv_bisect = math::bisect(excess, 0.01, 3.0, 1e-14);
    std::printf("  bisection %.15f in %d evaluations\n", iv_bisect.x, iv_bisect.iterations);
    ok &= math::near(iv_bisect.x, 0.2, 1e-13) && iv.iterations < iv_bisect.iterations;

    // A bracket found outward from a guess of 20% for a deep out-of-the-money call priced at 0.5.
    const auto otm = [](double sigma) { return bs_call(100, 150, 0.25, 0.05, sigma) - 0.5; };
    const auto b = math::bracket(otm, 0.2, 0.02);
    const auto iv_otm = math::brent(otm, b.lo, b.hi);
    std::printf("  bracket [%.4f, %.4f] after %d expansions; implied vol %.10f\n", b.lo, b.hi, b.iterations, iv_otm.x);
    ok &= b.status == math::root_status::converged && iv_otm.status == math::root_status::converged;
    ok &= math::near(bs_call(100, 150, 0.25, 0.05, iv_otm.x), 0.5, 1e-12);

    // The other endings: a price below intrinsic value has no volatility, and a budget can run out.
    const auto none = math::brent([](double sigma) { return bs_call(100, 50, 1, 0.05, sigma) - 1.0; }, 0.01, 3.0);
    const auto short_budget = math::brent(excess, 0.01, 3.0, 0.0, 3);
    const auto no_bracket = math::bracket([](double x) { return x * x + 1; }, 0.0, 1.0, 1.6, 10);
    std::printf("  no root: not_bracketed %d; three evaluations: max_iterations %d; x^2 + 1: no bracket %d\n",
                none.status == math::root_status::not_bracketed, short_budget.status == math::root_status::max_iterations,
                no_bracket.status == math::root_status::max_iterations);
    ok &= none.status == math::root_status::not_bracketed && short_budget.status == math::root_status::max_iterations;
    ok &= no_bracket.status == math::root_status::max_iterations;
    ok &= math::bracket(otm, 0.5, 0.0).status == math::root_status::not_bracketed;

    std::printf(ok ? "roots demo: ok\n" : "roots demo: FAILED\n");
    return ok ? 0 : 1;
}
