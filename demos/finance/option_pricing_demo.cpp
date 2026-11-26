//
//  option_pricing_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/option_pricing.hpp>: a call and a put priced by Black-Scholes-Merton with their
//  greeks, both by the compiler, then the same put American on a Cox-Ross-Rubinstein tree with its early-exercise
//  boundary. Exits non-zero if any figure is not the one documented.
//

#include <snicholls/core/finance/option_pricing.hpp>

#include <snicholls/core/math/constexpr_math.hpp>

#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core option pricing demo\n");
    namespace finance = snicholls::finance;
    namespace math = snicholls::math;
    bool ok = true;

    //! [black_scholes]
    // A three-month call on an index yielding 2%: S 100, K 105, r 4%, vol 25%. Designated members, any floating type.
    constexpr finance::vanilla_option<double> call{.kind = finance::option_kind::call, .spot = 100, .strike = 105, .expiry = 0.25,
                                                   .rate = 0.04, .yield = 0.02, .vol = 0.25};
    constexpr double premium = finance::black_scholes(call);
    constexpr finance::greeks<> risk = finance::black_scholes_greeks(call);   // delta, gamma, vega, vanna, volga, theta, rho

    // The put on the same terms, and put-call parity, proved by the compiler
    constexpr finance::vanilla_option<double> put{.kind = finance::option_kind::put, .spot = 100, .strike = 105, .expiry = 0.25,
                                                  .rate = 0.04, .yield = 0.02, .vol = 0.25};
    static_assert(math::near(premium - finance::black_scholes(put), 100 * math::exp(-0.02 * 0.25) - 105 * math::exp(-0.04 * 0.25), 1e-12));
    static_assert(risk.delta > 0 && risk.delta < 1 && risk.gamma > 0 && risk.vega > 0 && risk.theta < 0);
    //! [black_scholes]

    std::printf("  call %.4f: delta %.4f gamma %.5f vega %.3f vanna %.4f volga %.3f theta %.3f rho %.3f\n", premium, risk.delta, risk.gamma, risk.vega,
                risk.vanna, risk.volga, risk.theta, risk.rho);
    const auto put_risk = finance::black_scholes_greeks(put);
    std::printf("  put  %.4f: delta %.4f, the call's minus e^{-qT}\n", finance::black_scholes(put), put_risk.delta);
    ok &= premium > 2.9 && premium < 3.3 && math::near(risk.delta - put_risk.delta, math::exp(-0.02 * 0.25), 1e-12);
    ok &= finance::black_scholes(put) > premium && call.sign() == 1 && put.sign() == -1 && put.intrinsic(90) == 15;

    //! [binomial]
    // The put again, American, on a 100-step CRR tree solved by its Snell envelope: the price, delta and gamma off
    // the tree, and the critical spot below which to exercise at each step. O(steps) state, by the compiler.
    constexpr auto american = finance::binomial<100>(put);
    constexpr auto european = finance::binomial<100>(put, finance::exercise_style::european);
    static_assert(american.value > european.value, "the right to exercise early is worth something");
    static_assert(math::abs(european.value - finance::black_scholes(put)) < 0.02, "100 steps are within 2 cents of the closed form");
    static_assert(american.boundary[99] > american.boundary[50] && american.boundary[99] < put.strike, "the boundary rises to the strike");
    //! [binomial]

    std::printf("  American put %.4f (European %.4f, closed form %.4f), delta %.4f gamma %.5f\n", american.value, european.value,
                finance::black_scholes(put), american.delta, american.gamma);
    std::printf("  exercise below:");
    for (std::size_t t = 0; t < 100; t += 20)
        std::printf("  step %zu %.2f", t, american.boundary[t]);
    std::printf("  step 99 %.2f\n", american.boundary[99]);
    ok &= math::abs(american.delta - put_risk.delta) < 0.1 && european.boundary[99] == 0;

    std::printf(ok ? "option pricing demo: ok\n" : "option pricing demo: FAILED\n");
    return ok ? 0 : 1;
}
