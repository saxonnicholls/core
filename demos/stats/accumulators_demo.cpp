//
//  accumulators_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/accumulators.hpp>: live statistics of a return stream, tick by
//  tick, and two workers' results merged. Exits non-zero if a statistic is not the one documented.
//

#include <snicholls/core/stats/accumulators.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core accumulators demo\n");
    namespace stats = snicholls::stats;
    bool ok = true;
    const std::vector<double> portfolio{0.010, -0.004, 0.006, 0.002, -0.008, 0.012};
    const std::vector<double> index{0.008, -0.002, 0.004, 0.001, -0.006, 0.009};

    //! [accumulators]
    // Each tick updates the statistics in constant time: no history kept, nothing recomputed.
    stats::variance<double> returns;
    stats::covariance<double> against_index;
    stats::ewma<double> risk{.alpha = 0.06};   // RiskMetrics' lambda of 0.94
    stats::maximum<double> best;
    for (std::size_t t = 0; t < portfolio.size(); ++t) {
        returns.add(portfolio[t]);
        against_index.add(index[t], portfolio[t]);   // x the index, y the portfolio: beta is y on x
        risk.add(portfolio[t]);
        best.add(portfolio[t]);
    }

    // Two workers' accumulators merge into what one would have seen.
    stats::variance<double> first_half, second_half;
    for (std::size_t t = 0; t < 3; ++t)
        first_half.add(portfolio[t]);
    for (std::size_t t = 3; t < 6; ++t)
        second_half.add(portfolio[t]);
    first_half.merge(second_half);
    //! [accumulators]

    std::printf("  mean %.4f, volatility %.4f, beta %.2f, correlation %.2f, ewma volatility %.4f, best %.3f\n", returns.mean(),
                returns.standard_deviation(), against_index.beta(), against_index.correlation(), risk.volatility(), *best.value());
    ok &= std::abs(first_half.value() - returns.value()) < 1e-15 && *best.value() == 0.012 && std::abs(against_index.beta() - 1.3465) < 1e-4;

    std::printf(ok ? "accumulators demo: ok\n" : "accumulators demo: FAILED\n");
    return ok ? 0 : 1;
}
