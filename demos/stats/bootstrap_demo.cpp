//
//  bootstrap_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/bootstrap.hpp>: a confidence interval for a strategy's Sharpe
//  ratio by the Poisson bootstrap, the stationary bootstrap for autocorrelated returns, and a
//  median by the ordinary bootstrap. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/stats/bootstrap.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core bootstrap demo\n");
    namespace stats = snicholls::stats;
    bool ok = true;

    // A year of daily returns with a little drift
    stats::wyrand market(1);
    std::vector<double> returns(252);
    for (double& r : returns)
        r = 0.0006 + 0.01 * (stats::uniform_real(market) - 0.5) * 3.46;

    //! [bootstrap]
    // The Sharpe ratio is a function of the mean and the variance: bootstrap it from resampling counts,
    // with no resample ever gathered.
    const auto sharpe = [](const stats::weighted_moments& m) { return m.mean / m.standard_deviation() * std::sqrt(252.0); };
    stats::wyrand g(2026);
    const auto distribution = stats::bootstrap_moments(returns, 10000, g, sharpe);
    const auto [low, high] = distribution.percentile_interval(0.95);

    // Autocorrelated returns: resample in blocks of about 10 days instead of one at a time.
    std::vector<double> block_sharpes;
    stats::for_each_stationary_resample(returns.size(), 10.0, 2000, g, [&](std::span<const std::uint32_t> days) {
        std::vector<std::uint32_t> counts(returns.size());
        for (const std::uint32_t d : days)
            ++counts[d];
        block_sharpes.push_back(sharpe(stats::moments_of(returns, counts)));
    });
    //! [bootstrap]

    std::printf("  Sharpe %.2f, 95%% interval [%.2f, %.2f], standard error %.2f\n", distribution.estimate, low, high,
                distribution.standard_error());
    std::printf("  stationary bootstrap: %zu resamples\n", block_sharpes.size());
    ok &= low < distribution.estimate && distribution.estimate < high && block_sharpes.size() == 2000;

    const auto median = [](std::span<const double> v) {
        std::vector<double> c(v.begin(), v.end());
        std::nth_element(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(c.size() / 2), c.end());
        return c[c.size() / 2];
    };
    const auto medians = stats::bootstrap(returns, 1000, g, median);
    std::printf("  median daily return %.5f +/- %.5f\n", medians.estimate, medians.standard_error());
    ok &= medians.standard_error() > 0;

    std::printf(ok ? "bootstrap demo: ok\n" : "bootstrap demo: FAILED\n");
    return ok ? 0 : 1;
}
