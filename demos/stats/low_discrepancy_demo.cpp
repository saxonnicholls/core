//
//  low_discrepancy_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/low_discrepancy.hpp>: an option-style expectation by Sobol
//  points, with an error bar from randomly shifted copies, against plain Monte Carlo. Exits
//  non-zero if the estimate is not within its documented accuracy.
//

#include <snicholls/core/stats/low_discrepancy.hpp>

#include <array>
#include <cmath>
#include <cstdio>

int main()
{
    std::printf("snicholls-core low_discrepancy demo\n");
    namespace stats = snicholls::stats;
    bool ok = true;

    //! [sobol]
    // E[max(0, average of 4 uniforms - 0.5)], estimated from 2^12 points.
    const auto payoff = [](const std::array<double, 4>& x) { return std::max(0.0, (x[0] + x[1] + x[2] + x[3]) / 4 - 0.5); };
    constexpr int n = 1 << 12, copies = 8;

    // Eight randomly shifted Sobol copies: each is balanced, together they give an error bar.
    stats::wyrand g(7);
    double sum = 0, sum2 = 0;
    for (int c = 0; c < copies; ++c) {
        stats::sobol<4> points(g);
        double estimate = 0;
        for (int i = 0; i < n; ++i)
            estimate += payoff(points.next());
        estimate /= n;
        sum += estimate;
        sum2 += estimate * estimate;
    }
    const double qmc = sum / copies;
    const double qmc_error = std::sqrt((sum2 / copies - qmc * qmc) / (copies - 1));
    //! [sobol]

    // The same number of random points, for comparison
    double mc = 0;
    for (int i = 0; i < n * copies; ++i)
        mc += payoff({stats::uniform_real(g), stats::uniform_real(g), stats::uniform_real(g), stats::uniform_real(g)});
    mc /= n * copies;
    std::printf("  sobol %.6f +/- %.2e   monte carlo %.6f\n", qmc, qmc_error, mc);
    ok &= qmc_error < 2e-4 && std::abs(qmc - mc) < 5e-3;

    std::printf(ok ? "low_discrepancy demo: ok\n" : "low_discrepancy demo: FAILED\n");
    return ok ? 0 : 1;
}
