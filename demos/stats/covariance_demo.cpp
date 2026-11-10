//
//  covariance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/covariance.hpp>: a short history of many assets, its sample
//  covariance and Ledoit and Wolf's shrunk estimate. Exits non-zero if a property documented
//  does not hold.
//

#include <snicholls/core/stats/covariance.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core covariance demo\n");
    namespace stats = snicholls::stats;
    bool ok = true;

    // 40 days of 20 assets: a short history for so many
    stats::wyrand g(3);
    std::vector<std::array<double, 20>> returns(40);
    for (auto& day : returns) {
        const double market = stats::uniform_real(g) - 0.5;
        for (double& r : day)
            r = 0.01 * (market + stats::uniform_real(g) - 0.5);
    }

    //! [covariance]
    // The sample estimate is noisy with 40 dates and 20 assets; Ledoit and Wolf shrink it towards the
    // average variance by the amount that minimises the expected error.
    const stats::covariance_matrix sample = stats::sample_covariance(returns);
    const stats::covariance_matrix shrunk = stats::ledoit_wolf(returns);
    //! [covariance]

    std::printf("  shrinkage %.3f towards %.2e; cov(0, 1): sample %.2e, shrunk %.2e\n", shrunk.shrinkage, shrunk.target, sample(0, 1), shrunk(0, 1));
    ok &= shrunk.shrinkage > 0 && shrunk.shrinkage < 1 && shrunk(0, 1) == (1 - shrunk.shrinkage) * sample(0, 1);

    std::printf(ok ? "covariance demo: ok\n" : "covariance demo: FAILED\n");
    return ok ? 0 : 1;
}
