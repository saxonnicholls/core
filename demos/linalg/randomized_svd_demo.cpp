//
//  randomized_svd_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/linalg/randomized_svd.hpp>: the leading factors of a panel of returns (500 days of 120
//  assets driven by 4 factors and noise) by the truncated SVD, against the exact SVD; the variance the factors explain;
//  and the same call with another sketch and another Gaussian sampler. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/linalg/randomized_svd.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core randomized_svd demo\n");
    namespace la = snicholls::linalg;
    namespace st = snicholls::stats;
    using right = la::layout_right;
    bool ok = true;

    // Returns R = F Lᵀ + noise: 500 days, 120 assets, 4 factors of volatility 3%, 2%, 1.5% and 1%, idiosyncratic 0.5%
    constexpr std::size_t days = 500, assets = 120, factors = 4;
    const double vol[factors] = {0.03, 0.02, 0.015, 0.01};
    st::wyrand g(17);
    st::normal_inverse_cdf z;
    std::vector<double> f(days * factors), load(assets * factors), returns(days * assets);
    for (std::size_t t = 0; t < days; ++t)
        for (std::size_t j = 0; j < factors; ++j)
            f[t * factors + j] = vol[j] * z(g);
    for (double& x : load)
        x = z(g);
    for (std::size_t t = 0; t < days; ++t)
        for (std::size_t i = 0; i < assets; ++i) {
            double r = 0.005 * z(g);
            for (std::size_t j = 0; j < factors; ++j)
                r += f[t * factors + j] * load[i * factors + j];
            returns[t * assets + i] = r;
        }

    //! [truncated_svd]
    // The leading k = 4 singular triplets of the 500 x 120 panel: the factors' directions (V, by asset) and their
    // scores (U diag(s), by day). The working space is the caller's; nothing is allocated.
    constexpr std::size_t k = factors;
    std::vector<double> s(k), u(days * k), v(assets * k), work(la::truncated_svd_work_size(days, assets, k));
    const la::matrix_view<const double, right> r(returns.data(), {days, assets});
    const auto done = la::truncated_svd(la::reference, r, std::span<double>(s), la::matrix_view<double, right>(u.data(), {days, k}),
                                        la::matrix_view<double, right>(v.data(), {assets, k}), std::span<double>(work));

    // The randomness is a seam: another sketch (Rademacher ±1), or the Gaussian one with another sampler (Box-Muller)
    std::vector<double> s2(k), s3(k);
    const auto r2 = la::truncated_svd(la::reference, r, std::span<double>(s2), la::matrix_view<double, right>(u.data(), {days, k}),
                                      la::matrix_view<double, right>(v.data(), {assets, k}), std::span<double>(work), {},
                                      la::rademacher_sketch{1});
    const auto r3 = la::truncated_svd(la::reference, r, std::span<double>(s3), la::matrix_view<double, right>(u.data(), {days, k}),
                                      la::matrix_view<double, right>(v.data(), {assets, k}), std::span<double>(work),
                                      {.oversample = 10, .power_iterations = 2}, la::gaussian_sketch<st::normal_box_muller>{2});
    //! [truncated_svd]

    // The exact SVD of the whole panel, for comparison
    std::vector<double> all = returns, sall(assets), vall(assets * assets);
    const auto exact = la::svd(la::reference, la::matrix_view<double, right>(all.data(), {days, assets}), std::span<double>(sall),
                               la::matrix_view<double, right>(vall.data(), {assets, assets}));
    double total = 0, explained = 0;
    for (std::size_t i = 0; i < assets; ++i)
        total += sall[i] * sall[i];
    for (std::size_t i = 0; i < k; ++i)
        explained += s[i] * s[i];

    std::printf("  singular values (truncated / exact):");
    for (std::size_t i = 0; i < k; ++i)
        std::printf("  %.6f / %.6f", s[i], sall[i]);
    std::printf("\n  the 4 factors explain %.1f%% of the variance; the 5th singular value is %.4f\n", 100 * explained / total, sall[k]);
    std::printf("  Rademacher sketch: %.6f ...; Box-Muller sampler: %.6f ...\n", s2[0], s3[0]);
    ok &= done.is_ok() && r2.is_ok() && r3.is_ok() && exact.is_ok();
    for (std::size_t i = 0; i < k; ++i) {
        ok &= std::abs(s[i] - sall[i]) < 1e-6 * sall[0];   // the leading values, as the exact SVD's
        ok &= std::abs(s2[i] - sall[i]) < 1e-6 * sall[0] && std::abs(s3[i] - sall[i]) < 1e-6 * sall[0];
    }
    ok &= explained / total > 0.9 && sall[k] < 0.3 * sall[k - 1];   // four factors, then noise
    std::printf(ok ? "randomized_svd demo: ok\n" : "randomized_svd demo: FAILED\n");
    return ok ? 0 : 1;
}
