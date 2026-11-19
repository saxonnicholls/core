//
//  collapse_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/collapse.hpp>: healthy and collapsed embeddings told apart by their per-dimension
//  standard deviation and effective rank, two workers' moments merged; VICReg's variance and covariance terms; SIGReg
//  over random directions, its sums merged from two workers, and a few steps of its gradient spreading a collapsed
//  batch. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/collapse.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core collapse demo\n");
    namespace ml = snicholls::ml;
    namespace stats = snicholls::stats;
    const auto& backend = snicholls::linalg::reference;
    bool ok = true;
    constexpr std::size_t d = 4, n = 256;
    stats::wyrand g(7);
    const stats::normal_inverse_cdf normal{};
    std::vector<double> healthy(n * d), collapsed(n * d);
    for (std::size_t r = 0; r < n; ++r) {
        const double z = normal(g);
        for (std::size_t c = 0; c < d; ++c) {
            healthy[r * d + c] = normal(g);
            collapsed[r * d + c] = z * double(c + 1) * 0.01;   // one direction, and tiny
        }
    }

    //! [moments]
    // Two workers accumulate half the rows each and merge; then the spread per dimension and the effective rank
    std::vector<double> mean_a(d), m_a(d * d), mean_b(d), m_b(d * d), sd(d), var(d), cov(d * d), work(2 * d * d + d);
    ml::embedding_moments<double> a{mean_a, m_a}, b{mean_b, m_b};
    a.add_rows(std::span<const double>(healthy).first(n / 2 * d));
    for (std::size_t r = n / 2; r < n; ++r)
        b.add(std::span<const double>(healthy).subspan(r * d, d));
    a.merge(b);
    a.standard_deviation(sd);
    a.variance(var);
    a.covariance(cov);
    const double rank = *a.effective_rank(backend, std::span<double>(work));
    //! [moments]
    std::vector<double> mean_c(d), m_c(d * d), sd_c(d);
    ml::embedding_moments<double> c{mean_c, m_c};
    c.add_rows(collapsed);
    c.standard_deviation(sd_c);
    const double rank_c = *c.effective_rank(backend, std::span<double>(work));
    std::printf("  healthy: smallest std %.3f, effective rank %.2f of %zu; collapsed: largest std %.3f, rank %.2f\n",
                *std::min_element(sd.begin(), sd.end()), rank, d, *std::max_element(sd_c.begin(), sd_c.end()), rank_c);
    ok &= *std::min_element(sd.begin(), sd.end()) > 0.8 && rank > 3.8 && rank_c < 1.01 && *std::max_element(sd_c.begin(), sd_c.end()) < 0.05;
    ok &= var[0] == cov[0];

    // VICReg's terms: the variance hinge fires for the collapsed batch, the covariance term for its correlations
    std::vector<double> vwork(2 * d), cwork(d + d * d);
    const double v_h = ml::vicreg_variance_rows(backend, std::span<const double>(healthy), d, std::span<double>{}, std::span<double>(vwork));
    const double v_c = ml::vicreg_variance_rows(backend, std::span<const double>(collapsed), d, std::span<double>{}, std::span<double>(vwork));
    const double c_h = ml::vicreg_covariance_rows(backend, std::span<const double>(healthy), d, std::span<double>{}, std::span<double>(cwork));
    std::printf("  VICReg variance: healthy %.3f, collapsed %.3f; covariance healthy %.4f\n", v_h, v_c, c_h);
    ok &= v_h < 0.1 && v_c > 0.9 && c_h < 0.05;

    //! [sigreg]
    // SIGReg: 32 directions for this step, the statistic per direction averaged; two workers' sums add
    const ml::sigreg_quadrature<double> quad{};
    std::vector<double> directions(32 * d), sums(32 * 17 * 2), other(sums.size());
    ml::sigreg_directions<double>(2026, 0, d, directions);
    ml::sigreg_moments_rows(backend, std::span<const double>(healthy).first(n / 2 * d), d, std::span<const double>(directions), quad,
                            std::span<double>(sums));
    ml::sigreg_moments_rows(backend, std::span<const double>(healthy).subspan(n / 2 * d), d, std::span<const double>(directions), quad,
                            std::span<double>(other));
    for (std::size_t i = 0; i < sums.size(); ++i)
        sums[i] += other[i];   // the all-reduce
    const double healthy_sigreg = ml::sigreg_value(std::span<const double>(sums), n, quad);
    std::vector<double> dx(n * d);
    ml::sigreg_backward_rows(backend, std::span<const double>(healthy), d, std::span<const double>(directions), quad, std::span<const double>(sums), n,
                             std::span<double>(dx));
    //! [sigreg]
    // Descent on SIGReg alone spreads the collapsed batch towards a standard normal
    auto x = collapsed;
    const double before = ml::sigreg_rows(backend, std::span<const double>(x), d, std::span<const double>(directions), quad, std::span<double>(sums), {});
    for (std::uint64_t step = 0; step < 200; ++step) {
        ml::sigreg_directions<double>(2026, step, d, directions);
        std::fill(dx.begin(), dx.end(), 0.0);
        ml::sigreg_rows(backend, std::span<const double>(x), d, std::span<const double>(directions), quad, std::span<double>(sums), std::span<double>(dx));
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] -= 0.5 * dx[i];
    }
    const double after = ml::sigreg_rows(backend, std::span<const double>(x), d, std::span<const double>(directions), quad, std::span<double>(sums), {});
    std::printf("  SIGReg: healthy %.3f; collapsed %.1f, after 200 steps of its gradient %.3f\n", healthy_sigreg, before, after);
    ok &= healthy_sigreg < 4 && before > 50 && after < before / 10;

    std::printf(ok ? "collapse demo: ok\n" : "collapse demo: FAILED\n");
    return ok ? 0 : 1;
}
