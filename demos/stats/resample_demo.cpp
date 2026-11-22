//
//  resample_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/resample.hpp>: the resampling step of a particle filter. Log weights normalised
//  without underflow, the effective sample size deciding when to resample, and the four schemes (systematic,
//  stratified, multinomial, residual with each remainder method). Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/stats/resample.hpp>

#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <span>

namespace {

namespace st = snicholls::stats;

int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

template <std::size_t M>
std::array<std::size_t, 5> counts(const std::array<std::size_t, M>& idx)
{
    std::array<std::size_t, 5> c{};
    for (const std::size_t i : idx)
        ++c[i];
    return c;
}

} // namespace

int main()
{
    std::printf("resample: the resampling step of particle filters\n");

    //! [resample]
    // Five particles after a measurement update, as log-likelihoods far below exp's range
    std::array<double, 5> log_w{-1200.0, -1201.0, -1199.5, -1230.0, -1200.2};
    const double log_evidence = st::normalise_log_weights<double>(log_w);   // now Σ exp = 1
    std::array<double, 5> w{};
    for (std::size_t i = 0; i < 5; ++i)
        w[i] = std::exp(log_w[i]);

    // Resample (10 new particles from the 5) only when the effective sample size is below 3/4 of the count
    const double ess = st::effective_sample_size<double>(w);   // about 3.25
    std::array<std::size_t, 10> picked{};
    st::wyrand g(2026);   // the caller's generator: the same draw on every platform
    bool resampled = false;
    if (st::needs_resampling<double>(w, 0.75))
        resampled = st::systematic_resample<double>(w, picked, g);
    // picked holds 10 indices in non-decreasing order; particle 3 (weight 2e-14) is never copied
    //! [resample]
    double sum = 0;
    for (const double x : w)
        sum += x;
    std::printf("  log evidence %.4f, ESS %.3f of 5\n", log_evidence, ess);
    expect(std::abs(sum - 1) < 1e-12 && log_evidence < -1198 && log_evidence > -1200, "normalise_log_weights: weights sum to 1, the constant returned");
    expect(std::abs(st::log_sum_exp<double>(log_w)) < 1e-12, "log_sum_exp of the normalised weights is 0");
    expect(ess > 1 && ess < 5 && st::needs_resampling<double>(w, 0.9) && !st::needs_resampling<double>(w, 0.1), "effective_sample_size and needs_resampling");
    const auto sc = counts(picked);
    std::printf("  systematic copies:  %zu %zu %zu %zu %zu\n", sc[0], sc[1], sc[2], sc[3], sc[4]);
    bool floor_or_ceil = std::is_sorted(picked.begin(), picked.end());
    for (std::size_t j = 0; j < 5; ++j)
        floor_or_ceil = floor_or_ceil && (sc[j] == static_cast<std::size_t>(std::floor(10 * w[j])) || sc[j] == static_cast<std::size_t>(std::ceil(10 * w[j])));
    expect(resampled && floor_or_ceil && sc[3] == 0, "systematic: sorted, each particle floor or ceil of 10 w copies");

    std::array<std::size_t, 10> strat{}, multi{};
    expect(st::stratified_resample<double>(w, strat, g) && std::is_sorted(strat.begin(), strat.end()), "stratified: one uniform per stratum, sorted");
    expect(st::multinomial_resample<double>(w, multi, g) && std::is_sorted(multi.begin(), multi.end()), "multinomial: independent draws, generated sorted");

    // Residual: floor(M w) copies deterministically, the rest by the chosen method on the residual weights
    const std::array<double, 3> rw{5, 3, 2};   // M = 7: 3.5, 2.1, 1.4 expected, so 3, 2, 1 copies and one draw
    std::array<double, 3> residual{};
    bool residual_ok = true;
    for (const auto method : {st::residual_method::systematic, st::residual_method::stratified, st::residual_method::multinomial}) {
        std::array<std::size_t, 7> out{};
        residual_ok = residual_ok && st::residual_resample<double>(rw, residual, out, g, method) && out[0] == 0 && out[2] == 0 && out[3] == 1 && out[4] == 1
                      && out[5] == 2 && out[6] < 3;
    }
    expect(residual_ok, "residual: 0 0 0 1 1 2 then one draw, under each remainder method");

    // No draw is possible without a usable total weight
    const std::array<double, 2> none{0, 0};
    std::array<std::size_t, 3> out{};
    expect(!st::systematic_resample<double>(none, out, g) && !st::residual_resample<double>(none, residual, out, g), "zero total weight: false, nothing drawn");
    expect(st::log_sum_exp<double>(std::span<const double>{}) == -std::numeric_limits<double>::infinity(), "log_sum_exp of nothing is minus infinity");

    std::printf(failures == 0 ? "resample demo: ok\n" : "resample demo: FAILED\n");
    return failures == 0 ? 0 : 1;
}
