//
//  mixture_demo.cpp
//  snicholls-core
//
//  Demo for <snicholls/core/tracking/mixture.hpp>: a six-component Gaussian mixture, as a multi-hypothesis filter
//  leaves it, reduced by pruning, merging and truncation; the single moment-matched Gaussian; one merge and its
//  Runnalls cost; and the documented edge cases. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/tracking/mixture.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

namespace {

namespace la = snicholls::linalg;
namespace tr = snicholls::tracking;

using vec2 = la::fixed_vector<double, 2>;
using mat2 = la::fixed_matrix<double, 2, 2>;
using component = tr::weighted_gaussian<double, 2>;

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

constexpr component at(double w, double x, double y) { return {w, {vec2{{x, y}}, mat2{{1, 0, 0, 1}}}}; }

} // namespace

int main()
{
    std::printf("snicholls-core mixture demo\n");
    bool ok = true;

    //! [mixture]
    // Hypotheses about two targets: three near (0, 0), two near (10, 0), one negligible far away; weights sum to 1
    std::array<component, 6> mixture{at(0.30, 0, 0), at(0.15, 0.5, 0.2), at(0.05, -0.4, 0.3),
                                     at(0.30, 10, 0), at(0.1999999995, 10.6, -0.3), at(1e-9 / 2, 50, 50)};
    const auto single = tr::reduce_to_single<double, 2>(mixture);   // one Gaussian for display or a gate

    // How much information one merge would lose (Runnalls' bound on the KL divergence it adds)
    const auto near_cost = tr::runnalls_cost(mixture[0], mixture[1]);
    const auto far_cost = tr::runnalls_cost(mixture[0], mixture[3]);
    const auto merged = tr::merge_pair(mixture[0], mixture[1]);

    // Prune below 1e-9, merge within a squared Mahalanobis distance of 16, keep at most 4 (Stone Soup's defaults
    // but the cap)
    const std::size_t kept = tr::reduce_mixture<double, 2>(mixture, {1e-9, 16, 4});
    //! [mixture]

    std::printf("  single Gaussian: mean (%.4f, %.4f), variance x %.4f\n", single.mean[0], single.mean[1], single.covariance(0, 0));
    ok &= check(std::abs(single.mean[0] - (0.15 * 0.5 + 0.05 * -0.4 + 0.3 * 10 + 0.1999999995 * 10.6 + 5e-10 * 50)) < 1e-9, "reduce_to_single gives the weighted mean");
    ok &= check(single.covariance(0, 0) > 20, "its covariance spans both targets");
    ok &= check(near_cost.is_ok() && far_cost.is_ok() && *near_cost < *far_cost, "merging neighbours costs less than merging the two targets");
    std::printf("  Runnalls cost: neighbours %.4f, the two targets %.4f\n", *near_cost, *far_cost);
    ok &= check(std::abs(merged.weight - 0.45) < 1e-15 && std::abs(merged.state.mean[0] - 0.5 / 3) < 1e-15, "merge_pair adds the weights and matches the mean");
    ok &= check(tr::merge_pair(at(0.7, 0, 0), at(0.6, 1, 0)).weight == 1, "merge_pair caps the weight at 1");

    std::printf("  reduced to %zu components:\n", kept);
    for (std::size_t i = 0; i < kept; ++i)
        std::printf("    weight %.4f at (%.4f, %.4f)\n", mixture[i].weight, mixture[i].state.mean[0], mixture[i].state.mean[1]);
    ok &= check(kept == 2, "reduce_mixture leaves one component per target");
    ok &= check(mixture[0].weight >= mixture[1].weight, "heaviest first");
    ok &= check(std::abs(mixture[0].weight + mixture[1].weight - 1) < 1e-12, "the pruned weight is shared among the rest");
    const double x0 = mixture[0].state.mean[0], x1 = mixture[1].state.mean[0];
    ok &= check((x0 < 1 && x1 > 9) || (x0 > 9 && x1 < 1), "one component sits on each target");

    // The edge cases
    std::array<component, 3> three{at(0.5, 0, 0), at(0.3, 100, 0), at(0.2, 200, 0)};
    ok &= check(tr::reduce_mixture<double, 2>(three, {0, 16, 2}) == 2 && three[1].weight == 0.3, "max_components keeps the heaviest");
    ok &= check(tr::reduce_mixture<double, 2>(std::span<component>{}) == 0, "an empty mixture reduces to nothing");
    const std::array<component, 1> weightless{at(0, 4, 4)};
    ok &= check(tr::reduce_to_single<double, 2>(weightless) == tr::gaussian<double, 2>{}, "no total weight gives the zero Gaussian");
    component flat = at(0.4, 0, 0);
    flat.state.covariance = mat2{};
    std::array<component, 2> degenerate{at(0.6, 0, 0), flat};
    ok &= check(tr::reduce_mixture<double, 2>(degenerate) == 2, "a component without a Cholesky factor is not merged");
    const auto bad = tr::runnalls_cost(at(0.6, 0, 0), flat);
    ok &= check(bad.is_err() && bad.unwrap_err_unchecked() == la::error::not_positive_definite, "runnalls_cost reports it");

    std::printf(ok ? "mixture demo: ok\n" : "mixture demo: FAILED\n");
    return ok ? 0 : 1;
}
