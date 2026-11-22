//
//  fusion_demo.cpp
//  snicholls-core
//
//  Demo for <snicholls/core/tracking/fusion.hpp>: a radar track and a camera track of the same target fused by
//  covariance intersection, with ω chosen for the least trace, for the least determinant and within a trust bound;
//  three sources fused with the fast weights, bounded; and each refusal. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/tracking/fusion.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <span>

namespace {

namespace la = snicholls::linalg;
namespace tr = snicholls::tracking;

using vec2 = la::fixed_vector<double, 2>;
using mat2 = la::fixed_matrix<double, 2, 2>;
using g2 = tr::gaussian<double, 2>;

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core fusion demo\n");
    bool ok = true;

    //! [fusion]
    // The radar is good in range (x), poor across it; the camera the other way round. Their errors are correlated
    // by an unknown amount, so information is not simply added
    const g2 radar{vec2{{100.2, 49.0}}, mat2{{1.0, 0.0, 0.0, 9.0}}};
    const g2 camera{vec2{{99.0, 50.1}}, mat2{{6.0, 0.5, 0.5, 1.5}}};

    const double w_trace = tr::optimal_omega(radar, camera).unwrap();   // least mean squared error
    const double w_det = tr::optimal_omega(radar, camera, 0.0, 1.0, tr::ci_criterion::determinant).unwrap();
    const double w_trusted = tr::optimal_omega(radar, camera, 0.0, 0.3).unwrap();   // the radar no more than 0.3
    const auto fused = tr::covariance_intersection(radar, camera, w_trace).unwrap();

    // Three sources: the fast weights ∝ 1 / tr P, the third (a drifting sensor) held to at most 0.2
    const std::array<g2, 3> sources{radar, camera, g2{vec2{{100.5, 50.0}}, mat2{{0.5, 0.0, 0.0, 0.5}}}};
    std::array<double, 3> weights{};
    const std::array<double, 3> at_most{1.0, 1.0, 0.2};
    const bool weighted = tr::fast_ci_weights<double, 2>(sources, weights, {}, at_most);
    const auto fused3 = tr::covariance_intersection<double, 2>(sources, weights);
    //! [fusion]

    std::printf("  omega: trace %.4f, determinant %.4f, trusted %.4f\n", w_trace, w_det, w_trusted);
    std::printf("  fused (%.3f, %.3f), variances %.4f and %.4f\n", fused.mean[0], fused.mean[1], fused.covariance(0, 0), fused.covariance(1, 1));
    ok &= check(w_trace > 0 && w_trace < 1 && w_det > 0 && w_det < 1, "both optima are interior: neither sensor dominates");
    auto trace_at = [&](double w) { return la::trace(tr::covariance_intersection(radar, camera, w).unwrap().covariance); };
    ok &= check(trace_at(w_trace) <= trace_at(w_trace - 0.01) && trace_at(w_trace) <= trace_at(w_trace + 0.01), "the trace is least at the chosen omega");
    ok &= check(w_trusted <= 0.3 && (w_trace <= 0.3 ? std::abs(w_trusted - w_trace) < 1e-6 : std::abs(w_trusted - 0.3) < 1e-6), "the trust bound holds omega");
    ok &= check(la::trace(fused.covariance) < la::trace(radar.covariance) && la::trace(fused.covariance) < la::trace(camera.covariance), "the fused trace is below either input's");
    // Consistency: the fused information is the ω-weighted sum of the inputs'
    const auto lhs = la::spd_inverse(fused.covariance).unwrap();
    const auto rhs = la::spd_inverse(radar.covariance).unwrap() * w_trace + la::spd_inverse(camera.covariance).unwrap() * (1 - w_trace);
    ok &= check(std::abs(lhs(0, 0) - rhs(0, 0)) < 1e-12 && std::abs(lhs(0, 1) - rhs(0, 1)) < 1e-12 && std::abs(lhs(1, 1) - rhs(1, 1)) < 1e-12, "P_f^-1 = w P_a^-1 + (1 - w) P_b^-1");
    std::printf("  fast weights %.4f %.4f %.4f\n", weights[0], weights[1], weights[2]);
    ok &= check(weighted && std::abs(weights[0] + weights[1] + weights[2] - 1) < 1e-12, "fast_ci_weights sum to 1");
    ok &= check(weights[2] == 0.2 && std::abs(weights[0] / weights[1] - 7.5 / 10.0) < 1e-12, "the bound holds the third; the rest stay proportional to 1 / tr P");
    ok &= check(fused3.is_ok(), "covariance_intersection of n sources");
    const auto at_one = tr::covariance_intersection(radar, camera, 1.0).unwrap();
    ok &= check(std::abs(at_one.covariance(1, 1) - 9.0) < 1e-12 && std::abs(at_one.mean[0] - 100.2) < 1e-12, "omega = 1 returns the first estimate");

    // Each refusal
    const g2 broken{vec2{}, mat2{{1.0, 2.0, 2.0, 1.0}}};
    const auto bad = tr::covariance_intersection(radar, broken, 0.5);
    ok &= check(bad.is_err() && bad.unwrap_err_unchecked() == la::error::not_positive_definite, "an indefinite covariance is reported");
    const auto none = tr::covariance_intersection<double, 2>(std::span<const g2>{}, std::span<const double>{});
    ok &= check(none.is_err() && none.unwrap_err_unchecked() == la::error::dimension_mismatch, "an empty set is reported");
    ok &= check(tr::optimal_omega(radar, broken).is_err(), "optimal_omega reports an indefinite covariance");
    const std::array<double, 3> too_little{0.3, 0.3, 0.3};
    ok &= check(!tr::fast_ci_weights<double, 2>(sources, weights, {}, too_little), "fast_ci_weights refuses bounds that cannot sum to 1");
    ok &= check(!tr::fast_ci_weights<double, 2>(sources, std::span<double>(weights).first(2)), "fast_ci_weights refuses a span of the wrong length");

    std::printf(ok ? "fusion demo: ok\n" : "fusion demo: FAILED\n");
    return ok ? 0 : 1;
}
