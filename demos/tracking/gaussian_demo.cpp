//
//  gaussian_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/tracking/gaussian.hpp>: gating two measurements against a predicted position by the
//  Mahalanobis distance and a chi-square gate, scoring them by log-likelihood, comparing two estimates by Euclidean,
//  Hellinger and KL measures, the confidence ellipse of the prediction, and the error a non-positive-definite
//  covariance returns. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/tracking/gaussian.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <cmath>
#include <cstdio>
#include <numbers>

namespace {

namespace tr = snicholls::tracking;
namespace la = snicholls::linalg;

using v2 = la::fixed_vector<double, 2>;
using m2 = la::fixed_matrix<double, 2, 2>;

bool check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core gaussian demo\n");
    bool ok = true;

    //! [gaussian]
    // A predicted position and its innovation covariance S (prediction plus sensor noise), metres
    const tr::gaussian<double, 2> predicted{{{100, 50}}, {{4, 1, 1, 3}}};
    const double gate = tr::gate_threshold(0.99, 2);   // 9.21: a true measurement passes 99% of the time

    // Two candidate measurements: one close in the units of S, one far
    const v2 near_z{{101, 52}}, far_z{{106, 50}};
    const double d_near = *tr::squared_mahalanobis(near_z - predicted.mean, predicted.covariance);
    const double d_far = *tr::squared_mahalanobis(far_z - predicted.mean, predicted.covariance);
    const bool near_in = tr::in_gate(d_near, gate), far_in = tr::in_gate(d_far, gate);
    std::printf("  gate %.3f: near d2 %.3f (%s), far d2 %.3f (%s)\n", gate, d_near, near_in ? "in" : "out", d_far, far_in ? "in" : "out");

    // The near one's association score, and the 95% ellipse of the prediction for a display
    const double score = *tr::log_likelihood(near_z - predicted.mean, predicted.covariance);
    const auto plume = tr::confidence_ellipse(predicted.covariance, std::sqrt(tr::gate_threshold(0.95, 2)));
    std::printf("  log-likelihood %.4f; 95%% ellipse %.3f x %.3f m at %.1f deg\n", score, plume.semi_major, plume.semi_minor, plume.angle * 180 / std::numbers::pi);
    //! [gaussian]

    ok &= check(near_in && !far_in, "the near measurement gates in, the far one out");
    ok &= check(std::abs(d_near - 15.0 / 11) < 1e-12, "near distance^2 is (1, 2) S^-1 (1, 2) = 15/11");
    ok &= check(std::abs(d_far - 108.0 / 11) < 1e-12, "far distance^2 is (6, 0) S^-1 (6, 0) = 108/11");
    ok &= check(std::abs(score + (15.0 / 11 + std::log(11.0) + 2 * std::log(2 * std::numbers::pi)) / 2) < 1e-12, "log-likelihood is log N(r; 0, S)");
    ok &= check(std::abs(gate + 2 * std::log(0.01)) < 1e-9, "the 2-D 99% gate is -2 log 0.01");
    ok &= check(plume.semi_major > plume.semi_minor && plume.angle > 0 && plume.angle < std::numbers::pi / 4, "the ellipse leans toward x, positively correlated");

    // With the Cholesky factor in hand the same distance is one triangular solve
    const auto l = la::cholesky(predicted.covariance);
    const double d_whitened = tr::whitened_squared_norm(near_z - predicted.mean, *l);
    const double d_root = *tr::mahalanobis(near_z - predicted.mean, predicted.covariance);
    ok &= check(std::abs(d_whitened - d_near) < 1e-14 && std::abs(d_root * d_root - d_near) < 1e-14, "whitened norm and Mahalanobis agree");

    // Comparing two estimates of the same target
    const tr::gaussian<double, 2> a{{{100, 50}}, {{4, 0, 0, 1}}}, b{{{102, 50}}, {{4, 0, 0, 1}}};
    const double e = tr::euclidean(a.mean, b.mean);
    const double ew = tr::euclidean_weighted(a.mean, b.mean, v2{{0.25, 1}});
    const double h2 = *tr::squared_hellinger(a, b), h = *tr::hellinger(a, b);
    const double kl = *tr::kl_divergence(a, b);
    std::printf("  euclidean %.3f, weighted %.3f, hellinger %.4f, KL %.4f\n", e, ew, h, kl);
    ok &= check(e == 2 && ew == 1, "euclidean 2 m, weighted by 1/var_x 1");
    ok &= check(std::abs(h2 - (1 - std::exp(-4.0 / 32))) < 1e-14 && std::abs(h * h - h2) < 1e-15, "Hellinger^2 is 1 - exp(-d^2 / (8 var)) for equal covariances");
    ok &= check(std::abs(kl - 0.5) < 1e-14, "KL is d^2 / (2 var) = 0.5 for equal covariances");

    // An indefinite covariance is an error value, not a NaN
    const m2 bad{{1, 2, 2, 1}};
    const tr::gaussian<double, 2> broken{{{0, 0}}, bad};
    const bool all_err = tr::squared_mahalanobis(near_z, bad).is_err() && tr::mahalanobis(near_z, bad).is_err() && tr::log_likelihood(near_z, bad).is_err() &&
                         tr::squared_hellinger(a, broken).is_err() && tr::hellinger(a, broken).is_err() && tr::kl_divergence(a, broken).is_err();
    ok &= check(all_err && tr::kl_divergence(broken, a).unwrap_err() == la::error::not_positive_definite, "every factorising measure returns not_positive_definite");
    const auto flat = tr::confidence_ellipse(bad);
    ok &= check(flat.semi_minor == 0 && tr::confidence_ellipse(m2{{1, 0, 0, 1}}).angle == 0, "indefinite ellipse clamps its minor axis; a circle has angle 0");

    std::printf(ok ? "gaussian demo: ok\n" : "gaussian demo: FAILED\n");
    return ok ? 0 : 1;
}
