//
//  association_demo.cpp
//  snicholls-core
//
//  Demo for <snicholls/core/tracking/association.hpp>: a track in clutter updated by probabilistic data association
//  (a clutter density from a gate volume, chi-square gating, PDA weights, the moment-matched update), two crossing
//  tracks sharing detections by joint PDA, and each function's refusal. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/tracking/association.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/kalman.hpp>
#include <snicholls/core/tracking/measurement.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <numbers>
#include <span>

namespace {

namespace la = snicholls::linalg;
namespace tr = snicholls::tracking;

using vec2 = la::fixed_vector<double, 2>;
using mat2 = la::fixed_matrix<double, 2, 2>;
using vec4 = la::fixed_vector<double, 4>;
using mat4 = la::fixed_matrix<double, 4, 4>;

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core association demo\n");
    bool ok = true;

    //! [association]
    // A constant-velocity track (x, vx, y, vy) and a sensor that measures position
    const tr::gaussian<double, 4> prediction{vec4{{0, 1, 0, 1}}, mat4{{2, 0.3, 0, 0, 0.3, 0.5, 0, 0, 0, 0, 2, 0.3, 0, 0, 0.3, 0.5}}};
    const tr::linear_gaussian<double, 4, std::array{0, 2}> sensor{mat2{{0.5, 0, 0, 0.5}}};
    const auto mp = tr::predict_measurement(prediction, sensor);

    // Clutter: 2 false alarms expected inside the 99% gate, so a density of 2 / gate volume
    const double gate = tr::gate_threshold(0.99, 2);
    const double volume = tr::gate_volume(gate, mp.covariance).unwrap();
    const double clutter_density = 2 / volume;

    // Four detections: three inside the gate, one far outside (log-likelihood -∞, weight 0)
    const std::array<vec2, 4> detections{vec2{{0.4, -0.3}}, vec2{{-1.5, 1.0}}, vec2{{2.0, 2.2}}, vec2{{30.0, -25.0}}};
    std::array<vec2, 4> innovations{};
    std::array<double, 4> log_likelihoods{};
    for (std::size_t i = 0; i < detections.size(); ++i) {
        innovations[i] = detections[i] - mp.mean;
        const double d2 = tr::squared_mahalanobis(innovations[i], mp.covariance).unwrap();
        log_likelihoods[i] = tr::in_gate(d2, gate) ? tr::log_likelihood(innovations[i], mp.covariance).unwrap() : -std::numeric_limits<double>::infinity();
    }
    std::array<double, 5> weights{};   // missed detection first
    const bool formed = tr::pda_weights<double>(log_likelihoods, 0.9, 0.99, clutter_density, weights);
    const auto posterior = tr::pda_update<double, 4, 2>(prediction, mp, innovations, weights);

    // Two tracks, two detections: rows (missed, detection 1, detection 2), unnormalised; track 1 cannot see detection 1
    const std::array<double, 6> likelihoods{0.1, 2.0, 1.0,
                                            0.1, 0.0, 3.0};
    std::array<double, 6> marginals{};
    const bool joint = tr::jpda_marginals<4, 4, double>(likelihoods, 2, 2, marginals);
    //! [association]

    std::printf("  gate %.3f, gate volume %.3f, clutter density %.4f\n", gate, volume, clutter_density);
    std::printf("  PDA weights: missed %.4f, detections %.4f %.4f %.4f %.4f\n", weights[0], weights[1], weights[2], weights[3], weights[4]);
    ok &= check(formed, "pda_weights forms the weights");
    double sum = 0;
    for (const double w : weights)
        sum += w;
    ok &= check(std::abs(sum - 1) < 1e-12, "the weights sum to 1");
    ok &= check(weights[4] == 0, "the detection outside the gate has weight 0");
    ok &= check(weights[1] > weights[2] && weights[1] > weights[3], "the nearest detection is the likeliest");
    // The gate is an ellipse: π γ √det S
    const double det = mp.covariance(0, 0) * mp.covariance(1, 1) - mp.covariance(0, 1) * mp.covariance(1, 0);
    ok &= check(std::abs(volume - std::numbers::pi * gate * std::sqrt(det)) < 1e-9, "gate_volume is the ellipse's area");
    ok &= check(posterior.is_ok(), "pda_update gives a posterior");
    if (posterior) {
        std::printf("  posterior position (%.4f, %.4f), variance x %.4f (prior %.4f)\n", posterior->mean[0], posterior->mean[2], posterior->covariance(0, 0), prediction.covariance(0, 0));
        ok &= check(posterior->covariance(0, 0) < prediction.covariance(0, 0), "the posterior is tighter than the prediction");
        ok &= check(posterior->covariance(0, 2) == posterior->covariance(2, 0), "the posterior covariance is symmetric");
    }
    std::printf("  JPDA marginals: track 0 (%.4f, %.4f, %.4f), track 1 (%.4f, %.4f, %.4f)\n", marginals[0], marginals[1], marginals[2], marginals[3], marginals[4], marginals[5]);
    ok &= check(joint, "jpda_marginals forms the marginals");
    ok &= check(marginals[4] == 0, "a forbidden pairing has probability 0");
    ok &= check(std::abs(marginals[0] + marginals[1] + marginals[2] - 1) < 1e-12 && std::abs(marginals[3] + marginals[4] + marginals[5] - 1) < 1e-12, "each track's marginals sum to 1");
    // Events (t0, t1) by hand: (m, m) 0.01, (m, 2) 0.3, (1, m) 0.2, (1, 2) 6, (2, m) 0.1; total 6.61
    ok &= check(std::abs(marginals[1] - 6.2 / 6.61) < 1e-12 && std::abs(marginals[5] - 6.3 / 6.61) < 1e-12, "the marginals match the hand enumeration");

    // Each refusal
    std::array<double, 3> wrong{};
    ok &= check(!tr::pda_weights<double>(log_likelihoods, 0.9, 0.99, clutter_density, wrong), "pda_weights refuses weights of the wrong length");
    ok &= check(!tr::pda_weights<double>(log_likelihoods, 0.9, 0.99, 0.0, weights), "pda_weights refuses a zero clutter density");
    const std::array<double, 2> gated{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    std::array<double, 3> none{};
    ok &= check(!tr::pda_weights<double>(gated, 1.0, 1.0, clutter_density, none), "pda_weights refuses when no hypothesis has weight");
    const auto mismatched = tr::pda_update<double, 4, 2>(prediction, mp, innovations, std::span<const double>(weights).first(3));
    ok &= check(mismatched.is_err() && mismatched.unwrap_err_unchecked() == la::error::dimension_mismatch, "pda_update reports mismatched lengths");
    auto bad = mp;
    bad.covariance = mat2{{1, 2, 2, 1}};
    const auto indefinite = tr::pda_update<double, 4, 2>(prediction, bad, innovations, weights);
    ok &= check(indefinite.is_err() && indefinite.unwrap_err_unchecked() == la::error::not_positive_definite, "pda_update reports an S that is not positive definite");
    ok &= check(tr::gate_volume(gate, bad.covariance).is_err(), "gate_volume reports an S that is not positive definite");
    const std::array<double, 4> clash{0.0, 1.0, 0.0, 1.0};   // both tracks must take the one detection
    std::array<double, 4> m2{};
    ok &= check(!tr::jpda_marginals<2, 1, double>(clash, 2, 1, m2), "jpda_marginals refuses a cluster with no feasible event");
    ok &= check(!tr::jpda_marginals<1, 4, double>(likelihoods, 2, 2, marginals), "jpda_marginals refuses more tracks than it holds");
    ok &= check(tr::jpda_marginals<4, 4, double>(std::span<const double>{}, 0, 0, std::span<double>{}), "jpda_marginals of no tracks is trivially formed");

    std::printf(ok ? "association demo: ok\n" : "association demo: FAILED\n");
    return ok ? 0 : 1;
}
