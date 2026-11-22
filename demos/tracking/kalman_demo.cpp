//
//  kalman_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/tracking/kalman.hpp>: tracking a 2-D constant-velocity target from noisy position
//  measurements, gating each one, smoothing the track backwards with Rauch-Tung-Striebel, an extended filter on a
//  radar, a control input, the information form with two sensors, the GDOP of a sensor pair, central differences,
//  the alpha-beta filter and the errors a bad covariance returns. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/tracking/kalman.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/measurement.hpp>
#include <snicholls/core/tracking/transition.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

namespace tr = snicholls::tracking;
namespace la = snicholls::linalg;
namespace st = snicholls::stats;

bool check(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

constexpr std::array<std::size_t, 2> xy{0, 2};
constexpr std::array<std::size_t, 2> vxy{1, 3};
using state = la::fixed_vector<double, 4>;
using cv = tr::constant_velocity<double>;

} // namespace

int main()
{
    std::printf("snicholls-core kalman demo\n");
    bool ok = true;
    constexpr int steps = 60;
    st::wyrand g(2026);
    const st::normal_inverse_cdf normal{};

    //! [kalman]
    // The target moves at (3, -1) m/s from (0, 100); a sensor reports its position every second with 5 m of noise
    const auto model = tr::combine(cv{0.01}, cv{0.01});   // state (x, vx, y, vy)
    const tr::linear_gaussian<double, 4, xy> sensor{{{25, 0, 0, 25}}};
    const double gate = tr::gate_threshold(0.999, 2);

    tr::gaussian<double, 4> track{{{0, 0, 100, 0}}, la::fixed_matrix<double, 4, 4>::identity() * 100};
    std::array<tr::gaussian<double, 4>, steps> filtered{}, predicted{};
    std::array<state, steps> truth{};
    int gated_out = 0;
    for (int k = 0; k < steps; ++k) {
        truth[static_cast<std::size_t>(k)] = {{3.0 * (k + 1), 3, 100 - 1.0 * (k + 1), -1}};
        const la::fixed_vector<double, 2> z{{truth[static_cast<std::size_t>(k)][0] + 5 * normal(g), truth[static_cast<std::size_t>(k)][2] + 5 * normal(g)}};

        const auto prediction = tr::predict(track, model, 1.0);   // F x, F P F^T + Q
        const auto u = tr::update(prediction, z, sensor);        // posterior, innovation, gain, d^2, log-likelihood
        if (!u)
            return 1;   // S was not positive definite: cannot happen with R > 0
        gated_out += tr::in_gate(u->distance2, gate) ? 0 : 1;
        track = u->posterior;
        predicted[static_cast<std::size_t>(k)] = prediction;
        filtered[static_cast<std::size_t>(k)] = track;
    }

    // Smoothing backwards: each step refined by everything that came after it
    std::array<tr::gaussian<double, 4>, steps> smoothed{};
    smoothed[steps - 1] = filtered[steps - 1];
    for (std::size_t k = steps - 1; k-- > 0;)
        smoothed[k] = *tr::rts_smooth(filtered[k], predicted[k + 1], smoothed[k + 1], model.matrix(1.0));
    //! [kalman]

    double rms_measured = 5, rms_filtered = 0, rms_smoothed = 0;
    for (std::size_t k = 10; k < steps; ++k) {   // after the filter has settled
        const auto ef = filtered[k].mean - truth[k], es = smoothed[k].mean - truth[k];
        rms_filtered += ef[0] * ef[0] + ef[2] * ef[2];
        rms_smoothed += es[0] * es[0] + es[2] * es[2];
    }
    rms_filtered = std::sqrt(rms_filtered / (2 * (steps - 10)));
    rms_smoothed = std::sqrt(rms_smoothed / (2 * (steps - 10)));
    std::printf("  position error: measured %.2f m, filtered %.2f m, smoothed %.2f m; velocity (%.3f, %.3f)\n", rms_measured, rms_filtered, rms_smoothed, track.mean[1], track.mean[3]);
    ok &= check(rms_filtered < rms_measured && rms_smoothed < rms_filtered, "filtering beats the sensor, smoothing beats filtering");
    ok &= check(std::abs(track.mean[1] - 3) < 0.3 && std::abs(track.mean[3] + 1) < 0.3, "the velocity is estimated from positions alone");
    ok &= check(gated_out <= 1, "true measurements pass a 99.9% gate");
    ok &= check(smoothed[0].covariance(0, 0) < filtered[0].covariance(0, 0), "the smoothed first state is surer than the filtered one");

    // The same update in Joseph form, and from a measurement prediction made separately
    const auto pr = tr::predict(filtered[0], model, 1.0);
    const la::fixed_vector<double, 2> z1{{7, 97}};
    const auto standard = tr::update(pr, z1, sensor);
    const auto joseph = tr::update(pr, z1, sensor, tr::covariance_form::joseph);
    const auto mp = tr::predict_measurement(pr, sensor);
    const auto separate = tr::update<decltype(sensor)::kinds>(pr, z1, mp);
    ok &= check(std::abs(standard->posterior.covariance(0, 0) - joseph->posterior.covariance(0, 0)) < 1e-12 && separate->posterior == standard->posterior,
                "Joseph and standard forms agree; so does the two-part update");
    ok &= check(std::abs(standard->log_likelihood - *tr::log_likelihood(standard->innovation, standard->innovation_covariance)) < 1e-12 && standard->gain(0, 0) > 0,
                "the update carries the log-likelihood of its innovation");

    // An extended filter: a radar, linearised at the prediction; the Jacobian checked by central differences
    const tr::bearing_range<double, 4, xy> radar{{{1e-4, 0, 0, 4}}, {{-50, 0}}};
    const auto rmp = tr::predict_measurement(pr, radar);
    const auto ru = tr::update(pr, radar.measure(truth[1]), radar);
    const auto numeric = tr::numerical_jacobian<2>([&](const state& x) { return radar.measure(x); }, pr.mean);
    const auto analytic = radar.jacobian(pr.mean);
    double worst = 0;
    for (std::size_t i = 0; i < 8; ++i)
        worst = std::fmax(worst, std::abs(numeric.elements[i] - analytic.elements[i]));
    ok &= check(ru.is_ok() && rmp.mean == radar.measure(pr.mean) && worst < 1e-8, "extended update; the analytic Jacobian matches central differences");

    // A nonlinear motion model and a known control input (a 2 m/s current pushing east for 1 s)
    const tr::gaussian<double, 5> turning{{{0, 10, 0, 0, 0.1}}, la::fixed_matrix<double, 5, 5>::identity()};
    const auto tp = tr::predict(turning, tr::constant_turn<double>{0.1, 0.1, 0.01}, 1.0);
    const la::fixed_matrix<double, 4, 1> b{{1, 0, 0, 0}};
    const auto pushed = tr::predict(filtered[0], model, 1.0, b, la::fixed_vector<double, 1>{{2}}, la::fixed_matrix<double, 1, 1>{{0.25}});
    ok &= check(std::abs(tp.mean[2] - 10 * (1 - std::cos(0.1)) / 0.1) < 1e-12, "an extended predict through a constant turn");
    ok &= check(std::abs(pushed.mean[0] - pr.mean[0] - 2) < 1e-12 && std::abs(pushed.covariance(0, 0) - pr.covariance(0, 0) - 0.25) < 1e-12, "the control input moves the mean and adds its variance");

    // Information form: two sensors' updates add; the result is the covariance filter's
    const tr::linear_gaussian<double, 4, xy> second{{{9, 0, 0, 16}}};
    const la::fixed_vector<double, 2> z2{{6, 98}};
    const auto info = *tr::to_information(pr);
    const auto both = *tr::information_update(*tr::information_update(info, z1, sensor.matrix(), sensor.covariance()), z2, second.matrix(), second.covariance());
    const auto from_info = *tr::from_information(both);
    const auto kalman_two = tr::update(standard->posterior, z2, second)->posterior;
    ok &= check(std::abs(from_info.mean[0] - kalman_two.mean[0]) < 1e-9 && std::abs(from_info.covariance(2, 2) - kalman_two.covariance(2, 2)) < 1e-9,
                "two information updates equal two Kalman updates");

    // Where to put two bearing sensors: GDOP of a pair at right angles from 10 km, 1 mrad each
    using bearing2 = tr::bearing<double, 2, std::array<std::size_t, 2>{0, 1}>;
    const bearing2 west{{{1e-6}}, {{-1e4, 0}}}, south{{{1e-6}}, {{0, -1e4}}};
    const la::fixed_vector<double, 2> origin{};
    const auto fisher = *tr::fisher_information(west.jacobian(origin), west.covariance()) + *tr::fisher_information(south.jacobian(origin), south.covariance());
    const double d = *tr::gdop(fisher);
    std::printf("  GDOP of two orthogonal bearings at 10 km, 1 mrad: %.2f m\n", d);
    ok &= check(std::abs(d - 10 * std::sqrt(2.0)) < 1e-9, "GDOP = range x sigma x sqrt 2");

    // A fixed-gain alpha-beta filter on the first measurement
    const auto ab = tr::alpha_beta_update<xy, vxy>(pr.mean, z1, 0.5, 0.1, 1.0);
    ok &= check(std::abs(ab[0] - (pr.mean[0] + 0.5 * (7 - pr.mean[0]))) < 1e-12 && std::abs(ab[1] - (pr.mean[1] + 0.1 * (7 - pr.mean[0]))) < 1e-12, "alpha-beta: fixed gains on the residual");

    // Errors: a covariance with no Cholesky factor is an error value, never a NaN
    const tr::gaussian<double, 4> degenerate{};
    const tr::linear_gaussian<double, 4, xy> perfect{};
    const bool errors = tr::update(degenerate, z1, perfect).is_err() && tr::update<decltype(perfect)::kinds>(degenerate, z1, tr::predict_measurement(degenerate, perfect)).is_err() &&
                        tr::to_information(degenerate).is_err() && tr::from_information(tr::information_state<double, 4>{}).is_err() &&
                        tr::information_update(info, z1, perfect.matrix(), perfect.covariance()).is_err() && tr::fisher_information(perfect.matrix(), perfect.covariance()).is_err() &&
                        tr::gdop(*tr::fisher_information(west.jacobian(origin), west.covariance())).is_err() && tr::rts_smooth(filtered[0], degenerate, filtered[1], model.matrix(1.0)).is_err();
    ok &= check(errors, "every documented not_positive_definite error is returned");

    std::printf(ok ? "kalman demo: ok\n" : "kalman demo: FAILED\n");
    return ok ? 0 : 1;
}
