//
//  unscented_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/tracking/unscented.hpp>: tracking a crossing target with a bearing-and-range radar by the
//  unscented and the cubature Kalman filters, the sigma and cubature points and their weights, the transform of a
//  Gaussian through a polar conversion, exactness on a linear model, and the error a bad covariance returns. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/tracking/unscented.hpp>

#include <snicholls/core/geometry/angle.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/tracking/gaussian.hpp>
#include <snicholls/core/tracking/kalman.hpp>
#include <snicholls/core/tracking/measurement.hpp>
#include <snicholls/core/tracking/transition.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numbers>

namespace {

namespace tr = snicholls::tracking;
namespace la = snicholls::linalg;
namespace ge = snicholls::geometry;
namespace st = snicholls::stats;

bool check(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

constexpr std::array<std::size_t, 2> xy{0, 2};
using state = la::fixed_vector<double, 4>;
using cv = tr::constant_velocity<double>;
constexpr double deg = std::numbers::pi / 180;

} // namespace

int main()
{
    std::printf("snicholls-core unscented demo\n");
    bool ok = true;
    st::wyrand g(7);
    const st::normal_inverse_cdf normal{};
    constexpr int steps = 80;

    //! [unscented]
    // A radar at the origin, 0.5 degree and 2 m of noise; a target crossing west to east 300 m north of it, so the
    // bearing sweeps from about 150 to 30 degrees
    const tr::bearing_range<double, 4, xy> radar{{{0.5 * deg * 0.5 * deg, 0, 0, 4}}};
    const auto model = tr::combine(cv{0.05}, cv{0.05});
    tr::gaussian<double, 4> ukf{{{-500, 0, 300, 0}}, la::fixed_matrix<double, 4, 4>::identity() * 400}, ckf = ukf;
    double ukf_error = 0, ckf_error = 0;
    for (int k = 1; k <= steps; ++k) {
        const state truth{{-500 + 12.0 * k, 12, 300, 0}};
        const auto clean = radar.measure(truth);
        const la::fixed_vector<double, 2> z{{ge::wrap_pi(clean[0] + 0.5 * deg * normal(g)), clean[1] + 2 * normal(g)}};

        // Unscented: sigma points through the motion and through the radar, then the Kalman update
        const auto up = tr::unscented_predict(ukf, model, 1.0);
        const auto ump = tr::unscented_predict_measurement(*up, radar);
        ukf = tr::update<decltype(radar)::kinds>(*up, z, *ump)->posterior;   // the bearing innovation wraps

        // Cubature: 2n equally weighted points, the same steps
        const auto cp = tr::cubature_predict(ckf, model, 1.0);
        const auto cmp = tr::cubature_predict_measurement(*cp, radar);
        ckf = tr::update<decltype(radar)::kinds>(*cp, z, *cmp)->posterior;

        if (k > 20) {
            ukf_error += std::hypot(ukf.mean[0] - truth[0], ukf.mean[2] - truth[2]);
            ckf_error += std::hypot(ckf.mean[0] - truth[0], ckf.mean[2] - truth[2]);
        }
    }
    ukf_error /= steps - 20;
    ckf_error /= steps - 20;
    std::printf("  mean position error: UKF %.2f m, CKF %.2f m; UKF velocity (%.2f, %.2f)\n", ukf_error, ckf_error, ukf.mean[1], ukf.mean[3]);
    //! [unscented]

    // A cross-range error of 0.5 degree at 300-700 m is 2.6-6 m; filtering brings it well below
    ok &= check(ukf_error < 3 && ckf_error < 3, "both filters track the crossing target to a few metres");
    ok &= check(std::abs(ukf.mean[1] - 12) < 1 && std::abs(ukf.mean[3]) < 1, "the velocity is recovered from bearings and ranges");

    // Sigma points for n = 4 with Stone Soup's defaults (alpha 0.5, beta 2, kappa 3 - n): weights sum to 1
    const auto sigma = tr::make_sigma_points(ukf);
    double sum = 0;
    for (const double w : sigma->mean_weights)
        sum += w;
    std::printf("  sigma weights: centre %.4f (mean), %.4f (covariance), others %.4f\n", sigma->mean_weights[0], sigma->covariance_weights[0], sigma->mean_weights[1]);
    ok &= check(std::abs(sum - 1) < 1e-14 && std::abs(sigma->mean_weights[0] + 3.25 / 0.75) < 1e-14 && sigma->points[0] == ukf.mean, "Stone Soup's default sigma weights");
    tr::unscented_parameters<double> wide{};
    wide.alpha = 1;
    wide.kappa = 0.0;   // lambda = 0: the centre has no mean weight
    const auto sigma_wide = tr::make_sigma_points(ukf, wide);
    ok &= check(sigma_wide->mean_weights[0] == 0 && sigma_wide->covariance_weights[0] == 2, "a caller's alpha and kappa");
    const auto cub = tr::make_cubature_points(ukf);
    ok &= check(cub->mean_weights[7] == 1.0 / 8 && cub->covariance_weights[0] == 1.0 / 8, "cubature points weigh 1 / 2n each");

    // The polar-to-Cartesian conversion of a radar report, by the transform: the classic banana
    const tr::gaussian<double, 2> report{{{90 * deg, 1000}}, {{(2 * deg) * (2 * deg), 0, 0, 25}}};
    const auto pts = tr::make_sigma_points(report);
    const auto cart = tr::point_transform<ge::all_linear<2>, 2>(*pts, [](const la::fixed_vector<double, 2>& p) { return la::fixed_vector<double, 2>{{p[1] * std::cos(p[0]), p[1] * std::sin(p[0])}}; },
                                                                la::fixed_matrix<double, 2, 2>{});
    std::printf("  2-degree, 5 m report at 1 km: mean (%.2f, %.2f), sd (%.2f, %.2f) m\n", cart.mean[0], cart.mean[1], std::sqrt(cart.covariance(0, 0)), std::sqrt(cart.covariance(1, 1)));
    ok &= check(std::abs(cart.mean[0]) < 1e-9 && cart.mean[1] < 1000 && cart.mean[1] > 999 && std::abs(std::sqrt(cart.covariance(0, 0)) - 1000 * 2 * deg) < 0.5,
                "the mean is pulled inside the arc; cross-range sd is range x sigma");
    ok &= check(std::abs(cart.cross_covariance(1, 1) - 25) < 0.5 && std::abs(cart.cross_covariance(0, 1)) < 1e-9, "the cross-covariance links range to y");

    // On a linear model both are the Kalman filter, to rounding
    const tr::linear_gaussian<double, 4, xy> gps{{{4, 0, 0, 4}}};
    const la::fixed_vector<double, 2> zg{{ukf.mean[0] + 3, ukf.mean[2] - 2}};
    const auto kf = tr::update(tr::predict(ukf, model, 1.0), zg, gps)->posterior;
    const auto up = tr::unscented_predict(ukf, model, 1.0, wide);
    const auto uf = tr::update<decltype(gps)::kinds>(*up, zg, *tr::unscented_predict_measurement(*up, gps, wide))->posterior;
    const auto cp = tr::cubature_predict(ukf, model, 1.0, 0.5);
    const auto cf = tr::update<decltype(gps)::kinds>(*cp, zg, *tr::cubature_predict_measurement(*cp, gps, 2.0))->posterior;
    ok &= check(std::abs(uf.mean[0] - kf.mean[0]) < 1e-9 && std::abs(uf.covariance(0, 0) - kf.covariance(0, 0)) < 1e-9 && std::abs(cf.mean[2] - kf.mean[2]) < 1e-9 &&
                    std::abs(cf.covariance(2, 2) - kf.covariance(2, 2)) < 1e-9,
                "unscented and cubature equal Kalman on a linear model");

    // A covariance without a Cholesky factor is an error value
    const tr::gaussian<double, 4> flat{};
    ok &= check(tr::make_sigma_points(flat).is_err() && tr::make_cubature_points(flat).is_err() && tr::unscented_predict(flat, model, 1.0).is_err() &&
                    tr::cubature_predict(flat, model, 1.0).is_err() && tr::unscented_predict_measurement(flat, radar).is_err() &&
                    tr::cubature_predict_measurement(flat, radar).unwrap_err() == la::error::not_positive_definite,
                "not_positive_definite from every function");

    std::printf(ok ? "unscented demo: ok\n" : "unscented demo: FAILED\n");
    return ok ? 0 : 1;
}
