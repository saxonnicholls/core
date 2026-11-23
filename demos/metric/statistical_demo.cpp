//
//  statistical_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/metric/statistical.hpp>: gating detections against a track's innovation covariance with
//  mahalanobis (made once by make_mahalanobis, compared in its reduced form), the refusal of a covariance that is not
//  positive definite, and two track estimates compared by hellinger, squared_hellinger and kl_divergence, with NaN
//  for an indefinite covariance. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/metric/statistical.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/tracking/gaussian.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

namespace la = snicholls::linalg;
namespace me = snicholls::metric;
namespace tr = snicholls::tracking;

using v2 = la::fixed_vector<double, 2>;
using m2 = la::fixed_matrix<double, 2, 2>;

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core statistical demo\n");
    bool ok = true;

    //! [statistical]
    // A track predicted at the origin, uncertain along x (σ = 10) and sure along y (σ = 1)
    const v2 predicted{{0, 0}};
    const auto gate = me::make_mahalanobis(m2{{100, 0, 0, 1}}).unwrap();   // factorised once

    // Gating is a radius search: compare reduced (squared) distances against the contracted radius, no square roots
    const std::array<v2, 3> detections{v2{{15, 0}}, v2{{0, 4}}, v2{{-20, 1}}};
    const double radius = 3;   // three sigma
    std::size_t inside = 0;
    for (const auto& d : detections)
        inside += gate.reduced(predicted, d) <= gate.contract(radius) ? 1 : 0;

    // Two estimates of one target, compared as distributions
    const tr::gaussian<double, 2> a{v2{{0, 0}}, m2{{1, 0, 0, 1}}}, b{v2{{1, 0}}, m2{{2, 0, 0, 1}}};
    const double h = me::hellinger{}(a, b), h2 = me::squared_hellinger{}(a, b);
    const double kl_ab = me::kl_divergence{}(a, b), kl_ba = me::kl_divergence{}(b, a);
    //! [statistical]

    std::printf("  detections inside the 3-sigma gate: %zu of %zu; hellinger %.4f, KL(a|b) %.4f, KL(b|a) %.4f\n", inside, detections.size(), h, kl_ab, kl_ba);
    ok &= check(std::abs(gate(predicted, detections[0]) - 1.5) < 1e-12, "(15, 0) is 1.5 sigma away");
    ok &= check(std::abs(gate(predicted, detections[1]) - 4.0) < 1e-12, "(0, 4) is 4 sigma away, though nearer in metres");
    ok &= check(inside == 2, "two detections pass the gate: (15, 0) and (-20, 1)");
    ok &= check(std::abs(gate.expand(gate.reduced(predicted, detections[2])) - gate(predicted, detections[2])) < 1e-12, "expand(reduced) is the distance");
    ok &= check(std::abs(me::reduced_distance(gate, predicted, detections[2]) - 5.0) < 1e-12, "reduced_distance: 400 / 100 + 1 = 5");

    const auto refused = me::make_mahalanobis(m2{{1, 2, 2, 1}});
    ok &= check(refused.is_err() && refused.unwrap_err_unchecked() == la::error::not_positive_definite, "an indefinite covariance is not_positive_definite");

    ok &= check(h > 0 && h < 1 && std::abs(h * h - h2) < 1e-14, "hellinger in (0, 1), its square squared_hellinger");
    ok &= check(h == *tr::hellinger(a, b) && kl_ab == *tr::kl_divergence(a, b), "the same values as tracking::hellinger and kl_divergence");
    // KL(a ‖ b) in closed form, per axis: x: log √2 + (1 + 1) / 4 - 1/2; y: 0
    ok &= check(std::abs(kl_ab - std::log(std::sqrt(2.0))) < 1e-12, "KL(a | b) by hand: log sqrt 2");
    ok &= check(std::abs(kl_ab - kl_ba) > 0.1, "KL is not symmetric");
    const tr::gaussian<double, 2> broken{v2{{0, 0}}, m2{{1, 2, 2, 1}}};
    ok &= check(std::isnan(me::hellinger{}(a, broken)) && std::isnan(me::squared_hellinger{}(a, broken)) && std::isnan(me::kl_divergence{}(broken, a)),
                "an indefinite covariance gives NaN");

    std::printf(ok ? "statistical demo: ok\n" : "statistical demo: FAILED\n");
    return ok ? 0 : 1;
}
