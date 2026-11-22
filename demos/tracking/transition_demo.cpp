//
//  transition_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/tracking/transition.hpp>: a 2-D constant-velocity target composed from two 1-D models,
//  the noise each motion model adds over a step (random walk, constant velocity and acceleration, Ornstein-Uhlenbeck,
//  Singer and its approximation), a known turn and a turn whose rate is part of the state, with their Jacobians.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/tracking/transition.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numbers>

namespace {

namespace tr = snicholls::tracking;
namespace la = snicholls::linalg;

bool check(bool ok, const char* what)
{
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

template <std::size_t R, std::size_t C>
double max_abs_diff(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b)
{
    double m = 0;
    for (std::size_t i = 0; i < R * C; ++i)
        m = std::fmax(m, std::abs(a.elements[i] - b.elements[i]));
    return m;
}

} // namespace

int main()
{
    std::printf("snicholls-core transition demo\n");
    bool ok = true;

    //! [transition]
    // A ship in the plane, state (x, vx, y, vy): constant velocity on each axis, the composition free at run time
    const auto ship = tr::combine(tr::constant_velocity<double>{0.05}, tr::constant_velocity<double>{0.05});
    const la::fixed_vector<double, 4> now{{0, 5, 0, 2}};   // at the origin, 5 m/s east, 2 m/s north
    const auto in_ten = ship.apply(now, 10);               // (50, 5, 20, 2)
    const auto f = ship.matrix(10);                        // block diagonal [[1, 10], [0, 1]]
    const auto q = ship.covariance(10);                    // q dt^3 / 3 = 16.7 m^2 of position variance per axis
    std::printf("  ship after 10 s: (%.1f, %.1f), position sd %.2f m\n", in_ten[0], in_ten[2], std::sqrt(q(0, 0)));

    // A manoeuvring target with its turn rate in the state, (x, vx, y, vy, w): nonlinear, so it has a Jacobian
    const tr::constant_turn<double> turning{0.05, 0.05, 1e-4};
    const la::fixed_vector<double, 5> banking{{0, 10, 0, 0, std::numbers::pi / 20}};   // 9 degrees a second
    const auto after_turn = turning.apply(banking, 20);    // half a circle: heading west
    const auto jt = turning.jacobian(banking, 20);
    std::printf("  turning target after 20 s: (%.2f, %.2f), velocity (%.2f, %.2f)\n", after_turn[0], after_turn[2], after_turn[1], after_turn[3]);
    //! [transition]

    ok &= check(max_abs_diff(in_ten, la::fixed_vector<double, 4>{{50, 5, 20, 2}}) < 1e-14 && f(0, 1) == 10 && f(2, 3) == 10 && f(0, 2) == 0,
                "constant velocity moves (x, y) by v dt; F is block diagonal");
    ok &= check(std::abs(q(0, 0) - 0.05 * 1000 / 3) < 1e-12 && q(0, 1) == 0.05 * 100 / 2 && q(1, 1) == 0.5 && q(0, 2) == 0, "Q per axis is q [[dt^3/3, dt^2/2], [dt^2/2, dt]]");
    ok &= check(ship.jacobian(now, 10) == f, "a linear model's Jacobian is F");
    const double r = 10 / (std::numbers::pi / 20);   // the turn's radius
    ok &= check(std::abs(after_turn[0]) < 1e-12 && std::abs(after_turn[2] - 2 * r) < 1e-12 && std::abs(after_turn[1] + 10) < 1e-12, "half a turn ends 2r north, heading west");
    ok &= check(jt(4, 4) == 1 && jt(0, 0) == 1 && std::abs(jt(1, 1) + 1) < 1e-15, "constant turn's Jacobian at w dt = pi");
    const auto jt0 = turning.jacobian({{0, 10, 0, 0, 0}}, 10);   // zero rate: the small-angle series
    ok &= check(jt0(2, 4) == 10 * 100 / 2.0 && jt0(0, 1) == 10, "at zero rate y gains vx dt^2 / 2 per unit of w");
    const auto tq = turning.covariance(10);
    ok &= check(std::abs(tq(4, 4) - 1e-3) < 1e-18 && tq(0, 0) == q(0, 0), "constant turn's noise: constant velocity's plus q_w dt");

    // The noise each model adds over 2 s, from the same diffusion coefficient
    const double dt = 2;
    const tr::random_walk<double> walk{1};
    const tr::constant_acceleration<double> accel{1};
    const tr::ornstein_uhlenbeck<double> ou{1, 0.5};
    const tr::singer<double> sing{1, 0.5};
    const tr::singer_approximate<double> sing_approx{1, 0.5};
    std::printf("  position variance over 2 s: walk %.3f, CV %.3f, CA %.3f, OU %.3f, Singer %.3f, approximate %.3f\n", walk.covariance(dt)(0, 0),
                tr::constant_velocity<double>{1}.covariance(dt)(0, 0), accel.covariance(dt)(0, 0), ou.covariance(dt)(0, 0), sing.covariance(dt)(0, 0),
                sing_approx.covariance(dt)(0, 0));
    ok &= check(walk.covariance(dt)(0, 0) == 2 && walk.matrix(dt)(0, 0) == 1 && walk.apply({{3}}, dt)[0] == 3 && walk.jacobian({{3}}, dt)(0, 0) == 1, "a random walk diffuses q dt and stays put");
    ok &= check(accel.matrix(dt)(0, 2) == 2 && std::abs(accel.covariance(dt)(0, 0) - 32.0 / 20) < 1e-14 && accel.apply({{0, 0, 1}}, dt)[0] == 2 &&
                    accel.jacobian({}, dt) == accel.matrix(dt),
                "constant acceleration: F(0, 2) = dt^2/2, Q(0, 0) = q dt^5 / 20");
    const double e = std::exp(-0.5 * dt);
    ok &= check(std::abs(ou.matrix(dt)(0, 1) - (1 - e) / 0.5) < 1e-15 && std::abs(ou.covariance(dt)(1, 1) - (1 - e * e)) < 1e-14 && ou.jacobian({}, dt) == ou.matrix(dt) &&
                    std::abs(ou.apply({{0, 1}}, dt)[1] - e) < 1e-15,
                "Ornstein-Uhlenbeck: velocity decays as e^(-k dt), variance (q / 2k)(1 - e^(-2k dt))");
    // Singer's acceleration noise is Ornstein-Uhlenbeck's velocity noise; with a decorrelation time much shorter than
    // the step the integration splits into panels; a zero step adds nothing
    ok &= check(std::abs(sing.covariance(dt)(2, 2) - ou.covariance(dt)(1, 1)) < 1e-14 && sing.jacobian({}, dt) == sing.matrix(dt) &&
                    sing.apply({{0, 0, 1}}, dt) == sing.matrix(dt) * la::fixed_vector<double, 3>{{0, 0, 1}},
                "Singer's lower corner is Ornstein-Uhlenbeck's");
    ok &= check(std::abs(tr::singer<double>{1, 100}.covariance(dt)(2, 2) - 1.0 / 200) < 1e-15 && sing.covariance(0) == la::fixed_matrix<double, 3, 3>{},
                "fast decay (64 panels): acceleration variance q / 2k; zero step zero");
    ok &= check(sing_approx.matrix(dt) == sing.matrix(dt) && sing_approx.covariance(dt) == accel.covariance(dt) && sing_approx.jacobian({}, dt) == sing.matrix(dt) &&
                    sing_approx.apply({{0, 0, 1}}, dt) == sing.apply({{0, 0, 1}}, dt),
                "Singer approximate: Singer's matrix, constant acceleration's noise");
    ok &= check(accel.covariance(-dt) == accel.covariance(dt) && sing.covariance(-dt) == sing.covariance(dt), "a negative step (retrodiction) has the same noise");
    const tr::constant_nth_derivative<double, 3> jerk{1};
    ok &= check(std::abs(jerk.matrix(dt)(0, 3) - 8.0 / 6) < 1e-15 && jerk.covariance(dt)(3, 3) == 2 && jerk.jacobian({}, dt) == jerk.matrix(dt) && jerk.apply({{1, 0, 0, 0}}, dt)[0] == 1,
                "N = 3: F(0, 3) = dt^3 / 3!");

    // A known turn rate: a linear model; at zero rate, constant velocity
    const tr::known_turn_rate<double> known{0.05, 0.05, std::numbers::pi / 20};
    const auto kf = known.matrix(20);
    const auto ky = known.apply({{0, 10, 0, 0}}, 20);
    ok &= check(std::abs(ky[2] - 2 * r) < 1e-12 && known.jacobian({}, 20) == kf && known.covariance(20) == ship.covariance(20), "the known turn agrees with constant turn; noise per axis");
    ok &= check(max_abs_diff(tr::known_turn_rate<double>{0, 0, 0}.matrix(10), f) == 0, "a known turn at zero rate is constant velocity");

    // A combination with a nonlinear block: apply, Jacobian and noise by block
    const auto mixed = tr::combine(turning, walk);
    const la::fixed_vector<double, 6> xm{{0, 10, 0, 0, std::numbers::pi / 20, 7}};
    const auto ym = mixed.apply(xm, 20);
    const auto jm = mixed.jacobian(xm, 20);
    ok &= check(ym[5] == 7 && std::abs(ym[2] - 2 * r) < 1e-12 && jm(5, 5) == 1 && jm(0, 4) == jt(0, 4) && mixed.covariance(20)(5, 5) == 20, "a combination applies each block's model");

    std::printf(ok ? "transition demo: ok\n" : "transition demo: FAILED\n");
    return ok ? 0 : 1;
}
