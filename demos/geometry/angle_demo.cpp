//
//  angle_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/geometry/angle.hpp>: the residual of a bearing-elevation-range measurement across the
//  ±180° cut, the circular mean of sigma points with a negative centre weight, and each wrap in every quadrant.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/geometry/angle.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <span>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
using geo::angle_kind;
constexpr double pi = std::numbers::pi;

bool check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol; }

} // namespace

int main()
{
    std::printf("snicholls-core geometry angle demo\n");
    bool ok = true;

    //! [angle]
    // A radar measures (bearing, elevation, range); the kinds are part of the measurement model's type
    constexpr geo::angle_kinds<3> kinds{angle_kind::bearing, angle_kind::elevation, angle_kind::linear};
    const auto measured = la::make_vector(geo::radians(179.0), geo::radians(10.0), 1500.0);
    const auto predicted = la::make_vector(geo::radians(-179.0), geo::radians(12.0), 1480.0);
    // -2° the short way round, not 358°
    const auto nu = geo::residual<kinds>(measured, predicted);

    // The unscented transform's mean of sigma points, with a negative centre weight
    const std::array<la::fixed_vector<double, 3>, 3> sigma{la::make_vector(geo::radians(180.0), 0.1, 1000.0), la::make_vector(geo::radians(175.0), 0.1, 1010.0),
                                                          la::make_vector(geo::radians(-175.0), 0.1, 990.0)};
    const std::array<double, 3> w{-1.0, 1.0, 1.0};
    const auto mean = geo::weighted_mean<kinds>(sigma, w);
    //! [angle]
    std::printf("  residual: bearing %+.3f deg, elevation %+.3f deg, range %+.1f\n", geo::degrees(nu[0]), geo::degrees(nu[1]), nu[2]);
    std::printf("  sigma-point mean: bearing %+.3f deg, range %.1f\n", geo::degrees(mean[0]), mean[2]);
    ok &= check(near(geo::degrees(nu[0]), -2.0) && near(geo::degrees(nu[1]), -2.0) && nu[2] == 20, "residual wraps the bearing only");
    // Python: math.atan2(sum w sin, sum w cos) for those bearings and weights
    const double s = -std::sin(pi) + std::sin(geo::radians(175.0)) + std::sin(geo::radians(-175.0));
    const double c = -std::cos(pi) + std::cos(geo::radians(175.0)) + std::cos(geo::radians(-175.0));
    ok &= check(near(std::abs(mean[0]), std::abs(std::atan2(s, c))) && near(mean[2], 1000.0) && near(mean[1], 0.1), "weighted_mean: circular in the angles");

    // The wraps
    ok &= check(geo::wrap_pi(pi) == -pi && near(geo::wrap_pi(4.0), 4.0 - 2 * pi) && near(geo::wrap_pi(-4.0), 2 * pi - 4.0), "wrap_pi into [-pi, pi)");
    ok &= check(near(geo::wrap_two_pi(-pi / 2), 1.5 * pi) && geo::wrap_two_pi(0.0) == 0, "wrap_two_pi into [0, 2 pi)");
    ok &= check(near(geo::wrap_elevation(0.3), 0.3) && near(geo::wrap_elevation(pi / 2 + 0.2), pi / 2 - 0.2) && near(geo::wrap_elevation(pi + 0.3), -0.3)
                    && near(geo::wrap_elevation(1.5 * pi + 0.2), -pi / 2 + 0.2),
                "wrap_elevation in every quadrant");
    ok &= check(std::isnan(geo::wrap_pi(std::numeric_limits<double>::infinity())), "wrap_pi of infinity is NaN");
    ok &= check(geo::wrap(angle_kind::linear, 7.0) == 7.0 && geo::wrap(angle_kind::bearing, 7.0) == geo::wrap_pi(7.0) && geo::wrap(angle_kind::elevation, 2.0) == geo::wrap_elevation(2.0),
                "wrap by kind");
    ok &= check(near(geo::difference(angle_kind::linear, 3.0, -3.0), 6.0) && near(geo::difference(angle_kind::bearing, 3.0, -3.0), 6.0 - 2 * pi), "difference by kind");
    const auto wv = geo::wrapped<kinds>(la::make_vector(3 * pi, pi + 0.3, 5.0));
    ok &= check(wv[0] == -pi && near(wv[1], -0.3) && wv[2] == 5.0, "wrapped per component");
    ok &= check(geo::wrapped<geo::all_linear<2>>(la::make_vector(10.0, -10.0)) == la::make_vector(10.0, -10.0), "all_linear leaves a vector alone");
    ok &= check(near(geo::radians(90.0), pi / 2) && near(geo::degrees(pi), 180.0), "radians and degrees");

    // The circular mean: equal weights, given weights, nothing, and a zero resultant
    const std::array<double, 2> across{geo::radians(170.0), geo::radians(-170.0)};
    ok &= check(near(std::abs(geo::circular_mean<double>(across)), pi), "circular_mean of 170 and -170 deg is 180 deg");
    const std::array<double, 2> wts{3.0, 1.0};
    ok &= check(near(geo::circular_mean<double>(across, wts), std::atan2(2 * std::sin(geo::radians(170.0)), 4 * std::cos(geo::radians(170.0)))), "circular_mean with weights");
    const std::array<double, 2> same{0.4, 0.4}, cancel{1.0, -1.0};
    ok &= check(geo::circular_mean<double>({}) == 0 && geo::circular_mean<double>(same, cancel) == 0, "no angles, or a zero resultant, give 0");

    std::printf(ok ? "geometry angle demo: ok\n" : "geometry angle demo: FAILED\n");
    return ok ? 0 : 1;
}
