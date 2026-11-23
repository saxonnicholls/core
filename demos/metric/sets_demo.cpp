//
//  sets_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/metric/sets.hpp>: a tracker's output scored against the truth at one instant by
//  OSPA and by GOSPA, GOSPA split into localisation, missed targets and false tracks, and by Hausdorff; the same
//  scene under another base metric and with other point and range types; positions on the Earth under haversine;
//  empty sets; a large solver on the heap; and each refusal. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/metric/sets.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/geodesic.hpp>
#include <snicholls/core/metric/minkowski.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace me = snicholls::metric;

using pt = la::fixed_vector<double, 2>;

bool check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core sets demo\n");
    bool ok = true;

    //! [metrics]
    // Three targets; the tracker has two of them, slightly off, one missed, and one false track
    const std::array<pt, 3> truths{pt{{0, 0}}, pt{{20, 0}}, pt{{40, 0}}};
    const std::array<pt, 3> estimates{pt{{19, 0}}, pt{{0, 2}}, pt{{70, 10}}};
    const double c = 10, p = 2;   // an error over 10 counts as a miss; order 2

    me::set_metric_solver<double, 8, 8> solver;   // sized at compile time: up to 8 points each side
    const auto ospa = solver.ospa(truths, estimates, c, p).unwrap();
    const auto gospa = solver.gospa(truths, estimates, c, p).unwrap();   // α = 2
    const double worst = me::hausdorff(truths, estimates);               // the worst nearest-neighbour distance

    // Any base metric, any random-access range of any point type
    const std::vector<std::array<double, 2>> plain{{0, 0}, {20, 0}, {40, 0}};
    const auto ospa_l1 = solver.ospa(plain, estimates, c, p, me::manhattan{}).unwrap();
    //! [metrics]

    std::printf("  OSPA %.4f; GOSPA %.4f = localisation %.1f, missed %zu, false %zu\n", ospa, gospa.distance, gospa.localisation, gospa.missed, gospa.false_tracks);
    // By hand: pairs at 2 and 1, the third pair beyond c costs c^p: OSPA² = (4 + 1 + 100) / 3 = 35
    ok &= check(std::abs(ospa - std::sqrt(35.0)) < 1e-12, "OSPA by hand");
    ok &= check(ospa >= 0 && ospa <= c, "OSPA lies in [0, c]");
    ok &= check(std::abs(gospa.distance - std::sqrt(105.0)) < 1e-12, "GOSPA by hand: 4 + 1 + 100");
    ok &= check(gospa.localisation == 5 && gospa.missed == 1 && gospa.false_tracks == 1, "GOSPA's parts: localisation 5, one missed, one false");
    ok &= check(std::abs(gospa.distance * gospa.distance - (gospa.localisation + c * c / 2 * 2)) < 1e-12, "distance^p = localisation + c^p / 2 (missed + false)");
    const double gospa_a1 = solver.gospa(truths, std::span<const pt>(estimates).first(2), c, p, 1.0).unwrap().distance;
    ok &= check(std::abs(gospa_a1 - std::sqrt(5.0 + 100.0)) < 1e-12, "alpha = 1: a cardinality error costs the full c^p");

    std::printf("  Hausdorff %.4f; OSPA under manhattan %.4f\n", worst, ospa_l1);
    ok &= check(std::abs(worst - std::sqrt(30.0 * 30.0 + 10.0 * 10.0)) < 1e-12, "Hausdorff: the false track (70, 10) is sqrt 1000 from (40, 0)");
    ok &= check(std::abs(ospa_l1 - std::sqrt(35.0)) < 1e-12, "manhattan: the same pairs, the same distances along an axis");
    ok &= check(std::abs(solver.ospa(plain, estimates, c, p).unwrap() - ospa) < 1e-14, "a vector of std::array scores as the fixed_vectors do");
    ok &= check(me::hausdorff(std::span<const pt>{}, std::span<const pt>{}) == 0 && me::hausdorff(truths, std::span<const pt>{}) == std::numeric_limits<double>::infinity(),
                "Hausdorff of two empty sets is 0, of one empty set infinity");

    // Positions on the Earth, (latitude, longitude) in radians, scored in metres
    const auto deg = [](double lat, double lon) { return std::array<double, 2>{lat * std::numbers::pi / 180, lon * std::numbers::pi / 180}; };
    const std::array<std::array<double, 2>, 1> london{deg(51.5074, -0.1278)}, paris{deg(48.8566, 2.3522)};
    const double geo = solver.ospa(london, paris, 1e6, 1.0, me::haversine{}).unwrap();
    ok &= check(std::abs(geo - 343556.535) < 0.01, "haversine: London against Paris, 343.557 km");
    ok &= check(std::abs(solver.gospa(london, paris, 1e6, 1.0, 2.0, me::haversine{}).unwrap().localisation - geo) < 1e-9, "and the same as GOSPA's localisation");

    // Empty sets
    ok &= check(solver.ospa(std::span<const pt>{}, std::span<const pt>{}, c).unwrap() == 0, "OSPA of two empty sets is 0");
    ok &= check(solver.ospa(truths, std::span<const pt>{}, c).unwrap() == c, "OSPA with nothing estimated is c");
    const auto nothing = solver.gospa(truths, std::span<const pt>{}, c, p).unwrap();
    ok &= check(nothing.missed == 3 && std::abs(nothing.distance - std::sqrt(150.0)) < 1e-12, "GOSPA with nothing estimated: every truth missed");

    // A big scene: allocate the solver once, on the heap
    auto big = std::make_unique<me::set_metric_solver<double, 128, 128>>();
    std::vector<pt> grid, shifted;
    for (int i = 0; i < 100; ++i) {
        grid.push_back(pt{{10.0 * (i % 10), 10.0 * (i / 10)}});
        shifted.push_back(pt{{10.0 * (i % 10) + 0.5, 10.0 * (i / 10)}});
    }
    ok &= check(std::abs(big->ospa(grid, shifted, 3.0).unwrap() - 0.5) < 1e-12, "100 points each shifted by 0.5: OSPA 0.5");

    // Each refusal
    const auto bad_c = solver.ospa(truths, estimates, 0.0);
    ok &= check(bad_c.is_err() && bad_c.unwrap_err_unchecked() == me::set_metric_error::invalid_parameters, "c = 0 is invalid_parameters");
    const auto bad_alpha = solver.gospa(truths, estimates, c, p, 3.0);
    ok &= check(bad_alpha.is_err() && bad_alpha.unwrap_err_unchecked() == me::set_metric_error::invalid_parameters, "alpha outside (0, 2] is invalid_parameters");
    ok &= check(solver.gospa(truths, estimates, c, 0.5).is_err(), "p < 1 is invalid_parameters");
    me::set_metric_solver<double, 2, 2> tiny;
    const auto too_large = tiny.ospa(truths, estimates, c);
    ok &= check(too_large.is_err() && too_large.unwrap_err_unchecked() == me::set_metric_error::too_large, "more points than the solver holds is too_large");
    ok &= check(tiny.gospa(truths, estimates, c).is_err(), "and for GOSPA");

    std::printf(ok ? "sets demo: ok\n" : "sets demo: FAILED\n");
    return ok ? 0 : 1;
}
