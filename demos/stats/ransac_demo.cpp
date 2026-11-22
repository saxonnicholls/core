//
//  ransac_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/ransac.hpp>: registering two sensors' frames when a quarter of the track
//  correspondences are wrong. RANSAC over align_2d on minimal samples finds the transform and the good pairs, a final
//  least-squares fit uses them all; ransac_iterations gives the budget, and the two errors. Exits non-zero if a result
//  is not what it documents.
//

#include <snicholls/core/stats/ransac.hpp>

#include <snicholls/core/geometry/alignment.hpp>
#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <span>
#include <vector>

namespace {

namespace st = snicholls::stats;
namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;

using v2 = la::fixed_vector<double, 2>;

int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

} // namespace

int main()
{
    std::printf("ransac: a robust fit that ignores the data that do not fit\n");

    // Sensor A sees 32 targets; sensor B sees them turned by 0.4 rad and shifted by (10, 2), with small noise, but
    // every fourth pair was associated with the wrong target
    st::wyrand g(2026);
    const double turn = 0.4;
    const geo::transform_2d<double> truth{{{std::cos(turn), -std::sin(turn), std::sin(turn), std::cos(turn)}}, v2{{10, 2}}, 1.0};
    std::vector<v2> a, b;
    for (int i = 0; i < 32; ++i) {
        a.push_back(v2{{st::uniform_real(g) * 50, st::uniform_real(g) * 50}});
        const v2 noise{{(st::uniform_real(g) - 0.5) * 0.02, (st::uniform_real(g) - 0.5) * 0.02}};
        b.push_back(i % 4 == 0 ? v2{{st::uniform_real(g) * 50, st::uniform_real(g) * 50}} : truth.apply(a.back()) + noise);
    }

    //! [ransac]
    // The model: align_2d on a minimal sample of two pairs. Its result tests as a model and dereferences to one
    auto fit = [&](std::span<const std::size_t> s) {
        const std::array<v2, 2> from{a[s[0]], a[s[1]]}, to{b[s[0]], b[s[1]]};
        return geo::align_2d<double>(from, to);
    };
    // The residual of pair i under a model, in metres
    auto residual = [&](const geo::transform_2d<double>& x, std::size_t i) { return la::norm(b[i] - x.apply(a[i])); };

    std::vector<std::size_t> inliers(a.size());
    const auto r = st::ransac<2>(a.size(), fit, residual, 0.1, inliers, g);
    // r->model is the best two-pair fit; inliers[0 .. r->inliers) are the pairs within 0.1 m of it

    // The final least-squares fit on every inlier
    std::vector<v2> ia, ib;
    for (std::size_t k = 0; r && k < r->inliers; ++k) {
        ia.push_back(a[inliers[k]]);
        ib.push_back(b[inliers[k]]);
    }
    const auto refined = geo::align_2d<double>(ia, ib);
    //! [ransac]

    bool none_wrong = r.is_ok();
    for (std::size_t k = 0; r && k < r->inliers; ++k)
        none_wrong = none_wrong && inliers[k] % 4 != 0;
    if (r)
        std::printf("    %zu inliers of %zu after %zu tries; angle %.5f, refined %.5f\n", r->inliers, a.size(), r->tries, r->model.angle(),
                    refined ? refined->angle() : 0.0);
    expect(r && r->inliers == 24 && none_wrong, "ransac: the 24 good pairs, none of the 8 wrong ones");
    expect(refined && std::abs(refined->angle() - turn) < 1e-3 && la::norm(refined->translation - truth.translation) < 1e-2,
           "the least-squares refit on the inliers recovers the transform");

    // The adaptive budget: ⌈log(1 - p) / log(1 - w^s)⌉
    expect(st::ransac_iterations(0.99, 0.75, 2) == 6 && st::ransac_iterations(0.99, 0.5, 4) == 72, "ransac_iterations: 6 tries for 75 % inliers, 72 for 4 of 50 %");
    expect(st::ransac_iterations(0.99, 1.0, 2) == 1 && st::ransac_iterations(0.99, 0.0, 2) == static_cast<std::size_t>(-1),
           "ransac_iterations: 1 with no outliers, SIZE_MAX with no inliers");
    expect(r && r->tries <= 1000, "the tries never exceed max_iterations");

    // The errors
    const auto few = st::ransac<2>(1, fit, residual, 0.1, inliers, g);
    expect(!few && few.unwrap_err_unchecked() == st::ransac_error::too_few_data, "one datum for a sample of two: too_few_data");
    const auto none = st::ransac<2>(a.size(), [](std::span<const std::size_t>) { return std::optional<geo::transform_2d<double>>{}; }, residual, 0.1,
                                    inliers, g, 20);
    expect(!none && none.unwrap_err_unchecked() == st::ransac_error::no_model, "every sample degenerate: no_model");

    std::printf(failures == 0 ? "ransac demo: ok\n" : "ransac demo: FAILED\n");
    return failures == 0 ? 0 : 1;
}
