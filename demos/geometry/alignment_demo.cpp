//
//  alignment_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/geometry/alignment.hpp>: registering one sensor's frame to another's from the targets both
//  see, in 2-D (rigid and with scale, weighted) and in 3-D (Horn's quaternion), the inverse transform, the RMS misfit,
//  and the two errors. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/geometry/alignment.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <span>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;

using v2 = la::fixed_vector<double, 2>;
using v3 = la::fixed_vector<double, 3>;

int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok)
        ++failures;
}

bool near(double a, double b, double t = 1e-9) { return std::abs(a - b) <= t; }

} // namespace

int main()
{
    std::printf("alignment: the least-squares transform between corresponding points\n");

    //! [alignment]
    // Four swimmers seen by phone A (source) and phone B (target). B's frame is A's turned 30 degrees, scaled by 2
    // (a different zoom) and shifted by (5, -1)
    const std::array<v2, 4> seen_by_a{v2{{0, 0}}, v2{{3, 0}}, v2{{3, 1}}, v2{{0, 2}}};
    const double turn = std::numbers::pi / 6;
    const geo::transform_2d<double> truth{{{std::cos(turn), -std::sin(turn), std::sin(turn), std::cos(turn)}}, v2{{5, -1}}, 2.0};
    std::array<v2, 4> seen_by_b{};
    for (std::size_t i = 0; i < 4; ++i)
        seen_by_b[i] = truth.apply(seen_by_a[i]);

    // Sim(2): rotation, translation and scale. The span's element type is named, since std::array does not deduce it
    const auto a_to_b = geo::align_2d<double>(seen_by_a, seen_by_b, {}, true);
    // a_to_b->angle() is pi / 6, a_to_b->scale is 2; a_to_b->apply(p) maps any point of A's frame into B's
    const auto b_to_a = a_to_b->inverse();
    //! [alignment]
    expect(a_to_b && near(a_to_b->angle(), turn) && near(a_to_b->scale, 2.0) && near(a_to_b->translation[0], 5) && near(a_to_b->translation[1], -1),
           "align_2d with scale: 30 degrees, scale 2, translation (5, -1)");
    if (a_to_b)
        std::printf("    angle %.6f rad, scale %.6f, translation (%.6f, %.6f)\n", a_to_b->angle(), a_to_b->scale, a_to_b->translation[0], a_to_b->translation[1]);
    const v2 back = b_to_a.apply(seen_by_b[2]);
    expect(near(back[0], 3) && near(back[1], 1), "inverse() maps B's view of swimmer 2 back to A's (3, 1)");
    expect(near(geo::rms_error(*a_to_b, std::span<const v2>(seen_by_a), std::span<const v2>(seen_by_b)), 0), "rms_error of the exact fit is 0");

    // SE(2): without scale, the angle is still found; the misfit is what the scale would have absorbed
    const auto rigid = geo::align_2d<double>(seen_by_a, seen_by_b);
    expect(rigid && rigid->scale == 1.0 && near(rigid->angle(), turn), "align_2d rigid: scale fixed at 1, the same angle");
    expect(geo::rms_error(*rigid, std::span<const v2>(seen_by_a), std::span<const v2>(seen_by_b)) > 1.0, "  and a non-zero rms_error");

    // Weights: a wrong correspondence given weight 0 does not move the fit
    std::array<v2, 4> with_mistake = seen_by_b;
    with_mistake[3] = v2{{-40, 40}};
    const std::array<double, 4> w{1, 1, 1, 0};
    const auto weighted = geo::align_2d<double>(seen_by_a, with_mistake, w, true);
    expect(weighted && near(weighted->angle(), turn) && near(weighted->scale, 2.0), "weights: a zero-weight mismatch is ignored");
    expect(near(geo::rms_error(*weighted, std::span<const v2>(seen_by_a), std::span<const v2>(with_mistake), std::span<const double>(w)), 0),
           "  and the weighted rms_error is 0");

    // 3-D: a camera against four surveyed landmarks, a quarter turn about z and a lift of 1
    const std::array<v3, 4> landmarks{v3{{0, 0, 0}}, v3{{1, 0, 0}}, v3{{0, 2, 0}}, v3{{0, 0, 3}}};
    const std::array<v3, 4> in_camera{v3{{0, 0, 1}}, v3{{0, 1, 1}}, v3{{-2, 0, 1}}, v3{{0, 0, 4}}};
    const auto pose = geo::align_3d<double>(landmarks, in_camera);
    expect(pose && near(pose->rotation(0, 1), -1) && near(pose->rotation(1, 0), 1) && near(pose->rotation(2, 2), 1) && near(pose->translation[2], 1),
           "align_3d: a quarter turn about z and (0, 0, 1)");
    const v3 l2 = pose->inverse().apply(in_camera[2]);
    expect(near(l2[0], 0) && near(l2[1], 2) && near(l2[2], 0), "transform_3d::inverse() returns landmark 2");
    std::array<v3, 4> zoomed{};
    for (std::size_t i = 0; i < 4; ++i)
        zoomed[i] = in_camera[i] * 3.0;
    const auto sim3 = geo::align_3d<double>(landmarks, zoomed, {}, true);
    expect(sim3 && near(sim3->scale, 3.0) && near(geo::rms_error(*sim3, std::span<const v3>(landmarks), std::span<const v3>(zoomed)), 0),
           "align_3d with scale: Sim(3), scale 3, rms_error 0");

    // The errors
    const auto few = geo::align_2d<double>(std::span<const v2>(seen_by_a).first(1), std::span<const v2>(seen_by_b).first(1));
    expect(!few && few.unwrap_err_unchecked() == geo::alignment_error::too_few_points, "one point in 2-D: too_few_points");
    const std::array<v3, 3> line{v3{{0, 0, 0}}, v3{{1, 1, 0}}, v3{{2, 2, 0}}};
    const auto collinear = geo::align_3d<double>(line, line);
    expect(!collinear && collinear.unwrap_err_unchecked() == geo::alignment_error::degenerate, "collinear points in 3-D: degenerate");

    const auto few3 = geo::align_3d<double>(std::span<const v3>(landmarks).first(2), std::span<const v3>(in_camera).first(2));
    expect(!few3 && few3.unwrap_err_unchecked() == geo::alignment_error::too_few_points, "two points in 3-D: too_few_points");
    const std::array<v2, 3> one_spot{v2{{1, 1}}, v2{{1, 1}}, v2{{1, 1}}};
    const auto coincident = geo::align_2d<double>(one_spot, std::span<const v2>(seen_by_b).first(3));
    expect(!coincident && coincident.unwrap_err_unchecked() == geo::alignment_error::degenerate, "coincident points in 2-D: degenerate");
    expect(geo::rms_error(geo::transform_2d<double>{}, std::span<const v2>{}, std::span<const v2>{}) == 0.0, "rms_error over no points is 0");

    std::printf(failures == 0 ? "alignment demo: ok\n" : "alignment demo: FAILED\n");
    return failures == 0 ? 0 : 1;
}
