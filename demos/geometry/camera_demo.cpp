//
//  camera_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/geometry/camera.hpp>: a camera on a 10 m pole looking out over water turns a detection's
//  pixel into a point on the surface with its covariance, then projects it back; the grazing angle, a target riding
//  above the surface, and both errors. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/geometry/camera.hpp>

#include <snicholls/core/geometry/coordinates.hpp>

#include <cmath>
#include <cstdio>
#include <numbers>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
using v2 = la::fixed_vector<double, 2>;
using v3 = la::fixed_vector<double, 3>;
using m22 = la::fixed_matrix<double, 2, 2>;
using m33 = la::fixed_matrix<double, 3, 3>;
constexpr double pi = std::numbers::pi;

bool check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    return ok;
}

} // namespace

int main()
{
    std::printf("snicholls-core geometry camera demo\n");
    bool ok = true;

    //! [camera]
    // A 1280 x 720 camera, 10 m up, its optical axis 20° below the horizon, looking along world y (north)
    constexpr geo::pinhole<double> k{1000, 1000, 640, 360};
    const m33 level{{1, 0, 0, 0, 0, 1, 0, -1, 0}};                // camera x east, y down, z north
    const m33 r = geo::rotation_x(-20 * pi / 180) * level;        // tilt the axis down
    const v3 c{{0, 0, 10}};
    const v3 up{{0, 0, 1}};                                       // the water: z = 0
    // A swimmer detected at pixel (700, 420), to 2 pixels; the camera's pointing known to 1 mrad (1 pixel)
    const v2 u{{700, 420}};
    const m22 pixel_cov = m22::identity() * (2.0 * 2.0 + 1.0 * 1.0);
    const auto fix = geo::plane_fix(k, r, c, u, pixel_cov, up, 0.0);
    if (!fix)
        return 1;
    // The error ellipse: long along the line of sight, short across it
    const auto e = la::symmetric_eigen(la::block<0, 0, 2, 2>(fix->covariance));
    //! [camera]
    std::printf("  fix (%.2f, %.2f, %.2f) m, range %.1f m, grazing %.2f deg\n", fix->point[0], fix->point[1], fix->point[2], fix->range, fix->grazing * 180 / pi);
    std::printf("  1-sigma ellipse: %.3f m along the line of sight, %.3f m across\n", std::sqrt(e->values[1]), std::sqrt(e->values[0]));
    ok &= check(std::abs(fix->point[2]) < 1e-12 && fix->point[1] > 0, "the fix is on the water, ahead of the camera");
    ok &= check(e->values[1] > 4 * e->values[0], "long along the ray, short across it");
    const auto back = geo::project(k, r, c, fix->point);
    ok &= check(back.is_ok() && std::abs((*back)[0] - 700) < 1e-9 && std::abs((*back)[1] - 420) < 1e-9, "project takes the fix back to its pixel");

    // The pieces: K, the pixel's ray, the intersection, the grazing angle
    ok &= check(k.matrix() == m33{{1000, 0, 640, 0, 1000, 360, 0, 0, 1}}, "pinhole::matrix is K");
    const auto d = geo::pixel_ray(k, r, u);
    const auto x = geo::ray_plane(c, d, up, 0.0);
    ok &= check(x.is_ok() && la::norm(*x - fix->point) < 1e-12 && std::abs(fix->range - la::norm(*x - c)) < 1e-12, "pixel_ray and ray_plane give the same point");
    ok &= check(std::abs(geo::grazing_angle(d, up) - fix->grazing) < 1e-15, "grazing_angle");
    const auto axis = geo::plane_fix(k, r, c, v2{{640, 360}}, pixel_cov, up, 0.0);
    ok &= check(std::abs(axis->grazing - 20 * pi / 180) < 1e-14 && std::abs(axis->range - 10 / std::sin(20 * pi / 180)) < 1e-9, "on the axis: grazing 20 deg, range 10 / sin 20");

    // A target riding 0.3 m above the water, give or take 0.2 m: the height adds a third axis to the covariance
    const auto riding = geo::plane_fix(k, r, c, u, pixel_cov, up, 0.3, 0.04);
    ok &= check(riding.is_ok() && std::abs(riding->point[2] - 0.3) < 1e-12 && la::symmetric_eigen(riding->covariance)->values[0] > 1e-6, "a height variance makes the covariance rank 3");

    // The errors
    ok &= check(geo::project(k, r, c, v3{{0, -50, 5}}).unwrap_err_unchecked() == geo::camera_error::behind_camera, "project of a point behind the camera");
    ok &= check(geo::plane_fix(k, level, c, v2{{640, 360}}, pixel_cov, up, 0.0).unwrap_err_unchecked() == geo::camera_error::parallel_to_plane, "a level ray never meets the water");
    ok &= check(geo::plane_fix(k, level, c, v2{{640, 0}}, pixel_cov, up, 0.0).unwrap_err_unchecked() == geo::camera_error::behind_camera, "a pixel above the horizon looks at the sky");
    ok &= check(geo::ray_plane(c, v3{{1, 0, 0}}, up, 0.0).unwrap_err_unchecked() == geo::camera_error::parallel_to_plane, "ray_plane: parallel");
    ok &= check(geo::ray_plane(c, v3{{0, 0, -1}}, up, 20.0).unwrap_err_unchecked() == geo::camera_error::behind_camera, "ray_plane: the plane is behind");

    std::printf(ok ? "geometry camera demo: ok\n" : "geometry camera demo: FAILED\n");
    return ok ? 0 : 1;
}
