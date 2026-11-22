//
//  camera_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/geometry/camera.hpp>: the pinhole matrix, project and pixel_ray as inverses, ray_plane's
//  intersection and both its errors (parallel at the documented tolerance, behind the camera), the grazing angle, and
//  plane_fix: its point, range and grazing angle, its covariance against a central-difference Jacobian with and
//  without a height variance, symmetry, rank and the plane normal in its null space, its errors, and constant evaluation.
//

#include <snicholls/core/geometry/camera.hpp>

#include <snicholls/core/geometry/coordinates.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
using v2 = la::fixed_vector<double, 2>;
using v3 = la::fixed_vector<double, 3>;
using m22 = la::fixed_matrix<double, 2, 2>;
using m33 = la::fixed_matrix<double, 3, 3>;
using geo::camera_error;
constexpr double pi = std::numbers::pi;

static_assert(sizeof(camera_error) == 1 && static_cast<int>(camera_error::behind_camera) == 1 && static_cast<int>(camera_error::parallel_to_plane) == 2);
static_assert(std::is_aggregate_v<geo::pinhole<double>> && std::is_trivially_copyable_v<geo::pinhole<double>> && sizeof(geo::pinhole<double>) == 4 * sizeof(double));
static_assert(std::is_aggregate_v<geo::plane_fix_result<double>> && std::is_trivially_copyable_v<geo::plane_fix_result<double>>);
static_assert(std::is_trivially_copyable_v<snicholls::result<geo::plane_fix_result<double>, camera_error>>);

constexpr geo::pinhole<double> k{800, 820, 640, 360};
// Looking straight down from 10 m: camera x is world x, camera y (down the image) is world -y, the optical axis is -z
constexpr m33 down{{1, 0, 0, 0, -1, 0, 0, 0, -1}};
constexpr v3 centre{{0, 0, 10}};
constexpr v3 up{{0, 0, 1}};
// Looking along world y, level: the optical axis is parallel to the ground
constexpr m33 level{{1, 0, 0, 0, 0, 1, 0, -1, 0}};

static_assert(noexcept(k.matrix()) && noexcept(geo::project(k, down, centre, centre)) && noexcept(geo::pixel_ray(k, down, v2{})));
static_assert(noexcept(geo::ray_plane(centre, up, up, 0.0)) && noexcept(geo::grazing_angle(up, up)) && noexcept(geo::plane_fix(k, down, centre, v2{}, m22{}, up, 0.0)));

// Constant evaluation: the point (1, 2, 0) is 10 m below on the camera's (0.1, -0.2) ray
static_assert(k.matrix() == m33{{800, 0, 640, 0, 820, 360, 0, 0, 1}});
static_assert(*geo::project(k, down, centre, v3{{1, 2, 0}}) == v2{{720, 196}});
static_assert(geo::project(k, down, centre, v3{{0, 0, 20}}).unwrap_err_unchecked() == camera_error::behind_camera);
static_assert(geo::pixel_ray(k, down, v2{{720, 196}}) == v3{{0.1, 0.2, -1}});
static_assert(*geo::ray_plane(centre, v3{{0.1, 0.2, -1}}, up, 0.0) == v3{{1, 2, 0}});
static_assert(geo::ray_plane(centre, v3{{1, 0, 0}}, up, 0.0).unwrap_err_unchecked() == camera_error::parallel_to_plane);
static_assert(geo::grazing_angle(v3{{0, 0, -1}}, up) == pi / 2);
constexpr auto cfix = geo::plane_fix(k, down, centre, v2{{640, 360}}, m22{{1, 0, 0, 1}}, up, 0.0);
static_assert(cfix.is_ok() && cfix->point == v3{{0, 0, 0}} && cfix->range == 10 && cfix->grazing == pi / 2);

template <std::size_t R, std::size_t C>
void expect_near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double tol)
{
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            EXPECT_NEAR(a(i, j), b(i, j), tol) << "(" << i << ", " << j << ")";
}

// A camera 10 m up, tilted 0.6 rad from straight down about the world x axis
const m33 tilted = geo::rotation_x(0.6) * down;

v3 fix_point(const m33& r, const v2& u, double h)
{
    return *geo::ray_plane(centre, geo::pixel_ray(k, r, u), up, h);
}

TEST(GeometryCameraPinhole, MatrixIsK)
{
    const geo::pinhole<double> p{500, 510, 320.5, 240.25};
    EXPECT_EQ(p.matrix(), (m33{{500, 0, 320.5, 0, 510, 240.25, 0, 0, 1}}));
    const geo::pinhole<float> f{1, 2, 3, 4};
    EXPECT_EQ(f.matrix()(1, 2), 4.0f);
}

TEST(GeometryCameraProject, PixelOfAPointBelowTheCamera)
{
    const auto u = geo::project(k, down, centre, v3{{1, 2, 0}});
    ASSERT_TRUE(u.is_ok());
    EXPECT_EQ(*u, (v2{{720, 196}}));
    EXPECT_EQ(*geo::project(k, down, centre, v3{{0, 0, -5}}), (v2{{640, 360}}));   // on the optical axis: the principal point
}

TEST(GeometryCameraProject, BehindOrBesideTheCameraIsAnError)
{
    EXPECT_EQ(geo::project(k, down, centre, v3{{0, 0, 11}}).unwrap_err_unchecked(), camera_error::behind_camera);
    EXPECT_EQ(geo::project(k, down, centre, v3{{3, 4, 10}}).unwrap_err_unchecked(), camera_error::behind_camera);   // in the camera's plane
    EXPECT_EQ(geo::project(k, down, centre, centre).unwrap_err_unchecked(), camera_error::behind_camera);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(geo::project(k, down, centre, v3{{0, 0, nan}}).unwrap_err_unchecked(), camera_error::behind_camera);
}

TEST(GeometryCameraPixelRay, HasCameraZOneAndInvertsProject)
{
    for (double u = 0; u <= 1280; u += 160)
        for (double v = 0; v <= 720; v += 120) {
            const v2 px{{u, v}};
            const auto d = geo::pixel_ray(k, tilted, px);
            EXPECT_NEAR((la::transpose(tilted) * d)[2], 1, 1e-15);
            // Any point along the ray projects back to the pixel
            const auto back = geo::project(k, tilted, centre, centre + d * 7.5);
            ASSERT_TRUE(back.is_ok());
            EXPECT_NEAR((*back)[0], u, 1e-9);
            EXPECT_NEAR((*back)[1], v, 1e-9);
        }
}

TEST(GeometryCameraRayPlane, MeetsThePlane)
{
    const auto x = geo::ray_plane(centre, v3{{0.1, 0.2, -1}}, up, 0.0);
    ASSERT_TRUE(x.is_ok());
    expect_near(*x, v3{{1, 2, 0}}, 1e-15);
    // A plane given by a normal that is not a unit vector: 2 z = 4, the plane z = 2
    expect_near(*geo::ray_plane(centre, v3{{0, 0, -3}}, v3{{0, 0, 2}}, 4.0), v3{{0, 0, 2}}, 1e-15);
    // A sloping plane x + z = 0
    const auto s = geo::ray_plane(centre, v3{{0, 0, -1}}, v3{{1, 0, 1}}, 0.0);
    expect_near(*s, v3{{0, 0, 0}}, 1e-15);
}

TEST(GeometryCameraRayPlane, ParallelAtTheDocumentedTolerance)
{
    EXPECT_EQ(geo::ray_plane(centre, v3{{1, 0, 0}}, up, 0.0).unwrap_err_unchecked(), camera_error::parallel_to_plane);
    EXPECT_EQ(geo::ray_plane(centre, v3{{0, 0, 0}}, up, 0.0).unwrap_err_unchecked(), camera_error::parallel_to_plane);   // no direction
    // |n·d| at or below 1e-12 |n| |d| is parallel; a little above meets the plane (far away, behind or in front)
    EXPECT_EQ(geo::ray_plane(centre, v3{{1, 0, -0.5e-12}}, up, 0.0).unwrap_err_unchecked(), camera_error::parallel_to_plane);
    EXPECT_TRUE(geo::ray_plane(centre, v3{{1, 0, -1e-11}}, up, 0.0).is_ok());
    EXPECT_EQ(geo::ray_plane(centre, v3{{1, 0, 1e-11}}, up, 0.0).unwrap_err_unchecked(), camera_error::behind_camera);
}

TEST(GeometryCameraRayPlane, BehindTheCameraIsAnError)
{
    EXPECT_EQ(geo::ray_plane(centre, v3{{0, 0, 1}}, up, 0.0).unwrap_err_unchecked(), camera_error::behind_camera);    // looking up at a floor below
    EXPECT_EQ(geo::ray_plane(centre, v3{{0, 0, -1}}, up, 20.0).unwrap_err_unchecked(), camera_error::behind_camera);  // a ceiling above, looking down
    EXPECT_EQ(geo::ray_plane(centre, v3{{0, 0, -1}}, up, 10.0).unwrap_err_unchecked(), camera_error::behind_camera);  // the camera is on the plane
}

TEST(GeometryCameraGrazingAngle, IsTheAngleBetweenRayAndPlane)
{
    EXPECT_DOUBLE_EQ(geo::grazing_angle(v3{{0, 0, -1}}, up), pi / 2);
    EXPECT_DOUBLE_EQ(geo::grazing_angle(v3{{0, 0, 5}}, up), pi / 2);       // either side, any length
    EXPECT_DOUBLE_EQ(geo::grazing_angle(v3{{1, 0, -1}}, up), pi / 4);
    EXPECT_DOUBLE_EQ(geo::grazing_angle(v3{{1, 0, -1}}, v3{{0, 0, -3}}), pi / 4);
    EXPECT_EQ(geo::grazing_angle(v3{{1, 2, 0}}, up), 0.0);
    EXPECT_NEAR(geo::grazing_angle(v3{{std::sqrt(3.0), 0, -1}}, up), pi / 6, 1e-15);
}

TEST(GeometryCameraPlaneFix, PointRangeAndGrazing)
{
    const v2 u{{900, 500}};
    const auto f = geo::plane_fix(k, tilted, centre, u, m22{{1, 0, 0, 1}}, up, 0.0);
    ASSERT_TRUE(f.is_ok());
    const auto x = fix_point(tilted, u, 0.0);
    expect_near(f->point, x, 1e-12);
    EXPECT_NEAR(f->point[2], 0, 1e-12);
    EXPECT_NEAR(f->range, la::norm(x - centre), 1e-12);
    EXPECT_NEAR(f->grazing, geo::grazing_angle(geo::pixel_ray(k, tilted, u), up), 1e-15);
    // project and plane_fix agree: the fix projects back to the pixel
    const auto back = geo::project(k, tilted, centre, f->point);
    EXPECT_NEAR((*back)[0], 900, 1e-9);
    EXPECT_NEAR((*back)[1], 500, 1e-9);
    // On the optical axis, tilted 0.6 rad from the vertical: grazing pi/2 - 0.6, range 10 / cos 0.6
    const auto axis = geo::plane_fix(k, tilted, centre, v2{{640, 360}}, m22{}, up, 0.0);
    EXPECT_NEAR(axis->grazing, pi / 2 - 0.6, 1e-14);
    EXPECT_NEAR(axis->range, 10 / std::cos(0.6), 1e-12);
}

TEST(GeometryCameraPlaneFix, CovarianceMatchesAFiniteDifferenceJacobian)
{
    const v2 u{{300, 650}};
    const m22 sigma{{4, 1, 1, 9}};
    const auto f = geo::plane_fix(k, tilted, centre, u, sigma, up, 0.0);
    ASSERT_TRUE(f.is_ok());
    la::fixed_matrix<double, 3, 2> j{};
    const double step = 1e-3;
    for (std::size_t q = 0; q < 2; ++q) {
        v2 a = u, b = u;
        a[q] += step;
        b[q] -= step;
        const auto col = (fix_point(tilted, a, 0.0) - fix_point(tilted, b, 0.0)) / (2 * step);
        for (std::size_t i = 0; i < 3; ++i)
            j(i, q) = col[i];
    }
    const auto ref = j * sigma * la::transpose(j);
    expect_near(f->covariance, ref, 1e-7 * la::norm(ref));
    // Symmetric, the plane's normal in its null space (the point stays on the plane), positive semi-definite, rank 2
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_EQ(f->covariance(r, c), f->covariance(c, r));
    EXPECT_NEAR(la::norm(f->covariance * up), 0, 1e-12 * la::norm(f->covariance));
    const auto e = la::symmetric_eigen(f->covariance);
    ASSERT_TRUE(e.is_ok());
    EXPECT_NEAR(e->values[0], 0, 1e-12 * e->values[2]);
    EXPECT_GT(e->values[1], 0);
}

TEST(GeometryCameraPlaneFix, HeightVarianceAddsTheHeightDerivative)
{
    const v2 u{{1000, 200}};
    const m22 sigma{{2.25, 0, 0, 2.25}};
    const double var_h = 0.04;
    const auto without = geo::plane_fix(k, tilted, centre, u, sigma, up, 0.3);
    const auto with = geo::plane_fix(k, tilted, centre, u, sigma, up, 0.3, var_h);
    ASSERT_TRUE(without.is_ok() && with.is_ok());
    EXPECT_EQ(with->point, without->point);
    EXPECT_NEAR(with->point[2], 0.3, 1e-12);
    const double step = 1e-4;
    const auto dh = (fix_point(tilted, u, 0.3 + step) - fix_point(tilted, u, 0.3 - step)) / (2 * step);
    expect_near(with->covariance - without->covariance, la::outer(dh, dh) * var_h, 1e-8);
    const auto e = la::symmetric_eigen(with->covariance);
    EXPECT_GT(e->values[0], 1e-6);   // rank 3
    EXPECT_EQ(geo::plane_fix(k, tilted, centre, u, sigma, up, 0.3, -1.0)->covariance, without->covariance);   // only a positive variance counts
}

TEST(GeometryCameraPlaneFix, ErrorLongAlongTheRayAtGrazingAngles)
{
    // The same pixel noise, a shallower look: the error grows like r sigma / tan(gamma)
    const m22 sigma{{1, 0, 0, 1}};
    const auto steep = geo::plane_fix(k, geo::rotation_x(0.2) * down, centre, v2{{640, 360}}, sigma, up, 0.0);
    const auto shallow = geo::plane_fix(k, geo::rotation_x(1.3) * down, centre, v2{{640, 360}}, sigma, up, 0.0);
    ASSERT_TRUE(steep.is_ok() && shallow.is_ok());
    const double ls = la::symmetric_eigen(steep->covariance)->values[2], lw = la::symmetric_eigen(shallow->covariance)->values[2];
    EXPECT_GT(lw, 100 * ls);
    // Along the ray, ∂x/∂θ = r / sin(gamma), and one pixel is 1/fy radians: the long axis is r / (fy sin gamma)
    EXPECT_NEAR(std::sqrt(lw), shallow->range / (k.fy * std::sin(shallow->grazing)), 1e-9 * std::sqrt(lw));
}

TEST(GeometryCameraPlaneFix, ErrorsComeFromTheRay)
{
    const m22 sigma{{1, 0, 0, 1}};
    // A level camera: the principal ray is parallel to the ground, and a pixel above the horizon looks up
    EXPECT_EQ(geo::plane_fix(k, level, centre, v2{{640, 360}}, sigma, up, 0.0).unwrap_err_unchecked(), camera_error::parallel_to_plane);
    EXPECT_EQ(geo::plane_fix(k, level, centre, v2{{640, 100}}, sigma, up, 0.0).unwrap_err_unchecked(), camera_error::behind_camera);
    EXPECT_TRUE(geo::plane_fix(k, level, centre, v2{{640, 700}}, sigma, up, 0.0).is_ok());   // below the horizon
}

TEST(GeometryCameraFloat, WorksInFloat)
{
    const geo::pinhole<float> kf{800, 820, 640, 360};
    const la::fixed_matrix<float, 3, 3> df{{1, 0, 0, 0, -1, 0, 0, 0, -1}};
    const la::fixed_vector<float, 3> cf{{0, 0, 10}}, nf{{0, 0, 1}};
    const auto f = geo::plane_fix(kf, df, cf, la::fixed_vector<float, 2>{{720, 196}}, la::fixed_matrix<float, 2, 2>{{1, 0, 0, 1}}, nf, 0.0f);
    ASSERT_TRUE(f.is_ok());
    EXPECT_NEAR(f->point[0], 1.0f, 1e-5f);
    EXPECT_NEAR(f->point[1], 2.0f, 1e-5f);
}

} // namespace
