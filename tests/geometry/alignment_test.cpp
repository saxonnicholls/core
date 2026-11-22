//
//  alignment_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/geometry/alignment.hpp>: exact recovery of rotation, translation and scale from noise-free
//  points in 2-D and 3-D (fixed and random transforms, half turns, minimal point sets, coplanar points), weights,
//  a proper rotation for mirrored data, inverse(), rms_error against closed forms, each error, float, and the
//  constexpr and noexcept claims.
//

#include <snicholls/core/geometry/alignment.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace geo = snicholls::geometry;
namespace la = snicholls::linalg;
namespace st = snicholls::stats;

using v2 = la::fixed_vector<double, 2>;
using v3 = la::fixed_vector<double, 3>;
using m3 = la::fixed_matrix<double, 3, 3>;

constexpr double tol = 1e-10;

la::fixed_matrix<double, 2, 2> rot2(double a)
{
    return {{std::cos(a), -std::sin(a), std::sin(a), std::cos(a)}};
}

// Rodrigues: the rotation by angle a about the unit axis n
m3 rot3(v3 n, double a)
{
    n = n / la::norm(n);
    const m3 k{{0, -n[2], n[1], n[2], 0, -n[0], -n[1], n[0], 0}};
    return m3::identity() + k * std::sin(a) + k * k * (1 - std::cos(a));
}

// A uniformly random rotation from a uniformly random unit quaternion (Shoemake's three uniforms)
m3 random_rotation(st::wyrand& g)
{
    const double u1 = st::uniform_real(g), u2 = st::uniform_real(g), u3 = st::uniform_real(g);
    const double tau = 2 * std::numbers::pi;
    const double w = std::sqrt(1 - u1) * std::sin(tau * u2), x = std::sqrt(1 - u1) * std::cos(tau * u2);
    const double y = std::sqrt(u1) * std::sin(tau * u3), z = std::sqrt(u1) * std::cos(tau * u3);
    return {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
             2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
             2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
}

template <std::size_t N>
std::vector<la::fixed_vector<double, N>> random_points(st::wyrand& g, std::size_t n)
{
    std::vector<la::fixed_vector<double, N>> p(n);
    for (auto& x : p)
        for (std::size_t a = 0; a < N; ++a)
            x[a] = st::uniform_real(g) * 20 - 10;
    return p;
}

template <typename X, std::size_t N>
std::vector<la::fixed_vector<double, N>> apply_all(const X& x, const std::vector<la::fixed_vector<double, N>>& p)
{
    std::vector<la::fixed_vector<double, N>> out;
    for (const auto& q : p)
        out.push_back(x.apply(q));
    return out;
}

template <std::size_t R, std::size_t C>
void expect_near(const la::fixed_matrix<double, R, C>& a, const la::fixed_matrix<double, R, C>& b, double t = tol)
{
    for (std::size_t i = 0; i < R * C; ++i)
        EXPECT_NEAR(a.elements[i], b.elements[i], t) << "element " << i;
}

double det3(const m3& r)
{
    return r(0, 0) * (r(1, 1) * r(2, 2) - r(1, 2) * r(2, 1)) - r(0, 1) * (r(1, 0) * r(2, 2) - r(1, 2) * r(2, 0))
           + r(0, 2) * (r(1, 0) * r(2, 1) - r(1, 1) * r(2, 0));
}

// ---- 2-D

TEST(GeometryAlign2d, RecoversARigidTransformExactly)
{
    st::wyrand g(1);
    const auto src = random_points<2>(g, 12);
    const geo::transform_2d<double> truth{rot2(0.7), v2{{3, -2}}, 1.0};
    const auto dst = apply_all(truth, src);
    const auto x = geo::align_2d<double>(src, dst);
    ASSERT_TRUE(x);
    EXPECT_NEAR(x->angle(), 0.7, tol);
    expect_near(x->rotation, truth.rotation);
    expect_near(x->translation, truth.translation);
    EXPECT_EQ(x->scale, 1.0);
    EXPECT_NEAR(geo::rms_error(*x, std::span<const v2>(src), std::span<const v2>(dst)), 0.0, tol);
}

TEST(GeometryAlign2d, RecoversASimilarityWithScale)
{
    st::wyrand g(2);
    const auto src = random_points<2>(g, 9);
    const geo::transform_2d<double> truth{rot2(-2.4), v2{{-1, 5}}, 2.5};
    const auto dst = apply_all(truth, src);
    const auto x = geo::align_2d<double>(src, dst, {}, true);
    ASSERT_TRUE(x);
    EXPECT_NEAR(x->angle(), -2.4, tol);
    EXPECT_NEAR(x->scale, 2.5, tol);
    expect_near(x->translation, truth.translation);
    // Without with_scale the scale stays 1 and the angle is still found (it does not depend on the scale)
    const auto rigid = geo::align_2d<double>(src, dst);
    ASSERT_TRUE(rigid);
    EXPECT_EQ(rigid->scale, 1.0);
    EXPECT_NEAR(rigid->angle(), -2.4, tol);
    EXPECT_GT(geo::rms_error(*rigid, std::span<const v2>(src), std::span<const v2>(dst)), 1.0);
}

TEST(GeometryAlign2d, RecoversEveryAngleOfAFullTurnFromTwoPoints)
{
    const std::vector<v2> src{v2{{0, 0}}, v2{{1, 0}}};   // the minimum: two distinct points
    for (int k = -11; k <= 12; ++k) {
        const double angle = k * std::numbers::pi / 12;
        const geo::transform_2d<double> truth{rot2(angle), v2{{0.5, 0.25}}, 1.0};
        const auto x = geo::align_2d<double>(src, apply_all(truth, src));
        ASSERT_TRUE(x) << k;
        EXPECT_NEAR(std::remainder(x->angle() - angle, 2 * std::numbers::pi), 0.0, tol) << k;   // a half turn may come back as -pi
        expect_near(x->translation, truth.translation);
    }
}

TEST(GeometryAlign2d, CollinearPointsDetermineThePlaneRotation)
{
    const std::vector<v2> src{v2{{0, 0}}, v2{{1, 1}}, v2{{2, 2}}, v2{{5, 5}}};
    const geo::transform_2d<double> truth{rot2(1.1), v2{{2, 3}}, 0.5};
    const auto x = geo::align_2d<double>(src, apply_all(truth, src), {}, true);
    ASSERT_TRUE(x);
    EXPECT_NEAR(x->angle(), 1.1, tol);
    EXPECT_NEAR(x->scale, 0.5, tol);
}

TEST(GeometryAlign2d, AZeroWeightIgnoresAWrongCorrespondence)
{
    st::wyrand g(3);
    const auto src = random_points<2>(g, 8);
    const geo::transform_2d<double> truth{rot2(0.3), v2{{1, 1}}, 1.0};
    auto dst = apply_all(truth, src);
    dst[5] = v2{{100, -100}};   // a mismatch
    std::vector<double> w(8, 1.0);
    w[5] = 0;
    const auto weighted = geo::align_2d<double>(src, dst, w);
    ASSERT_TRUE(weighted);
    EXPECT_NEAR(weighted->angle(), 0.3, tol);
    expect_near(weighted->translation, truth.translation);
    EXPECT_NEAR(geo::rms_error(*weighted, std::span<const v2>(src), std::span<const v2>(dst), std::span<const double>(w)), 0.0, tol);
    const auto plain = geo::align_2d<double>(src, dst);
    ASSERT_TRUE(plain);
    EXPECT_GT(std::abs(plain->angle() - 0.3), 1e-3);   // unweighted, the outlier pulls the fit
}

TEST(GeometryAlign2d, UniformWeightsGiveTheUnweightedAnswer)
{
    st::wyrand g(4);
    const auto src = random_points<2>(g, 6);
    auto dst = random_points<2>(g, 6);   // unrelated points: a genuine least-squares compromise
    const std::vector<double> w(6, 3.0);
    const auto a = geo::align_2d<double>(src, dst, {}, true);
    const auto b = geo::align_2d<double>(src, dst, w, true);
    ASSERT_TRUE(a && b);
    expect_near(a->rotation, b->rotation);
    expect_near(a->translation, b->translation);
    EXPECT_NEAR(a->scale, b->scale, tol);
}

TEST(GeometryAlign2d, ReportsTooFewPointsAndMismatchedLengths)
{
    const std::vector<v2> one{v2{{1, 2}}}, two{v2{{1, 2}}, v2{{3, 4}}}, three{v2{{1, 2}}, v2{{3, 4}}, v2{{5, 7}}};
    const std::vector<double> w1{1.0};
    for (const auto& r : {geo::align_2d<double>(one, one), geo::align_2d<double>(two, three), geo::align_2d<double>(two, two, w1),
                          geo::align_2d<double>(std::span<const v2>{}, std::span<const v2>{})}) {
        ASSERT_FALSE(r);
        EXPECT_EQ(r.unwrap_err_unchecked(), geo::alignment_error::too_few_points);
    }
}

TEST(GeometryAlign2d, ReportsDegenerateForCoincidentPointsOrNoWeight)
{
    const std::vector<v2> same{v2{{1, 2}}, v2{{1, 2}}, v2{{1, 2}}}, spread{v2{{0, 0}}, v2{{1, 0}}, v2{{0, 1}}};
    const std::vector<double> zero(3, 0.0);
    for (const auto& r : {geo::align_2d<double>(same, spread),      // source coincides
                          geo::align_2d<double>(spread, same),      // target coincides: no rotation determined
                          geo::align_2d<double>(spread, spread, zero)}) {
        ASSERT_FALSE(r);
        EXPECT_EQ(r.unwrap_err_unchecked(), geo::alignment_error::degenerate);
    }
}

TEST(GeometryTransform2d, DefaultIsTheIdentityAndInverseUndoesApply)
{
    const geo::transform_2d<double> id{};
    expect_near(id.apply(v2{{3, -4}}), v2{{3, -4}});
    EXPECT_EQ(id.angle(), 0.0);
    const geo::transform_2d<double> x{rot2(0.9), v2{{4, -1}}, 3.0};
    const auto inv = x.inverse();
    EXPECT_NEAR(inv.scale, 1.0 / 3, tol);
    EXPECT_NEAR(inv.angle(), -0.9, tol);
    for (const v2 p : {v2{{0, 0}}, v2{{1, 2}}, v2{{-7, 3}}}) {
        expect_near(inv.apply(x.apply(p)), p);
        expect_near(x.apply(inv.apply(p)), p);
    }
    // apply is s R x + t, written out
    expect_near(x.apply(v2{{1, 0}}), v2{{4 + 3 * std::cos(0.9), -1 + 3 * std::sin(0.9)}});
}

TEST(GeometryRmsError, IsTheWeightedRootMeanSquareMisfit)
{
    const geo::transform_2d<double> id{};
    const std::vector<v2> src{v2{{0, 0}}, v2{{1, 0}}, v2{{0, 1}}};
    const std::vector<v2> dst{v2{{3, 4}}, v2{{1, 0}}, v2{{0, 1}}};   // misfits 5, 0, 0
    EXPECT_NEAR(geo::rms_error(id, std::span<const v2>(src), std::span<const v2>(dst)), std::sqrt(25.0 / 3), tol);
    const std::vector<double> w{2, 1, 1};
    EXPECT_NEAR(geo::rms_error(id, std::span<const v2>(src), std::span<const v2>(dst), std::span<const double>(w)), std::sqrt(50.0 / 4), tol);
    EXPECT_EQ(geo::rms_error(id, std::span<const v2>{}, std::span<const v2>{}), 0.0);
    // The shorter of source and target bounds the sum
    EXPECT_NEAR(geo::rms_error(id, std::span<const v2>(src).first(1), std::span<const v2>(dst)), 5.0, tol);
}

// ---- 3-D

TEST(GeometryAlign3d, RecoversARigidTransformExactly)
{
    st::wyrand g(5);
    const auto src = random_points<3>(g, 10);
    const geo::transform_3d<double> truth{rot3(v3{{1, 2, 3}}, 0.8), v3{{-4, 1, 2}}, 1.0};
    const auto dst = apply_all(truth, src);
    const auto x = geo::align_3d<double>(src, dst);
    ASSERT_TRUE(x);
    expect_near(x->rotation, truth.rotation);
    expect_near(x->translation, truth.translation);
    EXPECT_EQ(x->scale, 1.0);
    EXPECT_NEAR(geo::rms_error(*x, std::span<const v3>(src), std::span<const v3>(dst)), 0.0, tol);
}

TEST(GeometryAlign3d, RecoversRandomRotationsAndScales)
{
    st::wyrand g(6);
    for (int trial = 0; trial < 200; ++trial) {
        const auto src = random_points<3>(g, 3 + st::uniform_index(g, 10));
        const geo::transform_3d<double> truth{random_rotation(g), v3{{st::uniform_real(g) * 10, -3, st::uniform_real(g)}}, 0.1 + 5 * st::uniform_real(g)};
        const auto dst = apply_all(truth, src);
        const auto x = geo::align_3d<double>(src, dst, {}, true);
        ASSERT_TRUE(x) << trial;
        expect_near(x->rotation, truth.rotation, 1e-8);
        expect_near(x->translation, truth.translation, 1e-7);
        EXPECT_NEAR(x->scale, truth.scale, 1e-9) << trial;
        EXPECT_NEAR(det3(x->rotation), 1.0, 1e-12);
    }
}

TEST(GeometryAlign3d, RecoversHalfTurnsAboutEachAxis)
{
    const std::vector<v3> src{v3{{0, 0, 0}}, v3{{1, 0, 0}}, v3{{0, 2, 0}}, v3{{0, 0, 3}}};
    for (const v3 axis : {v3{{1, 0, 0}}, v3{{0, 1, 0}}, v3{{0, 0, 1}}, v3{{1, 1, 0}}, v3{{1, -1, 1}}}) {
        const geo::transform_3d<double> truth{rot3(axis, std::numbers::pi), v3{{1, 2, 3}}, 1.0};
        const auto x = geo::align_3d<double>(src, apply_all(truth, src));
        ASSERT_TRUE(x);
        expect_near(x->rotation, truth.rotation);
        expect_near(x->translation, truth.translation);
    }
}

TEST(GeometryAlign3d, ThreeNonCollinearPointsAreEnoughAndCoplanarPointsWork)
{
    const std::vector<v3> three{v3{{0, 0, 0}}, v3{{1, 0, 0}}, v3{{0, 1, 0}}};
    const geo::transform_3d<double> truth{rot3(v3{{0, 1, 1}}, -1.3), v3{{0, 0, 7}}, 2.0};
    const auto x = geo::align_3d<double>(three, apply_all(truth, three), {}, true);
    ASSERT_TRUE(x);
    expect_near(x->rotation, truth.rotation);
    expect_near(x->translation, truth.translation);
    EXPECT_NEAR(x->scale, 2.0, tol);

    const std::vector<v3> plane{v3{{0, 0, 1}}, v3{{4, 0, 1}}, v3{{0, 3, 1}}, v3{{2, 2, 1}}, v3{{-1, 5, 1}}};
    const auto y = geo::align_3d<double>(plane, apply_all(truth, plane), {}, true);
    ASSERT_TRUE(y);
    expect_near(y->rotation, truth.rotation);
}

TEST(GeometryAlign3d, AZeroWeightIgnoresAWrongCorrespondence)
{
    st::wyrand g(8);
    const auto src = random_points<3>(g, 7);
    const geo::transform_3d<double> truth{rot3(v3{{3, -1, 2}}, 2.0), v3{{1, 1, 1}}, 1.0};
    auto dst = apply_all(truth, src);
    dst[2] = v3{{50, 50, 50}};
    std::vector<double> w(7, 0.5);
    w[2] = 0;
    const auto x = geo::align_3d<double>(src, dst, w);
    ASSERT_TRUE(x);
    expect_near(x->rotation, truth.rotation);
    expect_near(x->translation, truth.translation);
    EXPECT_NEAR(geo::rms_error(*x, std::span<const v3>(src), std::span<const v3>(dst), std::span<const double>(w)), 0.0, tol);
}

TEST(GeometryAlign3d, MirroredDataStillGivesAProperRotation)
{
    st::wyrand g(9);
    const auto src = random_points<3>(g, 8);
    std::vector<v3> dst;
    for (const auto& p : src)
        dst.push_back(v3{{p[0], p[1], -p[2]}});   // a reflection: no rotation fits exactly
    const auto x = geo::align_3d<double>(src, dst);
    ASSERT_TRUE(x);
    EXPECT_NEAR(det3(x->rotation), 1.0, 1e-12);
    expect_near(la::transpose(x->rotation) * x->rotation, m3::identity(), 1e-12);
    EXPECT_GT(geo::rms_error(*x, std::span<const v3>(src), std::span<const v3>(dst)), 0.1);
}

TEST(GeometryAlign3d, ReportsTooFewPointsAndMismatchedLengths)
{
    const std::vector<v3> two{v3{{0, 0, 0}}, v3{{1, 0, 0}}}, three{v3{{0, 0, 0}}, v3{{1, 0, 0}}, v3{{0, 1, 0}}};
    const std::vector<v3> four{v3{{0, 0, 0}}, v3{{1, 0, 0}}, v3{{0, 1, 0}}, v3{{0, 0, 1}}};
    const std::vector<double> w2{1.0, 1.0};
    for (const auto& r : {geo::align_3d<double>(two, two), geo::align_3d<double>(three, four), geo::align_3d<double>(three, three, w2)}) {
        ASSERT_FALSE(r);
        EXPECT_EQ(r.unwrap_err_unchecked(), geo::alignment_error::too_few_points);
    }
}

TEST(GeometryAlign3d, ReportsDegenerateForCoincidentOrCollinearPoints)
{
    const std::vector<v3> same(4, v3{{1, 2, 3}});
    const std::vector<v3> line{v3{{0, 0, 0}}, v3{{1, 1, 1}}, v3{{2, 2, 2}}, v3{{-3, -3, -3}}};
    const geo::transform_3d<double> truth{rot3(v3{{1, 0, 0}}, 0.5), v3{{1, 2, 3}}, 1.0};
    const std::vector<double> zero(4, 0.0);
    for (const auto& r : {geo::align_3d<double>(same, line), geo::align_3d<double>(line, apply_all(truth, line)),
                          geo::align_3d<double>(line, line), geo::align_3d<double>(line, apply_all(truth, line), zero)}) {
        ASSERT_FALSE(r);
        EXPECT_EQ(r.unwrap_err_unchecked(), geo::alignment_error::degenerate);
    }
}

TEST(GeometryTransform3d, DefaultIsTheIdentityAndInverseUndoesApply)
{
    const geo::transform_3d<double> id{};
    expect_near(id.apply(v3{{3, -4, 5}}), v3{{3, -4, 5}});
    const geo::transform_3d<double> x{rot3(v3{{2, 1, -1}}, 1.7), v3{{4, -1, 0.5}}, 0.25};
    const auto inv = x.inverse();
    EXPECT_NEAR(inv.scale, 4.0, tol);
    expect_near(inv.rotation, la::transpose(x.rotation));
    for (const v3 p : {v3{{0, 0, 0}}, v3{{1, 2, 3}}, v3{{-7, 3, 0}}}) {
        expect_near(inv.apply(x.apply(p)), p);
        expect_near(x.apply(inv.apply(p)), p);
    }
}

TEST(GeometryAlign2d, FloatPointsAreSupported)
{
    using f2 = la::fixed_vector<float, 2>;
    const std::vector<f2> src{f2{{0, 0}}, f2{{2, 0}}, f2{{0, 1}}};
    const std::vector<f2> dst{f2{{1, 1}}, f2{{1, 3}}, f2{{0, 1}}};   // a quarter turn, then (1, 1)
    const auto x = geo::align_2d<float>(src, dst);
    ASSERT_TRUE(x);
    EXPECT_NEAR(x->angle(), std::numbers::pi_v<float> / 2, 1e-5f);
    EXPECT_NEAR(x->translation[0], 1.0f, 1e-5f);
    EXPECT_NEAR(x->translation[1], 1.0f, 1e-5f);
}

// ---- Compile time: real alignments in constant expressions

constexpr bool compile_time_2d()
{
    // A quarter turn and (1, 1): (0, 0) -> (1, 1), (2, 0) -> (1, 3), (0, 1) -> (0, 1)
    const std::array<v2, 3> src{v2{{0, 0}}, v2{{2, 0}}, v2{{0, 1}}};
    const std::array<v2, 3> dst{v2{{1, 1}}, v2{{1, 3}}, v2{{0, 1}}};
    const auto x = geo::align_2d<double>(src, dst);
    const auto e = geo::align_2d<double>(std::span<const v2>(src).first(1), std::span<const v2>(dst).first(1));
    const auto close = [](double a, double b) { return a - b < 1e-12 && b - a < 1e-12; };
    return x && close(x->rotation(0, 0), 0) && close(x->rotation(1, 0), 1) && close(x->translation[0], 1) && close(x->translation[1], 1)
           && close(geo::rms_error(*x, std::span<const v2>(src), std::span<const v2>(dst)), 0) && !e
           && e.unwrap_err_unchecked() == geo::alignment_error::too_few_points;
}
static_assert(compile_time_2d());

constexpr bool compile_time_3d()
{
    // A quarter turn about z, then (0, 0, 1)
    const std::array<v3, 4> src{v3{{0, 0, 0}}, v3{{1, 0, 0}}, v3{{0, 2, 0}}, v3{{0, 0, 3}}};
    const std::array<v3, 4> dst{v3{{0, 0, 1}}, v3{{0, 1, 1}}, v3{{-2, 0, 1}}, v3{{0, 0, 4}}};
    const auto x = geo::align_3d<double>(src, dst);
    const auto close = [](double a, double b) { return a - b < 1e-9 && b - a < 1e-9; };
    const auto inv = x->inverse();
    return x && close(x->rotation(0, 1), -1) && close(x->rotation(1, 0), 1) && close(x->rotation(2, 2), 1) && close(x->translation[2], 1)
           && close(inv.apply(dst[2])[1], 2);
}
static_assert(compile_time_3d());

static_assert(noexcept(geo::align_2d<double>(std::span<const v2>{}, std::span<const v2>{})));
static_assert(noexcept(geo::align_3d<double>(std::span<const v3>{}, std::span<const v3>{}, std::span<const double>{}, true)));
static_assert(noexcept(geo::rms_error(std::declval<const geo::transform_3d<double>&>(), std::span<const v3>{}, std::span<const v3>{})));
static_assert(noexcept(std::declval<const geo::transform_2d<double>&>().inverse()));
static_assert(noexcept(std::declval<const geo::transform_3d<double>&>().apply(v3{})));

} // namespace
