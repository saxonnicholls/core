//
//  angular_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/metric/angular.hpp>: dot, squared_norm, norm and cosine_similarity, and the metrics
//  cosine_distance, angular, angle and inner_product, on every point kind and mixes of them, against Python; their
//  capabilities and noexcept claims; angular's reduced round trip; zero vectors, empty spans and the clamped cosine;
//  the axioms on random points for angular and angle, and explicit counterexamples to the triangle inequality for
//  cosine_distance and inner_product; and constant evaluation.
//

#include <snicholls/core/metric/angular.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

struct xyz {
    double x, y, z;
};

struct checked_row {
    std::array<double, 3> v;
};

} // namespace

template <>
struct snicholls::metric::point_traits<xyz> {
    using value_type = double;
    static constexpr std::size_t extent = 3;
    static constexpr std::size_t dimension(const xyz&) noexcept { return 3; }
    static constexpr double at(const xyz& p, std::size_t i) noexcept { return i == 0 ? p.x : i == 1 ? p.y : p.z; }
};

template <>
struct snicholls::metric::point_traits<checked_row> {
    using value_type = double;
    static constexpr std::size_t extent = 3;
    static constexpr std::size_t dimension(const checked_row&) noexcept { return 3; }
    static double at(const checked_row& p, std::size_t i)
    {
        if (i >= 3)
            throw std::out_of_range("checked_row");
        return p.v[i];
    }
};

namespace {

namespace mt = snicholls::metric;
namespace la = snicholls::linalg;
namespace st = snicholls::stats;

using a2d = std::array<double, 2>;
using a3d = std::array<double, 3>;
using a3f = std::array<float, 3>;
using a3i = std::array<int, 3>;
using a4d = std::array<double, 4>;

// ---- capabilities ----
static_assert(mt::distance_for<mt::cosine_distance, a3d> && mt::distance_for<mt::angle, a3f, xyz> && mt::distance_for<mt::inner_product, a3i>);
static_assert(mt::distance_for<mt::angular, std::vector<double>, la::fixed_vector<float, 3>>);
static_assert(mt::reducible<mt::angular, a3d> && mt::reducible<mt::angular, a3f, std::span<const double>>);
static_assert(!mt::reducible<mt::cosine_distance, a3d> && !mt::reducible<mt::angle, a3d> && !mt::reducible<mt::inner_product, a3d>);
static_assert(!mt::decomposable<mt::angular, double> && !mt::decomposable<mt::cosine_distance, double>);
static_assert(mt::true_metric<mt::angular> && mt::true_metric<mt::angle>);
static_assert(!mt::true_metric<mt::cosine_distance> && !mt::true_metric<mt::inner_product>);
static_assert(!mt::distance_for<mt::angle, a3d, a4d> && !mt::distance_for<mt::cosine_distance, int>);
static_assert(std::is_empty_v<mt::cosine_distance> && std::is_empty_v<mt::angular> && std::is_empty_v<mt::angle> && std::is_empty_v<mt::inner_product>);

// ---- types ----
static_assert(std::is_same_v<decltype(mt::dot(a3f{}, a3f{})), float> && std::is_same_v<decltype(mt::norm(a3i{})), double>);
static_assert(std::is_same_v<decltype(mt::cosine_similarity(a3f{}, a3d{})), double> && std::is_same_v<decltype(mt::angle{}(a3f{}, a3f{})), float>);

// ---- noexcept ----
static_assert(noexcept(mt::dot(std::declval<const a3d&>(), std::declval<const xyz&>())) && noexcept(mt::norm(std::declval<const a3d&>())));
static_assert(noexcept(mt::angular{}(std::declval<const a3d&>(), std::declval<const a3d&>())) && noexcept(mt::angular{}.expand(1.0)));
static_assert(!noexcept(mt::dot(std::declval<const checked_row&>(), std::declval<const a3d&>())));
static_assert(!noexcept(mt::squared_norm(std::declval<const checked_row&>())));
static_assert(!noexcept(mt::cosine_similarity(std::declval<const a3d&>(), std::declval<const checked_row&>())));
static_assert(!noexcept(mt::angle{}(std::declval<const checked_row&>(), std::declval<const checked_row&>())));
static_assert(!noexcept(mt::inner_product{}(std::declval<const checked_row&>(), std::declval<const a3d&>())));

// ---- constant evaluation ----
constexpr bool near(double x, double y, double tol = 1e-12) { return x - y < tol && y - x < tol; }
constexpr a2d ex{1, 0}, diag{1, 1}, ey{0, 1};
static_assert(mt::dot(ex, diag) == 1.0 && mt::squared_norm(diag) == 2.0 && near(mt::norm(a2d{3, 4}), 5.0));
static_assert(near(mt::cosine_similarity(ex, diag), 0.7071067811865476));       // 1 / sqrt(2), python3
static_assert(near(mt::cosine_distance{}(ex, diag), 0.29289321881345254));     // 1 - 1/sqrt(2), python3
static_assert(near(mt::angular{}(ex, diag), 0.7653668647301795));              // sqrt(2 - sqrt(2)), python3
static_assert(near(mt::angle{}(ex, diag), 0.7853981633974483));                // pi / 4, python3
static_assert(mt::inner_product{}(ex, diag) == -1.0 && mt::angular{}(ex, ex) == 0.0 && mt::cosine_similarity(ex, ey) == 0.0);

// The reference pair x = (1, 2, 3), y = (4, 5, 6)
constexpr double ref_dot = 32;                          // by hand
constexpr double ref_norm_x = 3.7416573867739413;       // sqrt(14), python3
constexpr double ref_norm_y = 8.774964387392123;        // sqrt(77), python3
constexpr double ref_cos = 0.9746318461970762;          // numpy: x @ y / norm(x) / norm(y)
constexpr double ref_cosine_distance = 0.025368153802923787;  // 1 - ref_cos, numpy
constexpr double ref_angular = 0.225247214424169;       // sqrt(2 - 2 ref_cos), python3
constexpr double ref_angle = 0.2257261285527342;        // acos(ref_cos), python3

template <typename F>
void every_kind(F f)
{
    const a3d ad{1, 2, 3}, bd{4, 5, 6};
    const a3f af{1, 2, 3}, bf{4, 5, 6};
    const a3i ai{1, 2, 3};
    const std::vector<double> bv{4, 5, 6};
    const std::span<const double, 3> as(ad);
    const std::span<const double> bs(bv);
    const double ac[3] = {1, 2, 3};
    const float bc[3] = {4, 5, 6};
    const auto afv = la::fixed_vector<float, 3>{{1, 2, 3}};
    const auto brow = la::fixed_matrix<double, 1, 3>{{4, 5, 6}};
    const xyz ax{1, 2, 3}, bx{4, 5, 6};
    f(ad, bd);
    f(af, bf);
    f(af, bd);
    f(ai, bv);
    f(as, bs);
    f(ac, bc);
    f(afv, brow);
    f(ax, bx);
    f(ax, bf);
    f(ac, brow);
}

template <typename F>
void expect_everywhere(F f, double expected)
{
    every_kind([&](const auto& a, const auto& b) {
        using T = decltype(f(a, b));
        static_assert(std::is_same_v<T, mt::distance_t<decltype(a), decltype(b)>>);
        const double tol = (std::is_same_v<T, float> ? 1e-5 : 1e-12) * std::abs(expected);
        EXPECT_NEAR(static_cast<double>(f(a, b)), expected, tol);
        EXPECT_NEAR(static_cast<double>(f(b, a)), expected, tol);
    });
}

TEST(MetricAngularValues, DotAndCosineAgreeWithNumpyOnEveryPointKind)
{
    expect_everywhere([](const auto& a, const auto& b) { return mt::dot(a, b); }, ref_dot);
    expect_everywhere([](const auto& a, const auto& b) { return mt::cosine_similarity(a, b); }, ref_cos);
}

TEST(MetricAngularValues, NormsAgreeWithPythonOnEveryPointKind)
{
    every_kind([](const auto& a, const auto& b) {
        EXPECT_NEAR(static_cast<double>(mt::norm(a)), ref_norm_x, 1e-5);
        EXPECT_NEAR(static_cast<double>(mt::norm(b)), ref_norm_y, 1e-5);
        EXPECT_NEAR(static_cast<double>(mt::squared_norm(a)), 14.0, 1e-5);
        EXPECT_NEAR(static_cast<double>(mt::squared_norm(b)), 77.0, 1e-4);
    });
    EXPECT_EQ(mt::norm(a3d{2, 3, 6}), 7.0);  // 4 + 9 + 36 = 49
}

TEST(MetricAngularValues, EveryMetricAgreesWithPythonOnEveryPointKind)
{
    expect_everywhere(mt::cosine_distance{}, ref_cosine_distance);
    expect_everywhere(mt::angular{}, ref_angular);
    expect_everywhere(mt::angle{}, ref_angle);
    expect_everywhere(mt::inner_product{}, -ref_dot);
}

TEST(MetricAngularValues, OppositeAndOrthogonalDirections)
{
    const a3d x{1, 2, 3};
    const std::vector<double> minus_x{-1, -2, -3};
    const xyz orth{3, 0, -1};  // x . orth = 0
    EXPECT_EQ(mt::cosine_similarity(x, minus_x), -1.0);
    EXPECT_EQ(mt::cosine_distance{}(x, minus_x), 2.0);
    EXPECT_EQ(mt::angular{}(x, minus_x), 2.0);
    EXPECT_DOUBLE_EQ(mt::angle{}(x, minus_x), 3.141592653589793);   // pi, python3
    EXPECT_EQ(mt::cosine_similarity(x, orth), 0.0);
    EXPECT_EQ(mt::cosine_distance{}(x, orth), 1.0);
    EXPECT_DOUBLE_EQ(mt::angular{}(x, orth), 1.4142135623730951);   // sqrt(2), python3
    EXPECT_DOUBLE_EQ(mt::angle{}(x, orth), 1.5707963267948966);     // pi / 2, python3
    // (1, 2, 3) against (4, -2, 1): -(4 - 4 + 3) / sqrt(14 * 21), python3
    EXPECT_DOUBLE_EQ(mt::cosine_similarity(a3d{-1, -2, -3}, a3d{4, -2, 1}), -0.1749635530559413);
}

TEST(MetricAngularReduced, AngularRoundTripsThroughTwoMinusTwoCosine)
{
    const mt::angular m;
    const a3d x{1, 2, 3};
    const a3f y{4, 5, 6};
    const double r = m.reduced(x, y);
    EXPECT_NEAR(r, 2 - 2 * ref_cos, 1e-7);
    EXPECT_DOUBLE_EQ(mt::reduced_distance(m, x, y), r);
    EXPECT_DOUBLE_EQ(m.expand(r), m(x, y));
    EXPECT_DOUBLE_EQ(mt::contract(m, m(x, y)), r);
    EXPECT_DOUBLE_EQ(mt::expand(m, mt::contract(m, 1.25)), 1.25);
    EXPECT_EQ(m.expand(-1e-17), 0.0);  // a reduced value rounded below zero expands to zero, not NaN
    EXPECT_EQ(m.contract(0.5f), 0.25f);
    // the others have no cheaper form: their reduced distance is the distance
    EXPECT_EQ(mt::reduced_distance(mt::angle{}, x, y), mt::angle{}(x, y));
    EXPECT_EQ(mt::reduced_distance(mt::cosine_distance{}, x, y), mt::cosine_distance{}(x, y));
    EXPECT_EQ(mt::expand(mt::inner_product{}, -3.0), -3.0);
    EXPECT_EQ(mt::contract(mt::angle{}, 0.5), 0.5);
}

TEST(MetricAngularEdge, AZeroVectorHasCosineZeroWithEverything)
{
    const a3d zero{0, 0, 0}, x{1, 2, 3};
    const la::fixed_vector<float, 3> fzero{};
    EXPECT_EQ(mt::cosine_similarity(zero, x), 0.0);
    EXPECT_EQ(mt::cosine_similarity(x, fzero), 0.0);
    EXPECT_EQ(mt::cosine_similarity(zero, zero), 0.0);
    EXPECT_EQ(mt::cosine_distance{}(zero, x), 1.0);
    EXPECT_DOUBLE_EQ(mt::angular{}(zero, zero), 1.4142135623730951);   // sqrt(2), as documented
    EXPECT_DOUBLE_EQ(mt::angle{}(x, zero), 1.5707963267948966);        // pi / 2
    EXPECT_EQ(mt::inner_product{}(zero, x), 0.0);
    EXPECT_EQ(mt::norm(zero), 0.0);
}

TEST(MetricAngularEdge, AnEmptySpanIsAZeroVector)
{
    const std::span<const double> e;
    EXPECT_EQ(mt::dot(e, e), 0.0);
    EXPECT_EQ(mt::norm(e), 0.0);
    EXPECT_EQ(mt::cosine_similarity(e, e), 0.0);
    EXPECT_EQ(mt::cosine_distance{}(e, e), 1.0);
    EXPECT_DOUBLE_EQ(mt::angle{}(e, e), 1.5707963267948966);
    EXPECT_EQ(mt::inner_product{}(e, e), 0.0);
}

TEST(MetricAngularEdge, InOneDimensionOnlyTheSignCounts)
{
    const std::array<double, 1> a{2.5};
    const double b[1] = {-0.1};
    EXPECT_EQ(mt::cosine_similarity(a, b), -1.0);
    EXPECT_EQ(mt::cosine_similarity(a, std::array<float, 1>{7}), 1.0);
    EXPECT_EQ(mt::angular{}(a, b), 2.0);
    EXPECT_DOUBLE_EQ(mt::inner_product{}(a, b), 0.25);
}

TEST(MetricAngularEdge, CosinesAreClampedSoParallelVectorsGiveExactlyOne)
{
    st::wyrand g(17);
    for (int k = 0; k < 2000; ++k) {
        a4d x{}, y{}, twice{};
        const double s = 0.1 + 10 * st::uniform_real(g);
        for (std::size_t i = 0; i < 4; ++i) {
            x[i] = 2 * st::uniform_real(g) - 1;
            y[i] = s * x[i];     // parallel, up to rounding
            twice[i] = 2 * x[i]; // parallel exactly
        }
        EXPECT_EQ(mt::cosine_similarity(x, x), 1.0);
        EXPECT_EQ(mt::cosine_similarity(x, twice), 1.0);
        EXPECT_EQ(mt::angle{}(x, twice), 0.0);
        EXPECT_EQ(mt::angular{}(x, twice), 0.0);   // a direction metric: x and 2x are the same direction
        const double c = mt::cosine_similarity(x, y);
        EXPECT_LE(c, 1.0);
        EXPECT_GE(c, 1.0 - 1e-15);
        EXPECT_FALSE(std::isnan(mt::angle{}(x, y)));
        EXPECT_FALSE(std::isnan(mt::angular{}(x, y)));
        for (auto& v : y)
            v = -v;
        EXPECT_GE(mt::cosine_similarity(x, y), -1.0);
        EXPECT_FALSE(std::isnan(mt::angle{}(x, y)));
    }
}

TEST(MetricAngularRelations, OnUnitVectorsEuclideanIsAngularAndTheOrdersAgree)
{
    st::wyrand g(23);
    for (int k = 0; k < 500; ++k) {
        a4d x{}, y{};
        for (std::size_t i = 0; i < 4; ++i) {
            x[i] = 2 * st::uniform_real(g) - 1;
            y[i] = 2 * st::uniform_real(g) - 1;
        }
        const double nx = mt::norm(x), ny = mt::norm(y);
        for (std::size_t i = 0; i < 4; ++i) {
            x[i] /= nx;
            y[i] /= ny;
        }
        const double c = mt::cosine_similarity(x, y);
        EXPECT_NEAR(mt::euclidean{}(x, y), mt::angular{}(x, y), 1e-12);
        EXPECT_NEAR(mt::angular{}.reduced(x, y), 2 * mt::cosine_distance{}(x, y), 1e-12);
        EXPECT_NEAR(mt::inner_product{}(x, y), mt::cosine_distance{}(x, y) - 1, 1e-12);
        EXPECT_NEAR(std::cos(mt::angle{}(x, y)), c, 1e-12);
    }
}

template <std::size_t D>
std::array<double, D> random_point(st::wyrand& g)
{
    std::array<double, D> p{};
    for (auto& x : p)
        x = 2 * st::uniform_real(g) - 1;
    return p;
}

template <typename M>
void expect_axioms(M m, std::uint64_t seed)
{
    static_assert(mt::true_metric<M>);
    st::wyrand g(seed);
    for (int k = 0; k < 500; ++k) {
        const auto x = random_point<5>(g), y = random_point<5>(g), z = random_point<5>(g);
        EXPECT_EQ(m(x, x), 0.0);
        EXPECT_GT(m(x, y), 0.0);
        EXPECT_EQ(m(x, y), m(y, x));
        EXPECT_LE(m(x, z), m(x, y) + m(y, z) + 1e-12);
    }
}

TEST(MetricAngularAxioms, AngularAndAngleObeyIdentitySymmetryAndTheTriangleInequality)
{
    expect_axioms(mt::angular{}, 31);
    expect_axioms(mt::angle{}, 32);
}

TEST(MetricAngularAxioms, CosineDistanceBreaksTheTriangleInequality)
{
    // (1, 0), (1, 1), (0, 1): d(x, z) = 1 > 2 (1 - 1/sqrt(2)) = 0.586, by hand
    const mt::cosine_distance m;
    EXPECT_GT(m(ex, ey), m(ex, diag) + m(diag, ey));
    EXPECT_NEAR(m(ex, diag) + m(diag, ey), 0.5857864376269049, 1e-12);  // 2 - sqrt(2), python3
    // angular and angle on the same three directions are fine
    EXPECT_LE(mt::angular{}(ex, ey), mt::angular{}(ex, diag) + mt::angular{}(diag, ey));
    EXPECT_NEAR(mt::angle{}(ex, ey), mt::angle{}(ex, diag) + mt::angle{}(diag, ey), 1e-12);  // on a great circle
}

TEST(MetricAngularAxioms, InnerProductIsNotADistance)
{
    // x = y = z = (1, 0): d(x, z) = -1 > d(x, y) + d(y, z) = -2, and d(x, x) is not zero
    const mt::inner_product m;
    EXPECT_GT(m(ex, ex), m(ex, ex) + m(ex, ex));
    EXPECT_EQ(m(ex, ex), -1.0);
    EXPECT_LT(m(a2d{2, 0}, ex), m(ex, ex));  // length counts: the longer vector is "nearer"
}

TEST(MetricAngularNoexcept, AThrowingCoordinateReadPropagates)
{
    const checked_row x{{1, 2, 3}};
    const a3d y{4, 5, 6};
    EXPECT_DOUBLE_EQ(mt::dot(x, y), ref_dot);
    EXPECT_DOUBLE_EQ(mt::angle{}(x, y), ref_angle);
}

} // namespace
