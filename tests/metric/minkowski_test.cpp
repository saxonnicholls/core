//
//  minkowski_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/metric/minkowski.hpp>: euclidean, manhattan, chebyshev, minkowski<P>, squared_euclidean
//  and weighted_minkowski on every point kind and mixes of them, against numpy; their capabilities and noexcept
//  claims; the reduced, expand and contract round trips; early-exit accumulation; the metric axioms on random points
//  (and squared_euclidean's failing triangle inequality); edge cases (empty, one dimension, unit weights, minkowski<1>
//  and <2>, large orders approaching chebyshev); and constant evaluation.
//

#include <snicholls/core/metric/minkowski.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/concepts.hpp>
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

using a3d = std::array<double, 3>;
using a3f = std::array<float, 3>;
using a3i = std::array<int, 3>;
using a4d = std::array<double, 4>;

// ---- capabilities ----
static_assert(mt::distance_for<mt::euclidean, a3d> && mt::reducible<mt::euclidean, a3d, a3f> && mt::decomposable<mt::euclidean, double>);
static_assert(mt::reducible<mt::manhattan, a3i> && mt::decomposable<mt::manhattan, float> && mt::decomposable<mt::minkowski<3>, double>);
static_assert(mt::reducible<mt::chebyshev, xyz, a3d> && mt::decomposable<mt::chebyshev, double>);
static_assert(mt::reducible<mt::squared_euclidean, a3d> && mt::decomposable<mt::squared_euclidean, double>);
static_assert(mt::reducible<mt::weighted_minkowski<2, a3d>, a3d> && mt::decomposable<mt::weighted_minkowski<1, la::fixed_vector<double, 3>>, double>);
static_assert(mt::true_metric<mt::euclidean> && mt::true_metric<mt::manhattan> && mt::true_metric<mt::chebyshev> && mt::true_metric<mt::minkowski<7>>);
static_assert(mt::true_metric<mt::weighted_minkowski<2, a3d>> && !mt::true_metric<mt::squared_euclidean>);
static_assert(!mt::distance_for<mt::euclidean, a3d, a4d> && !mt::distance_for<mt::chebyshev, int> && !mt::distance_for<mt::manhattan, std::vector<bool>>);
static_assert(std::is_same_v<mt::euclidean, mt::minkowski<2>> && std::is_same_v<mt::manhattan, mt::minkowski<1>>);
static_assert(std::is_empty_v<mt::euclidean> && std::is_empty_v<mt::chebyshev> && std::is_empty_v<mt::squared_euclidean>);
static_assert(std::is_trivially_copyable_v<mt::weighted_minkowski<2, a3d>> && sizeof(mt::weighted_minkowski<2, a3d>) == sizeof(a3d));

// ---- the distance type ----
static_assert(std::is_same_v<decltype(mt::euclidean{}(a3f{}, a3f{})), float>);
static_assert(std::is_same_v<decltype(mt::euclidean{}(a3f{}, a3d{})), double>);
static_assert(std::is_same_v<decltype(mt::chebyshev{}(a3i{}, a3i{})), double>);
static_assert(std::is_same_v<decltype(mt::manhattan{}.reduced(a3i{}, a3f{})), double>);

// ---- noexcept ----
static_assert(noexcept(mt::euclidean{}(std::declval<const a3d&>(), std::declval<const xyz&>())));
static_assert(noexcept(mt::chebyshev{}.reduced(std::declval<const std::vector<double>&>(), std::declval<const a3d&>())));
static_assert(noexcept(mt::weighted<2>(a3d{1, 1, 1})(std::declval<const a3d&>(), std::declval<const a3d&>())));
static_assert(!noexcept(mt::euclidean{}(std::declval<const checked_row&>(), std::declval<const a3d&>())));
static_assert(!noexcept(mt::squared_euclidean{}(std::declval<const a3d&>(), std::declval<const checked_row&>())));
static_assert(!noexcept(mt::weighted<1>(checked_row{})(std::declval<const a3d&>(), std::declval<const a3d&>())));
static_assert(noexcept(mt::euclidean{}.expand(2.0)) && noexcept(mt::minkowski<3>{}.contract(2.0f)));

// ---- constant evaluation (hand values) ----
constexpr a3d o{0, 0, 0};
constexpr a3d p{3, 4, 0};
constexpr bool near(double x, double y, double tol = 1e-12) { return x - y < tol && y - x < tol; }
static_assert(near(mt::euclidean{}(o, p), 5.0));               // 3-4-5
static_assert(mt::euclidean{}.reduced(o, p) == 25.0);
static_assert(mt::manhattan{}(o, p) == 7.0 && mt::chebyshev{}(o, p) == 4.0 && mt::squared_euclidean{}(o, p) == 25.0);
static_assert(near(mt::minkowski<3>{}(o, p), 4.497941445275415));  // 91^(1/3), python3
static_assert(near(mt::weighted<2>(a3d{1, 0.25, 9})(o, p), 3.605551275463989));  // sqrt(9 + 4), python3
static_assert(mt::accumulate_within(mt::euclidean{}, o, p, 1.0) == 9.0);

// The reference pair a = (1, -2, 3), b = (4, 2, -1): differences (-3, -4, 4)
constexpr double ref_euclidean = 6.4031242374328485;  // numpy.linalg.norm
constexpr double ref_manhattan = 11;                  // by hand
constexpr double ref_chebyshev = 4;                   // by hand
constexpr double ref_squared = 41;                    // 9 + 16 + 16
constexpr double ref_minkowski3 = 5.371685354944832;  // 155 ** (1/3), python3
constexpr double ref_minkowski64 = 4.04355714452557;  // numpy: sum(|d|**64) ** (1/64)

// Calls f on the reference pair in every point kind, mixed
template <typename F>
void every_kind(F f)
{
    const a3d ad{1, -2, 3}, bd{4, 2, -1};
    const a3f af{1, -2, 3}, bf{4, 2, -1};
    const a3i ai{1, -2, 3};
    const std::vector<double> bv{4, 2, -1};
    const std::span<const double, 3> as(ad);
    const std::span<const double> bs(bv);
    const double ac[3] = {1, -2, 3};
    const float bc[3] = {4, 2, -1};
    const auto afv = la::fixed_vector<float, 3>{{1, -2, 3}};
    const auto brow = la::fixed_matrix<double, 1, 3>{{4, 2, -1}};
    const xyz ax{1, -2, 3}, bx{4, 2, -1};
    f(ad, bd);
    f(af, bf);    // float in float
    f(af, bd);    // float against double
    f(ai, bv);    // int against a vector
    f(as, bs);    // a static span against a dynamic one
    f(ac, bc);    // C arrays of double and float
    f(afv, brow); // a column against a row
    f(ax, bx);    // the test's own struct
    f(ax, bf);
    f(ac, brow);
}

template <typename M>
void expect_everywhere(M m, double expected)
{
    every_kind([&](const auto& a, const auto& b) {
        using T = decltype(m(a, b));
        static_assert(std::is_same_v<T, mt::distance_t<decltype(a), decltype(b)>>);
        const double tol = std::is_same_v<T, float> ? 1e-5 * expected : 1e-12 * expected;
        EXPECT_NEAR(static_cast<double>(m(a, b)), expected, tol);
        EXPECT_NEAR(static_cast<double>(m(b, a)), expected, tol);
    });
}

TEST(MetricMinkowskiValues, EuclideanAgreesWithNumpyOnEveryPointKind) { expect_everywhere(mt::euclidean{}, ref_euclidean); }
TEST(MetricMinkowskiValues, ManhattanAgreesWithHandOnEveryPointKind) { expect_everywhere(mt::manhattan{}, ref_manhattan); }
TEST(MetricMinkowskiValues, ChebyshevAgreesWithHandOnEveryPointKind) { expect_everywhere(mt::chebyshev{}, ref_chebyshev); }
TEST(MetricMinkowskiValues, SquaredEuclideanAgreesWithHandOnEveryPointKind) { expect_everywhere(mt::squared_euclidean{}, ref_squared); }
TEST(MetricMinkowskiValues, OrderThreeAgreesWithPythonOnEveryPointKind) { expect_everywhere(mt::minkowski<3>{}, ref_minkowski3); }

TEST(MetricMinkowskiWeighted, AgreesWithPythonForEveryKindOfWeights)
{
    // sqrt(1*9 + 2*16 + 0.5*16) = 7, by hand
    expect_everywhere(mt::weighted<2>(a3d{1, 2, 0.5}), 7.0);
    expect_everywhere(mt::weighted<2>(la::fixed_vector<float, 3>{{1, 2, 0.5f}}), 7.0);
    const std::vector<double> w{1, 2, 0.5};
    expect_everywhere(mt::weighted<2>(std::span<const double>(w)), 7.0);
    // (0.5*27 + 64 + 2*64) ** (1/3), python3
    expect_everywhere(mt::weighted<3>(a3d{0.5, 1, 2}), 5.901158443409004);
    // 1*3 + 2*4 + 0.5*4 = 13, by hand
    expect_everywhere(mt::weighted<1>(xyz{1, 2, 0.5}), 13.0);
}

TEST(MetricMinkowskiWeighted, UnitWeightsGiveTheUnweightedMetric)
{
    st::wyrand g(11);
    const auto w2 = mt::weighted<2>(a4d{1, 1, 1, 1});
    const auto w1 = mt::weighted<1>(la::fixed_vector<double, 4>{{1, 1, 1, 1}});
    const auto w5 = mt::weighted<5>(std::array<int, 4>{1, 1, 1, 1});
    for (int k = 0; k < 100; ++k) {
        a4d a{}, b{};
        for (std::size_t i = 0; i < 4; ++i) {
            a[i] = 2 * st::uniform_real(g) - 1;
            b[i] = 2 * st::uniform_real(g) - 1;
        }
        EXPECT_DOUBLE_EQ(w2(a, b), mt::euclidean{}(a, b));
        EXPECT_DOUBLE_EQ(w1(a, b), mt::manhattan{}(a, b));
        EXPECT_DOUBLE_EQ(w5(a, b), mt::minkowski<5>{}(a, b));
        EXPECT_DOUBLE_EQ(w2.reduced(a, b), mt::squared_euclidean{}(a, b));
    }
}

TEST(MetricMinkowskiOrder, LargeOrdersApproachChebyshevFromAbove)
{
    const a3d a{1, -2, 3}, b{4, 2, -1};
    EXPECT_NEAR(mt::minkowski<64>{}(a, b), ref_minkowski64, 1e-12);
    const double l[] = {mt::minkowski<1>{}(a, b), mt::minkowski<2>{}(a, b), mt::minkowski<3>{}(a, b), mt::minkowski<8>{}(a, b),
                        mt::minkowski<64>{}(a, b), mt::minkowski<200>{}(a, b)};
    for (std::size_t i = 1; i < std::size(l); ++i)
        EXPECT_LT(l[i], l[i - 1]);
    // chebyshev <= L_P <= n^(1/P) chebyshev
    EXPECT_GT(l[5], ref_chebyshev);
    EXPECT_LT(l[5], std::pow(3.0, 1.0 / 200) * ref_chebyshev);
    EXPECT_NEAR(l[5], 4.013886994038011, 1e-12);  // (2 * 4**200 + 3**200) ** (1/200), python3
}

TEST(MetricMinkowskiOrder, OrdersOneAndTwoAreManhattanAndEuclidean)
{
    st::wyrand g(5);
    for (int k = 0; k < 50; ++k) {
        std::vector<double> a(7), b(7);
        for (std::size_t i = 0; i < 7; ++i) {
            a[i] = st::uniform_real(g);
            b[i] = st::uniform_real(g);
        }
        EXPECT_EQ(mt::minkowski<1>{}(a, b), mt::manhattan{}(a, b));
        EXPECT_EQ(mt::minkowski<2>{}(a, b), mt::euclidean{}(a, b));
        EXPECT_DOUBLE_EQ(mt::euclidean{}(a, b) * mt::euclidean{}(a, b), mt::squared_euclidean{}(a, b));
    }
}

TEST(MetricMinkowskiReduced, ExpandAndContractRoundTrip)
{
    const a3d a{1, -2, 3};
    const xyz b{4, 2, -1};
    const auto check = [&](auto m) {
        const double d = m(a, b), r = m.reduced(a, b);
        EXPECT_DOUBLE_EQ(mt::reduced_distance(m, a, b), r);
        EXPECT_DOUBLE_EQ(mt::expand(m, r), d);
        EXPECT_NEAR(mt::contract(m, d), r, 1e-12 * r);
        EXPECT_NEAR(mt::expand(m, mt::contract(m, 2.5)), 2.5, 1e-12);
        EXPECT_DOUBLE_EQ(mt::accumulate(m, a, b), r);
    };
    check(mt::euclidean{});
    check(mt::manhattan{});
    check(mt::chebyshev{});
    check(mt::squared_euclidean{});
    check(mt::minkowski<3>{});
    check(mt::minkowski<64>{});
    check(mt::weighted<2>(a3d{1, 2, 0.5}));
    check(mt::weighted<3>(a3d{0.5, 1, 2}));
    EXPECT_EQ(mt::euclidean{}.reduced(a, b), ref_squared);   // the squared distance
    EXPECT_EQ(mt::minkowski<3>{}.reduced(a, b), 155.0);      // 27 + 64 + 64
    EXPECT_EQ(mt::minkowski<3>{}.contract(-2.0), 8.0);       // |d|^P
    EXPECT_EQ(mt::minkowski<5>{}.expand(0.0), 0.0);          // the root of zero is zero, not NaN
    EXPECT_EQ(mt::minkowski<5>{}.expand(32.0f), 2.0f);
}

TEST(MetricMinkowskiReduced, AccumulateWithinExitsEarlyAboveTheBound)
{
    const a4d a{0, 0, 0, 0};
    const std::array<int, 4> b{1, 2, 3, 4};
    // squared terms 1, 4, 9, 16: partial sums 1, 5, 14, 30
    EXPECT_EQ(mt::accumulate_within(mt::euclidean{}, a, b, 30.0), 30.0);
    EXPECT_EQ(mt::accumulate_within(mt::euclidean{}, a, b, 4.0), 5.0);
    EXPECT_EQ(mt::accumulate_within(mt::euclidean{}, a, b, 5.0), 14.0);
    // chebyshev's terms combine by the maximum: 1, 2, 3, 4
    EXPECT_EQ(mt::accumulate_within(mt::chebyshev{}, a, b, 4.0), 4.0);
    EXPECT_EQ(mt::accumulate_within(mt::chebyshev{}, a, b, 1.5), 2.0);
    // a radius in reduced form
    EXPECT_EQ(mt::accumulate_within(mt::euclidean{}, a, b, mt::contract(mt::euclidean{}, 2.0)), 5.0);
}

TEST(MetricMinkowskiEdge, EmptyPointsAreAtDistanceZero)
{
    const std::span<const double> e;
    const std::vector<float> v;
    EXPECT_EQ(mt::euclidean{}(e, e), 0.0);
    EXPECT_EQ(mt::manhattan{}(e, v), 0.0);
    EXPECT_EQ(mt::chebyshev{}(v, v), 0.0f);
    EXPECT_EQ(mt::squared_euclidean{}(e, e), 0.0);
    EXPECT_EQ(mt::minkowski<3>{}(e, e), 0.0);
    EXPECT_EQ(mt::weighted<2>(std::span<const double>{})(e, e), 0.0);
}

TEST(MetricMinkowskiEdge, InOneDimensionEveryOrderIsTheAbsoluteDifference)
{
    const std::array<double, 1> a{-1.5};
    const double b[1] = {2.0};
    EXPECT_EQ(mt::euclidean{}(a, b), 3.5);
    EXPECT_EQ(mt::manhattan{}(a, b), 3.5);
    EXPECT_EQ(mt::chebyshev{}(a, b), 3.5);
    EXPECT_NEAR(mt::minkowski<7>{}(a, b), 3.5, 1e-12);
    EXPECT_EQ(mt::squared_euclidean{}(a, b), 12.25);
    EXPECT_EQ(mt::weighted<2>(std::array<double, 1>{4})(a, b), 7.0);  // sqrt(4 * 12.25)
}

// Random points in [-1, 1]^D
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

TEST(MetricMinkowskiAxioms, EveryMetricObeysIdentitySymmetryAndTheTriangleInequality)
{
    expect_axioms(mt::euclidean{}, 1);
    expect_axioms(mt::manhattan{}, 2);
    expect_axioms(mt::chebyshev{}, 3);
    expect_axioms(mt::minkowski<3>{}, 4);
    expect_axioms(mt::minkowski<16>{}, 5);
    st::wyrand g(6);
    std::array<double, 5> w{};
    for (auto& x : w)
        x = 0.1 + st::uniform_real(g);
    expect_axioms(mt::weighted<2>(w), 7);
    expect_axioms(mt::weighted<1>(w), 8);
    expect_axioms(mt::weighted<4>(w), 9);
}

TEST(MetricMinkowskiAxioms, SquaredEuclideanBreaksTheTriangleInequality)
{
    // 0, 1, 2 on a line: d(0, 2) = 4 > d(0, 1) + d(1, 2) = 2
    const std::array<double, 1> x{0}, y{1}, z{2};
    const mt::squared_euclidean m;
    EXPECT_GT(m(x, z), m(x, y) + m(y, z));
    EXPECT_LE(mt::euclidean{}(x, z), mt::euclidean{}(x, y) + mt::euclidean{}(y, z));
    // it is still symmetric and zero only at equal points
    EXPECT_EQ(m(x, x), 0.0);
    EXPECT_EQ(m(x, z), m(z, x));
}

TEST(MetricMinkowskiNoexcept, AThrowingCoordinateReadPropagates)
{
    const checked_row a{{1, -2, 3}};
    const a3d b{4, 2, -1};
    EXPECT_DOUBLE_EQ(mt::euclidean{}(a, b), ref_euclidean);
    EXPECT_EQ(mt::weighted<1>(a)(b, b), 0.0);
}

} // namespace
