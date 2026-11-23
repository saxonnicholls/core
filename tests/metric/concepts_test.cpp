//
//  concepts_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/metric/concepts.hpp>: which types are points (std::array, std::vector, std::span of both
//  extents, C arrays, fixed_vector, a one-row fixed_matrix, a struct of the test's own) and which are not (int,
//  std::vector<bool>, a list, a pointer), their value types, extents and common distance type, compatible extents,
//  the capability concepts on metrics of the test's own, the reduced / expand / contract fallbacks, accumulate and
//  accumulate_within with their early exit, noexcept propagated from the coordinate read, and constant evaluation.
//

#include <snicholls/core/metric/concepts.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

// A point type of the test's own: three named coordinates
struct xyz {
    double x, y, z;
};

// A point whose coordinate read may throw (a bounds-checked row, say)
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

using a3d = std::array<double, 3>;
using a3f = std::array<float, 3>;
using a3i = std::array<int, 3>;
using a4d = std::array<double, 4>;

// A metric of the test's own with only operator(): the taxicab distance rounded to whole units
struct rounded_taxicab {
    template <mt::point A, mt::point B>
    constexpr double operator()(const A& a, const B& b) const noexcept
    {
        double s = 0;
        for (std::size_t i = 0; i < mt::dimension(a); ++i) {
            const double d = static_cast<double>(mt::coordinate(a, i)) - static_cast<double>(mt::coordinate(b, i));
            s += d < 0 ? -d : d;
        }
        return static_cast<double>(static_cast<long long>(s + 0.5));
    }
};

// A decomposable metric of the test's own: squared Euclidean with axis 0 counted twice
struct heavy_x {
    static constexpr bool is_metric = false;
    template <typename T>
    constexpr T term(std::size_t axis, T d) const noexcept
    {
        return (axis == 0 ? T(2) : T(1)) * d * d;
    }
    template <typename T>
    constexpr T combine(T acc, T t) const noexcept
    {
        return acc + t;
    }
    template <typename T>
    constexpr T expand(T r) const noexcept
    {
        return r;
    }
    template <typename T>
    constexpr T contract(T d) const noexcept
    {
        return d;
    }
    template <mt::point A, mt::point B>
    constexpr mt::distance_t<A, B> operator()(const A& a, const B& b) const noexcept
    {
        return mt::accumulate(*this, a, b);
    }
};

// A reducible metric of the test's own: Euclidean through its square
struct root_of_square {
    static constexpr bool is_metric = true;
    template <mt::point A, mt::point B>
    constexpr mt::distance_t<A, B> reduced(const A& a, const B& b) const noexcept
    {
        mt::distance_t<A, B> s = 0;
        for (std::size_t i = 0; i < mt::dimension(a); ++i) {
            const auto d = static_cast<mt::distance_t<A, B>>(mt::coordinate(a, i)) - static_cast<mt::distance_t<A, B>>(mt::coordinate(b, i));
            s += d * d;
        }
        return s;
    }
    template <typename T>
    constexpr T expand(T r) const noexcept
    {
        // Newton's root, enough for the test's exact squares
        if (!(r > T(0)))
            return T(0);
        T x = r;
        for (int i = 0; i < 64; ++i)
            x = (x + r / x) / T(2);
        return x;
    }
    template <typename T>
    constexpr T contract(T d) const noexcept
    {
        return d * d;
    }
    template <mt::point A, mt::point B>
    constexpr mt::distance_t<A, B> operator()(const A& a, const B& b) const noexcept
    {
        return expand(reduced(a, b));
    }
};

struct claims_false {
    static constexpr bool is_metric = false;
};
struct no_claim {};

// ---- which types are points ----
static_assert(mt::point<a3d> && mt::point<a3f> && mt::point<a3i>);
static_assert(mt::point<std::vector<double>> && mt::point<std::vector<float>>);
static_assert(mt::point<std::span<const double>> && mt::point<std::span<const double, 3>> && mt::point<std::span<float>>);
static_assert(mt::point<double[3]> && mt::point<int[2]>);
static_assert(mt::point<la::fixed_vector<float, 3>> && mt::point<la::fixed_matrix<double, 1, 3>>);
static_assert(mt::point<xyz> && mt::point<checked_row>);
static_assert(mt::point<const a3d&> && mt::point<const xyz&> && mt::point<std::vector<double>&&>);  // cv and references ignored
static_assert(mt::point<std::string_view> && mt::point<std::array<bool, 4>>);  // char and bool coordinates read by reference
static_assert(!mt::point<int> && !mt::point<double>);
static_assert(!mt::point<std::vector<bool>>);           // a packed-bit proxy, on every standard library
static_assert(!mt::point<std::list<double>>);           // no operator[]
static_assert(!mt::point<double*>);                     // no length
static_assert(!mt::point<std::vector<std::string>>);    // coordinates are not arithmetic
static_assert(!mt::point<la::fixed_matrix<double, 2, 2>>);  // a matrix is not a vector

// ---- value type, extent, distance type ----
static_assert(std::is_same_v<mt::value_t<a3f>, float> && std::is_same_v<mt::value_t<a3i>, int> && std::is_same_v<mt::value_t<xyz>, double>);
static_assert(std::is_same_v<mt::value_t<std::span<const double>>, double> && std::is_same_v<mt::value_t<const int[4]>, int>);
static_assert(std::is_same_v<mt::value_t<la::fixed_vector<float, 2>>, float>);
static_assert(mt::extent_v<a3d> == 3 && mt::extent_v<double[5]> == 5 && mt::extent_v<std::span<const double, 2>> == 2);
static_assert(mt::extent_v<la::fixed_vector<float, 4>> == 4 && mt::extent_v<la::fixed_matrix<double, 1, 6>> == 6 && mt::extent_v<xyz> == 3);
static_assert(mt::extent_v<std::vector<double>> == std::dynamic_extent && mt::extent_v<std::span<const float>> == std::dynamic_extent);
static_assert(std::is_same_v<mt::distance_t<a3f>, float> && std::is_same_v<mt::distance_t<a3d>, double>);
static_assert(std::is_same_v<mt::distance_t<a3f, a3d>, double> && std::is_same_v<mt::distance_t<a3d, a3f>, double>);
static_assert(std::is_same_v<mt::distance_t<a3i>, double> && std::is_same_v<mt::distance_t<a3i, a3f>, double>);  // integers compute in double
static_assert(std::is_same_v<mt::distance_t<std::array<long double, 2>, std::vector<float>>, long double>);
static_assert(std::is_same_v<mt::distance_t<la::fixed_vector<float, 3>, std::span<const float>>, float>);

// ---- compatible extents ----
static_assert(mt::compatible_points<a3d, a3f> && mt::compatible_points<a3d, xyz> && mt::compatible_points<a3d, double[3]>);
static_assert(mt::compatible_points<a3d, std::vector<double>> && mt::compatible_points<std::span<const double>, a4d>);
static_assert(mt::compatible_points<la::fixed_vector<float, 3>, la::fixed_matrix<double, 1, 3>>);
static_assert(!mt::compatible_points<a3d, a4d> && !mt::compatible_points<std::span<const double, 3>, double[4]>);
static_assert(!mt::compatible_points<xyz, la::fixed_vector<double, 2>> && !mt::compatible_points<a3d, int>);

// ---- capabilities ----
static_assert(mt::distance_for<rounded_taxicab, a3d> && mt::distance_for<rounded_taxicab, a3i, std::vector<float>>);
static_assert(!mt::reducible<rounded_taxicab, a3d> && !mt::decomposable<rounded_taxicab, double> && !mt::true_metric<rounded_taxicab>);
static_assert(mt::distance_for<heavy_x, a3d, xyz> && mt::decomposable<heavy_x, double> && mt::decomposable<heavy_x, float>);
static_assert(!mt::reducible<heavy_x, a3d> && !mt::true_metric<heavy_x>);  // decomposable alone, without reduced()
static_assert(mt::reducible<root_of_square, a3d> && mt::reducible<root_of_square, a3f, xyz> && !mt::decomposable<root_of_square, double>);
static_assert(mt::true_metric<root_of_square> && !mt::true_metric<claims_false> && !mt::true_metric<no_claim> && !mt::true_metric<int>);
static_assert(!mt::distance_for<int, a3d> && !mt::distance_for<no_claim, a3d>);

// ---- noexcept ----
static_assert(noexcept(mt::dimension(std::declval<const a3d&>())) && noexcept(mt::coordinate(std::declval<const xyz&>(), 0)));
static_assert(!noexcept(mt::coordinate(std::declval<const checked_row&>(), 0)));
static_assert(noexcept(mt::accumulate(heavy_x{}, std::declval<const a3d&>(), std::declval<const xyz&>())));
static_assert(!noexcept(mt::accumulate(heavy_x{}, std::declval<const checked_row&>(), std::declval<const a3d&>())));
static_assert(!noexcept(mt::accumulate_within(heavy_x{}, std::declval<const a3d&>(), std::declval<const checked_row&>(), 1.0)));
static_assert(noexcept(mt::expand(rounded_taxicab{}, 1.0)) && noexcept(mt::contract(heavy_x{}, 1.0)));

// ---- constant evaluation ----
constexpr a3d ca{1, 2, 3};
constexpr xyz cb{4, 6, 3};
// heavy_x: 2*3^2 + 4^2 + 0 = 34, by hand
static_assert(mt::accumulate(heavy_x{}, ca, cb) == 34.0);
static_assert(mt::accumulate_within(heavy_x{}, ca, cb, 10.0) == 18.0);  // stops after axis 0's 18 > 10
static_assert(mt::reduced_distance(root_of_square{}, ca, cb) == 25.0);  // 9 + 16 + 0
static_assert(mt::expand(root_of_square{}, 25.0) == 5.0 && mt::contract(root_of_square{}, 5.0) == 25.0);
static_assert(mt::reduced_distance(rounded_taxicab{}, ca, cb) == 7.0);   // |3| + |4| + 0, the distance itself
static_assert(mt::expand(rounded_taxicab{}, 7.0) == 7.0 && mt::contract(rounded_taxicab{}, 7.0) == 7.0);
static_assert(mt::dimension(cb) == 3 && mt::coordinate(cb, 1) == 6.0);

// The same three coordinates in every point kind
constexpr double c_arr[3] = {1, 2, 3};

TEST(MetricConceptsPoint, EveryKindReadsItsDimensionAndCoordinates)
{
    const a3d ad{1, 2, 3};
    const a3f af{1, 2, 3};
    const a3i ai{1, 2, 3};
    const std::vector<double> v{1, 2, 3};
    const std::span<const double, 3> ss(ad);
    const std::span<const double> sd(v);
    const auto fv = la::fixed_vector<float, 3>{{1, 2, 3}};
    const auto row = la::fixed_matrix<double, 1, 3>{{1, 2, 3}};
    const xyz p{1, 2, 3};
    const auto check = [](const auto& q) {
        EXPECT_EQ(mt::dimension(q), 3u);
        for (std::size_t i = 0; i < 3; ++i)
            EXPECT_EQ(static_cast<double>(mt::coordinate(q, i)), static_cast<double>(i + 1));
    };
    check(ad);
    check(af);
    check(ai);
    check(v);
    check(ss);
    check(sd);
    check(c_arr);
    check(fv);
    check(row);
    check(p);
}

TEST(MetricConceptsPoint, AnEmptySpanHasDimensionZero)
{
    const std::span<const double> e;
    EXPECT_EQ(mt::dimension(e), 0u);
    EXPECT_EQ(mt::accumulate(heavy_x{}, e, e), 0.0);
    EXPECT_EQ(mt::accumulate_within(heavy_x{}, e, e, -1.0), 0.0);
}

TEST(MetricConceptsReduced, AMetricWithoutACheaperFormUsesItsDistance)
{
    const a3d a{0, 0, 0};
    const std::vector<double> b{1.2, 1.2, 0};  // taxicab 2.4, rounded to 2
    EXPECT_EQ(rounded_taxicab{}(a, b), 2.0);
    EXPECT_EQ(mt::reduced_distance(rounded_taxicab{}, a, b), 2.0);
    EXPECT_EQ(mt::expand(rounded_taxicab{}, 2.5), 2.5);
    EXPECT_EQ(mt::contract(rounded_taxicab{}, 2.5), 2.5);
}

TEST(MetricConceptsReduced, AReducibleMetricRoundTripsThroughItsReducedForm)
{
    const root_of_square m;
    const a3f a{1, 2, 2};
    const xyz b{0, 0, 0};
    // |(1, 2, 2)| = 3, by hand
    EXPECT_FLOAT_EQ(static_cast<float>(mt::reduced_distance(m, a, b)), 9.0f);
    EXPECT_DOUBLE_EQ(m(a, b), 3.0);
    EXPECT_DOUBLE_EQ(mt::expand(m, mt::contract(m, 3.0)), 3.0);
    EXPECT_DOUBLE_EQ(mt::contract(m, m(a, b)), mt::reduced_distance(m, a, b));
    EXPECT_FLOAT_EQ(mt::expand(m, 16.0f), 4.0f);  // the number type is the caller's
    static_assert(std::is_same_v<decltype(mt::expand(m, 16.0f)), float>);
}

TEST(MetricConceptsAccumulate, SumsTermsOverEveryAxisOfMixedPoints)
{
    const heavy_x m;
    const std::array<int, 4> a{0, 0, 0, 0};
    const std::vector<float> b{1, 2, 3, 4};
    // 2*1 + 4 + 9 + 16 = 31, by hand
    EXPECT_EQ(mt::accumulate(m, a, b), 31.0);
    EXPECT_EQ(m(b, a), 31.0);
    const double c[4] = {1, 2, 3, 4};
    EXPECT_EQ(mt::accumulate(m, std::span<const double, 4>(c), a), 31.0);
}

TEST(MetricConceptsAccumulate, WithinStopsAboveTheBoundAndIsExactAtOrBelowIt)
{
    const heavy_x m;
    const a4d a{0, 0, 0, 0}, b{1, 2, 3, 4};
    // partial sums 2, 6, 15, 31
    EXPECT_EQ(mt::accumulate_within(m, a, b, 31.0), 31.0);    // at the bound: exact
    EXPECT_EQ(mt::accumulate_within(m, a, b, 100.0), 31.0);   // below: exact
    EXPECT_EQ(mt::accumulate_within(m, a, b, 1.0), 2.0);      // above after one axis
    EXPECT_EQ(mt::accumulate_within(m, a, b, 2.0), 6.0);      // 2 is not above 2, so one more axis
    EXPECT_EQ(mt::accumulate_within(m, a, b, 14.9), 15.0);
    EXPECT_GT(mt::accumulate_within(m, a, b, 30.9), 30.9);
    EXPECT_EQ(mt::accumulate_within(m, a, b, -1.0), 0.0);     // zero is already above a negative bound: no axis read
}

TEST(MetricConceptsAccumulate, ACoordinateReadThatThrowsPropagates)
{
    const checked_row a{{1, 2, 3}};
    const a3d b{1, 2, 3};
    EXPECT_EQ(mt::accumulate(heavy_x{}, a, b), 0.0);
    EXPECT_THROW(static_cast<void>(mt::coordinate(a, 3)), std::out_of_range);
}

TEST(MetricConceptsUserPoint, AStructWithTraitsWorksWithEveryCapability)
{
    const xyz a{1, 2, 3};
    const la::fixed_vector<float, 3> b{{4, 6, 3}};
    EXPECT_EQ(mt::accumulate(heavy_x{}, a, b), 34.0);      // 2*9 + 16
    EXPECT_EQ(rounded_taxicab{}(a, b), 7.0);
    EXPECT_DOUBLE_EQ(root_of_square{}(a, b), 5.0);
    EXPECT_EQ(mt::accumulate_within(heavy_x{}, a, b, 0.5), 18.0);
}

} // namespace
