//
//  low_discrepancy_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/low_discrepancy.hpp>: Joe and Kuo's first points, the net
//  property that makes a Sobol sequence what it is, Halton's radical inverses, the random shift,
//  and convergence faster than random sampling.
//

#include <snicholls/core/stats/low_discrepancy.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

namespace stats = snicholls::stats;

// Joe and Kuo's sobol.cc in two dimensions: (0,0), (.5,.5), (.75,.25), (.25,.75), (.375,.375), ...
constexpr std::array<std::array<double, 2>, 8> first_points{{{0, 0},
                                                             {0.5, 0.5},
                                                             {0.75, 0.25},
                                                             {0.25, 0.75},
                                                             {0.375, 0.375},
                                                             {0.875, 0.875},
                                                             {0.625, 0.125},
                                                             {0.125, 0.625}}};
consteval bool first_points_match()
{
    stats::sobol<2> s;
    for (const auto& expected : first_points)
        if (s.next() != expected)
            return false;
    return true;
}
static_assert(first_points_match());
static_assert(stats::sobol<2>::point(5) == first_points[5]);
static_assert(stats::radical_inverse(2, 1) == 0.5 && stats::radical_inverse(2, 2) == 0.25 && stats::radical_inverse(2, 3) == 0.75);
static_assert(stats::halton<3>::point(1)[1] * 3 == 1.0 && stats::halton<3>::point(1)[2] * 5 == 1.0);
static_assert(sizeof(stats::sobol<4>) == 2 * 4 * sizeof(std::uint32_t) + sizeof(std::uint64_t));

TEST(Sobol, NextPointAndSeekAgree)
{
    stats::sobol<16> s;
    for (std::uint64_t i = 0; i < 1000; ++i)
        ASSERT_EQ(s.next(), stats::sobol<16>::point(i)) << i;
    stats::sobol<16> jumped;
    jumped.seek(777);
    EXPECT_EQ(jumped.next(), stats::sobol<16>::point(777));
    jumped.skip(10);
    EXPECT_EQ(jumped.index(), 788u);
}

// The defining property: the first 2^m points put exactly one point in each interval of width 2^-m,
// in every one of the 1024 dimensions.
TEST(Sobol, EveryDimensionIsStratifiedAtEveryPowerOfTwo)
{
    constexpr std::size_t m = 10, n = std::size_t{1} << m;
    stats::sobol<1024> s;
    std::vector<std::vector<int>> count(1024, std::vector<int>(n));
    for (std::size_t i = 0; i < n; ++i) {
        const auto p = s.next();
        for (std::size_t d = 0; d < 1024; ++d)
            ++count[d][static_cast<std::size_t>(p[d] * n)];
    }
    for (std::size_t d = 0; d < 1024; ++d)
        for (std::size_t c = 0; c < n; ++c)
            ASSERT_EQ(count[d][c], 1) << "dimension " << d + 1 << ", cell " << c;
}

// Dimensions 1 and 2 form a (0, m, 2)-net: every dyadic box of area 2^-m holds exactly one point.
TEST(Sobol, TheFirstTwoDimensionsAreAZeroMTwoNet)
{
    constexpr std::size_t m = 8, n = std::size_t{1} << m;
    std::vector<std::array<double, 2>> points;
    stats::sobol<2> s;
    for (std::size_t i = 0; i < n; ++i)
        points.push_back(s.next());
    for (std::size_t a = 0; a <= m; ++a) {
        const std::size_t cols = std::size_t{1} << a, rows = std::size_t{1} << (m - a);
        std::vector<int> boxes(cols * rows);
        for (const auto& p : points)
            ++boxes[static_cast<std::size_t>(p[0] * cols) * rows + static_cast<std::size_t>(p[1] * rows)];
        for (const int b : boxes)
            ASSERT_EQ(b, 1) << "boxes of " << cols << " x " << rows;
    }
}

TEST(Sobol, ARandomShiftKeepsTheStratification)
{
    stats::wyrand g(11);
    stats::sobol<8> shifted(g);
    constexpr std::size_t n = 512;
    std::vector<std::vector<int>> count(8, std::vector<int>(n));
    for (std::size_t i = 0; i < n; ++i) {
        const auto p = shifted.next();
        for (std::size_t d = 0; d < 8; ++d)
            ++count[d][static_cast<std::size_t>(p[d] * n)];
    }
    for (const auto& dim : count)
        for (const int c : dim)
            ASSERT_EQ(c, 1);
    EXPECT_NE(stats::sobol<8>(g).next()[0], 0.0);   // the origin is shifted away
}

TEST(Sobol, ConvergesFasterThanRandomPoints)
{
    // The integral of prod (1 + (x_d - 1/2)) over [0, 1]^6 is 1.
    const auto f = [](const std::array<double, 6>& x) {
        double v = 1;
        for (const double xi : x)
            v *= 1 + (xi - 0.5);
        return v;
    };
    constexpr std::size_t n = 4096;
    stats::sobol<6> s;
    s.skip(1);
    stats::wyrand g(1);
    double qmc = 0, mc = 0;
    for (std::size_t i = 0; i < n; ++i) {
        qmc += f(s.next());
        std::array<double, 6> r{};
        for (double& x : r)
            x = stats::uniform_real(g);
        mc += f(r);
    }
    const double qmc_error = std::abs(qmc / n - 1), mc_error = std::abs(mc / n - 1);
    EXPECT_LT(qmc_error, 1e-3);
    EXPECT_LT(qmc_error, mc_error);
}

TEST(Halton, PointsAreRadicalInversesInThePrimes)
{
    stats::halton<4> h;
    const auto p = h.next();   // index 1
    EXPECT_DOUBLE_EQ(p[0], 0.5);
    EXPECT_DOUBLE_EQ(p[1], 1.0 / 3);
    EXPECT_DOUBLE_EQ(p[2], 0.2);
    EXPECT_DOUBLE_EQ(p[3], 1.0 / 7);
    EXPECT_EQ(h.index(), 2u);
    EXPECT_DOUBLE_EQ(stats::halton<1>::point(6)[0], 0.375);   // 6 = 110 in binary: 0.011
}

} // namespace
