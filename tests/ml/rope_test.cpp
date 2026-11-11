//
//  rope_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/rope.hpp>: the frequencies, a table built at compile time, the rotation in both
//  layouts against the formula, rope_inverse undoing rope, lengths kept, position 0 the identity, the relative-
//  position property of the dot product, and rope_inverse as rope's backward pass.
//

#include <snicholls/core/ml/rope.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

constexpr ml::rope_table<float, 8, 4> small_table{};
static_assert(small_table.cos[0][0] == 1.0f && small_table.sin[0][1] == 0.0f);
static_assert(m::near<float>(small_table.cos[1][0], m::cos(1.0f), 1e-6f) && m::near<float>(small_table.sin[3][1], m::sin(0.03f), 1e-6f));
static_assert(std::is_trivially_copyable_v<ml::rope_table<float, 8, 4>> && sizeof(ml::rope_table<float, 8, 4>) == 8 * 4 * sizeof(float));
static_assert(ml::rope_frequency<double>(0, 64) == 1.0 && m::near(ml::rope_frequency<double>(16, 64), 0.01, 1e-15));

constexpr bool rotates_at_compile_time()
{
    std::array<double, 4> x{1, 2, 3, 4};
    std::array<double, 2> c{}, s{};
    ml::rope_angles(5, c, s);
    ml::rope(x, c, s);
    ml::rope_inverse(x, c, s);
    return m::near(x[0], 1.0, 1e-12) && m::near(x[3], 4.0, 1e-12);
}
static_assert(rotates_at_compile_time());

std::vector<double> head(std::size_t d, double phase)
{
    std::vector<double> v(d);
    for (std::size_t i = 0; i < d; ++i)
        v[i] = std::sin(0.9 * static_cast<double>(i) + phase);
    return v;
}

TEST(MlRope, AnglesArePositionTimesFrequency)
{
    std::array<double, 8> c{}, s{};
    ml::rope_angles(37, c, s, 500000.0);
    for (std::size_t i = 0; i < 8; ++i) {
        const double angle = 37.0 * std::pow(500000.0, -2.0 * static_cast<double>(i) / 16.0);
        EXPECT_NEAR(c[i], std::cos(angle), 1e-13);
        EXPECT_NEAR(s[i], std::sin(angle), 1e-13);
    }
}

TEST(MlRope, BothLayoutsRotateTheirPairsByTheFormula)
{
    const auto x = head(8, 0.3);
    std::array<double, 4> c{}, s{};
    ml::rope_angles(11, c, s);
    auto half = x, inter = x;
    ml::rope(half, c, s, ml::rope_layout::half);
    ml::rope(inter, c, s, ml::rope_layout::interleaved);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(half[i], x[i] * c[i] - x[i + 4] * s[i], 1e-15);
        EXPECT_NEAR(half[i + 4], x[i] * s[i] + x[i + 4] * c[i], 1e-15);
        EXPECT_NEAR(inter[2 * i], x[2 * i] * c[i] - x[2 * i + 1] * s[i], 1e-15);
        EXPECT_NEAR(inter[2 * i + 1], x[2 * i] * s[i] + x[2 * i + 1] * c[i], 1e-15);
    }
}

TEST(MlRope, InverseUndoesAndLengthsAreKept)
{
    for (const auto layout : {ml::rope_layout::half, ml::rope_layout::interleaved}) {
        const auto x = head(16, 1.1);
        std::array<double, 8> c{}, s{};
        ml::rope_angles(1000, c, s);
        auto y = x;
        ml::rope(y, c, s, layout);
        double nx = 0, ny = 0;
        for (std::size_t i = 0; i < 16; ++i) {
            nx += x[i] * x[i];
            ny += y[i] * y[i];
        }
        EXPECT_NEAR(nx, ny, 1e-12);
        ml::rope_inverse(y, c, s, layout);
        for (std::size_t i = 0; i < 16; ++i)
            EXPECT_NEAR(y[i], x[i], 1e-14);
    }
}

TEST(MlRope, PositionZeroIsTheIdentity)
{
    const auto x = head(8, 0.5);
    std::array<double, 4> c{}, s{};
    ml::rope_angles(0, c, s);
    auto y = x;
    ml::rope(y, c, s);
    EXPECT_EQ(y, x);
}

TEST(MlRope, TheDotProductDependsOnlyOnTheDistance)
{
    // q at 10 and k at 7 score as q at 103 and k at 100: the point of RoPE
    const auto q0 = head(16, 0.2), k0 = head(16, 2.0);
    const auto score = [&](std::size_t pq, std::size_t pk) {
        auto q = q0, k = k0;
        std::array<double, 8> c{}, s{};
        ml::rope_angles(pq, c, s);
        ml::rope(q, c, s);
        ml::rope_angles(pk, c, s);
        ml::rope(k, c, s);
        double d = 0;
        for (std::size_t i = 0; i < 16; ++i)
            d += q[i] * k[i];
        return d;
    };
    EXPECT_NEAR(score(10, 7), score(103, 100), 1e-12);
    EXPECT_NEAR(score(4, 4), score(500, 500), 1e-12);
}

TEST(MlRope, InverseIsTheBackwardPass)
{
    // L = g · rope(x); dL/dx by central differences equals rope_inverse(g)
    const auto x = head(8, 0.7), g = head(8, 1.9);
    std::array<double, 4> c{}, s{};
    ml::rope_angles(23, c, s);
    auto dx = g;
    ml::rope_inverse(dx, c, s);
    for (std::size_t i = 0; i < 8; ++i) {
        const auto loss = [&](double h) {
            auto y = x;
            y[i] += h;
            ml::rope(y, c, s);
            double l = 0;
            for (std::size_t k = 0; k < 8; ++k)
                l += g[k] * y[k];
            return l;
        };
        EXPECT_NEAR(dx[i], (loss(1e-6) - loss(-1e-6)) / 2e-6, 1e-8);
    }
}

} // namespace
