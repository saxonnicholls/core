//
//  recurrence_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/recurrence.hpp>: the driver over a small recurrence (two exponential moving averages
//  read by a weight): the forward pass and its checkpoints, the backward pass against central differences in double,
//  its result independent of the chunk to the bit, several sequences as separate calls, the sizes and the concept at
//  compile time, and a whole forward pass in a constant expression.
//

#include <snicholls/core/ml/recurrence.hpp>

#include "../support/mixer_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::core_testing::central_difference;
using snicholls::core_testing::uniform;

// s_i <- a_i s_i + x_t, y_t = Σ_i c_i s_i: two channels, the decays and readout parameters, x and y per row
template <typename T>
struct ema
{
    using value_type = T;
    static constexpr std::size_t state_size = 2, work_size = 0;
    std::span<const T> x{}, a{}, c{};
    std::span<T> y{};
    std::span<const T> dy{};
    std::span<T> dx{}, da{}, dc{};

    constexpr void advance(std::size_t t, std::span<T> s) const
    {
        for (std::size_t i = 0; i < 2; ++i)
            s[i] = a[i] * s[i] + x[t];
    }
    constexpr void step(std::size_t t, std::span<T> s) const
    {
        advance(t, s);
        y[t] = c[0] * s[0] + c[1] * s[1];
    }
    constexpr void step_backward(std::size_t t, std::span<const T> prev, std::span<T> ds, std::span<T>) const
    {
        T g = T(0);
        for (std::size_t i = 0; i < 2; ++i) {
            const T st = a[i] * prev[i] + x[t];
            dc[i] += dy[t] * st;
            const T d = ds[i] + c[i] * dy[t];
            da[i] += d * prev[i];
            g += d;
            ds[i] = d * a[i];
        }
        dx[t] = g;
    }
};

static_assert(ml::recurrence<ema<double>>);
static_assert(!ml::recurrence<int>);
static_assert(ml::recurrence_checkpoints(10, 4) == 3 && ml::recurrence_checkpoints(8, 4) == 2 && ml::recurrence_checkpoints(0, 4) == 0);
static_assert(ml::recurrence_backward_work<ema<double>>(4) == 8);

// A whole forward pass at compile time
constexpr double compile_time_output()
{
    const std::array<double, 4> x{1, 2, 3, 4}, a{0.5, 0.25}, c{1, 2};
    std::array<double, 4> y{};
    std::array<double, 2> s{};
    ml::recurrence_forward(ema<double>{x, a, c, y, {}, {}, {}, {}}, 4, 1, 2, std::span<double>(s));
    return y[3];
}
// s0: 1, 2.5, 4.25, 6.125; s1: 1, 2.25, 3.5625, 4.890625; y = s0 + 2 s1
static_assert(compile_time_output() == 6.125 + 2 * 4.890625);

TEST(MlRecurrence, TheForwardPassSavesTheStateAtEachChunkStart)
{
    const std::vector<double> x{1, 2, 3, 4, 5}, a{0.5, 0.25}, c{1, 2};
    std::vector<double> y(5), s(2, 0.0), cps(2 * 3);
    ml::recurrence_forward(ema<double>{x, a, c, y, {}, {}, {}, {}}, 5, 1, 2, std::span<double>(s), std::span<double>(cps));
    std::vector<double> t(2, 0.0), y2(5);
    for (std::size_t i = 0; i < 5; ++i) {
        if (i % 2 == 0) {
            EXPECT_EQ(cps[(i / 2) * 2], t[0]);
            EXPECT_EQ(cps[(i / 2) * 2 + 1], t[1]);
        }
        ema<double>{x, a, c, y2, {}, {}, {}, {}}.step(i, t);
    }
    EXPECT_EQ(y, y2);   // the step is the forward pass, to the bit
    EXPECT_EQ(s, t);
}

TEST(MlRecurrence, TheBackwardPassMatchesCentralDifferences)
{
    constexpr std::size_t n = 7, seqs = 2;
    std::vector<double> x = uniform<double>(n * seqs, 1), a{0.7, -0.4}, c{0.9, 1.3}, s0 = uniform<double>(2 * seqs, 2);
    const auto wy = uniform<double>(n * seqs, 3), ws = uniform<double>(2 * seqs, 4);
    const auto loss = [&] {
        std::vector<double> y(n * seqs), s = s0;
        ml::recurrence_forward(ema<double>{x, a, c, y, {}, {}, {}, {}}, n, seqs, 3, std::span<double>(s));
        return snicholls::core_testing::weighted(y, wy) + snicholls::core_testing::weighted(s, ws);
    };
    std::vector<double> y(n * seqs), s = s0, cps(seqs * 3 * 2), dx(n * seqs), da(2), dc(2), ds = ws, work(3 * 2);
    ml::recurrence_forward(ema<double>{x, a, c, y, {}, {}, {}, {}}, n, seqs, 3, std::span<double>(s), std::span<double>(cps));
    ml::recurrence_backward(ema<double>{x, a, c, {}, wy, dx, da, dc}, n, seqs, 3, std::span<const double>(cps), std::span<double>(ds),
                            std::span<double>(work));
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(dx[i], central_difference(x, i, loss), 1e-7) << "x " << i;
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(da[i], central_difference(a, i, loss), 1e-7) << "a " << i;
        EXPECT_NEAR(dc[i], central_difference(c, i, loss), 1e-7) << "c " << i;
    }
    for (std::size_t i = 0; i < s0.size(); ++i)
        EXPECT_NEAR(ds[i], central_difference(s0, i, loss), 1e-7) << "initial state " << i;
}

TEST(MlRecurrence, TheBackwardPassDoesNotDependOnTheChunk)
{
    constexpr std::size_t n = 9;
    const auto x = uniform<double>(n, 5);
    const std::vector<double> a{0.6, 0.9}, c{1.1, -0.7}, wy = uniform<double>(n, 6);
    std::vector<double> reference_dx;
    for (const std::size_t chunk : {1u, 2u, 4u, 9u, 16u}) {
        const std::size_t m = ml::recurrence_checkpoints(n, chunk);
        std::vector<double> y(n), s(2, 0.0), cps(m * 2), dx(n), da(2), dc(2), ds(2, 0.0), work(chunk * 2);
        ml::recurrence_forward(ema<double>{x, a, c, y, {}, {}, {}, {}}, n, 1, chunk, std::span<double>(s), std::span<double>(cps));
        ml::recurrence_backward(ema<double>{x, a, c, {}, wy, dx, da, dc}, n, 1, chunk, std::span<const double>(cps), std::span<double>(ds),
                                std::span<double>(work));
        if (reference_dx.empty())
            reference_dx = dx;
        EXPECT_EQ(dx, reference_dx) << "chunk " << chunk;
    }
}

TEST(MlRecurrence, SequencesAreIndependent)
{
    const auto x = uniform<double>(8, 7);
    const std::vector<double> a{0.5, 0.8}, c{1, 1};
    std::vector<double> together(8), s(4, 0.0);
    ml::recurrence_forward(ema<double>{x, a, c, together, {}, {}, {}, {}}, 4, 2, 4, std::span<double>(s));
    for (std::size_t q = 0; q < 2; ++q) {
        const std::span<const double> xs(x.data() + 4 * q, 4);
        std::vector<double> alone(4), t(2, 0.0);
        ml::recurrence_forward(ema<double>{xs, a, c, alone, {}, {}, {}, {}}, 4, 1, 4, std::span<double>(t));
        for (std::size_t i = 0; i < 4; ++i)
            EXPECT_EQ(alone[i], together[4 * q + i]);
        EXPECT_EQ(t[0], s[2 * q]);
    }
}

} // namespace
