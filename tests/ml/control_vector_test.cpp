//
//  control_vector_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/control_vector.hpp>: llama.cpp's buffer layout and layer range
//  (src/llama-adapter.cpp, 39039aa), application to a batch, disabling, the sum at strengths, the backward pass, and
//  compile-time use.
//

#include <snicholls/core/ml/control_vector.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

TEST(MlControlVector, TheBufferStartsAtLayerOneAndTheRangeIsInclusive)
{
    std::vector<float> data(3 * 2);   // layers 1, 2, 3 of width 2
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = float(i + 1);
    const ml::control_vector_view<float> cv{data, 2, 2, 3};
    EXPECT_TRUE(cv.direction(0).empty());
    EXPECT_TRUE(cv.direction(1).empty());   // in the buffer, outside the range
    EXPECT_EQ(cv.direction(2)[0], 3.0f);    // layer 2 at width · (2 - 1)
    EXPECT_EQ(cv.direction(3)[1], 6.0f);
    EXPECT_TRUE(cv.direction(4).empty());
    const ml::control_vector_view<float> past{data, 2, 1, 9};
    EXPECT_TRUE(past.direction(4).empty());   // past the buffer: the loader's zeros
    const ml::control_vector_view<float> off{data, 2, -1, -1};
    EXPECT_TRUE(off.direction(2).empty());    // disabled
}

TEST(MlControlVector, ApplyAddsToEveryRowAndBackwardSumsThem)
{
    const std::array<double, 4> data{0.5, -1.0, 2.0, 0.25};
    const ml::control_vector_view<double> cv{data, 2, 1, 2};
    std::array<double, 6> rows{1, 1, 2, 2, 3, 3};
    ml::apply_control_vector(cv, 2, std::span<double>(rows));
    EXPECT_EQ(rows, (std::array<double, 6>{3, 1.25, 4, 2.25, 5, 3.25}));
    ml::apply_control_vector(cv, 3, std::span<double>(rows));   // outside the range: unchanged
    EXPECT_EQ(rows[0], 3.0);
    std::array<double, 2> dd{};
    const std::array<double, 6> drows{1, 2, 3, 4, 5, 6};
    ml::control_vector_backward(cv, 1, drows, std::span<double>(dd));
    EXPECT_EQ(dd, (std::array<double, 2>{9, 12}));
    ml::control_vector_backward(cv, 0, drows, std::span<double>(dd));
    EXPECT_EQ(dd, (std::array<double, 2>{9, 12}));
}

TEST(MlControlVector, VectorsSumAtTheirStrengths)
{
    std::array<float, 3> sum{}, a{1, 2, 3}, b{-1, 0, 1};
    ml::add_control_vector(std::span<float>(sum), a, 0.5f);
    ml::add_control_vector(std::span<float>(sum), b, 2.0f);
    EXPECT_EQ(sum, (std::array<float, 3>{-1.5f, 1.0f, 3.5f}));
}

constexpr bool compile_time()
{
    const std::array<int, 1> unused{};
    (void)unused;
    const std::array<double, 2> data{1, 2};
    std::array<double, 2> row{0, 0};
    ml::apply_control_vector(ml::control_vector_view<double>{data, 2, 1, 1}, 1, std::span<double>(row));
    return row[0] == 1 && row[1] == 2;
}
static_assert(compile_time());

} // namespace
