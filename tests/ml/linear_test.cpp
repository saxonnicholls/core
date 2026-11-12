//
//  linear_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/linear.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

static_assert(ml::weight_matrix<ml::dense_weights<float>> && ml::weight_matrix<ml::dense_weights<ml::bfloat16>>
              && ml::weight_matrix<ml::quantized_weights<ml::block_q4_0>>);

constexpr bool compile_time()
{
    const std::array<double, 6> w{1, 2, 3, 4, 5, 6};   // 2 x 3
    const std::array<double, 3> x{1, 0, -1};
    const std::array<double, 2> b{10, 20};
    std::array<double, 2> y{};
    ml::linear(ml::dense_weights<double>{w, 2, 3}, x, y, b);
    return y[0] == 10 + 1 - 3 && y[1] == 20 + 4 - 6;
}
static_assert(compile_time());

std::vector<float> pattern(std::size_t n, float scale, std::size_t salt = 0)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = scale * (float((i * 37 + salt * 11) % 29) / 29.0f - 0.5f);
    return v;
}

// Each output starts at its bias and adds the products in input order, as llm.c's matmul_forward
TEST(MlLinear, DenseFloatIsTheBiasThenTheProductsInOrder)
{
    const auto w = pattern(4 * 32, 0.5f, 1), x = pattern(32, 2.0f, 2), b = pattern(4, 0.1f, 3);
    std::vector<float> y(4);
    ml::linear(ml::dense_weights<float>{w, 4, 32}, ml::prepare_input<32>(ml::dense_weights<float>{w, 4, 32}, x), y, b);
    for (std::size_t i = 0; i < 4; ++i) {
        float acc = b[i];
        for (std::size_t j = 0; j < 32; ++j) {
            const float p = w[i * 32 + j] * x[j];
            acc += p;
        }
        EXPECT_EQ(y[i], acc);
    }
}

TEST(MlLinear, NoBiasStartsAtZero)
{
    const std::array<float, 2> w{2, 3}, x{5, 7};
    std::array<float, 1> y{};
    ml::linear(ml::dense_weights<float>{w, 1, 2}, x, y);
    EXPECT_EQ(y[0], 31.0f);
}

TEST(MlLinear, Bfloat16WeightsAreTheirDecodedValues)
{
    const auto f = pattern(3 * 8, 1.0f);
    std::vector<ml::bfloat16> h(f.size());
    std::vector<float> back(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) {
        h[i] = ml::to_minifloat<ml::bfloat16>(f[i]);
        back[i] = ml::to_float(h[i]);
    }
    const auto x = pattern(8, 1.0f, 4);
    std::array<float, 3> got{}, want{};
    ml::linear(ml::dense_weights<ml::bfloat16>{h, 3, 8}, x, got);
    ml::linear(ml::dense_weights<float>{back, 3, 8}, x, want);
    EXPECT_EQ(got, want);
}

// A quantised product is ggml's: the row's block dot product against the input quantised once to Q8_0
template <typename Block>
void quantised_is_the_block_dot()
{
    constexpr std::size_t rows = 3, cols = 64;
    const auto f = pattern(rows * cols, 1.0f), x = pattern(cols, 2.0f, 5), b = pattern(rows, 0.2f, 6);
    std::vector<Block> wq(rows * cols / 32);
    ml::quantize(f, std::span<Block>(wq));
    const ml::quantized_weights<Block> w{wq, rows, cols};
    const auto in = ml::prepare_input<cols>(w, x);
    std::array<ml::block_q8_0, 2> xq{};
    ml::quantize(x, std::span<ml::block_q8_0, 2>(xq));
    EXPECT_EQ(std::memcmp(in.data(), xq.data(), sizeof xq), 0);
    std::array<float, rows> y{};
    ml::linear(w, in, y, b);
    for (std::size_t i = 0; i < rows; ++i)
        EXPECT_EQ(y[i], b[i] + ml::dot(w.row(i), std::span<const ml::block_q8_0>(xq)));
    // And near the float product: within the quantisation error
    std::array<float, rows> exact{};
    ml::linear(ml::dense_weights<float>{f, rows, cols}, x, exact, b);
    for (std::size_t i = 0; i < rows; ++i)
        EXPECT_NEAR(y[i], exact[i], 0.25);
}

TEST(MlLinear, QuantisedProductsAreTheBlockDotProducts)
{
    quantised_is_the_block_dot<ml::block_q8_0>();
    quantised_is_the_block_dot<ml::block_q4_0>();
    quantised_is_the_block_dot<ml::block_mxfp4>();
}

TEST(MlLinear, RowsAreViewsIntoTheMatrix)
{
    const auto f = pattern(2 * 3, 1.0f);
    const ml::dense_weights<float> w{f, 2, 3};
    EXPECT_EQ(w.row(1).data(), f.data() + 3);
    EXPECT_EQ(w.row(1).size(), 3u);
}

} // namespace
