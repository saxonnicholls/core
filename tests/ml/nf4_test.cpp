//
//  nf4_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/nf4.hpp>: the sixteen NF4 values reproduced from the normal quantiles by bitsandbytes'
//  construction; the nearest code with ties to the lower; the sizes; exact reconstruction when every block shares its
//  maximum (no second-level error) and the codes are NF4 values; the round-trip error within the stated bound for
//  normal weights over two second-level groups; a zero block; the on-the-fly products equal to the dense products of
//  the dequantised matrix to the bit (forward, the input's gradient, the sequence forms); compile-time use.
//

#include <snicholls/core/ml/nf4.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
namespace math = snicholls::math;
constexpr auto& backend = snicholls::linalg::reference;

static_assert(ml::nf4_values[7] == 0.0f && ml::nf4_values[0] == -1.0f && ml::nf4_values[15] == 1.0f);
static_assert(ml::nf4_code(0.0f) == 7 && ml::nf4_code(1.5f) == 15 && ml::nf4_code(-0.7f) == 1);
static_assert(ml::nf4_code_bytes(128) == 64 && ml::nf4_block_count(128) == 2 && ml::nf4_group_count(64 * 257) == 2);
static_assert(noexcept(ml::nf4_code(0.5f)));

constexpr bool at_compile_time()
{
    std::array<float, 64> w{};
    for (std::size_t i = 0; i < 64; ++i)
        w[i] = ml::nf4_values[i % 16] * 2.0f;
    std::array<std::uint8_t, 32> codes{};
    std::array<ml::float8_e4m3, 1> scales{};
    std::array<float, 1> groups{};
    const float mean = ml::nf4_quantize(w, codes, scales, groups);
    const ml::nf4_weights q{codes, scales, groups, mean, 1, 64};
    return mean == 2.0f && q.at(0, 0) == -2.0f && q.at(0, 15) == 2.0f && q.at(0, 23) == 0.0f;
}
static_assert(at_compile_time());

TEST(MlNf4Values, AreTheNormalQuantilesOfBitsandbytesConstruction)
{
    // offset 0.9677083: 8 positive quantiles from linspace(offset, 0.5, 9)[:-1], 7 negative from linspace(offset, 0.5,
    // 8)[:-1], zero, sorted, divided by the largest
    const double offset = 0.9677083;
    std::vector<double> v;
    for (int i = 0; i < 8; ++i)
        v.push_back(math::normal_quantile(offset + (0.5 - offset) * i / 8.0));
    for (int i = 0; i < 7; ++i)
        v.push_back(-math::normal_quantile(offset + (0.5 - offset) * i / 7.0));
    v.push_back(0.0);
    std::sort(v.begin(), v.end());
    const double top = v.back();
    for (std::size_t i = 0; i < 16; ++i)
        EXPECT_NEAR(ml::nf4_values[i], v[i] / top, 2e-7) << i;
}

TEST(MlNf4Code, IsTheNearestValueWithTiesToTheLowerCode)
{
    for (std::uint8_t c = 0; c < 16; ++c)
        EXPECT_EQ(ml::nf4_code(ml::nf4_values[c]), c);
    for (std::uint8_t c = 0; c + 1 < 16; ++c) {
        const float mid = (ml::nf4_values[c] + ml::nf4_values[c + 1]) / 2;
        const float d_lo = mid - ml::nf4_values[c], d_hi = ml::nf4_values[c + 1] - mid;
        EXPECT_EQ(ml::nf4_code(mid), d_hi < d_lo ? c + 1 : c) << int(c);
    }
}

struct quantised
{
    std::vector<std::uint8_t> codes;
    std::vector<ml::float8_e4m3> blocks;
    std::vector<float> groups;
    ml::nf4_weights w;

    quantised(const std::vector<float>& x, std::size_t rows, std::size_t cols)
        : codes(ml::nf4_code_bytes(x.size())), blocks(ml::nf4_block_count(x.size())), groups(ml::nf4_group_count(x.size()))
    {
        const float mean = ml::nf4_quantize(x, codes, blocks, groups);
        w = ml::nf4_weights{codes, blocks, groups, mean, rows, cols};
    }
};

TEST(MlNf4Quantize, RoundTripsWithinTheStatedBound)
{
    stats::wyrand g(15);
    constexpr std::size_t rows = 300, cols = 64;   // 300 blocks: two second-level groups
    std::vector<float> x(rows * cols);
    for (auto& v : x)
        v = static_cast<float>(0.02 * math::normal_quantile((static_cast<double>(g() >> 11) + 0.5) * 0x1.0p-53));
    const quantised q(x, rows, cols);
    EXPECT_EQ(q.groups.size(), 2u);
    std::vector<float> back(x.size());
    ml::nf4_dequantize(q.w, back);
    double half_gap = 0;   // the largest half-gap between neighbouring values: 0.152, between -1 and -0.696
    for (std::size_t c = 0; c + 1 < 16; ++c)
        half_gap = std::max(half_gap, (static_cast<double>(ml::nf4_values[c + 1]) - ml::nf4_values[c]) / 2);
    double worst = 0;
    for (std::size_t b = 0; b < rows; ++b) {
        float c2 = 0;
        for (std::size_t j = 0; j < 64; ++j)
            c2 = std::max(c2, std::fabs(x[b * 64 + j]));
        const double s1 = q.groups[b / ml::nf4_group_blocks];
        const double scale_error = std::fabs(static_cast<double>(q.w.block_scale(b)) - c2);
        EXPECT_LE(scale_error, std::fabs(c2 - q.w.scale_mean) / 16 + s1 / 1024 + 1e-6 * c2) << b;
        for (std::size_t j = 0; j < 64; ++j) {
            const double e = std::fabs(static_cast<double>(back[b * 64 + j]) - x[b * 64 + j]);
            EXPECT_LE(e, half_gap * c2 + scale_error + 1e-6 * c2);
            worst = std::max(worst, e / c2);
        }
    }
    EXPECT_LT(worst, 0.2);
}

TEST(MlNf4Quantize, IsExactWhenEveryBlockSharesItsMaximumAndHandlesAZeroBlock)
{
    constexpr std::size_t rows = 4, cols = 128;
    std::vector<float> x(rows * cols);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = 0.5f * ml::nf4_values[(i * 7) % 16];
    const quantised q(x, rows, cols);
    EXPECT_EQ(q.w.scale_mean, 0.5f);
    EXPECT_EQ(q.groups[0], 0.0f);
    std::vector<float> back(x.size());
    ml::nf4_dequantize(q.w, back);
    EXPECT_EQ(back, x);
    std::vector<float> z(64, 0.0f);
    z.resize(128, 1.0f);
    const quantised qz(z, 1, 128);
    EXPECT_EQ(qz.w.at(0, 5), 0.0f);
    EXPECT_EQ(qz.w.code(5), 7);
    EXPECT_NEAR(qz.w.at(0, 100), 1.0f, 1.0f / 16);
}

TEST(MlNf4Linear, IsTheDenseProductOfTheDequantisedMatrixToTheBit)
{
    stats::wyrand g(16);
    constexpr std::size_t rows = 5, cols = 128, n = 3;
    std::vector<float> x(rows * cols), in(n * cols), bias(rows), dy(n * rows);
    for (auto& v : x)
        v = static_cast<float>(stats::uniform_real(g) - 0.5);
    for (auto& v : in)
        v = static_cast<float>(stats::uniform_real(g) - 0.5);
    for (auto& v : bias)
        v = static_cast<float>(stats::uniform_real(g));
    for (auto& v : dy)
        v = static_cast<float>(stats::uniform_real(g) - 0.5);
    const quantised q(x, rows, cols);
    std::vector<float> deq(x.size());
    ml::nf4_dequantize(q.w, deq);
    const ml::dense_weights<float> dense{deq, rows, cols};
    std::vector<float> y_nf4(n * rows), y_dense(n * rows), dx_nf4(n * cols, 0.0f), dx_dense(n * cols, 0.0f), dw(rows * cols, 0.0f);
    ml::linear_sequence(backend, q.w, in, y_nf4, bias);
    for (std::size_t t = 0; t < n; ++t) {
        const auto xt = std::span<const float>(in).subspan(t * cols, cols);
        ml::linear(dense, ml::prepare_input<cols>(dense, xt), std::span<float>(y_dense).subspan(t * rows, rows), bias);
        ml::linear_backward(dense, xt, std::span<const float>(dy).subspan(t * rows, rows), std::span<float>(dx_dense).subspan(t * cols, cols), dw);
    }
    ml::linear_sequence_backward_input(backend, q.w, dy, dx_nf4);
    for (std::size_t i = 0; i < y_nf4.size(); ++i)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(y_nf4[i]), std::bit_cast<std::uint32_t>(y_dense[i])) << i;
    for (std::size_t i = 0; i < dx_nf4.size(); ++i)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(dx_nf4[i]), std::bit_cast<std::uint32_t>(dx_dense[i])) << i;
    std::vector<float> one(rows), one_dense(rows);
    ml::linear(q.w, ml::prepare_input<cols>(q.w, std::span<const float>(in).first(cols)), one);
    ml::linear(dense, std::span<const float>(in).first(cols), one_dense);
    EXPECT_EQ(one, one_dense);   // without a bias
}

} // namespace
