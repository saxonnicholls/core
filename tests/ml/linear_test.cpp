//
//  linear_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/linear.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
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

// linear_backward: dx += W^T dy, dw += dy x^T, db += dy
template <typename W>
concept has_backward = requires(const W& w, std::span<const float> x, std::span<float> g) { ml::linear_backward(w, x, x, g, g, g); };
static_assert(has_backward<ml::dense_weights<float>> && has_backward<ml::dense_weights<ml::bfloat16>>);
static_assert(!has_backward<ml::quantized_weights<ml::block_q4_0>>, "quantised weights are inference only");

constexpr bool backward_compile_time()
{
    const std::array<double, 6> w{1, 2, 3, 4, 5, 6};   // 2 x 3
    const std::array<double, 3> x{1, 0, -1};
    const std::array<double, 2> dy{1, 10};
    std::array<double, 3> dx{100, 0, 0};
    std::array<double, 6> dw{};
    std::array<double, 2> db{};
    ml::linear_backward(ml::dense_weights<double>{w, 2, 3}, x, dy, dx, dw, db);
    return dx == std::array<double, 3>{100 + 1 + 40, 2 + 50, 3 + 60} && dw == std::array<double, 6>{1, 0, -1, 10, 0, -10}
           && db == std::array<double, 2>{1, 10};
}
static_assert(backward_compile_time());

TEST(MlLinear, BackwardIsTheTransposeTheOuterProductAndTheGradient)
{
    constexpr std::size_t rows = 3, cols = 5;
    const auto w = pattern(rows * cols, 0.5f, 1), x = pattern(cols, 2.0f, 2), dy = pattern(rows, 1.0f, 3);
    std::vector<float> dx(cols), dw(rows * cols), db(rows);
    ml::linear_backward(ml::dense_weights<float>{w, rows, cols}, x, dy, dx, dw, db);
    std::vector<float> want_dx(cols);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            const float p = w[i * cols + j] * dy[i];
            want_dx[j] += p;
            EXPECT_EQ(dw[i * cols + j], dy[i] * x[j]);
        }
    EXPECT_EQ(dx, want_dx);
    EXPECT_EQ(db, dy);
}

TEST(MlLinear, BackwardAccumulatesAndSkipsAnEmptyInputOrBiasGradient)
{
    constexpr std::size_t rows = 2, cols = 4;
    const auto w = pattern(rows * cols, 0.5f, 1), x = pattern(cols, 2.0f, 2), dy = pattern(rows, 1.0f, 3);
    const ml::dense_weights<float> m{w, rows, cols};
    std::vector<float> dx(cols), dw(rows * cols), db(rows), dw2(rows * cols);
    ml::linear_backward(m, x, dy, dx, dw, db);
    ml::linear_backward(m, x, dy, dx, dw, db);
    ml::linear_backward(m, x, dy, std::span<float>{}, dw2);
    for (std::size_t i = 0; i < dw.size(); ++i)
        EXPECT_EQ(dw[i], dw2[i] + dw2[i]);
    for (std::size_t i = 0; i < rows; ++i)
        EXPECT_EQ(db[i], dy[i] + dy[i]);
}

TEST(MlLinear, BackwardMatchesCentralDifferences)
{
    // L = g . (W x + b): dL/dx, dL/dW, dL/db by central differences in double
    constexpr std::size_t rows = 3, cols = 4;
    std::vector<double> w(rows * cols), x(cols), b(rows), g(rows);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = std::sin(0.3 * double(i) + 0.2);
    for (std::size_t i = 0; i < cols; ++i) x[i] = std::cos(0.7 * double(i));
    for (std::size_t i = 0; i < rows; ++i) { b[i] = 0.1 * double(i); g[i] = std::sin(1.3 * double(i) + 0.5); }
    auto loss = [&] {
        std::vector<double> y(rows);
        ml::linear(ml::dense_weights<double>{w, rows, cols}, x, y, b);
        double l = 0;
        for (std::size_t i = 0; i < rows; ++i) l += g[i] * y[i];
        return l;
    };
    std::vector<double> dx(cols), dw(rows * cols), db(rows);
    ml::linear_backward(ml::dense_weights<double>{w, rows, cols}, x, g, dx, dw, db);
    auto check = [&](std::vector<double>& p, const std::vector<double>& grad) {
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double keep = p[i], h = 1e-6;
            p[i] = keep + h; const double up = loss();
            p[i] = keep - h; const double down = loss();
            p[i] = keep;
            EXPECT_NEAR(grad[i], (up - down) / (2 * h), 1e-8) << i;
        }
    };
    check(x, dx);
    check(w, dw);
    check(b, db);
}

TEST(MlLinear, BackwardWidensBFloat16Weights)
{
    constexpr std::size_t rows = 2, cols = 3;
    const std::vector<float> w{0.5f, -1.0f, 2.0f, 0.25f, 4.0f, -0.125f};   // exact in bfloat16
    std::vector<ml::bfloat16> wb;
    for (float v : w) wb.push_back(ml::to_minifloat<ml::bfloat16>(v));
    const std::vector<float> x{1, 2, 3}, dy{1, -2};
    std::vector<float> dx(cols), dw(rows * cols), dxb(cols), dwb(rows * cols);
    ml::linear_backward(ml::dense_weights<float>{w, rows, cols}, x, dy, dx, dw);
    ml::linear_backward(ml::dense_weights<ml::bfloat16>{wb, rows, cols}, x, dy, dxb, dwb);
    EXPECT_EQ(dx, dxb);
    EXPECT_EQ(dw, dwb);
}

// ---- the batched forms through a linalg backend ----

namespace la = snicholls::linalg;

constexpr bool sequence_compile_time()
{
    const std::array<double, 6> w{1, 2, 3, 4, 5, 6};   // 2 x 3
    const std::array<double, 6> xs{1, 0, -1, 0, 1, 0};   // two rows
    const std::array<double, 2> b{10, 20};
    std::array<double, 4> ys{};
    ml::linear_sequence(la::reference, ml::dense_weights<double>{w, 2, 3}, xs, ys, b);
    const std::array<double, 4> dys{1, 0, 0, 1};
    std::array<double, 6> dxs{}, dw{};
    std::array<double, 2> db{};
    ml::linear_sequence_backward(la::reference, ml::dense_weights<double>{w, 2, 3}, xs, dys, dxs, dw, db);
    return ys[0] == 8 && ys[1] == 18 && ys[2] == 12 && ys[3] == 25          // W x + b, row by row
           && dxs[0] == 1 && dxs[2] == 3 && dxs[3] == 4 && dxs[5] == 6      // row t of dY picks row t of W
           && dw[0] == 1 && dw[2] == -1 && dw[4] == 1 && db[0] == 1 && db[1] == 1;
}
static_assert(sequence_compile_time());

TEST(MlLinear, SequenceAgreesWithLinearRowByRow)
{
    constexpr std::size_t rows = 24, cols = 40, n = 7;
    const auto w = pattern(rows * cols, 0.3f, 1), xs = pattern(n * cols, 1.0f, 2), b = pattern(rows, 0.1f, 3);
    const ml::dense_weights<float> m{w, rows, cols};
    std::vector<float> ys(n * rows), row(rows);
    ml::linear_sequence(la::reference, m, xs, ys, b);
    for (std::size_t t = 0; t < n; ++t) {
        ml::linear(m, std::span<const float>(xs).subspan(t * cols, cols), row, b);
        for (std::size_t i = 0; i < rows; ++i)   // the same products; only where the bias joins the sum differs
            EXPECT_NEAR(ys[t * rows + i], row[i], 1e-5f) << t << " " << i;
    }
}

TEST(MlLinear, SequenceBackwardAgreesWithLinearBackwardRowByRow)
{
    constexpr std::size_t rows = 24, cols = 40, n = 7;
    const auto w = pattern(rows * cols, 0.3f, 1), xs = pattern(n * cols, 1.0f, 2), dys = pattern(n * rows, 0.5f, 4);
    const ml::dense_weights<float> m{w, rows, cols};
    std::vector<float> dxs(n * cols, 0.25f), dw(rows * cols, 0.5f), db(rows, 1.0f);   // accumulated into
    auto dxs_r = dxs, dw_r = dw, db_r = db;
    ml::linear_sequence_backward(la::reference, m, xs, dys, dxs, dw, db);
    for (std::size_t t = 0; t < n; ++t)
        ml::linear_backward(m, std::span<const float>(xs).subspan(t * cols, cols), std::span<const float>(dys).subspan(t * rows, rows),
                            std::span<float>(dxs_r).subspan(t * cols, cols), dw_r, db_r);
    for (std::size_t i = 0; i < dxs.size(); ++i)
        EXPECT_NEAR(dxs[i], dxs_r[i], 1e-5f);
    for (std::size_t i = 0; i < dw.size(); ++i)
        EXPECT_NEAR(dw[i], dw_r[i], 1e-5f);
    for (std::size_t i = 0; i < db.size(); ++i)
        EXPECT_NEAR(db[i], db_r[i], 1e-5f);
}

TEST(MlLinear, SequenceSkipsEmptyGradientsAndAnEmptySequence)
{
    const std::vector<float> w{1, 2, 3, 4, 5, 6}, xs{1, 1, 1}, dys{1, 1};
    const ml::dense_weights<float> m{w, 2, 3};
    std::vector<float> dw(6);
    ml::linear_sequence_backward(la::reference, m, xs, dys, {}, dw);   // no dX, no db
    EXPECT_EQ(dw, (std::vector<float>{1, 1, 1, 1, 1, 1}));
    std::vector<float> none;
    ml::linear_sequence(la::reference, m, {}, none);                    // zero rows: nothing to do
    ml::linear_sequence_backward(la::reference, m, {}, {}, {}, dw);
    EXPECT_EQ(dw, (std::vector<float>{1, 1, 1, 1, 1, 1}));
}

} // namespace
