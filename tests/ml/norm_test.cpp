//
//  norm_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/norm.hpp>: layer_norm and rms_norm against long double, a row with a large
//  mean, the backward passes against central differences with and without parameters, accumulation over
//  rows, in-place use, empty rows, and compile-time use.
//

#include <snicholls/core/ml/norm.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;
using none = std::span<const double>;

constexpr bool compile_time()
{
    std::array<double, 4> x{1, 2, 3, 4}, y{}, z{};
    const auto s = ml::layer_norm(x, none{}, none{}, y);
    const double r = ml::rms_norm(x, none{}, z);
    return s.mean == 2.5 && m::near(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3], 4.0, 1e-4) && m::near(r, 1 / m::sqrt(7.5), 1e-12);
}
static_assert(compile_time());
static_assert(std::is_trivially_copyable_v<ml::norm_stats<float>> && sizeof(ml::norm_stats<float>) == 8);

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.7 * static_cast<double>(i) + b);
    return v;
}

TEST(MlNorm, LayerNormAgreesWithLongDouble)
{
    const auto x = wave(64, 3.0, 0.1), w = wave(64, 1.0, 1.0), b = wave(64, 0.5, 2.0);
    std::vector<double> y(64);
    const auto s = ml::layer_norm(x, w, b, y);
    long double mean = 0, var = 0;
    for (const double v : x)
        mean += v;
    mean /= 64;
    for (const double v : x)
        var += (v - mean) * (v - mean);
    var /= 64;
    const long double rstd = 1 / std::sqrt(var + 1e-5L);
    EXPECT_NEAR(s.mean, static_cast<double>(mean), 1e-15);
    EXPECT_NEAR(s.rstd, static_cast<double>(rstd), 1e-14);
    for (std::size_t i = 0; i < 64; ++i)
        EXPECT_NEAR(y[i], static_cast<double>((x[i] - mean) * rstd * w[i] + b[i]), 1e-14);
}

TEST(MlNorm, LayerNormKeepsTheVarianceOfARowWithALargeMean)
{
    std::vector<float> x(16);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = 10000.0f + (i % 2 == 0 ? 1.0f : -1.0f);   // variance exactly 1
    std::vector<float> y(16);
    const auto s = ml::layer_norm(x, std::span<const float>{}, std::span<const float>{}, y, 0.0f);
    EXPECT_NEAR(s.rstd, 1.0f, 1e-6f);   // E[x²] - E[x]² would cancel to 0 or garbage in float
    EXPECT_NEAR(y[0], 1.0f, 1e-6f);
}

TEST(MlNorm, RmsNormAgreesWithLongDouble)
{
    const auto x = wave(48, 2.0, 0.3), w = wave(48, 1.5, 0.9);
    std::vector<double> y(48);
    const double r = ml::rms_norm(x, w, y, 1e-6);
    long double ms = 0;
    for (const double v : x)
        ms += static_cast<long double>(v) * v;
    const long double rstd = 1 / std::sqrt(ms / 48 + 1e-6L);
    EXPECT_NEAR(r, static_cast<double>(rstd), 1e-15);
    for (std::size_t i = 0; i < 48; ++i)
        EXPECT_NEAR(y[i], static_cast<double>(x[i] * rstd * w[i]), 1e-14);
}

// loss(x, w, b) = sum_i c_i out_i, so dout = c; every gradient against a central difference
TEST(MlNorm, LayerNormBackwardMatchesCentralDifferences)
{
    const auto x = wave(10, 2.0, 0.4), w = wave(10, 1.2, 1.3), b = wave(10, 0.3, 2.2), c = wave(10, 1.0, 0.8);
    auto loss = [&](const std::vector<double>& xx, const std::vector<double>& ww, const std::vector<double>& bb) {
        std::vector<double> y(xx.size());
        ml::layer_norm(xx, ww, bb, y);
        double s = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            s += c[i] * y[i];
        return s;
    };
    std::vector<double> y(10), dx(10), dw(10, 0.0), db(10, 0.0);
    const auto stats = ml::layer_norm(x, w, b, y);
    ml::layer_norm_backward(c, x, w, stats, dx, dw, db);
    const double h = 1e-6;
    for (std::size_t k = 0; k < 10; ++k) {
        auto xu = x, xd = x, wu = w, wd = w, bu = b, bd = b;
        xu[k] += h; xd[k] -= h; wu[k] += h; wd[k] -= h; bu[k] += h; bd[k] -= h;
        EXPECT_NEAR(dx[k], (loss(xu, w, b) - loss(xd, w, b)) / (2 * h), 1e-7) << k;
        EXPECT_NEAR(dw[k], (loss(x, wu, b) - loss(x, wd, b)) / (2 * h), 1e-7) << k;
        EXPECT_NEAR(db[k], (loss(x, w, bu) - loss(x, w, bd)) / (2 * h), 1e-7) << k;
    }
    // A second row accumulates into dweight and dbias
    const auto dw1 = dw, db1 = db;
    ml::layer_norm_backward(c, x, w, stats, dx, dw, db);
    for (std::size_t k = 0; k < 10; ++k) {
        EXPECT_DOUBLE_EQ(dw[k], 2 * dw1[k]);
        EXPECT_DOUBLE_EQ(db[k], 2 * db1[k]);
    }
}

TEST(MlNorm, RmsNormBackwardMatchesCentralDifferencesWithAndWithoutAWeight)
{
    const auto x = wave(12, 1.7, 0.2), w = wave(12, 0.9, 1.7), c = wave(12, 1.0, 0.5);
    auto loss = [&](const std::vector<double>& xx, std::span<const double> ww) {
        std::vector<double> y(xx.size());
        ml::rms_norm(xx, ww, y, 1e-6);
        double s = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            s += c[i] * y[i];
        return s;
    };
    const double h = 1e-6;
    for (const bool weighted : {true, false}) {
        const std::span<const double> ww = weighted ? std::span<const double>(w) : none{};
        std::vector<double> y(12), dx(12), dw(weighted ? 12 : 0, 0.0);
        const double r = ml::rms_norm(x, ww, y, 1e-6);
        ml::rms_norm_backward(c, x, ww, r, dx, dw);
        for (std::size_t k = 0; k < 12; ++k) {
            auto xu = x, xd = x;
            xu[k] += h; xd[k] -= h;
            EXPECT_NEAR(dx[k], (loss(xu, ww) - loss(xd, ww)) / (2 * h), 1e-7) << k;
            if (weighted) {
                auto wu = w, wd = w;
                wu[k] += h; wd[k] -= h;
                EXPECT_NEAR(dw[k], (loss(x, wu) - loss(x, wd)) / (2 * h), 1e-7) << k;
            }
        }
    }
}

TEST(MlNorm, InPlaceAndEmptyRows)
{
    auto x = wave(8, 1.0, 0.0);
    std::vector<double> y(8);
    const auto s1 = ml::layer_norm(x, none{}, none{}, y);
    const auto s2 = ml::layer_norm(x, none{}, none{}, x);
    EXPECT_EQ(x, y);
    EXPECT_EQ(s1.rstd, s2.rstd);
    auto z = wave(8, 1.0, 0.5);
    std::vector<double> q(8);
    ml::rms_norm(z, none{}, q);
    ml::rms_norm(z, none{}, z);
    EXPECT_EQ(z, q);
    // dx in place over dout
    const auto in = wave(8, 2.0, 0.7);
    auto d = wave(8, 1.0, 0.1);
    std::vector<double> o(8), dx(8), none_w;
    const auto st = ml::layer_norm(in, none{}, none{}, o);
    ml::layer_norm_backward(d, in, none{}, st, dx, none_w, none_w);
    ml::layer_norm_backward(d, in, none{}, st, d, none_w, none_w);
    EXPECT_EQ(d, dx);
    // Empty rows: nothing written, nothing read
    std::vector<double> e;
    EXPECT_EQ(ml::layer_norm(e, none{}, none{}, e).mean, 0.0);
    ml::layer_norm_backward(e, e, none{}, ml::norm_stats<double>{0, 1}, e, e, e);
    ml::rms_norm_backward(e, e, none{}, 1.0, e, e);
    EXPECT_NEAR(ml::rms_norm(e, none{}, e, 1e-6), 1000.0, 1e-9);
}

// ggml's group_norm (ggml/src/ggml-cpu/ops.cpp, llama.cpp 39039aa), its sums in long double for float's ggml_float
std::vector<double> ggml_group_norm(const std::vector<double>& x, std::size_t channels, std::size_t groups, double eps)
{
    const std::size_t spatial = x.size() / channels, per = (channels + groups - 1) / groups;
    std::vector<double> y(x.size());
    for (std::size_t g = 0; g < groups; ++g) {
        const std::size_t start = g * per, end = std::min(start + per, channels);
        if (start >= end)
            continue;
        long double sum = 0;
        for (std::size_t i = start * spatial; i < end * spatial; ++i)
            sum += x[i];
        const long double mean = sum / static_cast<long double>((end - start) * spatial);
        long double sum2 = 0;
        for (std::size_t i = start * spatial; i < end * spatial; ++i)
            sum2 += (x[i] - mean) * (x[i] - mean);
        const long double scale = 1 / std::sqrt(sum2 / static_cast<long double>((end - start) * spatial) + eps);
        for (std::size_t i = start * spatial; i < end * spatial; ++i)
            y[i] = static_cast<double>((x[i] - mean) * scale);
    }
    return y;
}

TEST(MlNormGroup, GroupNormIsGgmlsAndSplitsChannelsAsGgmlDoes)
{
    // 10 channels in 4 groups: ceil(10 / 4) = 3 each, the last group 1
    EXPECT_EQ(ml::group_channels(10, 4, 0).first, 0u);
    EXPECT_EQ(ml::group_channels(10, 4, 3).first, 9u);
    EXPECT_EQ(ml::group_channels(10, 4, 3).last, 10u);
    EXPECT_EQ(ml::group_channels(4, 3, 2).first, ml::group_channels(4, 3, 2).last);   // ceil(4 / 3) = 2: the third group is empty
    for (const auto [c, g] : {std::pair<std::size_t, std::size_t>{10, 4}, {8, 2}, {4, 3}, {6, 6}, {5, 1}}) {
        const auto x = wave(c * 7, 2.5, 0.4);
        std::vector<double> y(x.size());
        ml::group_norm(x, c, g, none{}, none{}, y, 1e-6);
        const auto want = ggml_group_norm(x, c, g, 1e-6);
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(y[i], want[i], 1e-13) << c << " " << g << " " << i;
    }
}

TEST(MlNormGroup, OneGroupIsLayerNormAndOneChannelAGroupIsLayerNormPerChannelToTheBit)
{
    const std::size_t c = 6, s = 5;
    const auto x = wave(c * s, 3.0, 0.2), w = wave(c, 1.0, 1.0), b = wave(c, 0.5, 2.0);
    std::vector<double> wide_w(c * s), wide_b(c * s), y(c * s), want(c * s);
    for (std::size_t i = 0; i < c * s; ++i) {
        wide_w[i] = w[i / s];
        wide_b[i] = b[i / s];
    }
    ml::group_norm(x, c, 1, w, b, y);
    ml::layer_norm(x, wide_w, wide_b, want);
    EXPECT_EQ(y, want);
    ml::group_norm(x, c, c, w, b, y);
    for (std::size_t ch = 0; ch < c; ++ch)
        ml::layer_norm(std::span(x).subspan(ch * s, s), std::span(wide_w).subspan(ch * s, s), std::span(wide_b).subspan(ch * s, s),
                       std::span(want).subspan(ch * s, s));
    EXPECT_EQ(y, want);
}

TEST(MlNormGroup, GroupNormBackwardMatchesCentralDifferences)
{
    const std::size_t c = 6, s = 4, g = 3;
    auto x = wave(c * s, 2.0, 0.7);
    auto w = wave(c, 1.0, 0.3), b = wave(c, 0.5, 1.1);
    const auto coeff = wave(c * s, 1.0, 2.9);
    auto loss = [&] {
        std::vector<double> y(c * s);
        ml::group_norm(x, c, g, w, b, y);
        double l = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            l += coeff[i] * y[i];
        return l;
    };
    std::vector<double> dx(c * s), dw(c), db(c);
    ml::group_norm_backward(coeff, x, c, g, w, dx, dw, db);
    auto check = [&](std::vector<double>& v, const std::vector<double>& grad, const char* name) {
        for (std::size_t i = 0; i < v.size(); ++i) {
            const double keep = v[i], h = 1e-6;
            v[i] = keep + h;
            const double up = loss();
            v[i] = keep - h;
            const double down = loss();
            v[i] = keep;
            EXPECT_NEAR(grad[i], (up - down) / (2 * h), 1e-7) << name << " " << i;
        }
    };
    check(x, dx, "x");
    check(w, dw, "weight");
    check(b, db, "bias");
    // No parameters: the gradients without a weight are those with a weight of ones
    std::vector<double> dx1(c * s), dx2(c * s), ones(c, 1.0);
    ml::group_norm_backward(coeff, x, c, g, none{}, dx1, std::span<double>{}, std::span<double>{});
    ml::group_norm_backward(coeff, x, c, g, ones, dx2, std::span<double>{}, std::span<double>{});
    EXPECT_EQ(dx1, dx2);
}

TEST(MlNormL2, L2NormIsGgmlsAndItsBackwardMatchesCentralDifferences)
{
    auto x = wave(9, 1.5, 0.8);
    std::vector<double> y(9);
    const double n = ml::l2_norm(x, y);
    double sum = 0;
    for (const double v : x)
        sum += v * v;
    EXPECT_EQ(n, std::sqrt(sum));
    const double scale = 1.0 / std::fmax(std::sqrt(sum), 1e-12);   // ggml_compute_forward_l2_norm_f32
    for (std::size_t i = 0; i < 9; ++i)
        EXPECT_EQ(y[i], x[i] * scale);
    const auto coeff = wave(9, 1.0, 0.1);
    std::vector<double> dx(9);
    ml::l2_norm_backward(coeff, x, n, dx);
    for (std::size_t i = 0; i < 9; ++i) {
        auto f = [&](double v) {
            auto z = x;
            z[i] = v;
            std::vector<double> out(9);
            ml::l2_norm(z, out);
            double l = 0;
            for (std::size_t j = 0; j < 9; ++j)
                l += coeff[j] * out[j];
            return l;
        };
        EXPECT_NEAR(dx[i], (f(x[i] + 1e-6) - f(x[i] - 1e-6)) / 2e-6, 1e-8) << i;
    }
    // Below eps the scale is the constant 1 / eps
    const std::array<double, 2> tiny{1e-14, 0.0};
    std::array<double, 2> t{}, dt{};
    const double tn = ml::l2_norm(tiny, t);
    EXPECT_EQ(t[0], 1e-14 / 1e-12);
    ml::l2_norm_backward(std::array<double, 2>{1.0, 2.0}, tiny, tn, dt);
    EXPECT_EQ(dt[1], 2.0 / 1e-12);
}

constexpr bool group_compile_time()
{
    std::array<double, 4> x{1, 3, 10, 30}, y{}, z{};
    ml::group_norm(x, 2, 2, std::span<const double>{}, std::span<const double>{}, y, 0.0);   // each channel [a, 3a] to [-1, 1]
    const double n = ml::l2_norm(std::array<double, 2>{3, 4}, std::span(z).first(2));
    return y[0] == -1 && y[1] == 1 && m::near(y[2], -1.0, 1e-15) && m::near(y[3], 1.0, 1e-15) && n == 5 && m::near(z[0], 0.6, 1e-15)
           && m::near(z[1], 0.8, 1e-15);
}
static_assert(group_compile_time());

} // namespace
