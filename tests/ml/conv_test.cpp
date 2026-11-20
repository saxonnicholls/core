//
//  conv_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/conv.hpp>: shapes and their compile-time sizes; im2col against ggml's index formula
//  and col2im as its adjoint; the direct and lowered forwards against each other and a loop written out here, to the
//  bit, over ranks 1 to 3 with strides, padding, dilation and groups; depthwise as a per-channel convolution; the
//  backward passes against central differences; the transposed convolution against ggml's scatter and as the adjoint of
//  the convolution; and compile-time use.
//

#include <snicholls/core/ml/conv.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

// Compile time: the sizes, and a 1-D convolution evaluated by the compiler
constexpr ml::conv_shape<2> vit_patch{.in_channels = 3, .out_channels = 8, .input = {224, 224}, .kernel = {16, 16}, .stride = {16, 16}};
static_assert(vit_patch.valid() && vit_patch.output() == std::array<std::size_t, 2>{14, 14} && vit_patch.output_size() == 196
              && vit_patch.column_width() == 768 && vit_patch.weight_size() == 8 * 768);
constexpr ml::conv_shape<1> whisper{.in_channels = 80, .out_channels = 4, .input = {3000}, .kernel = {3}, .stride = {2}, .padding = {1}};
static_assert(whisper.output()[0] == 1500);
static_assert(!ml::conv_shape<1>{.in_channels = 3, .out_channels = 2, .input = {4}, .kernel = {3}, .groups = 2}.valid());
static_assert(!ml::conv_shape<1>{.in_channels = 1, .out_channels = 1, .input = {2}, .kernel = {3}}.valid());
static_assert(ml::depthwise_shape(4, 8, 8, 3, 3, 1, 1).groups == 4 && ml::depthwise_shape(4, 8, 8, 3, 3, 1, 1).weight_size() == 36);
constexpr ml::conv_transpose_shape<2> up2{.in_channels = 4, .out_channels = 2, .input = {5, 7}, .kernel = {2, 2}, .stride = {2, 2}};
static_assert(up2.valid() && up2.output() == std::array<std::size_t, 2>{10, 14} && up2.as_conv().output() == up2.input);

constexpr bool compile_time_conv()
{
    constexpr ml::conv_shape<1> s{.in_channels = 1, .out_channels = 1, .input = {4}, .kernel = {2}};
    const std::array<double, 4> x{1, 2, 3, 4};
    const std::array<double, 2> w{1, -1};
    const std::array<double, 1> b{0.5};
    std::array<double, s.y_size()> y{}, z{};
    std::array<double, s.column_size()> cols{};
    ml::conv_forward(s, x, w, b, y);
    ml::conv_forward_im2col(s, x, w, b, z, cols);
    return y == std::array<double, 3>{-0.5, -0.5, -0.5} && y == z && cols == std::array<double, 6>{1, 2, 2, 3, 3, 4};
}
static_assert(compile_time_conv());

constexpr bool compile_time_transpose()
{
    constexpr ml::conv_transpose_shape<1> s{.in_channels = 1, .out_channels = 1, .input = {2}, .kernel = {2}, .stride = {2}};
    const std::array<double, 2> x{1, 2}, w{3, 4};
    std::array<double, s.y_size()> y{};
    ml::conv_transpose_forward(s, x, w, std::span<const double>{}, y);
    return y == std::array<double, 4>{3, 4, 6, 8};
}
static_assert(compile_time_transpose());

template <typename T>
std::vector<T> noise(std::size_t n, std::uint64_t seed)
{
    std::vector<T> v(n);
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ull + 1;
    for (auto& e : v) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        e = static_cast<T>(static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return v;
}

template <typename T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

template <typename T>
T dot(const std::vector<T>& a, const std::vector<T>& b)
{
    T s = T(0);
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

// A 2-D convolution written out: oc, oy, ox, then ic, ky, kx, out-of-bounds taps skipped, the bias last
std::vector<float> conv2d_by_hand(const ml::conv_shape<2>& s, const std::vector<float>& x, const std::vector<float>& w,
                                  const std::vector<float>& b)
{
    const auto o = s.output();
    const std::size_t cg = s.in_channels / s.groups, og = s.out_channels / s.groups;
    std::vector<float> y(s.out_channels * o[0] * o[1]);
    for (std::size_t oc = 0; oc < s.out_channels; ++oc)
        for (std::size_t oy = 0; oy < o[0]; ++oy)
            for (std::size_t ox = 0; ox < o[1]; ++ox) {
                float sum = 0.0f;
                for (std::size_t c = 0; c < cg; ++c)
                    for (std::size_t ky = 0; ky < s.kernel[0]; ++ky)
                        for (std::size_t kx = 0; kx < s.kernel[1]; ++kx) {
                            const auto iy = static_cast<std::ptrdiff_t>(oy * s.stride[0] + ky * s.dilation[0]) - static_cast<std::ptrdiff_t>(s.padding[0]);
                            const auto ix = static_cast<std::ptrdiff_t>(ox * s.stride[1] + kx * s.dilation[1]) - static_cast<std::ptrdiff_t>(s.padding[1]);
                            if (iy < 0 || ix < 0 || iy >= static_cast<std::ptrdiff_t>(s.input[0]) || ix >= static_cast<std::ptrdiff_t>(s.input[1]))
                                continue;
                            const std::size_t ic = (oc / og) * cg + c;
                            const float term = w[((oc * cg + c) * s.kernel[0] + ky) * s.kernel[1] + kx]
                                               * x[(ic * s.input[0] + static_cast<std::size_t>(iy)) * s.input[1] + static_cast<std::size_t>(ix)];
                            sum += term;
                        }
                y[(oc * o[0] + oy) * o[1] + ox] = b.empty() ? sum : sum + b[oc];
            }
    return y;
}

const std::vector<ml::conv_shape<2>> shapes2{
    {.in_channels = 3, .out_channels = 4, .input = {7, 9}, .kernel = {3, 3}},
    {.in_channels = 3, .out_channels = 4, .input = {7, 9}, .kernel = {3, 2}, .stride = {2, 3}, .padding = {1, 2}},
    {.in_channels = 4, .out_channels = 6, .input = {8, 8}, .kernel = {3, 3}, .stride = {1, 1}, .padding = {2, 2}, .dilation = {2, 2}, .groups = 2},
    {.in_channels = 2, .out_channels = 3, .input = {9, 9}, .kernel = {2, 2}, .stride = {4, 4}},   // stride beyond the kernel
    {.in_channels = 5, .out_channels = 3, .input = {4, 6}, .kernel = {1, 1}},                     // 1x1
    {.in_channels = 4, .out_channels = 4, .input = {6, 5}, .kernel = {3, 3}, .stride = {1, 1}, .padding = {1, 1}, .dilation = {1, 1}, .groups = 4},
};

// The backward pass of either layer, by overload
template <typename... A>
void backward(const ml::conv_shape<2>& s, A&&... a) { ml::conv_backward(s, a...); }
template <typename... A>
void backward(const ml::conv_shape<1>& s, A&&... a) { ml::conv_backward(s, a...); }
template <typename... A>
void backward(const ml::conv_transpose_shape<2>& s, A&&... a) { ml::conv_transpose_backward(s, a...); }

TEST(ConvForward, DirectAndIm2colAgreeWithTheLoopByHandToTheBit)
{
    for (const auto& s : shapes2) {
        ASSERT_TRUE(s.valid());
        const auto x = noise<float>(s.x_size(), 1), w = noise<float>(s.weight_size(), 2), b = noise<float>(s.out_channels, 3);
        std::vector<float> direct(s.y_size()), lowered(s.y_size()), cols(s.column_size());
        ml::conv_forward(s, x, w, b, direct);
        ml::conv_forward_im2col(s, x, w, b, lowered, cols);
        EXPECT_TRUE(same_bits(direct, conv2d_by_hand(s, x, w, b)));
        EXPECT_TRUE(same_bits(direct, lowered));
        ml::conv_forward(s, x, w, std::span<const float>{}, direct);
        EXPECT_TRUE(same_bits(direct, conv2d_by_hand(s, x, w, {})));
    }
}

TEST(ConvIm2col, MatchesGgmlsIndexFormula)
{
    // ggml_compute_forward_im2col_f32: dst[(oh*OW + ow)*(IC*KH*KW) + ic*KH*KW + kh*KW + kw] = src[ic][oh*s1 + kh*d1 - p1][ow*s0 + kw*d0 - p0], 0 outside
    const ml::conv_shape<2> s{.in_channels = 2, .out_channels = 1, .input = {5, 6}, .kernel = {3, 2}, .stride = {2, 1}, .padding = {1, 0}, .dilation = {1, 2}};
    const auto x = noise<float>(s.x_size(), 4);
    std::vector<float> cols(s.column_size());
    ml::im2col(s, x, cols);
    const auto o = s.output();
    for (std::size_t oh = 0; oh < o[0]; ++oh)
        for (std::size_t ow = 0; ow < o[1]; ++ow)
            for (std::size_t ic = 0; ic < 2; ++ic)
                for (std::size_t kh = 0; kh < 3; ++kh)
                    for (std::size_t kw = 0; kw < 2; ++kw) {
                        const long iih = long(oh * 2 + kh) - 1, iiw = long(ow + kw * 2);
                        const float expected = iih < 0 || iih >= 5 || iiw >= 6 ? 0.0f : x[(ic * 5 + std::size_t(iih)) * 6 + std::size_t(iiw)];
                        EXPECT_EQ(cols[(oh * o[1] + ow) * 12 + ic * 6 + kh * 2 + kw], expected);
                    }
}

TEST(ConvIm2col, Col2imIsItsAdjointInEveryRank)
{
    const ml::conv_shape<1> s1{.in_channels = 2, .out_channels = 1, .input = {11}, .kernel = {3}, .stride = {2}, .padding = {2}, .dilation = {2}};
    const ml::conv_shape<3> s3{.in_channels = 2, .out_channels = 1, .input = {4, 5, 3}, .kernel = {2, 3, 2}, .stride = {1, 2, 1}, .padding = {1, 1, 0}};
    const auto check = [](const auto& s) {
        const auto x = noise<double>(s.x_size(), 5), c = noise<double>(s.column_size(), 6);
        std::vector<double> cx(s.column_size()), xc(s.x_size());
        ml::im2col(s, x, cx);
        ml::col2im(s, c, xc);
        EXPECT_NEAR(dot(cx, c), dot(x, xc), 1e-12);
    };
    check(s1);
    check(shapes2[2]);
    check(s3);
}

TEST(ConvForward, Rank1And3AgreeBetweenTheTwoForms)
{
    const ml::conv_shape<1> s1{.in_channels = 4, .out_channels = 6, .input = {17}, .kernel = {3}, .stride = {2}, .padding = {1}, .dilation = {1}, .groups = 2};
    const ml::conv_shape<3> s3{.in_channels = 2, .out_channels = 3, .input = {4, 5, 6}, .kernel = {2, 2, 3}, .stride = {2, 1, 2}, .padding = {1, 0, 1}};
    const auto check = [](const auto& s) {
        const auto x = noise<float>(s.x_size(), 7), w = noise<float>(s.weight_size(), 8), b = noise<float>(s.out_channels, 9);
        std::vector<float> a(s.y_size()), l(s.y_size()), cols(s.column_size());
        ml::conv_forward(s, x, w, b, a);
        ml::conv_forward_im2col(s, x, w, b, l, cols);
        EXPECT_TRUE(same_bits(a, l));
    };
    check(s1);
    check(s3);
    // A 1-D convolution of a 1-D signal by hand: y[o] = sum_k w[k] x[2o + k - 1]
    const ml::conv_shape<1> s{.in_channels = 1, .out_channels = 1, .input = {6}, .kernel = {3}, .stride = {2}, .padding = {1}};
    const std::vector<double> x{1, 2, 3, 4, 5, 6}, w{1, 10, 100};
    std::vector<double> y(s.y_size());
    ml::conv_forward(s, x, w, std::span<const double>{}, y);
    EXPECT_EQ(y, (std::vector<double>{0 + 10 + 200, 2 * 1 + 30 + 400, 4 + 50 + 600}));
}

TEST(ConvDepthwise, IsEachChannelConvolvedWithItsOwnFilter)
{
    const auto s = ml::depthwise_shape(3, 6, 7, 3, 3, 2, 1);
    const auto x = noise<float>(s.x_size(), 10), w = noise<float>(s.weight_size(), 11);
    std::vector<float> y(s.y_size());
    ml::conv_forward(s, x, w, std::span<const float>{}, y);
    const ml::conv_shape<2> one{.in_channels = 1, .out_channels = 1, .input = {6, 7}, .kernel = {3, 3}, .stride = {2, 2}, .padding = {1, 1}};
    for (std::size_t c = 0; c < 3; ++c) {
        std::vector<float> yc(one.y_size());
        ml::conv_forward(one, std::span(x).subspan(c * 42, 42), std::span(w).subspan(c * 9, 9), std::span<const float>{}, yc);
        EXPECT_TRUE(same_bits(yc, std::vector<float>(y.begin() + long(c * yc.size()), y.begin() + long((c + 1) * yc.size()))));
    }
}

template <typename Shape, typename Forward>
void check_gradients(const Shape& s, Forward forward)
{
    // The loss L = <y, r>; dx, dw, db by the backward pass against central differences
    auto x = noise<double>(s.x_size(), 12), w = noise<double>(s.weight_size(), 13), b = noise<double>(s.out_channels, 14);
    const auto r = noise<double>(s.y_size(), 15);
    const auto loss = [&] {
        std::vector<double> y(s.y_size());
        forward(x, w, b, y);
        return dot(y, r);
    };
    std::vector<double> dx(s.x_size()), dw(s.weight_size(), 0.0), db(s.out_channels, 0.0);
    backward(s, x, w, r, dx, dw, db);
    const double h = 1e-6;
    const auto central = [&](std::vector<double>& v, const std::vector<double>& grad) {
        for (std::size_t i = 0; i < v.size(); ++i) {
            const double keep = v[i];
            v[i] = keep + h;
            const double up = loss();
            v[i] = keep - h;
            const double down = loss();
            v[i] = keep;
            EXPECT_NEAR(grad[i], (up - down) / (2 * h), 1e-7);
        }
    };
    central(x, dx);
    central(w, dw);
    central(b, db);
}

TEST(ConvBackward, MatchesCentralDifferences)
{
    const ml::conv_shape<2> s = shapes2[2];
    check_gradients(s, [&](const auto& x, const auto& w, const auto& b, auto& y) { ml::conv_forward(s, x, w, b, y); });
    const ml::conv_shape<1> s1{.in_channels = 2, .out_channels = 2, .input = {9}, .kernel = {4}, .stride = {3}, .padding = {2}};
    check_gradients(s1, [&](const auto& x, const auto& w, const auto& b, auto& y) { ml::conv_forward(s1, x, w, b, y); });
}

TEST(ConvBackward, EmptyGradientsAreSkippedAndWeightsAccumulate)
{
    const ml::conv_shape<2> s = shapes2[0];
    const auto x = noise<double>(s.x_size(), 16), w = noise<double>(s.weight_size(), 17), dy = noise<double>(s.y_size(), 18);
    std::vector<double> dw1(s.weight_size(), 0.0), dw2(s.weight_size(), 0.0);
    ml::conv_backward(s, x, w, dy, std::span<double>{}, dw1, std::span<double>{});
    ml::conv_backward(s, x, std::span<const double>{}, dy, std::span<double>{}, dw2, std::span<double>{});
    ml::conv_backward(s, x, std::span<const double>{}, dy, std::span<double>{}, dw2, std::span<double>{});
    for (std::size_t i = 0; i < dw1.size(); ++i)
        EXPECT_DOUBLE_EQ(dw2[i], 2 * dw1[i]);
}

TEST(ConvTranspose, MatchesGgmlsScatter)
{
    // ggml_compute_forward_conv_transpose_2d: dst[oc][iy*s + ky][ix*s + kx] += sum_ic src[ic][iy][ix] * w[ic][oc][ky][kx]
    const ml::conv_transpose_shape<2> s{.in_channels = 3, .out_channels = 2, .input = {3, 4}, .kernel = {3, 2}, .stride = {2, 2}};
    const auto x = noise<double>(s.x_size(), 19), w = noise<double>(s.weight_size(), 20);
    std::vector<double> y(s.y_size());
    ml::conv_transpose_forward(s, x, w, std::span<const double>{}, y);
    const auto o = s.output();
    ASSERT_EQ(o, (std::array<std::size_t, 2>{7, 8}));
    std::vector<double> ref(y.size(), 0.0);
    for (std::size_t oc = 0; oc < 2; ++oc)
        for (std::size_t iy = 0; iy < 3; ++iy)
            for (std::size_t ix = 0; ix < 4; ++ix)
                for (std::size_t ky = 0; ky < 3; ++ky)
                    for (std::size_t kx = 0; kx < 2; ++kx)
                        for (std::size_t ic = 0; ic < 3; ++ic)
                            ref[(oc * 7 + iy * 2 + ky) * 8 + ix * 2 + kx] += x[(ic * 3 + iy) * 4 + ix] * w[((ic * 2 + oc) * 3 + ky) * 2 + kx];
    for (std::size_t i = 0; i < y.size(); ++i)
        EXPECT_NEAR(y[i], ref[i], 1e-14);
    // 1-D, as ggml_conv_transpose_1d: output (in - 1) s + k
    const ml::conv_transpose_shape<1> t{.in_channels = 1, .out_channels = 1, .input = {3}, .kernel = {3}, .stride = {2}};
    const std::vector<double> xs{1, 2, 3}, ws{1, 10, 100};
    std::vector<double> ys(t.y_size());
    ml::conv_transpose_forward(t, xs, ws, std::vector<double>{0.5}, ys);
    EXPECT_EQ(ys, (std::vector<double>{1.5, 10.5, 102.5, 20.5, 203.5, 30.5, 300.5}));
}

TEST(ConvTranspose, IsTheAdjointOfTheConvolutionAndHasItsGradients)
{
    const ml::conv_transpose_shape<2> s{.in_channels = 4, .out_channels = 6, .input = {4, 3}, .kernel = {3, 3}, .stride = {2, 1},
                                        .padding = {1, 1}, .dilation = {1, 2}, .groups = 2};
    ASSERT_TRUE(s.valid());
    const auto c = s.as_conv();
    const auto w = noise<double>(s.weight_size(), 21), u = noise<double>(c.x_size(), 22), v = noise<double>(s.x_size(), 23);
    std::vector<double> cu(c.y_size()), tv(s.y_size());
    ml::conv_forward(c, u, w, std::span<const double>{}, cu);
    ml::conv_transpose_forward(s, v, w, std::span<const double>{}, tv);
    EXPECT_NEAR(dot(cu, v), dot(u, tv), 1e-12);
    check_gradients(s, [&](const auto& x, const auto& ww, const auto& b, auto& y) { ml::conv_transpose_forward(s, x, ww, b, y); });
    EXPECT_FALSE((ml::conv_transpose_shape<1>{.in_channels = 1, .out_channels = 1, .input = {1}, .kernel = {1}, .padding = {1}}.valid()));
}

} // namespace
