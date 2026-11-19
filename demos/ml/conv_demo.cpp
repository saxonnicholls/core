//
//  conv_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/conv.hpp>: a 2-D convolution with padding, direct and through im2col; its backward
//  pass; a depthwise convolution; a 1-D convolution as Whisper's stem uses it; and a transposed convolution that
//  doubles a feature map. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/conv.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core conv demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [conv]
    // A 3x3 convolution of one 4x4 channel into two, padding 1: the sizes are known to the compiler.
    constexpr ml::conv_shape<2> s{.in_channels = 1, .out_channels = 2, .input = {4, 4}, .kernel = {3, 3}, .padding = {1, 1}};
    static_assert(s.output() == std::array<std::size_t, 2>{4, 4} && s.column_size() == 16 * 9);
    std::array<float, s.x_size()> x{};
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(i);
    std::array<float, s.weight_size()> w{};
    w[4] = 1.0f;                          // filter 0: the identity (the centre tap)
    for (std::size_t t = 9; t < 18; ++t)  // filter 1: the 3x3 box sum
        w[t] = 1.0f;
    const std::array<float, 2> bias{0.0f, -1.0f};
    std::array<float, s.y_size()> y{}, y2{};
    std::array<float, s.column_size()> cols{};
    ml::conv_forward(s, x, w, bias, y);               // the direct loop
    ml::conv_forward_im2col(s, x, w, bias, y2, cols);  // im2col and a matrix product: the same bits

    // The backward pass: dx written, dw and db added.
    std::array<float, s.y_size()> dy{};
    dy.fill(1.0f);
    std::array<float, s.x_size()> dx{};
    std::array<float, s.weight_size()> dw{};
    std::array<float, 2> db{};
    ml::conv_backward(s, x, w, dy, dx, dw, db);

    // Depthwise: one 3x3 filter per channel, stride 2.
    const auto dws = ml::depthwise_shape(3, 6, 6, 3, 3, 2, 1);
    std::vector<float> dwx(dws.x_size(), 1.0f), dww(dws.weight_size(), 1.0f), dwy(dws.y_size());
    ml::conv_forward(dws, dwx, dww, std::span<const float>{}, dwy);

    // 1-D, Whisper's second stem convolution: kernel 3, stride 2, padding 1, 3000 frames to 1500.
    constexpr ml::conv_shape<1> stem{.in_channels = 2, .out_channels = 2, .input = {3000}, .kernel = {3}, .stride = {2}, .padding = {1}};
    static_assert(stem.output()[0] == 1500);

    // A transposed convolution doubling a 2x2 map with a 2x2 kernel at stride 2.
    constexpr ml::conv_transpose_shape<2> up{.in_channels = 1, .out_channels = 1, .input = {2, 2}, .kernel = {2, 2}, .stride = {2, 2}};
    const std::array<float, 4> small{1, 2, 3, 4}, ones{1, 1, 1, 1};
    std::array<float, up.y_size()> big{};
    ml::conv_transpose_forward(up, small, ones, std::span<const float>{}, big);
    //! [conv]

    std::printf("  identity filter: y[5] %.1f (x[5] %.1f), box sum y[16 + 5] %.1f\n", static_cast<double>(y[5]), static_cast<double>(x[5]),
                static_cast<double>(y[16 + 5]));
    ok &= y[5] == x[5] && y[16 + 5] == 0 + 1 + 2 + 4 + 5 + 6 + 8 + 9 + 10 - 1.0f && y == y2;
    std::printf("  backward: db %.0f %.0f, dx[0] %.0f (corner: 1 + 4 taps)\n", static_cast<double>(db[0]), static_cast<double>(db[1]),
                static_cast<double>(dx[0]));
    ok &= db[0] == 16 && db[1] == 16 && dx[0] == 1 + 4;
    std::printf("  depthwise: %zu outputs, interior %.0f, corner %.0f\n", dwy.size(), static_cast<double>(dwy[4]), static_cast<double>(dwy[0]));
    ok &= dwy.size() == 27 && dwy[4] == 9 && dwy[0] == 4;
    std::printf("  transposed: %zu x %zu, row 0: %.0f %.0f %.0f %.0f\n", up.output()[0], up.output()[1], static_cast<double>(big[0]),
                static_cast<double>(big[1]), static_cast<double>(big[2]), static_cast<double>(big[3]));
    ok &= big[0] == 1 && big[1] == 1 && big[2] == 2 && big[3] == 2 && big[15] == 4;

    std::printf(ok ? "conv demo: ok\n" : "conv demo: FAILED\n");
    return ok ? 0 : 1;
}
