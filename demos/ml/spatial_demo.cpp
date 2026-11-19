//
//  spatial_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/spatial.hpp>: max and average pooling with the backward pass, upscaling in all three
//  modes, zero and reflect padding, roll, SAM's window partition, and SAM's decomposed relative positions. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/spatial.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core spatial demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [spatial]
    // A 4x4 plane, 2x2 pools at stride 2 (ggml's pool_2d: the average counts every tap).
    std::array<float, 16> x{};
    for (std::size_t i = 0; i < 16; ++i)
        x[i] = static_cast<float>(i);
    constexpr auto pool = ml::pool_2d_shape(1, 4, 4, 2, 2, 2, 2);
    std::array<float, pool.y_size()> mx{}, avg{};
    ml::pool_forward(pool, ml::pool_kind::max, x, mx);
    ml::pool_forward(pool, ml::pool_kind::avg, x, avg);
    std::array<float, 16> dx{};
    ml::pool_backward(pool, ml::pool_kind::max, x, std::array<float, 4>{1, 1, 1, 1}, dx);   // to each window's maximum

    // Upscale a 2x2 plane to 4x4: nearest, bilinear at half-pixel centres, bicubic with corners aligned.
    const std::array<float, 4> small{0, 1, 2, 3};
    std::array<float, 16> nearest{}, bilinear{}, bicubic{};
    ml::upscale(ml::upscale_mode::nearest, false, small, 2, 2, nearest, 4, 4);
    ml::upscale(ml::upscale_mode::bilinear, false, small, 2, 2, bilinear, 4, 4);
    ml::upscale(ml::upscale_mode::bicubic, true, small, 2, 2, bicubic, 4, 4);

    // Zero padding of a 1x3 row by one on each side; reflect padding; a roll by one.
    std::array<float, 15> padded{};
    ml::pad<2>({1, 3}, {1, 1}, {1, 1}, std::array<float, 3>{1, 2, 3}, padded);
    std::array<float, 7> reflected{};
    ml::pad_reflect_1d(1, 3, 2, 2, std::array<float, 3>{1, 2, 3}, reflected);
    std::array<float, 3> rolled{};
    ml::roll<1>({3}, {1}, std::array<float, 3>{1, 2, 3}, rolled);

    // SAM: a 3x3 grid of one channel into windows of 2 (four windows, zero-padded), and back.
    std::array<float, 9> grid{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::array<float, ml::window_count(3, 3, 2) * 4> windows{};
    ml::window_partition(3, 3, 1, 2, grid, windows);
    std::array<float, 9> back{};
    ml::window_unpartition(3, 3, 1, 2, windows, back);

    // SAM's relative positions for a 2x2 window of one channel: tables gathered, scores added.
    std::array<float, 4> rh{}, rw{};
    ml::rel_pos_gather(std::array<float, 3>{-1, 0, 1}, 2, rh);   // rh[q][k] = table[q - k + 1]
    ml::rel_pos_gather(std::array<float, 3>{-2, 0, 2}, 2, rw);
    std::array<float, 8> rel_h{}, rel_w{};
    ml::rel_pos_scores(std::array<float, 4>{1, 1, 1, 1}, rh, rw, 2, 2, 1, rel_h, rel_w);
    std::array<float, 16> attn{};
    ml::add_rel_pos(attn, rel_h, rel_w, 2, 2);
    //! [spatial]

    std::printf("  max %.0f %.0f %.0f %.0f, avg %.1f %.1f %.1f %.1f, max's gradient to x[5] %.0f\n", static_cast<double>(mx[0]),
                static_cast<double>(mx[1]), static_cast<double>(mx[2]), static_cast<double>(mx[3]), static_cast<double>(avg[0]),
                static_cast<double>(avg[1]), static_cast<double>(avg[2]), static_cast<double>(avg[3]), static_cast<double>(dx[5]));
    ok &= mx == std::array<float, 4>{5, 7, 13, 15} && avg == std::array<float, 4>{2.5f, 4.5f, 10.5f, 12.5f} && dx[5] == 1 && dx[0] == 0;
    std::printf("  upscale row 1: nearest %.2f %.2f, bilinear %.2f %.2f, bicubic corners %.2f %.2f\n", static_cast<double>(nearest[4]),
                static_cast<double>(nearest[7]), static_cast<double>(bilinear[4]), static_cast<double>(bilinear[5]),
                static_cast<double>(bicubic[0]), static_cast<double>(bicubic[15]));
    ok &= nearest[4] == 0 && nearest[7] == 1 && bilinear[4] == 0.5f && bilinear[5] == 0.75f && bicubic[0] == 0 && bicubic[15] == 3;
    std::printf("  pad: %.0f %.0f %.0f %.0f %.0f (row 1), reflect: %.0f %.0f %.0f %.0f %.0f %.0f %.0f, roll: %.0f %.0f %.0f\n",
                static_cast<double>(padded[5]), static_cast<double>(padded[6]), static_cast<double>(padded[7]),
                static_cast<double>(padded[8]), static_cast<double>(padded[9]), static_cast<double>(reflected[0]),
                static_cast<double>(reflected[1]), static_cast<double>(reflected[2]), static_cast<double>(reflected[3]),
                static_cast<double>(reflected[4]), static_cast<double>(reflected[5]), static_cast<double>(reflected[6]),
                static_cast<double>(rolled[0]), static_cast<double>(rolled[1]), static_cast<double>(rolled[2]));
    ok &= padded[6] == 1 && padded[8] == 3 && padded[0] == 0 && reflected == std::array<float, 7>{3, 2, 1, 2, 3, 2, 1}
          && rolled == std::array<float, 3>{3, 1, 2};
    std::printf("  windows: %zu, the last %.0f %.0f %.0f %.0f; unpartition restores the grid: %s\n", windows.size() / 4,
                static_cast<double>(windows[12]), static_cast<double>(windows[13]), static_cast<double>(windows[14]),
                static_cast<double>(windows[15]), back == grid ? "yes" : "no");
    ok &= windows[12] == 9 && windows[13] == 0 && windows[15] == 0 && back == grid;
    std::printf("  relative positions: score (0,0)->(1,1) %.0f, (1,1)->(0,0) %.0f\n", static_cast<double>(attn[3]),
                static_cast<double>(attn[12]));
    ok &= attn[3] == -1 - 2 && attn[12] == 1 + 2;

    std::printf(ok ? "spatial demo: ok\n" : "spatial demo: FAILED\n");
    return ok ? 0 : 1;
}
