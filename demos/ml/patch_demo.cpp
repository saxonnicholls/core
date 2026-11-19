//
//  patch_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/patch.hpp>: an RGB image cut into patches and put back, a video into tubelets, a
//  spectrogram into rectangular patches; the 2-D and 3-D sine-cosine position embeddings, as functions and as tables
//  the compiler builds; axial RoPE over the grid and gathered at the context tokens' positions. Exits non-zero if a
//  result is not what it documents.
//

#include <snicholls/core/ml/patch.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/rope.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core patch demo\n");
    namespace ml = snicholls::ml;
    const auto& backend = snicholls::linalg::reference;
    bool ok = true;

    //! [patchify]
    // A 4 x 4 RGB image in patches of 2 x 2: 4 tokens of 12 values, raster order, each row (dy, dx, c)
    constexpr ml::patch_grid image{.height = 4, .width = 4, .channels = 3, .patch_height = 2, .patch_width = 2};
    static_assert(ml::valid(image) && image.tokens() == 4 && image.token_size() == 12);
    std::vector<float> pixels(image.values());
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = float(i);   // pixel (y, x) channel c is ((y · 4 + x) · 3 + c)
    std::vector<float> tokens(image.values()), back(image.values());
    ml::patchify_rows(backend, image, std::span<const float>(pixels), std::span<float>(tokens));
    ml::unpatchify_rows(backend, image, std::span<const float>(tokens), std::span<float>(back));   // the inverse, and the backward

    // A video of 4 frames in tubelets of 2 x 2 x 2, and a 1-channel spectrogram (frequency x time) in 4 x 3 patches
    constexpr ml::patch_grid video{.frames = 4, .height = 4, .width = 4, .channels = 3, .tubelet = 2, .patch_height = 2, .patch_width = 2};
    constexpr ml::patch_grid spectrogram{.height = 8, .width = 12, .channels = 1, .patch_height = 4, .patch_width = 3};
    //! [patchify]
    std::printf("  image: %zu tokens of %zu; token 1 starts at pixel value %g (pixel (0, 2))\n", image.tokens(), image.token_size(),
                double(tokens[12]));
    std::printf("  video: %zu tubelets of %zu; spectrogram: %zu patches of %zu\n", video.tokens(), video.token_size(), spectrogram.tokens(),
                spectrogram.token_size());
    ok &= back == pixels && tokens[12] == float((0 * 4 + 2) * 3) && ml::patch_source(image, 3, 11) == pixels.size() - 1;
    ok &= video.tokens() == 8 && video.token_size() == 24 && spectrogram.tokens() == 8 && spectrogram.token_size() == 12;

    //! [positions]
    // The fixed sine-cosine embedding of a 2 x 3 grid in 8 channels: [row, column], each [sin ..., cos ...]
    std::array<double, 2 * 3 * 8> pos{}, swapped{};
    ml::sincos_position_2d<double>(8, 2, 3, pos);
    ml::sincos_position_2d<double>(8, 2, 3, swapped, ml::sincos_layout::column_first);
    // The same as a table the compiler builds, and in three dimensions (frame, row, column)
    static constexpr ml::sincos_table_2d<double, 8, 2, 3> table{};
    static constexpr ml::sincos_table_3d<float, 16, 2, 2, 2> video_table{};
    std::array<float, 2 * 2 * 2 * 12> uniform{};
    ml::sincos_position_3d<float>(12, 2, 2, 2, uniform, true);   // 4 channels per axis

    // Axial RoPE over the 2 x 3 grid for heads of 8 (2 pairs per axis), read at the context tokens' positions
    static constexpr ml::axial_rope_table<double, 8, 1, 2, 3> rope{};
    const std::array<std::uint32_t, 4> context{0, 2, 4, 5};
    const ml::gathered_rope<ml::axial_rope_table<double, 8, 1, 2, 3>> at_context{&rope, context, 2};   // two sequences of two
    //! [positions]
    // Token 4 is (row 1, column 1): its row half is axis 1 at 1, its column half the same
    std::printf("  position (1, 1): sin of the first frequency %.6f in both halves; table error %.1e\n", pos[4 * 8], std::abs(table.row(4)[0] - pos[4 * 8]));
    ok &= std::abs(pos[4 * 8] - std::sin(1.0)) < 1e-15 && std::abs(pos[4 * 8 + 4] - std::sin(1.0)) < 1e-15 && swapped[1 * 8] == pos[1 * 8 + 4];
    ok &= std::abs(table.row(5)[1] - pos[5 * 8 + 1]) < 1e-15 && video_table.row(0)[8] == 0.0f && video_table.row(0)[10] == 1.0f && uniform[0] == 0.0f;
    // The second sequence's first token is token 4: its column pairs rotate by 1
    const double angle = rope.cos_at(4)[2];
    ok &= at_context.sequence(1).cos_at(0)[2] == angle && at_context.sin_at(1)[0] == rope.sin_at(2)[0];
    ok &= std::abs(angle - std::cos(1.0)) < 1e-15;
    std::printf("  axial RoPE at token 4, first column pair: cos %.6f\n", angle);

    std::printf(ok ? "patch demo: ok\n" : "patch demo: FAILED\n");
    return ok ? 0 : 1;
}
