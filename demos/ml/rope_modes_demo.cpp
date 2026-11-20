//
//  rope_modes_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/rope_modes.hpp>: ggml's names for core's layouts, Qwen2-VL's multimodal RoPE for an
//  image token, a vision encoder's table over a grid of patches, LongRoPE's factors, and a key moved to a new position
//  by the K-shift. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/rope_modes.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core rope_modes demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [rope_modes]
    // A GGUF's rope type names a core layout: NORM pairs adjacent channels, NEOX the halves
    static_assert(ml::rope_mode_layout(ml::rope_mode::normal) == ml::rope_layout::interleaved);
    static_assert(ml::rope_mode_of(ml::rope_layout::half) == ml::rope_mode::neox);

    // Qwen2-VL: a head of 16 here, sections of 2, 3 and 3 pairs rotated by time, height and width
    const ml::rope_sections qwen{2, 3, 3, 0};
    std::array<float, 8> c{}, s{};
    ml::rope_mode_angles(ml::rope_position{5, 2, 7, 0}, c, s, 10000.0f, ml::rope_mode::mrope, qwen);
    std::array<float, 16> q{};
    q.fill(1.0f);
    ml::rope(q, c, s, ml::rope_mode_layout(ml::rope_mode::mrope));

    // A vision encoder's table over a 2 x 3 grid of patches, positions (row, column), sections d/4 each
    std::array<ml::rope_position, 6> grid{};
    ml::vision_rope_positions(2, 3, grid);
    constexpr std::size_t head = 16;
    const ml::rope_mode_table<float, 6, head> vision(std::span<const ml::rope_position, 6>(grid), 10000.0f, ml::rope_mode::vision,
                                                     ml::rope_sections{head / 4, head / 4, head / 4, head / 4});

    // LongRoPE (Phi-3): long factors past the trained context, and the amplitude its converter writes
    const std::array<float, 8> shorts{1, 1, 1, 1, 1, 1, 1, 1}, longs{1, 1, 2, 4, 8, 16, 32, 64};
    const auto factors = ml::longrope_factors<float>(shorts, longs, 131072, 4096);
    const float amplitude = ml::longrope_attention_factor(32.0f, 4096);
    ml::rope_mode_angles(ml::rope_position{1000}, c, s, 10000.0f, ml::rope_mode::neox, {}, factors, amplitude);

    // Context shift: a key cached at position 40, moved to 32 after eight tokens before it were dropped
    std::array<float, 8> key{0.5f, -1.0f, 0.25f, 2.0f, 1.0f, 0.0f, -0.5f, 1.5f}, fresh = key;
    std::array<float, 4> kc{}, ks{};
    ml::rope_angles(40, kc, ks);
    ml::rope(key, kc, ks);
    ml::rope_shift(key, -8, 10000.0f);
    ml::rope_angles(32, kc, ks);
    ml::rope(fresh, kc, ks);
    //! [rope_modes]

    std::printf("  mrope: pair 0 axis %zu, pair 2 axis %zu, pair 5 axis %zu; q[0] %.4f\n", ml::rope_section_axis(ml::rope_mode::mrope, qwen, 0),
                ml::rope_section_axis(ml::rope_mode::mrope, qwen, 2), ml::rope_section_axis(ml::rope_mode::mrope, qwen, 5),
                static_cast<double>(q[0]));
    ok &= ml::rope_section_axis(ml::rope_mode::mrope, qwen, 2) == 1 && ml::rope_section_axis(ml::rope_mode::mrope, qwen, 5) == 2;
    ok &= std::abs(q[0] - (std::cos(5.0f) - std::sin(5.0f))) < 1e-6f;   // pair 0 turns by t = 5 at frequency 1
    std::printf("  vision: patch (1, 2) sin of its first row pair %.4f, first column pair %.4f\n", static_cast<double>(vision.sin_at(5)[0]),
                static_cast<double>(vision.sin_at(5)[4]));
    ok &= std::abs(vision.sin_at(5)[0] - std::sin(1.0f)) < 1e-6f && std::abs(vision.sin_at(5)[4] - std::sin(2.0f)) < 1e-6f;
    std::printf("  longrope: amplitude %.4f, cos[7] %.4f\n", static_cast<double>(amplitude), static_cast<double>(c[7]));
    ok &= factors.data() == longs.data() && std::abs(amplitude - std::sqrt(1.0f + std::log(32.0f) / std::log(4096.0f))) < 1e-6f;
    float worst = 0.0f;
    for (std::size_t i = 0; i < key.size(); ++i)
        worst = std::fmax(worst, std::abs(key[i] - fresh[i]));
    std::printf("  K-shift 40 -> 32: largest difference from a fresh rotation %.2g\n", static_cast<double>(worst));
    ok &= worst < 1e-5f;

    std::printf(ok ? "rope_modes demo: ok\n" : "rope_modes demo: FAILED\n");
    return ok ? 0 : 1;
}
