//
//  rope_modes_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/rope_modes.hpp>: ggml's mode names against core's layouts, the plain modes against
//  rope_angles to the bit, the sectioned modes (M-RoPE, interleaved M-RoPE, vision) against a transcription of
//  ggml_mrope_cache_init (llama.cpp 39039aa) in long double, vision mode against axial_rope_table to the bit, LongRoPE's
//  factors and attention factor, the K-shift against a fresh rotation, and compile-time tables.
//

#include <snicholls/core/ml/patch.hpp>
#include <snicholls/core/ml/rope_modes.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using ld = long double;

static_assert(int(ml::rope_mode::normal) == 0 && int(ml::rope_mode::neox) == 2 && int(ml::rope_mode::mrope) == 8
              && int(ml::rope_mode::vision) == 24 && int(ml::rope_mode::imrope) == 40);   // GGML_ROPE_TYPE_*
static_assert(ml::rope_mode_layout(ml::rope_mode::normal) == ml::rope_layout::interleaved
              && ml::rope_mode_layout(ml::rope_mode::neox) == ml::rope_layout::half
              && ml::rope_mode_layout(ml::rope_mode::imrope) == ml::rope_layout::half);
static_assert(ml::rope_mode_of(ml::rope_layout::interleaved) == ml::rope_mode::normal
              && ml::rope_mode_of(ml::rope_layout::half) == ml::rope_mode::neox);

// ggml_mrope_cache_init, transcribed: the running products in long double, the section choice as written there
std::vector<std::array<ld, 2>> ggml_mrope(const std::array<std::int64_t, 4>& p, const std::array<int, 4>& sections, bool imrope,
                                          bool vision, std::size_t pairs, ld base)
{
    const ld n_dims = vision ? ld(pairs) : ld(2 * pairs);
    const ld theta_scale = std::pow(base, -2.0L / n_dims);
    ld theta_t = ld(p[0]), theta_h = ld(p[1]), theta_w = ld(p[2]), theta_e = ld(p[3]);
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const int sec_w = sections[1] + sections[0], sec_e = sections[2] + sec_w;
    std::vector<std::array<ld, 2>> out(pairs);
    for (std::size_t i0 = 0; i0 < 2 * pairs; i0 += 2) {
        const int sector = int(i0 / 2) % sect_dims;
        if (vision) {
            if (sector == 0)
                theta_t = ld(p[0]);
            else if (sector == sections[0])
                theta_h = ld(p[1]);
            else if (sector == sec_w)
                theta_w = ld(p[2]);
            else if (sector == sec_e)
                theta_e = ld(p[3]);
        }
        ld theta = theta_t;
        if (imrope) {
            if (sector % 3 == 1 && sector < 3 * sections[1])
                theta = theta_h;
            else if (sector % 3 == 2 && sector < 3 * sections[2])
                theta = theta_w;
            else if (sector % 3 == 0 && sector < 3 * sections[0])
                theta = theta_t;
            else
                theta = theta_e;
        } else {
            if (sector >= sections[0] && sector < sec_w)
                theta = theta_h;
            else if (sector >= sec_w && sector < sec_w + sections[2])
                theta = theta_w;
            else if (sector >= sec_w + sections[2])
                theta = theta_e;
        }
        out[i0 / 2] = {std::cos(theta), std::sin(theta)};
        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
    return out;
}

void sectioned_is_ggml(ml::rope_mode mode, ml::rope_sections s, std::size_t pairs)
{
    const std::array<int, 4> sec{int(s.t), int(s.h), int(s.w), int(s.e)};
    for (const auto& p : {std::array<std::int64_t, 4>{0, 0, 0, 0}, {7, 3, 5, 0}, {120, 9, 31, 2}, {4095, 40, 17, 11}}) {
        std::vector<double> c(pairs), sn(pairs);
        ml::rope_mode_angles(ml::rope_position{p[0], p[1], p[2], p[3]}, c, sn, 10000.0, mode, s);
        const auto want = ggml_mrope(p, sec, mode == ml::rope_mode::imrope, mode == ml::rope_mode::vision, pairs, 10000.0L);
        for (std::size_t i = 0; i < pairs; ++i) {
            // the running product loses a few ulps a pair: at angle a, |a| * 1e-15 is the scale
            const double tol = 1e-12 * (1.0 + double(p[0] + p[1] + p[2] + p[3]));
            EXPECT_NEAR(c[i], double(want[i][0]), tol) << int(mode) << " " << i;
            EXPECT_NEAR(sn[i], double(want[i][1]), tol) << int(mode) << " " << i;
        }
    }
}

TEST(MlRopeModes, PlainModesAreRopeAnglesToTheBit)
{
    for (std::size_t pos : {0u, 1u, 17u, 4000u}) {
        std::array<float, 32> c{}, s{}, c2{}, s2{};
        ml::rope_angles(pos, c, s, 10000.0f);
        ml::rope_mode_angles(ml::rope_position{std::int64_t(pos)}, c2, s2, 10000.0f, ml::rope_mode::neox);
        EXPECT_EQ(c, c2);
        EXPECT_EQ(s, s2);
        ml::rope_mode_angles(ml::rope_position{std::int64_t(pos), 99, 99, 99}, c2, s2, 10000.0f, ml::rope_mode::normal);
        EXPECT_EQ(c, c2);   // only t is read
    }
}

TEST(MlRopeModes, SectionedModesAreGgmls)
{
    sectioned_is_ggml(ml::rope_mode::mrope, {16, 24, 24, 0}, 64);    // Qwen2-VL 7B: mrope_section [16, 24, 24], head 128
    sectioned_is_ggml(ml::rope_mode::imrope, {24, 20, 20, 0}, 64);   // Qwen3-VL: interleaved [24, 20, 20]
    sectioned_is_ggml(ml::rope_mode::mrope, {2, 3, 1, 2}, 11);       // a period shorter than the head, all four axes
    sectioned_is_ggml(ml::rope_mode::imrope, {3, 2, 2, 1}, 13);
    sectioned_is_ggml(ml::rope_mode::vision, {8, 8, 8, 8}, 16);      // ggml's vision encoders: d/4 each, head 32
    sectioned_is_ggml(ml::rope_mode::vision, {3, 5, 0, 0}, 8);
}

TEST(MlRopeModes, SectionAxesAreGgmlsChoice)
{
    const ml::rope_sections q2{16, 24, 24, 0};
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::mrope, q2, 15), 0u);
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::mrope, q2, 16), 1u);
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::mrope, q2, 40), 2u);
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::mrope, q2, 63), 2u);
    const ml::rope_sections q3{24, 20, 20, 0};
    // Interleaved: t h w t h w ... while each axis has pairs left, then t for the rest of t's share
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::imrope, q3, 0), 0u);
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::imrope, q3, 1), 1u);
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::imrope, q3, 2), 2u);
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::imrope, q3, 60), 0u);
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::imrope, q3, 61), 3u);   // 61 = 3 · 20 + 1: h is used up
    EXPECT_EQ(ml::rope_section_axis(ml::rope_mode::neox, q3, 61), 0u);
}

TEST(MlRopeModes, VisionModeIsAxialRopeToTheBit)
{
    constexpr std::size_t head = 32, rows = 3, columns = 4;
    const ml::axial_rope_table<double, head, 1, rows, columns> axial(10000.0);
    std::array<ml::rope_position, rows * columns> pos{};
    ml::vision_rope_positions(rows, columns, pos);
    const ml::rope_mode_table<double, rows * columns, head> table(std::span<const ml::rope_position, rows * columns>(pos), 10000.0,
                                                                  ml::rope_mode::vision, ml::rope_sections{head / 4, head / 4, head / 4, head / 4});
    for (std::size_t t = 0; t < rows * columns; ++t) {
        EXPECT_EQ(table.cos[t], axial.cos[t]) << t;
        EXPECT_EQ(table.sin[t], axial.sin[t]) << t;
    }
    static_assert(ml::rope_source<ml::rope_mode_table<double, 4, 8>>);
}

TEST(MlRopeModes, LongRopeFactorsDivideTheFrequenciesAndScaleTheAmplitude)
{
    std::array<double, 4> shorts{1.0, 1.0, 1.5, 2.0}, longs{1.0, 2.0, 8.0, 40.0}, c{}, s{};
    const auto f = ml::longrope_factors<double>(shorts, longs, 131072, 4096);
    EXPECT_EQ(f.data(), longs.data());
    EXPECT_EQ((ml::longrope_factors<double>(shorts, longs, 4096, 4096).data()), shorts.data());
    const double a = ml::longrope_attention_factor(32.0, 4096);
    EXPECT_NEAR(a, std::sqrt(1 + std::log(32.0) / std::log(4096.0)), 1e-15);   // Phi-3 128k: 1.1902...
    EXPECT_EQ(ml::longrope_attention_factor(1.0, 4096), 1.0);
    ml::rope_mode_angles(ml::rope_position{100}, c, s, 10000.0, ml::rope_mode::neox, {}, f, a);
    for (std::size_t i = 0; i < 4; ++i) {
        const double angle = 100.0 * (ml::rope_frequency<double>(i, 8, 10000.0) / longs[i]);
        EXPECT_EQ(c[i], a * std::cos(angle));
        EXPECT_EQ(s[i], a * std::sin(angle));
    }
}

TEST(MlRopeModes, AShiftedKeyIsTheKeyRotatedAtItsNewPosition)
{
    std::array<double, 16> key{};
    for (std::size_t i = 0; i < key.size(); ++i)
        key[i] = std::sin(0.3 * double(i) + 1.0);
    for (const auto layout : {ml::rope_layout::half, ml::rope_layout::interleaved})
        for (const auto& [from, delta] : {std::pair<std::size_t, std::int64_t>{10, 5}, {300, -120}, {7, 0}, {0, 4096}}) {
            std::array<double, 8> c{}, s{};
            auto moved = key, fresh = key;
            ml::rope_angles(from, c, s);
            ml::rope(moved, c, s, layout);
            ml::rope_shift(moved, delta, 10000.0, layout);
            ml::rope_angles(std::size_t(std::int64_t(from) + delta), c, s);
            ml::rope(fresh, c, s, layout);
            for (std::size_t i = 0; i < key.size(); ++i)
                EXPECT_NEAR(moved[i], fresh[i], 1e-11) << int(layout) << " " << from << " " << delta << " " << i;
        }
    // The shift is rope by rope_angles(delta), to the bit, and back by rope_inverse
    std::array<float, 8> k{1, 2, 3, 4, 5, 6, 7, 8}, by_rope = k, c{}, s{};
    std::array<float, 4> c4{}, s4{};
    ml::rope_angles(9, c4, s4, 500000.0f);
    ml::rope(by_rope, c4, s4);
    ml::rope_shift(k, 9, 500000.0f);
    EXPECT_EQ(k, by_rope);
    ml::rope_inverse(by_rope, c4, s4);
    ml::rope_shift(k, -9, 500000.0f);
    EXPECT_EQ(k, by_rope);
    (void)c;
    (void)s;
}

constexpr bool compile_time()
{
    constexpr std::array<ml::rope_position, 2> pos{ml::rope_position{0, 0, 0, 0}, ml::rope_position{3, 1, 2, 0}};
    const ml::rope_mode_table<double, 2, 8> t(std::span<const ml::rope_position, 2>(pos), 10000.0, ml::rope_mode::mrope,
                                              ml::rope_sections{2, 1, 1, 0});
    // Position 0 is the identity; at (3, 1, 2), pair 0 turns by 3, pair 2 by h = 1, pair 3 by w = 2
    return t.cos_at(0)[3] == 1.0 && t.sin_at(0)[1] == 0.0 && t.cos_at(1)[0] == snicholls::math::cos(3.0)
           && t.sin_at(1)[2] == snicholls::math::sin(1.0 * ml::rope_frequency<double>(2, 8, 10000.0))
           && t.sin_at(1)[3] == snicholls::math::sin(2.0 * ml::rope_frequency<double>(3, 8, 10000.0));
}
static_assert(compile_time());

} // namespace
