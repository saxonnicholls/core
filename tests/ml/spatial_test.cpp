//
//  spatial_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/spatial.hpp>: pooling against ggml's loops (1-D excluding the padding, 2-D including
//  it), its backward by central differences and to the first maximum; upscaling against hand-worked values and
//  ggml's formulas, with corners aligned or not, and its adjoint; padding, reflect padding, roll and SAM's window
//  partition against their definitions and as adjoints; SAM's relative positions against the einsums; and
//  compile-time use.
//

#include <snicholls/core/ml/spatial.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr bool compile_time()
{
    // 2x2 max and average pools of a 4x4 ramp, a roll, a reflection and a nearest upscale, by the compiler
    std::array<double, 16> x{};
    for (std::size_t i = 0; i < 16; ++i)
        x[i] = double(i);
    constexpr auto s = ml::pool_2d_shape(1, 4, 4, 2, 2, 2, 2);
    std::array<double, s.y_size()> mx{}, av{};
    ml::pool_forward(s, ml::pool_kind::max, x, mx);
    ml::pool_forward(s, ml::pool_kind::avg, x, av);
    std::array<double, 4> r{};
    ml::roll<1>({4}, {1}, std::array<double, 4>{1, 2, 3, 4}, r);
    std::array<double, 7> p{};
    ml::pad_reflect_1d(1, 4, 2, 1, std::array<double, 4>{1, 2, 3, 4}, p);
    std::array<double, 4> up{};
    ml::upscale(ml::upscale_mode::nearest, false, std::array<double, 1>{7}, 1, 1, up, 2, 2);
    return mx == std::array<double, 4>{5, 7, 13, 15} && av == std::array<double, 4>{2.5, 4.5, 10.5, 12.5}
           && r == std::array<double, 4>{4, 1, 2, 3} && p == std::array<double, 7>{3, 2, 1, 2, 3, 4, 3} && up == std::array<double, 4>{7, 7, 7, 7};
}
static_assert(compile_time());
static_assert(ml::pool_1d_shape(2, 10, 3, 2, 1).output()[0] == 5 && !ml::pool_1d_shape(2, 10, 3, 2, 1).count_padding);
static_assert(ml::pool_2d_shape(1, 5, 5, 3, 3, 2, 2, 1, 1).count_padding && ml::window_count(64, 64, 14) == 25);

std::vector<double> noise(std::size_t n, std::uint64_t seed)
{
    std::vector<double> v(n);
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ull + 1;
    for (auto& e : v) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        e = static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0;
    }
    return v;
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

TEST(SpatialPool, MatchesGgmlsLoops)
{
    // ggml_compute_forward_pool_2d: windows from -p every s, taps outside skipped, the average divided by k0 k1
    const auto s = ml::pool_2d_shape(2, 5, 6, 3, 2, 2, 3, 1, 1);
    const auto x = noise(s.x_size(), 1);
    std::vector<double> mx(s.y_size()), av(s.y_size());
    ml::pool_forward(s, ml::pool_kind::max, x, mx);
    ml::pool_forward(s, ml::pool_kind::avg, x, av);
    const auto o = s.output();
    ASSERT_EQ(o, (std::array<std::size_t, 2>{3, 3}));
    for (std::size_t c = 0; c < 2; ++c)
        for (std::size_t oy = 0; oy < o[0]; ++oy)
            for (std::size_t ox = 0; ox < o[1]; ++ox) {
                double m = -DBL_MAX, a = 0;
                for (long ky = 0; ky < 3; ++ky)
                    for (long kx = 0; kx < 2; ++kx) {
                        const long iy = long(oy) * 2 - 1 + ky, ix = long(ox) * 3 - 1 + kx;
                        if (iy < 0 || iy >= 5 || ix < 0 || ix >= 6)
                            continue;
                        const double v = x[(c * 5 + std::size_t(iy)) * 6 + std::size_t(ix)];
                        m = std::max(v, m);
                        a += v;
                    }
                EXPECT_EQ(mx[(c * 3 + oy) * 3 + ox], m);
                EXPECT_EQ(av[(c * 3 + oy) * 3 + ox], a / 6);
            }
    // ggml_compute_forward_pool_1d_ksp: the average divided by the taps inside the row
    const auto s1 = ml::pool_1d_shape(1, 5, 3, 2, 1);
    const std::vector<double> r{1, 2, 3, 4, 5};
    std::vector<double> a1(s1.y_size()), m1(s1.y_size());
    ml::pool_forward(s1, ml::pool_kind::avg, r, a1);
    ml::pool_forward(s1, ml::pool_kind::max, r, m1);
    EXPECT_EQ(a1, (std::vector<double>{1.5, 3, 4.5}));
    EXPECT_EQ(m1, (std::vector<double>{2, 4, 5}));
}

TEST(SpatialPool, BackwardMatchesCentralDifferencesAndRoutesToTheFirstMaximum)
{
    for (const auto kind : {ml::pool_kind::max, ml::pool_kind::avg}) {
        const ml::pool_shape<2> s{.channels = 2, .input = {5, 4}, .kernel = {2, 3}, .stride = {1, 2}, .padding = {1, 1}};
        auto x = noise(s.x_size(), 2);
        const auto r = noise(s.y_size(), 3);
        std::vector<double> dx(s.x_size()), y(s.y_size());
        ml::pool_backward(s, kind, x, r, dx);
        const double h = 1e-6;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double keep = x[i];
            x[i] = keep + h;
            ml::pool_forward(s, kind, x, y);
            const double up = dot(y, r);
            x[i] = keep - h;
            ml::pool_forward(s, kind, x, y);
            const double down = dot(y, r);
            x[i] = keep;
            EXPECT_NEAR(dx[i], (up - down) / (2 * h), 1e-7);
        }
    }
    const auto s = ml::pool_1d_shape(1, 4, 4, 1);
    std::vector<double> dx(4);
    ml::pool_backward(s, ml::pool_kind::max, std::vector<double>{1, 3, 3, 2}, std::vector<double>{1}, dx);
    EXPECT_EQ(dx, (std::vector<double>{0, 1, 0, 0}));
    // A window entirely in the padding is ggml's lowest value and passes no gradient
    const ml::pool_shape<1> pad_only{.channels = 1, .input = {1}, .kernel = {1}, .stride = {1}, .padding = {1}};
    std::vector<double> y(3), g(1);
    ml::pool_forward(pad_only, ml::pool_kind::max, std::vector<double>{4}, y);
    EXPECT_EQ(y, (std::vector<double>{-DBL_MAX, 4, -DBL_MAX}));
    ml::pool_backward(pad_only, ml::pool_kind::max, std::vector<double>{4}, std::vector<double>{1, 1, 1}, g);
    EXPECT_EQ(g[0], 1);
}

TEST(SpatialUpscale, NearestAndBilinearAsGgmlComputesThem)
{
    const std::vector<float> x{1, 2, 3, 4};   // one 2x2 plane
    std::vector<float> y(16);
    ml::upscale(ml::upscale_mode::nearest, false, x, 2, 2, y, 4, 4);
    EXPECT_EQ(y, (std::vector<float>{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));
    // Half-pixel bilinear: output 0 is at -0.25, clamped to the first row; output 1 at 0.25
    ml::upscale(ml::upscale_mode::bilinear, false, x, 2, 2, y, 4, 4);
    EXPECT_EQ(y[0], 1.0f);
    EXPECT_EQ(y[1], 1.25f);
    EXPECT_EQ(y[5], 1.0f * 0.75f * 0.75f + 2.0f * 0.25f * 0.75f + 3.0f * 0.75f * 0.25f + 4.0f * 0.25f * 0.25f);
    EXPECT_EQ(y[15], 4.0f);
    // Corners aligned: 2 -> 3 puts the middle exactly between
    std::vector<float> z(9);
    ml::upscale(ml::upscale_mode::bilinear, true, x, 2, 2, z, 3, 3);
    EXPECT_EQ(z, (std::vector<float>{1, 1.5f, 2, 2, 2.5f, 3, 3, 3.5f, 4}));
}

TEST(SpatialUpscale, BicubicReproducesLinearRampsInsideAndKeysWeightsSumToOne)
{
    // Keys' kernel reproduces linear functions away from the clamped border
    std::vector<double> x(8 * 8), y(15 * 15);
    for (std::size_t r = 0; r < 8; ++r)
        for (std::size_t c = 0; c < 8; ++c)
            x[r * 8 + c] = double(r) * 2 + double(c);
    ml::upscale(ml::upscale_mode::bicubic, true, x, 8, 8, y, 15, 15);
    for (std::size_t r = 2; r < 13; ++r)
        for (std::size_t c = 2; c < 13; ++c)
            EXPECT_NEAR(y[r * 15 + c], double(r) / 2 * 2 + double(c) / 2, 1e-12);
    const std::vector<double> ones(9, 1.0);
    std::vector<double> o(5 * 7);
    ml::upscale(ml::upscale_mode::bicubic, false, ones, 3, 3, o, 5, 7);
    for (const double v : o)
        EXPECT_NEAR(v, 1.0, 1e-14);
}

TEST(SpatialUpscale, BackwardIsTheAdjoint)
{
    for (const auto mode : {ml::upscale_mode::nearest, ml::upscale_mode::bilinear, ml::upscale_mode::bicubic})
        for (const bool align : {false, true}) {
            const auto x = noise(2 * 3 * 5, 4), g = noise(2 * 7 * 4, 5);
            std::vector<double> y(g.size()), dx(x.size());
            ml::upscale(mode, align, x, 3, 5, y, 7, 4);
            ml::upscale_backward(mode, align, g, 7, 4, dx, 3, 5);
            EXPECT_NEAR(dot(y, g), dot(x, dx), 1e-12);
        }
}

TEST(SpatialPad, ZeroPadAndCropAreAdjoint)
{
    const std::vector<double> x{1, 2, 3, 4, 5, 6};   // 2x3
    std::vector<double> y(4 * 4);
    ml::pad<2>({2, 3}, {1, 0}, {1, 1}, x, y);
    EXPECT_EQ(y, (std::vector<double>{0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 6, 0, 0, 0, 0, 0}));
    const auto g = noise(16, 6);
    std::vector<double> dx(6);
    ml::pad_backward<2>({2, 3}, {1, 0}, {1, 1}, g, dx);
    EXPECT_NEAR(dot(y, g), dot(x, dx), 1e-15);
    EXPECT_EQ(dx[0], g[4]);
}

TEST(SpatialPad, ReflectMirrorsWithoutTheEdgeAndFoldsBack)
{
    const std::vector<double> x{1, 2, 3, 4, 10, 20, 30, 40};
    std::vector<double> y(2 * 7);
    ml::pad_reflect_1d(2, 4, 3, 0, x, y);
    EXPECT_EQ(y, (std::vector<double>{4, 3, 2, 1, 2, 3, 4, 40, 30, 20, 10, 20, 30, 40}));
    std::vector<double> z(2 * 6);
    ml::pad_reflect_1d(2, 4, 0, 2, x, z);
    EXPECT_EQ(z, (std::vector<double>{1, 2, 3, 4, 3, 2, 10, 20, 30, 40, 30, 20}));
    const auto u = noise(8, 7), g = noise(2 * 9, 8);
    std::vector<double> yu(18), dx(8);
    ml::pad_reflect_1d(2, 4, 3, 2, u, yu);
    ml::pad_reflect_1d_backward(2, 4, 3, 2, g, dx);
    EXPECT_NEAR(dot(yu, g), dot(u, dx), 1e-14);
}

TEST(SpatialRoll, WrapsEachDimensionAndRollsBack)
{
    const std::vector<int> idx{0, 1, 2, 3, 4, 5};
    std::vector<double> x(6), y(6), back(6);
    for (std::size_t i = 0; i < 6; ++i)
        x[i] = double(idx[i]);
    ml::roll<2>({2, 3}, {1, -1}, x, y);   // rows down by one, columns left by one
    EXPECT_EQ(y, (std::vector<double>{4, 5, 3, 1, 2, 0}));
    ml::roll<2>({2, 3}, {-1, 7}, y, back);   // 7 = 1 mod 3
    EXPECT_EQ(back, x);
}

TEST(SpatialWindow, PartitionPadsAndUnpartitionIsItsAdjoint)
{
    // ggml_compute_forward_win_part_f32 on a 5x3 grid of 2 channels, windows of 2
    const std::size_t h = 5, w = 3, c = 2, win = 2;
    const auto x = noise(h * w * c, 9);
    std::vector<double> parts(ml::window_count(h, w, win) * win * win * c);
    ASSERT_EQ(ml::window_count(h, w, win), 6u);
    ml::window_partition(h, w, c, win, x, parts);
    for (std::size_t py = 0; py < 3; ++py)
        for (std::size_t px = 0; px < 2; ++px)
            for (std::size_t i2 = 0; i2 < win; ++i2)
                for (std::size_t i1 = 0; i1 < win; ++i1)
                    for (std::size_t i0 = 0; i0 < c; ++i0) {
                        const std::size_t i = (((py * 2 + px) * win + i2) * win + i1) * c + i0;
                        const std::size_t gy = py * win + i2, gx = px * win + i1;
                        EXPECT_EQ(parts[i], gy >= h || gx >= w ? 0.0 : x[(gy * w + gx) * c + i0]);
                    }
    std::vector<double> back(x.size());
    ml::window_unpartition(h, w, c, win, parts, back);
    EXPECT_EQ(back, x);
    const auto g = noise(parts.size(), 10);
    std::vector<double> gx(x.size());
    ml::window_unpartition(h, w, c, win, g, gx);
    EXPECT_NEAR(dot(parts, g), dot(x, gx), 1e-14);
}

TEST(SpatialRelPos, GatherAndDecomposedScoresMatchSam)
{
    const std::size_t h = 3, w = 2, c = 4;
    const auto th = noise((2 * h - 1) * c, 11), tw = noise((2 * w - 1) * c, 12), q = noise(h * w * c, 13);
    std::vector<double> rh(h * h * c), rw(w * w * c);
    ml::rel_pos_gather(th, h, rh);
    ml::rel_pos_gather(tw, w, rw);
    EXPECT_EQ(rh[(0 * h + 2) * c + 1], th[(0 + h - 1 - 2) * c + 1]);   // q - k + size - 1
    EXPECT_EQ(rh[(2 * h + 0) * c + 3], th[(2 + h - 1) * c + 3]);
    std::vector<double> rel_h(h * w * h), rel_w(h * w * w);
    ml::rel_pos_scores(q, rh, rw, h, w, c, rel_h, rel_w);
    const std::size_t n = h * w;
    auto attn = noise(n * n, 14);
    const auto before = attn;
    ml::add_rel_pos(attn, rel_h, rel_w, h, w);
    for (std::size_t qh = 0; qh < h; ++qh)
        for (std::size_t qw = 0; qw < w; ++qw)
            for (std::size_t kh = 0; kh < h; ++kh)
                for (std::size_t kw = 0; kw < w; ++kw) {
                    double sh = 0, sw = 0;
                    for (std::size_t i = 0; i < c; ++i) {
                        sh += q[(qh * w + qw) * c + i] * rh[(qh * h + kh) * c + i];
                        sw += q[(qh * w + qw) * c + i] * rw[(qw * w + kw) * c + i];
                    }
                    const std::size_t qi = qh * w + qw, ki = kh * w + kw;
                    EXPECT_EQ(attn[qi * n + ki], (before[qi * n + ki] + sh) + sw);
                }
    const auto g = noise(n * n, 15);
    std::vector<double> dh(rel_h.size(), 0.0), dw(rel_w.size(), 0.0);
    ml::add_rel_pos_backward(g, h, w, dh, dw);
    std::vector<double> zero(n * n, 0.0);
    ml::add_rel_pos(zero, rel_h, rel_w, h, w);   // the terms alone: <terms, g> = <rel_h, dh> + <rel_w, dw>
    EXPECT_NEAR(dot(zero, g), dot(rel_h, dh) + dot(rel_w, dw), 1e-13);
}

} // namespace
