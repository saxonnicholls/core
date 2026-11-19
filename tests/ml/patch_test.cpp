//
//  patch_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/patch.hpp>: patchify_rows against a loop by hand to the bit for images, video tubelets
//  and rectangular spectrogram patches over a batch; unpatchify_rows its inverse and its backward (the adjoint
//  identity, central differences); patch_source and valid at compile time; the 2-D and 3-D sine-cosine embeddings
//  against the formula in double, their layouts, uniform_power's cut, the constexpr tables; axial RoPE's split of the
//  pairs, its relative-position property per axis; gathered_rope and its sequences.
//

#include <snicholls/core/ml/patch.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/rope.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
const auto& backend = snicholls::linalg::reference;

constexpr ml::patch_grid image{.height = 4, .width = 6, .channels = 3, .patch_height = 2, .patch_width = 2};
constexpr ml::patch_grid video{.frames = 4, .height = 4, .width = 4, .channels = 2, .tubelet = 2, .patch_height = 2, .patch_width = 2};
constexpr ml::patch_grid spectrogram{.height = 8, .width = 12, .channels = 1, .patch_height = 4, .patch_width = 3};

static_assert(ml::valid(image) && ml::valid(video) && ml::valid(spectrogram));
static_assert(!ml::valid(ml::patch_grid{.height = 5, .width = 6, .patch_height = 2, .patch_width = 2}));
static_assert(image.tokens() == 6 && image.token_size() == 12 && video.tokens() == 8 && video.token_size() == 16
              && spectrogram.tokens() == 8 && spectrogram.token_size() == 12);
// Token 4 of the image is patch row 1, column 1; element 5 is (dy 0, dx 1, c 2): pixel (2, 3), channel 2
static_assert(ml::patch_source(image, 4, 5) == (2 * 6 + 3) * 3 + 2);
static_assert(std::is_trivially_copyable_v<ml::patch_grid>);
static_assert(noexcept(ml::patchify_rows(backend, image, std::span<const float>{}, std::span<float>{})));

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.31 * double(i) + b);
    return v;
}

// The patch rows written out: sample, token (gt, gy, gx), then (dt, dy, dx, c)
std::vector<double> by_hand(const ml::patch_grid& g, const std::vector<double>& v, std::size_t samples)
{
    std::vector<double> out;
    for (std::size_t s = 0; s < samples; ++s)
        for (std::size_t gt = 0; gt < g.grid_frames(); ++gt)
            for (std::size_t gy = 0; gy < g.grid_height(); ++gy)
                for (std::size_t gx = 0; gx < g.grid_width(); ++gx)
                    for (std::size_t dt = 0; dt < g.tubelet; ++dt)
                        for (std::size_t dy = 0; dy < g.patch_height; ++dy)
                            for (std::size_t dx = 0; dx < g.patch_width; ++dx)
                                for (std::size_t c = 0; c < g.channels; ++c) {
                                    const std::size_t f = gt * g.tubelet + dt, y = gy * g.patch_height + dy, x = gx * g.patch_width + dx;
                                    out.push_back(v[s * g.values() + ((f * g.height + y) * g.width + x) * g.channels + c]);
                                }
    return out;
}

TEST(MlPatchify, IsALoopByHandForImagesTubeletsAndSpectrograms)
{
    for (const auto& g : {image, video, spectrogram}) {
        const auto v = wave(3 * g.values(), 1.0, 0.4);
        std::vector<double> tokens(v.size());
        ml::patchify_rows(backend, g, std::span<const double>(v), std::span<double>(tokens), 3);
        EXPECT_EQ(tokens, by_hand(g, v, 3));
        std::vector<double> back(v.size());
        ml::unpatchify_rows(backend, g, std::span<const double>(tokens), std::span<double>(back), 3);
        EXPECT_EQ(back, v);
    }
}

TEST(MlPatchify, UnpatchifyIsItsBackward)
{
    auto v = wave(video.values(), 1.0, 0.4);
    const auto y = wave(video.values(), 0.6, 2.0);
    std::vector<double> grad(v.size());
    ml::unpatchify_rows(backend, video, std::span<const double>(y), std::span<double>(grad));
    auto loss = [&] {
        std::vector<double> t(v.size());
        ml::patchify_rows(backend, video, std::span<const double>(v), std::span<double>(t));
        double l = 0;
        for (std::size_t i = 0; i < t.size(); ++i)
            l += t[i] * y[i];
        return l;
    };
    for (std::size_t i = 0; i < v.size(); ++i) {
        const double keep = v[i], e = 1e-6;
        v[i] = keep + e;
        const double hi = loss();
        v[i] = keep - e;
        const double lo = loss();
        v[i] = keep;
        EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 1e-8) << i;
    }
}

// One axis of the formula: d channels of position p, [sin(p w_i)..., cos(p w_i)...], w_i = 10000^(-i / (d/2))
double axis(std::size_t d, std::size_t p, std::size_t i)
{
    const std::size_t h = d / 2;
    const double w = 1.0 / std::pow(10000.0, double(i % h) / double(h));
    return i < h ? std::sin(double(p) * w) : std::cos(double(p) * w);
}

TEST(MlSincosPosition, TwoDimensionsIsTheFormulaInEitherLayout)
{
    constexpr std::size_t dim = 8, gh = 3, gw = 5;
    std::vector<double> rows(gh * gw * dim), cols(rows.size());
    ml::sincos_position_2d<double>(dim, gh, gw, rows);
    ml::sincos_position_2d<double>(dim, gh, gw, cols, ml::sincos_layout::column_first);
    for (std::size_t y = 0; y < gh; ++y)
        for (std::size_t x = 0; x < gw; ++x)
            for (std::size_t i = 0; i < dim / 2; ++i) {
                const std::size_t t = y * gw + x;
                EXPECT_NEAR(rows[t * dim + i], axis(dim / 2, y, i), 1e-15);
                EXPECT_NEAR(rows[t * dim + dim / 2 + i], axis(dim / 2, x, i), 1e-15);
                EXPECT_EQ(cols[t * dim + i], rows[t * dim + dim / 2 + i]);
                EXPECT_EQ(cols[t * dim + dim / 2 + i], rows[t * dim + i]);
            }
    // Position (0, 0): sines 0, cosines 1
    EXPECT_EQ(rows[0], 0.0);
    EXPECT_EQ(rows[2], 1.0);
}

TEST(MlSincosPosition, ThreeDimensionsSplitsTheChannelsAndUniformPowerCutsTheLast)
{
    constexpr std::size_t gt = 2, gh = 2, gw = 3;
    {
        constexpr std::size_t dim = 16;   // frame 8, row 4, column 4
        std::vector<double> e(gt * gh * gw * dim);
        ml::sincos_position_3d<double>(dim, gt, gh, gw, e);
        for (std::size_t t = 0; t < gt; ++t)
            for (std::size_t y = 0; y < gh; ++y)
                for (std::size_t x = 0; x < gw; ++x) {
                    const double* r = &e[((t * gh + y) * gw + x) * dim];
                    for (std::size_t i = 0; i < 8; ++i)
                        EXPECT_NEAR(r[i], axis(8, t, i), 1e-15);
                    for (std::size_t i = 0; i < 4; ++i) {
                        EXPECT_NEAR(r[8 + i], axis(4, y, i), 1e-15);
                        EXPECT_NEAR(r[12 + i], axis(4, x, i), 1e-15);
                    }
                }
    }
    {
        constexpr std::size_t dim = 10;   // 2 ceil(10 / 6) = 4 each: frame 4, row 4, column 2 of its 4
        std::vector<double> e(gt * gh * gw * dim);
        ml::sincos_position_3d<double>(dim, gt, gh, gw, e, true);
        const double* r = &e[((1 * gh + 1) * gw + 2) * dim];
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(r[i], axis(4, 1, i), 1e-15);
            EXPECT_NEAR(r[4 + i], axis(4, 1, i), 1e-15);
        }
        EXPECT_NEAR(r[8], axis(4, 2, 0), 1e-15);
        EXPECT_NEAR(r[9], axis(4, 2, 1), 1e-15);
    }
}

constexpr ml::sincos_table_2d<double, 8, 2, 3> table2{};
constexpr ml::sincos_table_3d<float, 16, 2, 2, 2> table3{};
static_assert(table2.row(0)[0] == 0.0 && table2.row(0)[2] == 1.0 && table2.row(0)[6] == 1.0);

TEST(MlSincosPosition, TheCompileTimeTablesAreTheFunctionsWithinTheLastBits)
{
    std::vector<double> e2(2 * 3 * 8);
    ml::sincos_position_2d<double>(8, 2, 3, e2);
    for (std::size_t i = 0; i < e2.size(); ++i)
        EXPECT_NEAR(table2.values[i], e2[i], 4e-16);
    std::vector<float> e3(2 * 2 * 2 * 16);
    ml::sincos_position_3d<float>(16, 2, 2, 2, e3);
    for (std::size_t i = 0; i < e3.size(); ++i)
        EXPECT_NEAR(table3.values[i], e3[i], 1e-7f);
}

constexpr ml::axial_rope_table<double, 12, 1, 3, 4> rope2{};   // two axes, 3 pairs each
constexpr ml::axial_rope_table<double, 14, 2, 3, 4> rope3{};   // three axes, 2 pairs each, the seventh unrotated
static_assert(ml::rope_source<ml::axial_rope_table<double, 12, 1, 3, 4>>);
static_assert(rope2.axes == 2 && rope2.share == 3 && rope3.axes == 3 && rope3.share == 2);

TEST(MlAxialRope, EachAxisRotatesItsShareOfPairsByItsCoordinate)
{
    // Token (y 2, x 1) = 9: pairs 0-2 by the row, 3-5 by the column, each frequency of a 6-channel head
    for (std::size_t p = 0; p < 6; ++p) {
        const double angle = double(p < 3 ? 2 : 1) * ml::rope_frequency<double>(p % 3, 6);
        EXPECT_NEAR(rope2.cos_at(9)[p], std::cos(angle), 1e-15);
        EXPECT_NEAR(rope2.sin_at(9)[p], std::sin(angle), 1e-15);
    }
    // Token (t 1, y 2, x 3) = 23: frame, row, column, then one unrotated pair
    const std::size_t coords[3] = {1, 2, 3};
    for (std::size_t p = 0; p < 6; ++p)
        EXPECT_NEAR(rope3.cos_at(23)[p], std::cos(double(coords[p / 2]) * ml::rope_frequency<double>(p % 2, 4)), 1e-15);
    EXPECT_EQ(rope3.cos_at(23)[6], 1.0);
    EXPECT_EQ(rope3.sin_at(23)[6], 0.0);
}

TEST(MlAxialRope, AQueryKeyProductDependsOnlyOnTheirOffset)
{
    std::array<double, 12> q{}, k{};
    for (std::size_t i = 0; i < 12; ++i) {
        q[i] = std::sin(0.7 * double(i) + 0.1);
        k[i] = std::cos(0.3 * double(i) + 0.5);
    }
    auto product = [&](std::size_t a, std::size_t b) {
        auto qa = q, kb = k;
        ml::rope(qa, rope2.cos_at(a), rope2.sin_at(a));
        ml::rope(kb, rope2.cos_at(b), rope2.sin_at(b));
        double s = 0;
        for (std::size_t i = 0; i < 12; ++i)
            s += qa[i] * kb[i];
        return s;
    };
    // (0, 0) with (1, 2), and (1, 1) with (2, 3): the same offset (1, 1); (0, 0) with (1, 1) differs
    EXPECT_NEAR(product(0 * 4 + 0, 1 * 4 + 2), product(1 * 4 + 1, 2 * 4 + 3), 1e-12);
    EXPECT_GT(std::abs(product(0, 1 * 4 + 2) - product(0, 1 * 4 + 1)), 1e-6);
}

TEST(MlGatheredRope, ReadsTheSourceAtEachPositionAndSplitsIntoSequences)
{
    static const ml::rope_table<double, 16, 4> table{};
    const std::vector<std::uint32_t> positions{3, 7, 1, 0, 15, 2};
    const ml::gathered_rope<ml::rope_table<double, 16, 4>> g{&table, positions, 3};
    static_assert(ml::rope_source<ml::gathered_rope<ml::rope_table<double, 16, 4>>>);
    EXPECT_EQ(g.cos_at(1)[1], table.cos_at(7)[1]);
    const auto second = g.sequence(1);
    EXPECT_EQ(second.positions.size(), 3u);
    EXPECT_EQ(second.sin_at(1)[0], table.sin_at(15)[0]);
    const ml::gathered_rope<ml::rope_table<double, 16, 4>> one{&table, positions};   // one sequence of all six
    EXPECT_EQ(one.sequence(0).positions.size(), 6u);
}

} // namespace
