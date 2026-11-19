//
//  bitnet_kernels_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/bitnet_kernels.hpp>: I2_S bytes and ggml_vec_dot_i2_i8_s_1x1 equal bitnet.cpp's on
//  every row of the committed fixture (scripts/generate_bitnet_i2s_fixture.cpp, built from BitNet's C++); the TL1 and
//  TL2 tiled bytes equal BitNet's converter on real model tilings (scripts/generate_bitnet_lut_fixture.py); the
//  ARM I2_S layout; pack and unpack round trips; the LUT tables; the TL1 and TL2 GEMVs equal the plain ternary product
//  exactly; the kernels' activations; and compile-time use.
//

#include <snicholls/core/ml/bitnet_kernels.hpp>

#include "quant_rows.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr bool compile_time()
{
    // 128 weights through I2_S and both LUT forms: each gives the ternary dot product
    std::array<std::int8_t, 132> t{}, a{};
    for (std::size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<std::int8_t>(int(i * 5 % 3) - 1);
        a[i] = static_cast<std::int8_t>(int(i * 37 % 255) - 127);
    }
    const auto t128 = std::span<const std::int8_t>(t).first(128);
    const auto a128 = std::span<const std::int8_t>(a).first(128);
    std::array<std::uint8_t, ml::i2_s_bytes(128)> packed{};
    ml::pack_i2_s(t128, 0.25f, packed);
    const std::int32_t exact = ml::ternary_dot(t128, a128);
    std::array<std::uint8_t, 64> c1{};
    ml::tl1_encode(t128, c1);
    std::array<std::array<std::int16_t, 16>, 64> l1{};
    std::array<std::int32_t, 1> y1{};
    ml::tl1_gemv(c1, 1, 128, a128, l1, y1);
    std::array<ml::tl2_code_t, 44> c2{};
    ml::tl2_encode(t, c2);
    std::array<std::array<std::int16_t, 16>, 44> l2{};
    std::array<std::int32_t, 1> y2{};
    ml::tl2_gemv(c2, 1, 132, a, l2, y2);
    return ml::i2_s_dot(std::span<const std::uint8_t>(packed), a128) == exact && y1[0] == exact
           && y2[0] == ml::ternary_dot(t, a) && ml::i2_s_scale(packed, 128) == 0.25f && ml::i2_s_at(std::span<const std::uint8_t>(packed), 5) == t[5];
}
static_assert(compile_time());

constexpr std::size_t n = 128 * 37;

std::vector<float> ternary_row(int k)
{
    quant_rows::stream s(1000 + std::uint64_t(k));
    std::vector<float> r(n);
    for (auto& v : r) {
        const int t = k == 16 ? (s.next() >> 40) % 2 == 0 ? -1 : 1 : int((s.next() >> 40) % 3) - 1;
        v = float(t) * 0.37f;
    }
    return r;
}

std::vector<std::int8_t> activations(std::uint64_t seed, std::size_t count = n)
{
    quant_rows::stream s(seed);
    std::vector<std::int8_t> a(count);
    for (auto& v : a)
        v = static_cast<std::int8_t>(s.next() >> 56);
    return a;
}

// The LCG of scripts/generate_bitnet_lut_fixture.py
std::vector<std::int8_t> lcg(std::size_t m, std::size_t k, std::uint64_t seed)
{
    std::uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
    std::vector<std::int8_t> o(m * k);
    for (auto& v : o) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<std::int8_t>(int((s >> 33) % 3) - 1);
    }
    return o;
}

std::vector<std::string> lines(const char* file)
{
    std::vector<std::string> out;
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/" + file);
    EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/" << file;
    std::string text;
    while (std::getline(in, text))
        if (!text.empty() && text[0] != '#')
            out.push_back(text);
    return out;
}

TEST(MlBitnetKernels, I2sIsBitnetCppsOnEveryFixtureRow)
{
    const auto fixture = lines("bitnet_i2s_fixture.txt");
    ASSERT_EQ(fixture.size(), std::size_t(quant_rows::kinds + 2));
    for (const auto& text : fixture) {
        quant_rows::line l;
        ASSERT_TRUE(quant_rows::parse(text, l));
        const int k = l.kind;
        const auto x = k < quant_rows::kinds ? quant_rows::row(k, n, 4) : ternary_row(k - quant_rows::kinds + 15);
        const auto a = activations(7 + std::uint64_t(k));
        ASSERT_EQ(quant_rows::fnv_floats(x), l.values.at("x")) << k;
        ASSERT_EQ(quant_rows::fnv(a), l.values.at("a")) << k;
        std::vector<std::uint8_t> out(ml::i2_s_bytes(n));
        ml::quantize_i2_s(x, out);
        const std::int32_t d = ml::i2_s_dot_raw(std::span<const std::uint8_t>(out), a);
        EXPECT_EQ(std::uint32_t(d), l.values.at("dot")) << k;
        out.resize(n / 4 + 4);
        EXPECT_EQ(quant_rows::fnv(out), l.values.at("bytes")) << k;
    }
}

TEST(MlBitnetKernels, I2sRoundTripsAndItsDotIsTheTernaryDot)
{
    const auto x = ternary_row(15);
    std::vector<std::uint8_t> out(ml::i2_s_bytes(n));
    EXPECT_EQ(ml::quantize_i2_s(x, out), 0.37f);
    EXPECT_EQ(ml::i2_s_scale(out, n), 0.37f);
    std::vector<std::int8_t> t(n);
    for (std::size_t j = 0; j < n; ++j) {
        t[j] = static_cast<std::int8_t>(ml::i2_s_at(std::span<const std::uint8_t>(out), j));
        ASSERT_EQ(float(t[j]) * 0.37f, x[j]) << j;
    }
    std::vector<std::uint8_t> again(ml::i2_s_bytes(n));
    ml::pack_i2_s(t, 0.37f, again);
    EXPECT_EQ(again, out);
    const auto a = activations(3);
    EXPECT_EQ(ml::i2_s_dot(std::span<const std::uint8_t>(out), a), ml::ternary_dot(t, a));
}

TEST(MlBitnetKernels, TheArmLayoutInterleavesSixteenBytes)
{
    // On ARM a block is 64 weights in 16 bytes: weight j in byte j % 16, bits 6 - 2 (j / 16)
    std::vector<std::int8_t> t(64, 0);
    t[0] = 1;    // code 2 in byte 0's top bits
    t[17] = -1;  // code 0 in byte 1, bits 4-5
    std::vector<std::uint8_t> out(ml::i2_s_bytes(64));
    ml::pack_i2_s<ml::i2_s_block_arm>(t, 1.0f, out);
    EXPECT_EQ(out[0], 0b10010101);
    EXPECT_EQ(out[1], 0b01000101);
    EXPECT_EQ(out[2], 0b01010101);
    for (std::size_t j = 0; j < 64; ++j)
        EXPECT_EQ(ml::i2_s_at<ml::i2_s_block_arm>(std::span<const std::uint8_t>(out), j), t[j]) << j;
    const auto a = activations(5, 64);
    EXPECT_EQ(ml::i2_s_dot<ml::i2_s_block_arm>(std::span<const std::uint8_t>(out), a), ml::ternary_dot(t, a));
}

TEST(MlBitnetKernels, TlLayoutsAreTheConvertersOnModelTilings)
{
    const auto fixture = lines("bitnet_lut_fixture.txt");
    ASSERT_EQ(fixture.size(), 8u);
    for (const auto& text : fixture) {
        char kind[8] = {};
        std::size_t M = 0, K = 0, BM = 0, BK = 0, bm = 0;
        unsigned long long hin = 0, hout = 0;
        unsigned bytes = 0;
        ASSERT_EQ(std::sscanf(text.c_str(), "%7s %zu %zu %zu %zu %zu in=%llx out=%llx bytes=%x", kind, &M, &K, &BM, &BK, &bm, &hin, &hout, &bytes), 9);
        const auto t = lcg(M, K, M * 7 + K);
        ASSERT_EQ(quant_rows::fnv(t), hin) << text;
        std::vector<std::uint8_t> out;
        if (std::string(kind) == "tl1") {
            out.resize(ml::tl1_packed_bytes(M, K));
            ml::tl1_pack(t, M, K, {BM, BK, bm}, out);
        } else {
            out.resize(ml::tl2_packed_bytes(M, K));
            std::vector<std::uint8_t> scratch(M * (K % BK) / 2);
            ml::tl2_pack(t, M, K, {BM, BK, bm}, out, scratch);
        }
        EXPECT_EQ(out.size(), bytes) << text;
        EXPECT_EQ(quant_rows::fnv(out), hout) << text;
    }
}

TEST(MlBitnetKernels, TheTablesAreEveryCombination)
{
    for (int w0 = -1; w0 <= 1; ++w0)
        for (int w1 = -1; w1 <= 1; ++w1) {
            const auto lut = ml::tl1_lut(5, -7);
            EXPECT_EQ(lut[ml::tl1_code(w0, w1)], 5 * w0 - 7 * w1);
            for (int w2 = -1; w2 <= 1; ++w2) {
                const auto c = ml::tl2_code(w0, w1, w2);
                const int v = ml::tl2_lut(3, -11, 100)[c.magnitude];
                EXPECT_EQ(c.negative ? -v : v, 3 * w0 - 11 * w1 + 100 * w2);
            }
        }
    EXPECT_EQ(ml::tl1_lut(1, 1)[12], 0);
    EXPECT_EQ(ml::tl2_lut(1, 1, 1)[15], 0);
}

TEST(MlBitnetKernels, LutGemvsAreTheTernaryProduct)
{
    constexpr std::size_t rows = 7, cols = 96;
    const auto t = lcg(rows, cols, 99);
    const auto a = activations(11, cols);
    std::vector<std::uint8_t> c1(rows * cols / 2);
    std::vector<ml::tl2_code_t> c2(rows * cols / 3);
    ml::tl1_encode(t, c1);
    ml::tl2_encode(t, c2);
    std::vector<std::array<std::int16_t, 16>> l1(cols / 2), l2(cols / 3);
    std::vector<std::int32_t> y1(rows), y2(rows);
    ml::tl1_gemv(c1, rows, cols, a, l1, y1);
    ml::tl2_gemv(c2, rows, cols, a, l2, y2);
    for (std::size_t i = 0; i < rows; ++i) {
        const auto row = std::span<const std::int8_t>(t).subspan(i * cols, cols);
        EXPECT_EQ(y1[i], ml::ternary_dot(row, a)) << i;
        EXPECT_EQ(y2[i], ml::ternary_dot(row, a)) << i;
    }
}

TEST(MlBitnetKernels, TheKernelsActivationsHaveOneScaleAndNoFloor)
{
    const std::vector<float> x{0.5f, -2.0f, 1.0f, 0.0078125f};
    std::vector<std::int8_t> q(4);
    EXPECT_EQ(ml::lut_activation_quantize(x, q), 63.5f);
    EXPECT_EQ(q, (std::vector<std::int8_t>{32, -127, 64, 0}));   // 31.75 to 32; 0.496 to 0
    const std::vector<float> zero(3, 0.0f);
    std::vector<std::int8_t> z(3, 5);
    EXPECT_EQ(ml::lut_activation_quantize(zero, z), std::numeric_limits<float>::infinity());
    EXPECT_EQ(z[1], 0);
}

} // namespace
