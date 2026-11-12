//
//  quantize_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/quantize.hpp>: on every case of the committed fixture (written by ggml itself,
//  scripts/generate_quant_fixture.cpp), the Q8_0, Q4_0 and MXFP4 bytes equal ggml's quantize_row_*_ref and the
//  three dot products equal ggml's scalar ggml_vec_dot_*_generic bit for bit; dequantisation is d · q and lies
//  within half a step; the layouts; zero blocks; symmetric int8 rows; and compile-time use.
//

#include <snicholls/core/ml/quantize.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;

static_assert(std::is_trivially_copyable_v<ml::block_q8_0> && std::is_trivially_copyable_v<ml::block_q4_0>
              && std::is_trivially_copyable_v<ml::block_mxfp4>);
static_assert(alignof(ml::block_q8_0) == 2 && alignof(ml::block_q4_0) == 2 && alignof(ml::block_mxfp4) == 1);

constexpr bool compile_time()
{
    // 32 values k - 16: Q8_0 scale 16/127 and q = round(127 (k - 16) / 16); the dot of a row with itself
    std::array<float, 32> x{};
    for (std::size_t k = 0; k < 32; ++k)
        x[k] = float(k) - 16.0f;
    std::array<ml::block_q8_0, 1> q8{};
    std::array<ml::block_q4_0, 1> q4{};
    std::array<ml::block_mxfp4, 1> mx{};
    ml::quantize(x, std::span(q8));
    ml::quantize(x, std::span(q4));
    ml::quantize(x, std::span(mx));
    std::array<float, 32> back{};
    ml::dequantize(std::span<const ml::block_mxfp4>(mx), back);
    const float self = ml::dot(std::span<const ml::block_q8_0>(q8), std::span<const ml::block_q8_0>(q8));
    return q8[0].qs[0] == -127 && q8[0].qs[16] == 0 && (q4[0].qs[0] & 0x0f) == 0 && mx[0].e == 129 && back[0] == -16.0f
           && self > 2700.0f && self < 2780.0f;   // sum (k - 16)^2 = 2736, up to rounding
}
static_assert(compile_time());

std::vector<std::uint8_t> hex_bytes(const std::string& s)
{
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i + 1 < s.size(); i += 2)
        out.push_back(static_cast<std::uint8_t>(std::stoul(s.substr(i, 2), nullptr, 16)));
    return out;
}

struct quant_case {
    std::vector<float> x, y;
    std::vector<std::uint8_t> q8, q4, mx, yq8;
    std::array<std::uint32_t, 3> dot{};
};

std::vector<quant_case> load()
{
    std::vector<quant_case> cases;
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/quant_fixture.txt");
    EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/quant_fixture.txt";
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kind, word;
        ls >> kind;
        if (kind == "case") {
            cases.emplace_back();
            continue;
        }
        if (cases.empty())
            continue;
        auto& c = cases.back();
        if (kind == "x" || kind == "y") {
            auto& r = kind == "x" ? c.x : c.y;
            while (ls >> word)
                r.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(std::stoul(word, nullptr, 16))));
        } else if (kind == "dot") {
            for (auto& d : c.dot) {
                ls >> word;
                d = static_cast<std::uint32_t>(std::stoul(word, nullptr, 16));
            }
        } else {
            ls >> word;
            (kind == "q8_0" ? c.q8 : kind == "q4_0" ? c.q4 : kind == "mxfp4" ? c.mx : c.yq8) = hex_bytes(word);
        }
    }
    return cases;
}

const std::vector<quant_case>& data()
{
    static const auto cases = load();
    return cases;
}

template <typename Block>
std::vector<std::uint8_t> raw(const std::vector<Block>& b)
{
    std::vector<std::uint8_t> out(b.size() * sizeof(Block));
    std::memcpy(out.data(), b.data(), out.size());
    return out;
}

template <typename Block>
std::vector<Block> quantized(const std::vector<float>& x)
{
    std::vector<Block> b(x.size() / ml::quant_block_size);
    ml::quantize(x, std::span<Block>(b));
    return b;
}

TEST(MlQuantize, BytesAreGgmlsOnEveryFixtureCase)
{
    ASSERT_EQ(data().size(), 32u);
    for (std::size_t i = 0; i < data().size(); ++i) {
        const auto& c = data()[i];
        EXPECT_EQ(raw(quantized<ml::block_q8_0>(c.x)), c.q8) << "case " << i;
        EXPECT_EQ(raw(quantized<ml::block_q4_0>(c.x)), c.q4) << "case " << i;
        EXPECT_EQ(raw(quantized<ml::block_mxfp4>(c.x)), c.mx) << "case " << i;
        EXPECT_EQ(raw(quantized<ml::block_q8_0>(c.y)), c.yq8) << "case " << i;
    }
}

// The bits of a dot product, except that every NaN is one value: the sign of a NaN the hardware makes (inf - inf)
// is its own, negative on x86 and positive on ARM, and the fixture was generated on x86
std::uint32_t dot_bits(float d)
{
    return d != d ? 0x7fc00000u : std::bit_cast<std::uint32_t>(d);
}
std::uint32_t fixture_bits(std::uint32_t bits)
{
    return dot_bits(std::bit_cast<float>(bits));
}

TEST(MlQuantize, DotProductsAreGgmlsBitForBit)
{
    for (std::size_t i = 0; i < data().size(); ++i) {
        const auto& c = data()[i];
        const auto y = quantized<ml::block_q8_0>(c.y);
        const std::span<const ml::block_q8_0> ys(y);
        const auto x8 = quantized<ml::block_q8_0>(c.x);
        const auto x4 = quantized<ml::block_q4_0>(c.x);
        const auto xm = quantized<ml::block_mxfp4>(c.x);
        EXPECT_EQ(dot_bits(ml::dot(std::span<const ml::block_q8_0>(x8), ys)), fixture_bits(c.dot[0])) << "case " << i;
        EXPECT_EQ(dot_bits(ml::dot(std::span<const ml::block_q4_0>(x4), ys)), fixture_bits(c.dot[1])) << "case " << i;
        EXPECT_EQ(dot_bits(ml::dot(std::span<const ml::block_mxfp4>(xm), ys)), fixture_bits(c.dot[2])) << "case " << i;
    }
}

TEST(MlQuantize, ANanDotProductMatchesTheFixturesNanWhateverItsSign)
{
    EXPECT_EQ(dot_bits(std::bit_cast<float>(0xffc00000u)), dot_bits(std::bit_cast<float>(0x7fc00000u)));
    EXPECT_NE(dot_bits(1.0f), dot_bits(-1.0f));
    EXPECT_NE(dot_bits(0.0f), dot_bits(-0.0f));
}

TEST(MlQuantize, DequantisationIsWithinHalfAStep)
{
    for (const auto& c : data()) {
        const auto b = quantized<ml::block_q8_0>(c.x);
        std::vector<float> back(c.x.size());
        ml::dequantize(std::span<const ml::block_q8_0>(b), back);
        for (std::size_t i = 0; i < c.x.size(); ++i) {
            const float d = ml::to_float(b[i / 32].d);
            EXPECT_EQ(back[i], float(b[i / 32].qs[i % 32]) * d);
            if (d >= 6.1035156e-5f) {   // a normal float16 scale: subnormal ones lose relative precision
                EXPECT_LE(std::fabs(back[i] - c.x[i]), 0.5f * d * 1.01f + 1e-3f * std::fabs(c.x[i]));
            }
        }
    }
}

TEST(MlQuantize, AZeroBlockQuantisesToZero)
{
    const std::vector<float> zero(32, 0.0f);
    const auto q8 = quantized<ml::block_q8_0>(zero);
    const auto q4 = quantized<ml::block_q4_0>(zero);
    const auto mx = quantized<ml::block_mxfp4>(zero);
    EXPECT_EQ(q8[0].d.bits, 0u);
    EXPECT_EQ(q8[0].qs[3], 0);
    EXPECT_EQ(q4[0].qs[3], 0x88);   // 8 is zero in Q4_0
    EXPECT_EQ(mx[0].e, 0);
    std::vector<float> back(32, 1.0f);
    ml::dequantize(std::span<const ml::block_q4_0>(q4), back);
    EXPECT_EQ(back[31], 0.0f);
}

TEST(MlQuantize, Int8RowsUseOneScale)
{
    const std::vector<float> x{0.5f, -1.27f, 0.0f, 1.0f, -0.01f};
    std::vector<std::int8_t> q(x.size());
    const float scale = ml::quantize_int8(x, q);
    EXPECT_FLOAT_EQ(scale, 1.27f / 127.0f);
    EXPECT_EQ(q[1], -127);
    EXPECT_EQ(q[3], 100);
    EXPECT_EQ(q[4], -1);
    std::vector<float> back(x.size());
    ml::dequantize_int8(q, scale, back);
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(back[i], x[i], scale / 2 + 1e-7f);
    // roundf's rule: the largest float below one half rounds to 0, not up to 1
    std::vector<std::int8_t> edge(2);
    EXPECT_EQ(ml::quantize_int8(std::vector<float>{127.0f, 0.49999997f}, edge), 1.0f);
    EXPECT_EQ(edge[1], 0);
    std::vector<std::int8_t> zq(3);
    EXPECT_EQ(ml::quantize_int8(std::vector<float>(3, 0.0f), zq), 0.0f);
    EXPECT_EQ(zq[0], 0);
}

} // namespace
