//
//  quantize_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/quantize.hpp>: on every case of the committed fixture (written by ggml itself,
//  scripts/generate_quant_fixture.cpp), the Q8_0, Q4_0 and MXFP4 bytes equal ggml's quantize_row_*_ref and the
//  three dot products equal ggml's scalar ggml_vec_dot_*_generic bit for bit; dequantisation is d · q and lies
//  within half a step; the layouts; zero blocks; symmetric int8 rows; and compile-time use. On every row kind of
//  the second fixture (scripts/generate_quant_formats_fixture.cpp), Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, NVFP4,
//  Q1_0, Q2_0 and F8_E4M3 bytes (reference and importance-weighted), dequantised floats, activations and dot
//  products equal ggml's; the type table and block traits; the importance path's weighted error.
//

#include <snicholls/core/ml/quantize.hpp>

#include "quant_rows.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
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


// ggml's type table, as ggml.c's type_traits has it: names, block sizes and bytes
static_assert(ml::ggml_type_traits(ml::ggml_type::q4_k).name == "q4_K" && ml::ggml_type_traits(ml::ggml_type::q4_k).block_bytes == 144);
static_assert(ml::ggml_type_traits(ml::ggml_type::iq2_xxs).block_elements == 256 && ml::ggml_type_traits(ml::ggml_type::bf16).block_bytes == 2);
static_assert(ml::ggml_type_traits(ml::ggml_type::nvfp4).block_elements == 64 && !ml::ggml_type_traits(ml::ggml_type::f32).quantized);
static_assert(ml::ggml_type_traits(static_cast<ml::ggml_type>(4)).name.empty() && ml::ggml_type_traits(static_cast<ml::ggml_type>(99)).block_bytes == 0);
static_assert(ml::ggml_row_bytes(ml::ggml_type::q4_0, 4096) == 128 * 18 && ml::ggml_row_bytes(ml::ggml_type::f16, 10) == 20);
static_assert(ml::ggml_row_bytes(static_cast<ml::ggml_type>(31), 64) == 0);
static_assert(sizeof(ml::block_q4_1) == 20 && sizeof(ml::block_q5_0) == 22 && sizeof(ml::block_q5_1) == 24 && sizeof(ml::block_q8_1) == 36);
static_assert(sizeof(ml::block_nvfp4) == 36 && sizeof(ml::block_q1_0) == 18 && sizeof(ml::block_q2_0) == 18 && sizeof(ml::block_q8_k) == 292);
static_assert(ml::block_traits<ml::block_q4_1>::bits_per_weight == 5.0 && ml::block_traits<ml::block_q1_0>::bits_per_weight == 1.125);
static_assert(std::same_as<ml::block_traits<ml::block_q5_1>::dot_block, ml::block_q8_1> && ml::block_traits<ml::block_q8_k>::name == "q8_K");
static_assert(ml::quant_block<ml::block_q5_0> && ml::quant_block<ml::block_mxfp4> && !ml::quant_block<ml::block_q4_1> && !ml::quant_block<ml::block_nvfp4>);
static_assert(!ml::quantized_block<float> && !ml::quantized_block<ml::float8_e4m3>);

// The helpers the scale searches share are ggml's to the bit, and correct at compile time as at run time
static_assert(ml::detail::quantize::nearest_int(2.5f) == 2 && ml::detail::quantize::nearest_int(3.5f) == 4 && ml::detail::quantize::nearest_int(-2.5f) == -2);
static_assert(ml::detail::quantize::to_int(3e9f) == std::numeric_limits<int>::min() && ml::detail::quantize::to_int(-7.9f) == -7);
static_assert(ml::detail::quantize::round_to_int(-2.5f) == -3 && ml::detail::quantize::round_to_int(1e10f) == std::numeric_limits<int>::min());
static_assert(ml::detail::quantize::fp32_to_ue4m3(1.0f) == 0x38 && ml::detail::quantize::ue4m3_half(0x38) == 0.5f);
static_assert(ml::detail::quantize::fp32_to_ue4m3(1000.0f) == 0x7e && ml::detail::quantize::fp32_to_ue4m3(-1.0f) == 0);

TEST(MlQuantize, CompileTimeSqrtIsRunTimeSqrt)
{
    std::uint32_t bits = 0x00000001u;
    for (int i = 0; i < 200000; ++i, bits = bits * 2654435761u + 12345u) {
        const float x = std::bit_cast<float>(bits & 0x7fffffffu);
        if (!std::isfinite(x))
            continue;
        EXPECT_EQ(std::bit_cast<std::uint32_t>(static_cast<float>(snicholls::math::detail::sqrt_ct(static_cast<double>(x)))), std::bit_cast<std::uint32_t>(std::sqrt(x))) << x;   // the compile-time path
    }
}

// The fixture of the formats new in 0.17.0: hashes of ggml's outputs on the rows of tests/ml/quant_rows.hpp
std::vector<quant_rows::line> formats_fixture(const std::string& format)
{
    std::vector<quant_rows::line> out;
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/quant_formats_fixture.txt");
    EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/quant_formats_fixture.txt";
    std::string text;
    quant_rows::line l;
    while (std::getline(in, text))
        if (quant_rows::parse(text, l) && l.format == format)
            out.push_back(l);
    return out;
}

template <typename B>
std::vector<B> quantized_with(const std::vector<float>& x, const std::vector<float>& importance)
{
    std::vector<B> b(x.size() / B::elements);
    ml::quantize(x, importance, std::span<B>(b));
    return b;
}

template <typename B>
void matches_ggml(const char* format)
{
    using A = typename B::dot_block;
    const auto lines = formats_fixture(format);
    ASSERT_EQ(lines.size(), std::size_t(quant_rows::kinds)) << format;
    for (const auto& l : lines) {
        const auto x = quant_rows::row(l.kind, 512, 1);
        const auto y = quant_rows::row((l.kind + 1) % 15, 512, 2);
        const auto w = quant_rows::importance(512, 3);
        ASSERT_EQ(quant_rows::fnv_floats(x), l.values.at("x")) << "the rows of tests/ml/quant_rows.hpp changed";
        const auto ref = quantized_with<B>(x, {});
        const auto imat = quantized_with<B>(x, w);
        const auto act = quantized_with<A>(y, {});
        std::vector<float> deq(512);
        ml::dequantize(std::span<const B>(ref), deq);
        const float d = ml::dot(std::span<const B>(ref), std::span<const A>(act));
        const char* kind = quant_rows::name(l.kind);
        EXPECT_EQ(quant_rows::fnv(ref), l.values.at("ref")) << format << " " << kind;
        EXPECT_EQ(quant_rows::fnv(imat), l.values.at("imat")) << format << " " << kind;
        EXPECT_EQ(quant_rows::fnv_floats(deq), l.values.at("deq")) << format << " " << kind;
        EXPECT_EQ(quant_rows::fnv(act), l.values.at("act")) << format << " " << kind;
        EXPECT_EQ(quant_rows::float_bits(d), l.values.at("dot")) << format << " " << kind;
    }
}

TEST(MlQuantizeFormats, Q4_0IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q4_0>("q4_0"); }
TEST(MlQuantizeFormats, Q4_1IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q4_1>("q4_1"); }
TEST(MlQuantizeFormats, Q5_0IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q5_0>("q5_0"); }
TEST(MlQuantizeFormats, Q5_1IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q5_1>("q5_1"); }
TEST(MlQuantizeFormats, Q8_0IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q8_0>("q8_0"); }
TEST(MlQuantizeFormats, Mxfp4IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_mxfp4>("mxfp4"); }
TEST(MlQuantizeFormats, Nvfp4IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_nvfp4>("nvfp4"); }
TEST(MlQuantizeFormats, Q1_0IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q1_0>("q1_0"); }
TEST(MlQuantizeFormats, Q2_0IsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q2_0>("q2_0"); }

TEST(MlQuantizeFormats, F8E4m3RowsAreGgmlsOnEveryRowKind)
{
    const auto lines = formats_fixture("f8_e4m3");
    ASSERT_EQ(lines.size(), std::size_t(quant_rows::kinds));
    for (const auto& l : lines) {
        const auto x = quant_rows::row(l.kind, 512, 1);
        const auto y = quant_rows::row((l.kind + 1) % 15, 512, 2);
        std::vector<ml::float8_e4m3> q(512);
        ml::quantize_row(x, q);
        std::vector<float> deq(512);
        ml::dequantize_row(q, deq);
        const char* kind = quant_rows::name(l.kind);
        EXPECT_EQ(quant_rows::fnv(q), l.values.at("ref")) << kind;
        EXPECT_EQ(quant_rows::fnv_floats(deq), l.values.at("deq")) << kind;
        EXPECT_EQ(quant_rows::float_bits(ml::dot(std::span<const ml::float8_e4m3>(q), std::span<const float>(y))), l.values.at("dot")) << kind;
    }
}

TEST(MlQuantizeFormats, TheImportanceRowLowersTheWeightedError)
{
    const auto x = quant_rows::row(quant_rows::weights, 4096, 11);
    std::vector<float> w(x.size());
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = i % 7 == 0 ? 50.0f : 0.1f;
    const auto error = [&](auto block, bool weighted) {
        using B = decltype(block);
        const auto q = quantized_with<B>(x, weighted ? w : std::vector<float>{});
        std::vector<float> back(x.size());
        ml::dequantize(std::span<const B>(q), back);
        double e = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            e += double(w[i]) * double(back[i] - x[i]) * double(back[i] - x[i]);
        return e;
    };
    EXPECT_LT(error(ml::block_q4_0{}, true), error(ml::block_q4_0{}, false));
    EXPECT_LT(error(ml::block_q4_1{}, true), error(ml::block_q4_1{}, false));
    EXPECT_LT(error(ml::block_q5_0{}, true), error(ml::block_q5_0{}, false));
    EXPECT_LT(error(ml::block_q5_1{}, true), error(ml::block_q5_1{}, false));
    EXPECT_EQ(error(ml::block_q8_0{}, true), error(ml::block_q8_0{}, false));   // Q8_0 has no weighted path
}

TEST(MlQuantizeFormats, Q8_1SumsAndQ8KBlockSumsAreTheCodes)
{
    const auto x = quant_rows::row(quant_rows::uniform, 512, 5);
    const auto q1 = quantized_with<ml::block_q8_1>(x, {});
    for (const auto& b : q1) {
        int s = 0;
        for (const auto q : b.qs)
            s += q;
        EXPECT_NEAR(ml::to_float(b.s), float(s) * ml::to_float(b.d), 1e-3f * std::fabs(float(s) * ml::to_float(b.d)) + 1e-6f);   // s is d · sum before d rounds to float16
    }
    const auto qk = quantized_with<ml::block_q8_k>(x, {});
    for (const auto& b : qk)
        for (std::size_t j = 0; j < 16; ++j) {
            int s = 0;
            for (std::size_t k = 0; k < 16; ++k)
                s += b.qs[16 * j + k];
            EXPECT_EQ(b.bsums[j], s);
        }
    // a zero block: ggml leaves bsums unset there; core writes zeros
    const auto zero = quantized_with<ml::block_q8_k>(std::vector<float>(256, 0.0f), {});
    EXPECT_EQ(zero[0].d, 0.0f);
    EXPECT_EQ(zero[0].bsums[3], 0);
}

constexpr bool formats_at_compile_time()
{
    // 64 values k - 32: every small format quantises, dequantises and dots with its activations at compile time
    std::array<float, 128> x{};
    for (std::size_t k = 0; k < 128; ++k)
        x[k] = float(int(k % 64) - 32) / 8.0f;
    std::array<ml::block_q4_1, 4> q41{};
    std::array<ml::block_q8_1, 4> a81{};
    std::array<ml::block_q5_0, 4> q50{};
    std::array<ml::block_q8_0, 4> a80{};
    std::array<ml::block_nvfp4, 2> nv{};
    std::array<ml::block_q1_0, 1> q10{};
    ml::quantize(x, std::span(q41));
    ml::quantize(x, std::span(a81));
    ml::quantize(x, std::span(q50));
    ml::quantize(x, std::span(a80));
    ml::quantize(x, std::span(nv));
    ml::quantize(x, std::span(q10));
    const float d41 = ml::dot(std::span<const ml::block_q4_1>(q41), std::span<const ml::block_q8_1>(a81));
    const float d50 = ml::dot(std::span<const ml::block_q5_0>(q50), std::span<const ml::block_q8_0>(a80));
    const float dnv = ml::dot(std::span<const ml::block_nvfp4>(nv), std::span<const ml::block_q8_0>(a80));
    const float d10 = ml::dot(std::span<const ml::block_q1_0>(q10), std::span<const ml::block_q8_0>(a80));
    std::array<float, 128> back{};
    ml::dequantize(std::span<const ml::block_q5_0>(q50), back);
    // sum x^2 = 2 · sum_{k=-32}^{31} (k/8)^2 = 2 · 21856 / 64 = 683
    return back[0] == -4.0f && d41 > 670.0f && d41 < 695.0f && d50 > 660.0f && d50 < 695.0f && dnv > 650.0f && dnv < 710.0f && d10 > 400.0f;
}
static_assert(formats_at_compile_time());

} // namespace
