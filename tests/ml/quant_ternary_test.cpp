//
//  quant_ternary_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/quant_ternary.hpp>: on every row kind of the committed fixture (written by ggml
//  itself, scripts/generate_quant_ternary_fixture.cpp), TQ1_0 and TQ2_0 bytes, dequantised floats and dot products
//  with Q8_K equal ggml's; the layouts and traits; the decoding tables; exact round trips of ternary rows; the dot
//  product against dequantise-then-multiply; and compile-time use.
//

#include <snicholls/core/ml/quant_ternary.hpp>

#include "quant_rows.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;

static_assert(std::is_trivially_copyable_v<ml::block_tq1_0> && std::is_trivially_copyable_v<ml::block_tq2_0>);
static_assert(sizeof(ml::block_tq1_0) == 54 && sizeof(ml::block_tq2_0) == 66);
static_assert(alignof(ml::block_tq1_0) == 2 && alignof(ml::block_tq2_0) == 2);
static_assert(ml::block_traits<ml::block_tq1_0>::name == "tq1_0" && ml::block_traits<ml::block_tq2_0>::name == "tq2_0");
static_assert(ml::block_traits<ml::block_tq1_0>::bits_per_weight == 1.6875 && ml::block_traits<ml::block_tq2_0>::bits_per_weight == 2.0625);
static_assert(std::same_as<ml::block_traits<ml::block_tq1_0>::dot_block, ml::block_q8_k>);
static_assert(!ml::quant_block<ml::block_tq1_0>);   // 256 numbers and Q8_K activations: not a quantized_weights block

// The decoding table is ggml's pow3 trick: every byte the quantiser writes decodes to the trits it was made from
constexpr bool tables()
{
    if (ml::ternary_pow3 != std::array<std::uint8_t, 6>{1, 3, 9, 27, 81, 243})
        return false;
    for (unsigned n = 0; n < 243; ++n) {
        const auto b = static_cast<std::uint8_t>((n * 256 + 242) / 243);
        unsigned back = 0;
        for (std::size_t k = 0; k < 5; ++k)
            back = back * 3 + ml::ternary_trits[b][k];
        if (back != n)
            return false;
    }
    return true;
}
static_assert(tables());

constexpr bool compile_time()
{
    // A ternary row times 0.5 comes back exactly, and its dot product with itself (as Q8_K) is an integer sum
    std::array<float, 256> x{};
    for (std::size_t i = 0; i < 256; ++i)
        x[i] = 0.5f * float(int(i * 7 % 3) - 1);
    std::array<ml::block_tq1_0, 1> t1{};
    std::array<ml::block_tq2_0, 1> t2{};
    std::array<ml::block_q8_k, 1> a{};
    ml::quantize(x, std::span(t1));
    ml::quantize(x, std::span(t2));
    ml::quantize(x, std::span(a));
    std::array<float, 256> b1{}, b2{};
    ml::dequantize(std::span<const ml::block_tq1_0>(t1), b1);
    ml::dequantize(std::span<const ml::block_tq2_0>(t2), b2);
    const float d1 = ml::dot(std::span<const ml::block_tq1_0>(t1), std::span<const ml::block_q8_k>(a));
    const float d2 = ml::dot(std::span<const ml::block_tq2_0>(t2), std::span<const ml::block_q8_k>(a));
    return b1 == x && b2 == x && d1 == d2 && d1 > 42.0f && d1 < 43.5f;   // 171 nonzero values of 0.25: 42.75
}
static_assert(compile_time());

std::vector<quant_rows::line> fixture(const std::string& format)
{
    std::vector<quant_rows::line> out;
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/quant_ternary_fixture.txt");
    EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/quant_ternary_fixture.txt";
    std::string text;
    quant_rows::line l;
    while (std::getline(in, text))
        if (quant_rows::parse(text, l) && l.format == format)
            out.push_back(l);
    return out;
}

template <typename B>
std::vector<B> quantized(const std::vector<float>& x, const std::vector<float>* w = nullptr)
{
    std::vector<B> b(x.size() / B::elements);
    if (w)
        ml::quantize(x, *w, std::span<B>(b));
    else
        ml::quantize(x, std::span<B>(b));
    return b;
}

template <typename B>
void matches_ggml(const char* name)
{
    const auto lines = fixture(name);
    ASSERT_EQ(lines.size(), std::size_t(quant_rows::kinds)) << name;
    for (const auto& l : lines) {
        const auto x = quant_rows::row(l.kind, 512, 1), y = quant_rows::row((l.kind + 1) % 15, 512, 2);
        const auto w = quant_rows::importance(512, 3);
        const auto r = quantized<B>(x), m = quantized<B>(x, &w);
        const auto a = quantized<ml::block_q8_k>(y);
        std::vector<float> deq(512);
        ml::dequantize(std::span<const B>(r), deq);
        const float d = ml::dot(std::span<const B>(r), std::span<const ml::block_q8_k>(a));
        const char* kind = quant_rows::name(l.kind);
        ASSERT_EQ(quant_rows::fnv_floats(x), l.values.at("x")) << name << " " << kind << ": the input row";
        EXPECT_EQ(quant_rows::fnv(r), l.values.at("ref")) << name << " " << kind;
        EXPECT_EQ(quant_rows::fnv(m), l.values.at("imat")) << name << " " << kind;
        EXPECT_EQ(quant_rows::fnv_floats(deq), l.values.at("deq")) << name << " " << kind;
        EXPECT_EQ(quant_rows::fnv(a), l.values.at("act")) << name << " " << kind;
        EXPECT_EQ(quant_rows::float_bits(d), l.values.at("dot")) << name << " " << kind;
    }
}

TEST(MlQuantTernary, Tq1_0IsGgmlsOnEveryRowKind)
{
    matches_ggml<ml::block_tq1_0>("tq1_0");
}

TEST(MlQuantTernary, Tq2_0IsGgmlsOnEveryRowKind)
{
    matches_ggml<ml::block_tq2_0>("tq2_0");
}

TEST(MlQuantTernary, TernaryRowsComeBackExactly)
{
    quant_rows::stream s(9);
    std::vector<float> x(1024);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = 0.125f * float(int(s.next() % 3) - 1);
    x[3] = 0.125f;   // the block's amax is the scale
    std::vector<float> back(x.size());
    ml::dequantize(std::span<const ml::block_tq1_0>(quantized<ml::block_tq1_0>(x)), back);
    EXPECT_EQ(back, x);
    ml::dequantize(std::span<const ml::block_tq2_0>(quantized<ml::block_tq2_0>(x)), back);
    EXPECT_EQ(back, x);
}

TEST(MlQuantTernary, TheDotProductIsDequantiseThenMultiplyWithinRounding)
{
    const auto x = quant_rows::row(quant_rows::weights, 1024, 5), y = quant_rows::row(quant_rows::uniform, 1024, 6);
    const auto a = quantized<ml::block_q8_k>(y);
    std::vector<float> xa(1024), ya(1024);
    ml::dequantize(std::span<const ml::block_q8_k>(a), ya);
    const auto t1 = quantized<ml::block_tq1_0>(x);
    const auto t2 = quantized<ml::block_tq2_0>(x);
    for (const float d : {ml::dot(std::span<const ml::block_tq1_0>(t1), std::span<const ml::block_q8_k>(a)),
                          ml::dot(std::span<const ml::block_tq2_0>(t2), std::span<const ml::block_q8_k>(a))}) {
        ml::dequantize(std::span<const ml::block_tq1_0>(t1), xa);
        double exact = 0, mag = 0;
        for (std::size_t i = 0; i < 1024; ++i) {
            exact += double(xa[i]) * double(ya[i]);
            mag += std::fabs(double(xa[i]) * double(ya[i]));
        }
        EXPECT_NEAR(d, exact, 1e-5 * mag + 1e-6);
    }
    // TQ1_0 and TQ2_0 hold the same trits: the same dot product
    EXPECT_EQ(ml::dot(std::span<const ml::block_tq1_0>(t1), std::span<const ml::block_q8_k>(a)),
              ml::dot(std::span<const ml::block_tq2_0>(t2), std::span<const ml::block_q8_k>(a)));
}

TEST(MlQuantTernary, AZeroBlockIsAllTritOne)
{
    const std::vector<float> zero(256, 0.0f);
    const auto t1 = quantized<ml::block_tq1_0>(zero);
    const auto t2 = quantized<ml::block_tq2_0>(zero);
    EXPECT_EQ(t1[0].d.bits, 0u);
    EXPECT_EQ(t2[0].qs[0], 0x55);   // four codes 1
    EXPECT_EQ(ml::ternary_trits[t1[0].qs[0]], (std::array<std::uint8_t, 5>{1, 1, 1, 1, 1}));
}

TEST(MlQuantTernary, TheImportanceRowIsIgnoredAsGgmlIgnoresIt)
{
    const auto x = quant_rows::row(quant_rows::heavy, 256, 8);
    const auto w = quant_rows::importance(256, 9);
    EXPECT_EQ(quant_rows::fnv(quantized<ml::block_tq1_0>(x)), quant_rows::fnv(quantized<ml::block_tq1_0>(x, &w)));
    EXPECT_EQ(quant_rows::fnv(quantized<ml::block_tq2_0>(x)), quant_rows::fnv(quantized<ml::block_tq2_0>(x, &w)));
}

} // namespace
