//
//  quant_k_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/quant_k.hpp>: on every row kind of the committed fixture (written by ggml itself,
//  scripts/generate_quant_formats_fixture.cpp), the Q2_K to Q6_K bytes (reference and importance-weighted),
//  dequantised floats, Q8_K activations and dot products equal ggml's; the layouts and traits; the dot product
//  against dequantise-then-multiply; error bounds that fall with the bits; zero blocks; and compile-time use.
//

#include <snicholls/core/ml/quant_k.hpp>

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

static_assert(std::is_trivially_copyable_v<ml::block_q2_k> && std::is_trivially_copyable_v<ml::block_q3_k> && std::is_trivially_copyable_v<ml::block_q4_k>
              && std::is_trivially_copyable_v<ml::block_q5_k> && std::is_trivially_copyable_v<ml::block_q6_k>);
static_assert(sizeof(ml::block_q2_k) == 84 && sizeof(ml::block_q3_k) == 110 && sizeof(ml::block_q4_k) == 144 && sizeof(ml::block_q5_k) == 176
              && sizeof(ml::block_q6_k) == 210);
static_assert(alignof(ml::block_q2_k) == 2 && alignof(ml::block_q6_k) == 2);
static_assert(ml::block_traits<ml::block_q4_k>::name == "q4_K" && ml::block_traits<ml::block_q6_k>::bits_per_weight == 6.5625);
static_assert(ml::block_traits<ml::block_q2_k>::bits_per_weight == 2.625 && ml::block_traits<ml::block_q3_k>::bits_per_weight == 3.4375);
static_assert(std::same_as<ml::block_traits<ml::block_q5_k>::dot_block, ml::block_q8_k>);
static_assert(!ml::quant_block<ml::block_q4_k>);   // 256 numbers with Q8_K activations

// The 6-bit scale packing of Q4_K and Q5_K reads back what was packed, every value of every slot
constexpr bool scale_packing()
{
    for (int v = 0; v < 64; ++v) {
        std::array<std::uint8_t, 8> ls{}, lm{};
        for (int j = 0; j < 8; ++j) {
            ls[std::size_t(j)] = static_cast<std::uint8_t>((v + j) % 64);
            lm[std::size_t(j)] = static_cast<std::uint8_t>((v + 3 * j) % 64);
        }
        std::array<std::uint8_t, 12> packed{};
        ml::detail::quant_k::pack_scale_min_k4(ls.data(), lm.data(), packed);
        for (int j = 0; j < 8; ++j) {
            std::uint8_t s = 0, m = 0;
            ml::detail::quant_k::scale_min_k4(j, packed, s, m);
            if (s != ls[std::size_t(j)] || m != lm[std::size_t(j)])
                return false;
        }
        std::array<int, 16> l6{};
        for (int j = 0; j < 16; ++j)
            l6[std::size_t(j)] = (v + 5 * j) % 64;
        ml::detail::quant_k::pack_q3_scales(l6.data(), packed);
        for (int j = 0; j < 16; ++j)
            if (ml::detail::quant_k::q3_scale(packed, j) != l6[std::size_t(j)] - 32)
                return false;
    }
    return true;
}
static_assert(scale_packing());

// One Q6_K block quantised, dequantised and dotted at compile time: a ramp comes back within a step
constexpr bool compile_time()
{
    std::array<float, 256> x{};
    for (std::size_t i = 0; i < 256; ++i)
        x[i] = (float(i) - 128.0f) / 64.0f;
    std::array<ml::block_q6_k, 1> q{};
    std::array<ml::block_q8_k, 1> a{};
    ml::quantize(x, std::span(q));
    ml::quantize(x, std::span(a));
    std::array<float, 256> back{};
    ml::dequantize(std::span<const ml::block_q6_k>(q), back);
    for (std::size_t i = 0; i < 256; ++i) {
        const float e = back[i] - x[i];
        if (e > 0.03f || e < -0.03f)
            return false;
    }
    const float self = ml::dot(std::span<const ml::block_q6_k>(q), std::span<const ml::block_q8_k>(a));
    return self > 338.0f && self < 344.0f;   // sum ((i - 128) / 64)^2 = 341.33
}
static_assert(compile_time());

std::vector<quant_rows::line> fixture(const std::string& format)
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
std::vector<B> quantized(const std::vector<float>& x, const std::vector<float>& importance = {})
{
    std::vector<B> b(x.size() / B::elements);
    ml::quantize(x, importance, std::span<B>(b));
    return b;
}

template <typename B>
void matches_ggml(const char* format)
{
    const auto lines = fixture(format);
    ASSERT_EQ(lines.size(), std::size_t(quant_rows::kinds)) << format;
    for (const auto& l : lines) {
        const auto x = quant_rows::row(l.kind, 512, 1);
        const auto y = quant_rows::row((l.kind + 1) % 15, 512, 2);
        const auto w = quant_rows::importance(512, 3);
        ASSERT_EQ(quant_rows::fnv_floats(x), l.values.at("x")) << "the rows of tests/ml/quant_rows.hpp changed";
        ASSERT_EQ(quant_rows::fnv(w), l.values.at("w"));
        const auto ref = quantized<B>(x);
        const auto imat = quantized<B>(x, w);
        const auto act = quantized<ml::block_q8_k>(y);
        std::vector<float> deq(512);
        ml::dequantize(std::span<const B>(ref), deq);
        const float d = ml::dot(std::span<const B>(ref), std::span<const ml::block_q8_k>(act));
        const char* kind = quant_rows::name(l.kind);
        EXPECT_EQ(quant_rows::fnv(ref), l.values.at("ref")) << format << " " << kind;
        EXPECT_EQ(quant_rows::fnv(imat), l.values.at("imat")) << format << " " << kind;
        EXPECT_EQ(quant_rows::fnv_floats(deq), l.values.at("deq")) << format << " " << kind;
        EXPECT_EQ(quant_rows::fnv(act), l.values.at("act")) << format << " " << kind;
        EXPECT_EQ(quant_rows::float_bits(d), l.values.at("dot")) << format << " " << kind;
    }
}

TEST(MlQuantK, Q2KIsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q2_k>("q2_K"); }
TEST(MlQuantK, Q3KIsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q3_k>("q3_K"); }
TEST(MlQuantK, Q4KIsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q4_k>("q4_K"); }
TEST(MlQuantK, Q5KIsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q5_k>("q5_K"); }
TEST(MlQuantK, Q6KIsGgmlsOnEveryRowKind) { matches_ggml<ml::block_q6_k>("q6_K"); }

// The largest error of a format on a row of weights, relative to the row's largest magnitude
template <typename B>
float worst_error(const std::vector<float>& x, const std::vector<float>& importance = {})
{
    const auto q = quantized<B>(x, importance);
    std::vector<float> back(x.size());
    ml::dequantize(std::span<const B>(q), back);
    float err = 0, amax = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        err = std::fmax(err, std::fabs(back[i] - x[i]));
        amax = std::fmax(amax, std::fabs(x[i]));
    }
    return err / amax;
}

TEST(MlQuantK, MoreBitsGiveLessError)
{
    const auto x = quant_rows::row(quant_rows::weights, 4096, 7);
    const float e2 = worst_error<ml::block_q2_k>(x), e3 = worst_error<ml::block_q3_k>(x), e4 = worst_error<ml::block_q4_k>(x),
                e5 = worst_error<ml::block_q5_k>(x), e6 = worst_error<ml::block_q6_k>(x);
    EXPECT_LT(e6, e5);
    EXPECT_LT(e5, e4);
    EXPECT_LT(e4, e3);
    EXPECT_LT(e3, e2);
    EXPECT_LT(e2, 0.5f);
    EXPECT_LT(e6, 0.03f);
}

// The importance-weighted path lowers the importance-weighted squared error against the reference
template <typename B>
double weighted_error(const std::vector<float>& x, const std::vector<float>& w, bool weighted)
{
    const auto q = quantized<B>(x, weighted ? w : std::vector<float>{});
    std::vector<float> back(x.size());
    ml::dequantize(std::span<const B>(q), back);
    double e = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        e += double(w[i]) * double(back[i] - x[i]) * double(back[i] - x[i]);
    return e;
}

TEST(MlQuantK, TheImportanceMatrixLowersTheWeightedError)
{
    const auto x = quant_rows::row(quant_rows::weights, 4096, 11);
    std::vector<float> w(x.size());
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = i % 7 == 0 ? 50.0f : 0.1f;   // a few columns matter far more
    EXPECT_LT(weighted_error<ml::block_q2_k>(x, w, true), weighted_error<ml::block_q2_k>(x, w, false));
    EXPECT_LT(weighted_error<ml::block_q3_k>(x, w, true), weighted_error<ml::block_q3_k>(x, w, false));
    EXPECT_LT(weighted_error<ml::block_q4_k>(x, w, true), weighted_error<ml::block_q4_k>(x, w, false));
    EXPECT_LT(weighted_error<ml::block_q5_k>(x, w, true), weighted_error<ml::block_q5_k>(x, w, false));
    EXPECT_LT(weighted_error<ml::block_q6_k>(x, w, true), weighted_error<ml::block_q6_k>(x, w, false));
}

// The dot product is the dequantised rows' dot product, up to float rounding of the sum
template <typename B>
void dot_is_dequantised_dot()
{
    const auto x = quant_rows::row(quant_rows::weights, 2048, 13);
    const auto y = quant_rows::row(quant_rows::uniform, 2048, 17);
    const auto q = quantized<B>(x);
    const auto a = quantized<ml::block_q8_k>(y);
    std::vector<float> xd(x.size()), yd(y.size());
    ml::dequantize(std::span<const B>(q), xd);
    ml::dequantize(std::span<const ml::block_q8_k>(a), yd);
    double exact = 0, mass = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        exact += double(xd[i]) * double(yd[i]);
        mass += std::fabs(double(xd[i]) * double(yd[i]));
    }
    const float d = ml::dot(std::span<const B>(q), std::span<const ml::block_q8_k>(a));
    EXPECT_NEAR(d, exact, 1e-5 * mass) << ml::block_traits<B>::name;
}

TEST(MlQuantK, DotProductsAreTheDequantisedDotProduct)
{
    dot_is_dequantised_dot<ml::block_q2_k>();
    dot_is_dequantised_dot<ml::block_q3_k>();
    dot_is_dequantised_dot<ml::block_q4_k>();
    dot_is_dequantised_dot<ml::block_q5_k>();
    dot_is_dequantised_dot<ml::block_q6_k>();
}

TEST(MlQuantK, AZeroRowDecodesToZeros)
{
    const std::vector<float> zero(256, 0.0f);
    const auto check = [&](auto block) {
        using B = decltype(block);
        const auto q = quantized<B>(zero);
        std::vector<float> back(256, 1.0f);
        ml::dequantize(std::span<const B>(q), back);
        for (const float v : back)
            EXPECT_EQ(v, 0.0f) << ml::block_traits<B>::name;
    };
    check(ml::block_q2_k{});
    check(ml::block_q3_k{});
    check(ml::block_q4_k{});
    check(ml::block_q5_k{});
    check(ml::block_q6_k{});
}

} // namespace
