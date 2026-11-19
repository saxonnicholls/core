//
//  quant_iq_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/quant_iq.hpp>: on every row kind of tests/ml/quant_rows.hpp, the IQ formats' bytes
//  (ggml's quantize_<type> without and with an importance row), dequantised floats, activations and scalar dot
//  products equal the hashes ggml's generator recorded (tests/fixtures/ml/quant_iq_fixture.txt, from
//  scripts/generate_quant_iq_fixture.cpp), except the cases listed with their reasons; dot products against
//  dequantise-then-multiply; the error of a round trip; the layouts; compile-time use.
//

#include <snicholls/core/ml/quant_iq.hpp>

#include "quant_rows.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;

// The layouts: ggml's sizes (also static_asserted in the header against ggml_type_traits) and alignments
static_assert(sizeof(ml::block_iq1_s) == 50 && sizeof(ml::block_iq1_m) == 56 && sizeof(ml::block_iq2_xxs) == 66
              && sizeof(ml::block_iq2_xs) == 74 && sizeof(ml::block_iq2_s) == 82 && sizeof(ml::block_iq3_xxs) == 98
              && sizeof(ml::block_iq3_s) == 110 && sizeof(ml::block_iq4_nl) == 18 && sizeof(ml::block_iq4_xs) == 136);
static_assert(alignof(ml::block_iq1_s) == 2 && alignof(ml::block_iq1_m) == 1 && alignof(ml::block_iq2_xxs) == 2
              && alignof(ml::block_iq4_xs) == 2 && alignof(ml::block_iq4_nl) == 2);
static_assert(std::is_trivially_copyable_v<ml::block_iq1_s> && std::is_trivially_copyable_v<ml::block_iq1_m>
              && std::is_trivially_copyable_v<ml::block_iq2_xxs> && std::is_trivially_copyable_v<ml::block_iq3_s>
              && std::is_trivially_copyable_v<ml::block_iq4_xs> && std::is_standard_layout_v<ml::block_iq2_s>);
static_assert(ml::quant_block<ml::block_iq4_nl> && !ml::quant_block<ml::block_iq4_xs>);
static_assert(ml::block_traits<ml::block_iq2_xxs>::name == "iq2_xxs" && ml::block_traits<ml::block_iq2_xxs>::bits_per_weight == 2.0625
              && ml::block_traits<ml::block_iq1_s>::bits_per_weight == 1.5625 && ml::block_traits<ml::block_iq4_xs>::bits_per_weight == 4.25);
static_assert(std::is_same_v<ml::block_traits<ml::block_iq3_s>::dot_block, ml::block_q8_k>
              && std::is_same_v<ml::block_traits<ml::block_iq4_nl>::dot_block, ml::block_q8_0>);

// Compile time: quantise, dequantise and a dot product, the neighbour search included
constexpr float detail_abs(float v) { return v < 0 ? -v : v; }

constexpr std::array<float, 256> ramp()
{
    std::array<float, 256> x{};
    for (std::size_t i = 0; i < 256; ++i)
        x[i] = (float(i % 37) - 18.0f) / 16.0f;
    return x;
}

constexpr bool iq4_at_compile_time()
{
    const auto x = ramp();
    std::array<ml::block_iq4_nl, 8> nl{};
    std::array<ml::block_iq4_xs, 1> xs{};
    std::array<ml::block_q8_0, 8> a0{};
    std::array<ml::block_q8_k, 1> ak{};
    ml::quantize_row(x, {}, std::span(nl));
    ml::quantize_row(x, {}, std::span(xs));
    ml::quantize(x, std::span(a0));
    ml::quantize(x, std::span(ak));
    std::array<float, 256> back{};
    ml::dequantize(std::span<const ml::block_iq4_xs>(xs), back);
    float err = 0;
    for (std::size_t i = 0; i < 256; ++i)
        err = err < detail_abs(back[i] - x[i]) ? detail_abs(back[i] - x[i]) : err;
    const float d_nl = ml::dot(std::span<const ml::block_iq4_nl>(nl), std::span<const ml::block_q8_0>(a0));
    const float d_xs = ml::dot(std::span<const ml::block_iq4_xs>(xs), std::span<const ml::block_q8_k>(ak));
    float exact = 0;
    for (const float v : x)
        exact += v * v;
    return err < 0.15f && detail_abs(d_nl - exact) < 0.02f * exact && detail_abs(d_xs - exact) < 0.02f * exact;
}

// Quantising a whole IQ1, IQ2 or IQ3 block exceeds Clang's default 1,048,576 constant-evaluation steps; one
// neighbour search does not
constexpr bool neighbour_search_at_compile_time()
{
    // An off-grid IQ3_S point (levels 3, 4, 2, 7 packed 3 bits each), its nearest grid point in the weighted norm
    constexpr std::uint16_t u = 3 | (4 << 3) | (2 << 6) | (7 << 9);
    static_assert(ml::iq_grid_find(ml::iq3s_grid_index, u) < 0);
    const float xval[4] = {7.0f, 9.0f, 5.0f, 15.0f}, weight[4] = {1, 1, 1, 1};
    std::int8_t L[4] = {};
    const int g = ml::detail::iq::find_best_neighbour(ml::detail::iq::iq3s, u, xval, weight, 1.0f, L);
    return g >= 0 && ml::iq_grid_find(ml::iq3s_grid_index, ml::detail::iq::pack(ml::detail::iq::iq3s, L)) == g;
}

static_assert(iq4_at_compile_time());
static_assert(neighbour_search_at_compile_time());

// The fixture
struct fixture_line {
    int kind = 0;
    bool aborted = false;
    std::map<std::string, std::uint64_t> values;
};

const std::map<std::string, std::vector<fixture_line>>& fixture()
{
    static const auto lines = [] {
        std::map<std::string, std::vector<fixture_line>> out;
        std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/quant_iq_fixture.txt");
        EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/quant_iq_fixture.txt";
        std::string text;
        while (std::getline(in, text)) {
            if (text.empty() || text[0] == '#' || text.rfind("table ", 0) == 0)
                continue;
            std::istringstream ls(text);
            std::string format, word;
            fixture_line l;
            ls >> format >> l.kind;
            while (ls >> word) {
                if (word == "abort") {
                    l.aborted = true;
                    continue;
                }
                const auto eq = word.find('=');
                l.values[word.substr(0, eq)] = std::stoull(word.substr(eq + 1), nullptr, 16);
            }
            out[format].push_back(l);
        }
        return out;
    }();
    return lines;
}

constexpr std::size_t n = 512;

// What a case is not compared on, and why: "key kind" pairs
struct exclusions {
    std::set<std::pair<std::string, int>> keys;
    bool skips(const std::string& key, int kind) const { return keys.count({key, kind}) != 0; }
};

template <typename Block>
void matches_ggml(const char* name, bool needs_importance, const exclusions& skip = {})
{
    const auto it = fixture().find(name);
    ASSERT_NE(it, fixture().end()) << name;
    ASSERT_EQ(it->second.size(), std::size_t(quant_rows::kinds)) << name;
    using A = typename Block::dot_block;
    for (const auto& l : it->second) {
        const auto x = quant_rows::row(l.kind, n, 1);
        if (l.aborted) {
            // ggml aborts on this row (its "point not on grid" assertion); so does core, in a debug build
            EXPECT_EQ(l.kind, int(quant_rows::with_nan)) << name;
            continue;
        }
        const auto y = quant_rows::row((l.kind + 1) % 15, n, 2);
        const auto w = quant_rows::importance(n, 3);
        const std::vector<float> ones(n, 1.0f);
        std::vector<Block> ref(n / Block::elements), imat(ref.size());
        ml::quantize_row(x, needs_importance ? std::span<const float>(ones) : std::span<const float>{}, std::span<Block>(ref));
        ml::quantize(x, w, std::span<Block>(imat));
        std::vector<A> act(n / A::elements);
        ml::quantize(y, std::span<A>(act));
        std::vector<float> deq(n);
        ml::dequantize(std::span<const Block>(ref), deq);
        const float d = ml::dot(std::span<const Block>(ref), std::span<const A>(act));
        const auto expect = [&](const char* key, std::uint64_t got) {
            if (!skip.skips(key, l.kind))
                EXPECT_EQ(got, l.values.at(key)) << name << " " << key << " on " << quant_rows::name(l.kind);
        };
        expect("x", quant_rows::fnv_floats(x));
        expect("w", quant_rows::fnv(w));
        expect("ref", quant_rows::fnv(ref));
        expect("imat", quant_rows::fnv(imat));
        expect("deq", quant_rows::fnv_floats(deq));
        expect("act", quant_rows::fnv(act));
        expect("dot", quant_rows::float_bits(d));
        if constexpr (std::is_same_v<Block, ml::block_iq4_nl>) {
            std::vector<Block> fast(ref.size());
            ml::quantize_row_reference(x, std::span<Block>(fast));
            expect("rref", quant_rows::fnv(fast));
        }
    }
}

TEST(MlQuantIq, Iq4NlIsGgmlsOnEveryRow)
{
    // quantize_row_iq4_nl_ref leaves a block of zeros (and of subnormals, below its 1e-15 threshold) with the codes
    // of the block before it; before the first, ggml's are uninitialised memory and core's are 0
    matches_ggml<ml::block_iq4_nl>("iq4_nl", false, {{{"rref", quant_rows::zero_half}, {"rref", quant_rows::spike}, {"rref", quant_rows::subnormal}}});
}

TEST(MlQuantIq, Iq4XsIsGgmlsOnEveryRow) { matches_ggml<ml::block_iq4_xs>("iq4_xs", false); }
TEST(MlQuantIq, Iq3XxsIsGgmlsOnEveryRow) { matches_ggml<ml::block_iq3_xxs>("iq3_xxs", false); }
TEST(MlQuantIq, Iq3SIsGgmlsOnEveryRow) { matches_ggml<ml::block_iq3_s>("iq3_s", false); }
TEST(MlQuantIq, Iq2XxsIsGgmlsOnEveryRow) { matches_ggml<ml::block_iq2_xxs>("iq2_xxs", true); }
TEST(MlQuantIq, Iq2XsIsGgmlsOnEveryRow) { matches_ggml<ml::block_iq2_xs>("iq2_xs", true); }
TEST(MlQuantIq, Iq2SIsGgmlsOnEveryRow) { matches_ggml<ml::block_iq2_s>("iq2_s", false); }

// IQ1 sorts each group's weights with qsort on the value alone: the order of equal values is the C library's
// (the fixture's is macOS's), core's is stable. It matters only where equal values carry different weights (the
// constant rows with an importance row) or sit in different halves of an IQ1_M group (the halves row)
TEST(MlQuantIq, Iq1SIsGgmlsOnEveryRowWithoutTies)
{
    matches_ggml<ml::block_iq1_s>("iq1_s", true, {{{"imat", quant_rows::constant}, {"imat", quant_rows::negative}}});
}

TEST(MlQuantIq, Iq1MIsGgmlsOnEveryRowWithoutTies)
{
    matches_ggml<ml::block_iq1_m>("iq1_m", false,
                                  {{{"imat", quant_rows::constant}, {"imat", quant_rows::negative}, {"ref", quant_rows::halves},
                                    {"deq", quant_rows::halves}, {"dot", quant_rows::halves}}});
}

// The dot product is dequantise-then-multiply in integers: within float rounding of the exact sum
template <typename Block>
void dot_is_dequantised_product(double max_rms)
{
    using A = typename Block::dot_block;
    for (const int kind : {int(quant_rows::uniform), int(quant_rows::weights), int(quant_rows::outliers), int(quant_rows::heavy)}) {
        const auto x = quant_rows::row(kind, n, 7);
        const auto y = quant_rows::row(int(quant_rows::uniform), n, 8);
        const auto w = quant_rows::importance(n, 9);
        std::vector<Block> b(n / Block::elements);
        ml::quantize(x, w, std::span<Block>(b));
        std::vector<A> a(n / A::elements);
        ml::quantize(y, std::span<A>(a));
        std::vector<float> xd(n), yd(n);
        ml::dequantize(std::span<const Block>(b), xd);
        ml::dequantize(std::span<const A>(a), yd);
        double exact = 0, magnitude = 0, e2 = 0, s2 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            exact += double(xd[i]) * double(yd[i]);
            magnitude += std::fabs(double(xd[i]) * double(yd[i]));
            e2 += (double(xd[i]) - x[i]) * (double(xd[i]) - x[i]);
            s2 += double(x[i]) * x[i];
        }
        const float d = ml::dot(std::span<const Block>(b), std::span<const A>(a));
        EXPECT_NEAR(d, exact, 1e-6 * magnitude + 1e-12) << quant_rows::name(kind);
        if (kind == quant_rows::uniform || kind == quant_rows::weights)
            EXPECT_LT(std::sqrt(e2 / s2), max_rms) << quant_rows::name(kind);
    }
}

TEST(MlQuantIq, DotProductsAreTheDequantisedProductAndRoundTripsAreAsCloseAsTheFormatAllows)
{
    dot_is_dequantised_product<ml::block_iq1_s>(0.6);
    dot_is_dequantised_product<ml::block_iq1_m>(0.6);
    dot_is_dequantised_product<ml::block_iq2_xxs>(0.42);
    dot_is_dequantised_product<ml::block_iq2_xs>(0.4);
    dot_is_dequantised_product<ml::block_iq2_s>(0.3);
    dot_is_dequantised_product<ml::block_iq3_xxs>(0.25);
    dot_is_dequantised_product<ml::block_iq3_s>(0.17);
    dot_is_dequantised_product<ml::block_iq4_nl>(0.09);
    dot_is_dequantised_product<ml::block_iq4_xs>(0.09);
}

TEST(MlQuantIq, MoreBitsGiveLessError)
{
    const auto x = quant_rows::row(int(quant_rows::weights), n, 11);
    const auto w = quant_rows::importance(n, 12);
    const auto rms = [&](auto block) {
        using B = decltype(block);
        std::vector<B> b(n / B::elements);
        ml::quantize(x, w, std::span<B>(b));
        std::vector<float> d(n);
        ml::dequantize(std::span<const B>(b), d);
        double e = 0;
        for (std::size_t i = 0; i < n; ++i)
            e += (double(d[i]) - x[i]) * (double(d[i]) - x[i]);
        return e;
    };
    const double e1 = rms(ml::block_iq1_m{}), e2 = rms(ml::block_iq2_s{}), e3 = rms(ml::block_iq3_s{}), e4 = rms(ml::block_iq4_xs{});
    EXPECT_GT(e1, e2);
    EXPECT_GT(e2, e3);
    EXPECT_GT(e3, e4);
}

TEST(MlQuantIq, AZeroRowQuantisesToZero)
{
    const std::vector<float> zero(n, 0.0f), ones(n, 1.0f);
    const auto zero_back = [&](auto block, bool needs) {
        using B = decltype(block);
        std::vector<B> b(n / B::elements);
        ml::quantize_row(zero, needs ? std::span<const float>(ones) : std::span<const float>{}, std::span<B>(b));
        std::vector<float> d(n, 1.0f);
        ml::dequantize(std::span<const B>(b), d);
        for (const float v : d)
            if (v != 0.0f)
                return false;
        return true;
    };
    EXPECT_TRUE(zero_back(ml::block_iq2_xxs{}, true));
    EXPECT_TRUE(zero_back(ml::block_iq2_xs{}, true));
    EXPECT_TRUE(zero_back(ml::block_iq2_s{}, false));
    EXPECT_TRUE(zero_back(ml::block_iq3_xxs{}, false));
    EXPECT_TRUE(zero_back(ml::block_iq3_s{}, false));
    EXPECT_TRUE(zero_back(ml::block_iq4_nl{}, false));
    EXPECT_TRUE(zero_back(ml::block_iq4_xs{}, false));
    // IQ1's values are -1 ± 1/8, ±1/8, 1 ± 1/8: a zero scale decodes to zero
    EXPECT_TRUE(zero_back(ml::block_iq1_s{}, true));
    EXPECT_TRUE(zero_back(ml::block_iq1_m{}, false));
}

TEST(MlQuantIq, Iq4NlWeightsAreQuantBlocksForLinear)
{
    // IQ4_NL pairs with Q8_0, so quantized_weights (ml/linear.hpp) takes it like Q4_0
    const auto x = quant_rows::row(int(quant_rows::weights), 64, 13);
    std::array<ml::block_iq4_nl, 2> b{};
    ml::quantize(x, std::span(b));
    EXPECT_EQ(ml::block_traits<ml::block_iq4_nl>::elements, ml::quant_block_size);
    std::array<float, 64> back{};
    ml::dequantize(std::span<const ml::block_iq4_nl>(b), back);
    for (std::size_t i = 0; i < 64; ++i)
        EXPECT_NEAR(back[i], x[i], 0.01f);
}

} // namespace
