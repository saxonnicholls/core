//
//  binary_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/metric/binary.hpp>: which types are bit vectors (unsigned words of any width in any
//  point kind, a struct of the test's own) and which are not (signed words, bool, std::vector<bool>, int); hamming,
//  jaccard and mismatches against Python on every kind and mixed word widths; empty, all-zero, disjoint and full
//  sets; the axioms on random codes; noexcept propagated from the coordinate read; and constant evaluation.
//

#include <snicholls/core/metric/binary.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

// A 128-bit code of the test's own, as two named words
struct code128 {
    std::uint64_t lo, hi;
};

// Words read through a check that may throw
struct checked_words {
    std::array<std::uint64_t, 2> w;
};

} // namespace

template <>
struct snicholls::metric::point_traits<code128> {
    using value_type = std::uint64_t;
    static constexpr std::size_t extent = 2;
    static constexpr std::size_t dimension(const code128&) noexcept { return 2; }
    static constexpr std::uint64_t at(const code128& p, std::size_t i) noexcept { return i == 0 ? p.lo : p.hi; }
};

template <>
struct snicholls::metric::point_traits<checked_words> {
    using value_type = std::uint64_t;
    static constexpr std::size_t extent = 2;
    static constexpr std::size_t dimension(const checked_words&) noexcept { return 2; }
    static std::uint64_t at(const checked_words& p, std::size_t i)
    {
        if (i >= 2)
            throw std::out_of_range("checked_words");
        return p.w[i];
    }
};

namespace {

namespace mt = snicholls::metric;
namespace la = snicholls::linalg;
namespace st = snicholls::stats;

using u64x2 = std::array<std::uint64_t, 2>;

// ---- which types are bit vectors ----
static_assert(mt::bit_vector<u64x2> && mt::bit_vector<std::vector<std::uint8_t>> && mt::bit_vector<std::span<const std::uint32_t>>);
static_assert(mt::bit_vector<std::uint16_t[4]> && mt::bit_vector<code128> && mt::bit_vector<const u64x2&>);
static_assert(!mt::bit_vector<std::array<std::int64_t, 2>> && !mt::bit_vector<std::array<int, 4>>);  // signed words
static_assert(!mt::bit_vector<std::array<bool, 8>> && mt::point<std::array<bool, 8>>);              // bool is a point, not words
static_assert(!mt::bit_vector<std::vector<bool>> && !mt::point<std::vector<bool>>);
static_assert(!mt::bit_vector<int> && !mt::bit_vector<std::uint64_t> && !mt::bit_vector<std::array<double, 2>>);

// ---- capabilities ----
static_assert(mt::distance_for<mt::hamming, u64x2> && mt::distance_for<mt::hamming, std::array<std::uint8_t, 2>, u64x2>);
static_assert(mt::distance_for<mt::jaccard, code128, std::vector<std::uint64_t>>);
static_assert(!mt::distance_for<mt::hamming, std::array<std::int64_t, 2>> && !mt::distance_for<mt::jaccard, std::array<int, 2>>);
static_assert(!mt::distance_for<mt::hamming, std::vector<bool>> && !mt::distance_for<mt::hamming, int>);
static_assert(!mt::distance_for<mt::hamming, u64x2, std::array<std::uint64_t, 3>>);  // mismatched extents
static_assert(mt::distance_for<mt::mismatches, std::array<std::int64_t, 2>> && mt::distance_for<mt::mismatches, std::string_view>);
static_assert(mt::distance_for<mt::mismatches, std::array<bool, 3>> && mt::distance_for<mt::mismatches, la::fixed_vector<float, 3>, std::array<float, 3>>);
static_assert(!mt::distance_for<mt::mismatches, int> && !mt::distance_for<mt::mismatches, std::vector<bool>>);
static_assert(mt::true_metric<mt::hamming> && mt::true_metric<mt::jaccard> && mt::true_metric<mt::mismatches>);
static_assert(!mt::reducible<mt::hamming, u64x2> && !mt::decomposable<mt::jaccard, double>);
static_assert(std::is_same_v<decltype(mt::hamming{}(u64x2{}, u64x2{})), std::size_t> && std::is_same_v<decltype(mt::jaccard{}(u64x2{}, u64x2{})), double>);
static_assert(std::is_same_v<decltype(mt::mismatches{}(std::string_view{}, std::string_view{})), std::size_t>);

// ---- noexcept ----
static_assert(noexcept(mt::hamming{}(std::declval<const u64x2&>(), std::declval<const code128&>())));
static_assert(noexcept(mt::jaccard{}(std::declval<const u64x2&>(), std::declval<const u64x2&>())));
static_assert(noexcept(mt::mismatches{}(std::declval<const std::string_view&>(), std::declval<const std::string_view&>())));
static_assert(!noexcept(mt::hamming{}(std::declval<const checked_words&>(), std::declval<const u64x2&>())));
static_assert(!noexcept(mt::jaccard{}(std::declval<const u64x2&>(), std::declval<const checked_words&>())));
static_assert(!noexcept(mt::mismatches{}(std::declval<const checked_words&>(), std::declval<const checked_words&>())));

// The reference pair: a = {0b1011, 0xFF00}, b = {0b0110, 0x0F00}
// XOR 0b1101, 0xF000: 3 + 4 = 7 bits; AND 0b0010, 0x0F00: 5 bits; OR 0b1111, 0xFF00: 12 bits (python3 bin().count)
constexpr std::size_t ref_hamming = 7;
constexpr double ref_jaccard = 0.5833333333333333;  // 1 - 5/12, python3

// ---- constant evaluation ----
constexpr u64x2 ca{0b1011, 0xFF00}, cb{0b0110, 0x0F00};
static_assert(mt::hamming{}(ca, cb) == ref_hamming && mt::hamming{}(ca, ca) == 0);
static_assert(mt::jaccard{}(ca, cb) - ref_jaccard < 1e-15 && ref_jaccard - mt::jaccard{}(ca, cb) < 1e-15);
static_assert(mt::mismatches{}(std::string_view("karolin"), std::string_view("kathrin")) == 3);  // the textbook example
static_assert(mt::hamming{}(code128{~0ull, 0}, code128{0, ~0ull}) == 128);

TEST(MetricBinaryHamming, CountsDifferingBitsOnEveryKind)
{
    const std::vector<std::uint64_t> bv{0b0110, 0x0F00};
    const std::span<const std::uint64_t, 2> as(ca);
    const std::span<const std::uint64_t> bs(bv);
    const std::uint64_t ac[2] = {0b1011, 0xFF00};
    const std::array<std::uint16_t, 2> b16{0b0110, 0x0F00};
    const std::uint32_t b32[2] = {0b0110, 0x0F00};
    const code128 ax{0b1011, 0xFF00};
    EXPECT_EQ(mt::hamming{}(ca, cb), ref_hamming);
    EXPECT_EQ(mt::hamming{}(ca, bv), ref_hamming);
    EXPECT_EQ(mt::hamming{}(as, bs), ref_hamming);
    EXPECT_EQ(mt::hamming{}(ac, b16), ref_hamming);   // 64-bit words against 16-bit ones
    EXPECT_EQ(mt::hamming{}(b32, ax), ref_hamming);
    EXPECT_EQ(mt::hamming{}(ax, cb), ref_hamming);
    EXPECT_EQ(mt::hamming{}(cb, ax), ref_hamming);
}

TEST(MetricBinaryHamming, NarrowWordsDoNotCarryHighBits)
{
    // 0x0F, 0xF0 against 0x00, 0xFF: 4 + 4 bits, the uint8_t never widened into garbage
    const std::array<std::uint8_t, 2> a{0x0F, 0xF0};
    const std::vector<std::uint64_t> b{0x00, 0xFF};
    EXPECT_EQ(mt::hamming{}(a, b), 8u);
    const std::array<std::uint8_t, 3> full{0xFF, 0xFF, 0xFF}, none{0, 0, 0};
    EXPECT_EQ(mt::hamming{}(full, none), 24u);
    EXPECT_EQ(mt::hamming{}(u64x2{~0ull, ~0ull}, u64x2{0, 0}), 128u);
}

TEST(MetricBinaryJaccard, AgreesWithPythonOnEveryKind)
{
    const std::vector<std::uint64_t> bv{0b0110, 0x0F00};
    const std::array<std::uint16_t, 2> b16{0b0110, 0x0F00};
    const code128 ax{0b1011, 0xFF00};
    EXPECT_DOUBLE_EQ(mt::jaccard{}(ca, cb), ref_jaccard);
    EXPECT_DOUBLE_EQ(mt::jaccard{}(ca, bv), ref_jaccard);
    EXPECT_DOUBLE_EQ(mt::jaccard{}(ax, b16), ref_jaccard);
    EXPECT_DOUBLE_EQ(mt::jaccard{}(std::span<const std::uint64_t>(bv), ax), ref_jaccard);
}

TEST(MetricBinaryJaccard, EmptyDisjointAndEqualSets)
{
    const u64x2 none{0, 0}, a{0b1010, 0}, b{0b0101, 0}, a_and_more{0b1010, 1};
    EXPECT_EQ(mt::jaccard{}(none, none), 0.0);   // two empty sets
    EXPECT_EQ(mt::jaccard{}(none, a), 1.0);      // one empty
    EXPECT_EQ(mt::jaccard{}(a, b), 1.0);         // disjoint
    EXPECT_EQ(mt::jaccard{}(a, a), 0.0);
    EXPECT_DOUBLE_EQ(mt::jaccard{}(a, a_and_more), 1.0 / 3);  // 1 - 2/3
}

TEST(MetricBinaryMismatches, CountsDifferingCoordinatesOfAnyType)
{
    EXPECT_EQ(mt::mismatches{}(std::string_view("karolin"), std::string_view("kathrin")), 3u);
    EXPECT_EQ(mt::mismatches{}(std::string_view("1011101"), std::string_view("1001001")), 2u);  // python3
    const std::array<double, 3> x{1, 2, 3};
    const la::fixed_vector<double, 3> y{{1, 2.5, 3}};
    const std::vector<double> z{1, 2, 3};
    EXPECT_EQ(mt::mismatches{}(x, y), 1u);
    EXPECT_EQ(mt::mismatches{}(x, z), 0u);
    const std::array<int, 3> xi{1, 2, 4};
    EXPECT_EQ(mt::mismatches{}(xi, x), 1u);   // 1 == 1.0, 2 == 2.0, 4 != 3.0
    const std::array<bool, 4> p{true, false, true, true}, q{true, true, false, true};
    EXPECT_EQ(mt::mismatches{}(p, q), 2u);
    const std::array<std::int64_t, 2> s{-1, 5}, t{-1, -5};
    EXPECT_EQ(mt::mismatches{}(s, t), 1u);    // signed words are fine coordinate by coordinate
    EXPECT_EQ(mt::mismatches{}(ca, cb), 2u);  // two words differ, seven bits
}

TEST(MetricBinaryEdge, EmptyAndSingleWordCodes)
{
    const std::span<const std::uint64_t> e;
    const std::vector<std::uint8_t> v;
    EXPECT_EQ(mt::hamming{}(e, e), 0u);
    EXPECT_EQ(mt::hamming{}(e, v), 0u);
    EXPECT_EQ(mt::jaccard{}(e, e), 0.0);
    EXPECT_EQ(mt::mismatches{}(std::string_view(), std::string_view()), 0u);
    const std::array<std::uint32_t, 1> one{0xFFFF0000u}, other{0x0000FFFFu};
    EXPECT_EQ(mt::hamming{}(one, other), 32u);
    EXPECT_EQ(mt::jaccard{}(one, other), 1.0);
    EXPECT_EQ(mt::mismatches{}(one, other), 1u);
}

template <typename M, typename Make>
void expect_axioms(M m, Make make, std::uint64_t seed, double tol)
{
    static_assert(mt::true_metric<M>);
    st::wyrand g(seed);
    for (int k = 0; k < 1000; ++k) {
        const auto x = make(g), y = make(g), z = make(g);
        EXPECT_EQ(m(x, x), 0u);
        if (x != y) {
            EXPECT_GT(m(x, y), 0u);
        }
        EXPECT_EQ(m(x, y), m(y, x));
        EXPECT_LE(static_cast<double>(m(x, z)), static_cast<double>(m(x, y)) + static_cast<double>(m(y, z)) + tol);
    }
}

TEST(MetricBinaryAxioms, EveryMetricObeysIdentitySymmetryAndTheTriangleInequality)
{
    const auto words = [](st::wyrand& g) { return std::array<std::uint64_t, 3>{st::random_u64(g), st::random_u64(g), st::random_u64(g)}; };
    // sparse sets, so that the Jaccard distance spreads over [0, 1]
    const auto sparse = [](st::wyrand& g) { return u64x2{st::random_u64(g) & st::random_u64(g) & st::random_u64(g), st::random_u64(g) & st::random_u64(g)}; };
    // short codes over a small alphabet, so that coordinates often agree
    const auto symbols = [](st::wyrand& g) {
        std::array<int, 6> s{};
        for (auto& c : s)
            c = static_cast<int>(st::uniform_index(g, 3));
        return s;
    };
    expect_axioms(mt::hamming{}, words, 41, 0);
    expect_axioms(mt::hamming{}, sparse, 42, 0);
    expect_axioms(mt::jaccard{}, words, 43, 1e-12);
    expect_axioms(mt::jaccard{}, sparse, 44, 1e-12);
    expect_axioms(mt::mismatches{}, symbols, 45, 0);
}

TEST(MetricBinaryNoexcept, AThrowingCoordinateReadPropagates)
{
    const checked_words a{{0b1011, 0xFF00}};
    EXPECT_EQ(mt::hamming{}(a, cb), ref_hamming);
    EXPECT_DOUBLE_EQ(mt::jaccard{}(cb, a), ref_jaccard);
    EXPECT_EQ(mt::mismatches{}(a, a), 0u);
}

} // namespace
