//
//  string_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/metric/string.hpp>: levenshtein on the classic pairs, empty and identical sequences, mixed
//  sequence types (string_view with std::array of char, std::vector<int> with std::span<const int>), the MaxLength
//  boundary and the over-length sentinel, agreement with a full-matrix reference on random strings, the metric axioms
//  and the length bounds, and evaluation and noexcept claims at compile time.
//

#include <snicholls/core/metric/string.hpp>

#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace me = snicholls::metric;
namespace st = snicholls::stats;

using namespace std::string_view_literals;

constexpr std::size_t too_long = std::numeric_limits<std::size_t>::max();
using lev = me::levenshtein<32>;

static_assert(me::true_metric<lev>);
static_assert(me::distance_for<lev, std::string_view> && me::distance_for<lev, std::vector<int>, std::span<const int>>);
static_assert(lev{}("kitten"sv, "sitting"sv) == 3 && lev{}("flaw"sv, "lawn"sv) == 2 && lev{}(""sv, ""sv) == 0);
static_assert(me::levenshtein<3>{}("abcd"sv, "abcd"sv) == too_long && me::levenshtein<3>{}("abc"sv, "abcdefg"sv) == 4);
static_assert(lev{}("kitten"sv, std::array<char, 7>{'s', 'i', 't', 't', 'i', 'n', 'g'}) == 3);
static_assert(noexcept(lev{}("a"sv, "b"sv)));
constexpr int c_array_a[] = {1, 2, 3}, c_array_b[] = {1, 3};
static_assert(noexcept(lev{}(c_array_a, c_array_b)) && lev{}(c_array_a, c_array_b) == 1);

// The full (m + 1) x (n + 1) Wagner-Fischer matrix, for reference
std::size_t reference(std::string_view a, std::string_view b)
{
    std::vector<std::vector<std::size_t>> d(a.size() + 1, std::vector<std::size_t>(b.size() + 1));
    for (std::size_t i = 0; i <= a.size(); ++i)
        d[i][0] = i;
    for (std::size_t j = 0; j <= b.size(); ++j)
        d[0][j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i)
        for (std::size_t j = 1; j <= b.size(); ++j)
            d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
    return d[a.size()][b.size()];
}

std::string random_string(st::wyrand& g, std::size_t max_length)
{
    std::string s(static_cast<std::size_t>(st::uniform_index(g, max_length + 1)), 'a');
    for (auto& ch : s)
        ch = static_cast<char>('a' + st::uniform_index(g, 3));
    return s;
}

TEST(MetricLevenshtein, ClassicPairs)
{
    const lev l{};
    EXPECT_EQ(l("kitten"sv, "sitting"sv), 3u);
    EXPECT_EQ(l("sitting"sv, "kitten"sv), 3u);
    EXPECT_EQ(l("flaw"sv, "lawn"sv), 2u);
    EXPECT_EQ(l("saturday"sv, "sunday"sv), 3u);
    EXPECT_EQ(l("intention"sv, "execution"sv), 5u);
    EXPECT_EQ(l("abc"sv, "xyz"sv), 3u);
    EXPECT_EQ(l("ab"sv, "ba"sv), 2u);   // a transposition is two edits
}

TEST(MetricLevenshtein, EmptyAndIdenticalSequences)
{
    const lev l{};
    EXPECT_EQ(l(""sv, ""sv), 0u);
    EXPECT_EQ(l(""sv, "abc"sv), 3u);
    EXPECT_EQ(l("abcd"sv, ""sv), 4u);
    EXPECT_EQ(l("kitten"sv, "kitten"sv), 0u);
    const std::string s(32, 'q');
    EXPECT_EQ(l(std::string_view(s), std::string_view(s)), 0u);
    EXPECT_EQ(me::levenshtein<0>{}(""sv, "hello"sv), 5u);   // a working row of one count is enough for an empty side
    EXPECT_EQ(me::levenshtein<0>{}(""sv, ""sv), 0u);
}

TEST(MetricLevenshtein, MixesSequenceTypes)
{
    const lev l{};
    const std::array<char, 7> sitting{'s', 'i', 't', 't', 'i', 'n', 'g'};
    EXPECT_EQ(l("kitten"sv, sitting), 3u);
    EXPECT_EQ(l(sitting, "kitten"sv), 3u);
    const std::vector<int> tokens{1, 2, 3, 4, 5};
    const std::array<int, 4> other{1, 3, 4, 6};
    const std::span<const int> view(other);
    EXPECT_EQ(l(tokens, view), 2u);   // delete 2, substitute 5 by 6
    EXPECT_EQ(l(view, tokens), 2u);
    EXPECT_EQ(l(tokens, std::span<const int>(tokens)), 0u);
    EXPECT_EQ(l(tokens, std::span<const int>(tokens).first(2)), 3u);
    const std::string owned = "lawn";
    EXPECT_EQ(l("flaw"sv, owned), 2u);
}

TEST(MetricLevenshtein, TheMaxLengthBoundaryAndTheOverLengthSentinel)
{
    const me::levenshtein<4> l{};
    EXPECT_EQ(l("abcd"sv, "abcd"sv), 0u);           // both exactly MaxLength
    EXPECT_EQ(l("abcd"sv, "abce"sv), 1u);
    EXPECT_EQ(l("abcd"sv, "abcdefghij"sv), 6u);      // the longer may exceed MaxLength while the shorter fits
    EXPECT_EQ(l("abcdefghij"sv, "abcd"sv), 6u);
    EXPECT_EQ(l("abcde"sv, "abcde"sv), too_long);    // both over: the sentinel, never a wrong answer
    EXPECT_EQ(l("abcde"sv, "abcdefghij"sv), too_long);
    EXPECT_EQ(l(""sv, "abcdefghij"sv), 10u);
}

TEST(MetricLevenshtein, MatchesAFullMatrixReferenceOnRandomStrings)
{
    st::wyrand g(19);
    const me::levenshtein<12> l{};
    for (int trial = 0; trial < 2000; ++trial) {
        const std::string a = random_string(g, 12), b = random_string(g, 12);
        EXPECT_EQ(l(std::string_view(a), std::string_view(b)), reference(a, b)) << a << ' ' << b;
    }
}

TEST(MetricLevenshtein, IsAMetricOnRandomShortStrings)
{
    st::wyrand g(23);
    const me::levenshtein<8> l{};
    for (int trial = 0; trial < 3000; ++trial) {
        const std::string a = random_string(g, 8), b = random_string(g, 8), c = random_string(g, 8);
        const std::string_view va(a), vb(b), vc(c);
        const std::size_t ab = l(va, vb);
        EXPECT_EQ(ab == 0, a == b) << a << ' ' << b;
        EXPECT_EQ(ab, l(vb, va)) << a << ' ' << b;
        EXPECT_LE(ab, l(va, vc) + l(vc, vb)) << a << ' ' << b << ' ' << c;
        EXPECT_GE(ab, std::max(a.size(), b.size()) - std::min(a.size(), b.size()));
        EXPECT_LE(ab, std::max(a.size(), b.size()));
    }
}

} // namespace
