//
//  unicode_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/unicode.hpp>: UTF-8 decoding against the well-formed byte sequences of Unicode
//  Table 3-7 (every lead byte, the boundary continuation bytes, truncation), encoding round trips over every code
//  point, categories and their groups at the boundaries of the scripts tokenizers meet, White_Space, lowercase,
//  combining classes, and canonical decomposition including Hangul; at compile time where the claims are constexpr.
//

#include <snicholls/core/ml/unicode.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using cat = ml::unicode_category;

constexpr ml::utf8_decoded dec(std::string_view s) { return ml::decode_utf8(s, 0); }

// Decoding: the boundaries of Table 3-7
static_assert(dec("A").code_point == 'A' && dec("A").length == 1 && dec("A").valid);
static_assert(dec("\xC2\x80").code_point == 0x80 && dec("\xDF\xBF").code_point == 0x7FF);
static_assert(dec("\xE0\xA0\x80").code_point == 0x800 && dec("\xEF\xBF\xBF").code_point == 0xFFFF);
static_assert(dec("\xF0\x90\x80\x80").code_point == 0x10000 && dec("\xF4\x8F\xBF\xBF").code_point == 0x10FFFF);
static_assert(!dec("\xC0\x80").valid && !dec("\xC1\xBF").valid);           // overlong two-byte
static_assert(!dec("\xE0\x9F\xBF").valid && !dec("\xF0\x8F\xBF\xBF").valid);   // overlong three- and four-byte
static_assert(!dec("\xED\xA0\x80").valid && dec("\xED\x9F\xBF").valid);   // surrogates
static_assert(!dec("\xF4\x90\x80\x80").valid && !dec("\xF5\x80\x80\x80").valid);   // past U+10FFFF
static_assert(!dec("\x80").valid && !dec("\xFF").valid && dec("\x80").length == 1);
static_assert(!dec("\xE2\x82").valid && dec("\xE2\x82").length == 1 && dec("\xE2\x82").code_point == ml::replacement_character);
static_assert(ml::decode_utf8("", 0).length == 0 && ml::decode_utf8("ab", 5).length == 0);
static_assert(noexcept(ml::decode_utf8(std::string_view{}, 0)) && noexcept(ml::category(0)) && noexcept(ml::to_lower(0)));

// Categories and groups
static_assert(ml::category('a') == cat::lowercase_letter && ml::category('Z') == cat::uppercase_letter);
static_assert(ml::category('5') == cat::decimal_number && ml::category(' ') == cat::space_separator);
static_assert(ml::category('$') == cat::currency_symbol && ml::category('+') == cat::math_symbol);
static_assert(ml::category('!') == cat::other_punctuation && ml::category('\n') == cat::control);
static_assert(ml::category(0x00BD) == cat::other_number && ml::category(0x0301) == cat::nonspacing_mark);
static_assert(ml::category(0x01C5) == cat::titlecase_letter && ml::category(0x02B0) == cat::modifier_letter);
static_assert(ml::category(0x4E2D) == cat::other_letter && ml::category(0x0903) == cat::spacing_mark);
static_assert(ml::category(0x1F642) == cat::other_symbol && ml::category(0x200D) == cat::format);
static_assert(ml::category(0xE000) == cat::private_use && ml::category(0xD800) == cat::surrogate);
static_assert(ml::category(0x0378) == cat::unassigned && ml::category(0x110000) == cat::unassigned);
static_assert(ml::category(0x0897) == cat::nonspacing_mark);   // new in Unicode 16.0
static_assert(ml::is_letter(cat::other_letter) && !ml::is_letter(cat::nonspacing_mark) && ml::is_mark(cat::enclosing_mark));
static_assert(ml::is_number(cat::letter_number) && ml::is_punctuation(cat::dash_punctuation) && ml::is_symbol(cat::modifier_symbol));
static_assert(ml::is_separator(cat::line_separator) && ml::is_other(cat::unassigned) && ml::is_other(cat::private_use));
static_assert(ml::is_uppercase_class(cat::uppercase_letter) && ml::is_uppercase_class(cat::other_letter) &&
              !ml::is_uppercase_class(cat::lowercase_letter));
static_assert(ml::is_lowercase_class(cat::lowercase_letter) && ml::is_lowercase_class(cat::spacing_mark) &&
              !ml::is_lowercase_class(cat::titlecase_letter));

// White_Space, lowercase, combining classes, decomposition
static_assert(ml::is_whitespace('\t') && ml::is_whitespace(0x85) && ml::is_whitespace(0x3000) && ml::is_whitespace(0x2028));
static_assert(!ml::is_whitespace(0x1C) && !ml::is_whitespace(0x200B) && !ml::is_whitespace('a'));
static_assert(ml::to_lower('A') == 'a' && ml::to_lower('a') == 'a' && ml::to_lower(0x00C0) == 0x00E0);
static_assert(ml::to_lower(0x0130) == 'i' && ml::to_lower(0x03A3) == 0x03C3 && ml::to_lower(0x0410) == 0x0430);
static_assert(ml::to_lower(0x0100) == 0x0101 && ml::to_lower(0x0101) == 0x0101 && ml::to_lower(0x1E9E) == 0x00DF);
static_assert(ml::to_lower(0x10400) == 0x10428 && ml::to_lower(0x4E2D) == 0x4E2D && ml::to_lower(0x212A) == 'k');
static_assert(ml::combining_class('a') == 0 && ml::combining_class(0x0301) == 230 && ml::combining_class(0x0327) == 202);
static_assert(ml::combining_class(0x05B0) == 10 && ml::combining_class(0x10FFFF) == 0);

constexpr std::array<char32_t, 4> decomposed(char32_t cp, std::size_t& n)
{
    std::array<char32_t, ml::max_decomposition> out{};
    n = ml::decompose(cp, out);
    return out;
}
constexpr bool decomposes(char32_t cp, std::initializer_list<char32_t> want)
{
    std::size_t n = 0;
    const auto d = decomposed(cp, n);
    if (n != want.size())
        return false;
    std::size_t i = 0;
    for (const char32_t c : want)
        if (d[i++] != c)
            return false;
    return true;
}
static_assert(decomposes('a', {'a'}) && decomposes(0x00E9, {'e', 0x0301}) && decomposes(0x01D5, {'U', 0x0308, 0x0304}));
static_assert(decomposes(0x1E08, {'C', 0x0327, 0x0301}) && decomposes(0xF900, {0x8C48}));   // compatibility ideograph
static_assert(decomposes(0xAC00, {0x1100, 0x1161}) && decomposes(0xD55C, {0x1112, 0x1161, 0x11AB}));   // Hangul

TEST(MlUnicode, EveryCodePointRoundTripsThroughUtf8)
{
    std::array<char, 4> buf{};
    for (char32_t cp = 0; cp <= 0x10FFFF; ++cp) {
        if (cp >= 0xD800 && cp <= 0xDFFF)
            continue;
        const std::size_t n = ml::encode_utf8(cp, buf.data());
        ASSERT_EQ(n, ml::utf8_length(cp));
        const auto d = ml::decode_utf8(std::string_view(buf.data(), n), 0);
        ASSERT_TRUE(d.valid);
        ASSERT_EQ(d.code_point, cp);
        ASSERT_EQ(d.length, n);
    }
    // A surrogate or a value past the range encodes as U+FFFD
    EXPECT_EQ(ml::encode_utf8(0xD800, buf.data()), 3u);
    EXPECT_EQ(std::string_view(buf.data(), 3), "\xEF\xBF\xBD");
    EXPECT_EQ(ml::encode_utf8(0x110000, buf.data()), 3u);
}

TEST(MlUnicode, EveryTwoByteSequenceDecodesAsTable37Says)
{
    for (unsigned a = 0; a < 256; ++a)
        for (unsigned b = 0; b < 256; ++b) {
            const char s[2] = {static_cast<char>(a), static_cast<char>(b)};
            const auto d = ml::decode_utf8(std::string_view(s, 2), 0);
            const bool want = a < 0x80 || (a >= 0xC2 && a <= 0xDF && b >= 0x80 && b <= 0xBF);
            EXPECT_EQ(d.valid, want) << a << ' ' << b;
            EXPECT_EQ(d.length, a < 0x80 ? 1u : (want ? 2u : 1u));
        }
}

TEST(MlUnicode, RepairingInvalidTextReplacesEachBadByte)
{
    EXPECT_TRUE(ml::is_valid_utf8("plain \xC3\xA9 \xF0\x9F\x99\x82"));
    EXPECT_FALSE(ml::is_valid_utf8("a\x80" "b"));
    std::string out(30, '\0');
    out.resize(ml::replace_invalid_utf8("a\x80" "b\xE2\x82", std::span<char>(out)));
    EXPECT_EQ(out, "a\xEF\xBF\xBD" "b\xEF\xBF\xBD\xEF\xBF\xBD");
}

TEST(MlUnicode, CategoryRunsCoverTheirRanges)
{
    // Every code point of a sample of scripts has the category its block says
    for (char32_t cp = 0x4E00; cp <= 0x9FFF; ++cp)
        EXPECT_EQ(ml::category(cp), cat::other_letter);
    for (char32_t cp = 0xAC00; cp <= 0xD7A3; ++cp)
        EXPECT_EQ(ml::category(cp), cat::other_letter);
    for (char32_t cp = 0x0300; cp <= 0x036F; ++cp)
        EXPECT_EQ(ml::category(cp), cat::nonspacing_mark);
    for (char32_t cp = 0x0660; cp <= 0x0669; ++cp)
        EXPECT_EQ(ml::category(cp), cat::decimal_number);
    for (char32_t cp = 0xF0000; cp <= 0xFFFFD; ++cp)
        EXPECT_EQ(ml::category(cp), cat::private_use);
}

} // namespace
