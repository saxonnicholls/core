//
//  phone_number_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/phone_number.hpp>: the type's layout, every accepted spelling and every error, text
//  out, ordering and hashing, the literal, and agreement with the Python phonenumbers package on the committed
//  fixture tests/fixtures/e164/parse.tsv (scripts/generate_e164_fixtures.py, never core's own output).
//

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/acout.hpp>
#include <snicholls/core/phone_number.hpp>

#ifndef SNICHOLLS_CORE_FIXTURE_DIR
#error "SNICHOLLS_CORE_FIXTURE_DIR must name tests/fixtures (set by tests/CMakeLists.txt)"
#endif

namespace {

using snicholls::parse_phone_number;
using snicholls::phone_errc;
using snicholls::phone_number;
using namespace snicholls::literals;
using namespace std::string_view_literals;

constexpr phone_number must(std::string_view text) { return *parse_phone_number(text); }

constexpr phone_errc error_of(std::string_view text) { return parse_phone_number(text).unwrap_err_unchecked(); }

// ---- the type ------------------------------------------------------------------------------------------

TEST(PhoneNumberType, IsOneTriviallyCopyableWord)
{
    static_assert(sizeof(phone_number) == 8);
    static_assert(alignof(phone_number) == alignof(std::uint64_t));
    static_assert(std::is_trivially_copyable_v<phone_number>);
    static_assert(std::is_standard_layout_v<phone_number>);
    static_assert(std::is_aggregate_v<phone_number>);
    static_assert(std::is_nothrow_default_constructible_v<phone_number>);
    static_assert(std::is_trivially_copyable_v<snicholls::result<phone_number, phone_errc>>);
    static_assert(sizeof(snicholls::result<phone_number, phone_errc>) == 16);
    static_assert(std::is_trivially_copyable_v<snicholls::phone_number_text>);
    static_assert(sizeof(snicholls::phone_number_text) == 18);
    SUCCEED();
}

template <phone_number P>
constexpr std::uint16_t code_of = P.country_code();

TEST(PhoneNumberType, IsStructuralSoItCanBeATemplateArgument)
{
    static_assert(code_of<"+61412345678"_e164> == 61);
    static_assert(code_of<"+1 650 253 0000"_e164> == 1);
    SUCCEED();
}

TEST(PhoneNumberType, ValueInitialisedIsEmpty)
{
    constexpr phone_number p{};
    static_assert(p.empty() && p.digit_count() == 0 && p.country_code() == 0 && p.country_code_length() == 0);
    static_assert(p.national_number_length() == 0 && p.national_number().size() == 0);
    static_assert(snicholls::to_text(p).size() == 0);
    EXPECT_EQ(snicholls::to_string(p), "");
    char buf[1];
    const auto r = snicholls::to_chars(buf, buf, p);
    EXPECT_EQ(r.ec, std::errc{});
    EXPECT_EQ(r.ptr, buf);
}

TEST(PhoneNumberType, EveryMemberIsNoexcept)
{
    constexpr phone_number p{};
    static_assert(noexcept(parse_phone_number("+61412345678"sv)));
    static_assert(noexcept(p.empty()) && noexcept(p.digit_count()) && noexcept(p.country_code()));
    static_assert(noexcept(p.country_code_length()) && noexcept(p.national_number_length()));
    static_assert(noexcept(p.national_number()) && noexcept(p.digit(0)));
    static_assert(noexcept(snicholls::to_text(p)) && noexcept(snicholls::hash_value(p)));
    static_assert(noexcept(snicholls::to_chars(nullptr, nullptr, p)));
    static_assert(noexcept(p == p) && noexcept(p <=> p) && noexcept(std::hash<phone_number>{}(p)));
    static_assert(noexcept(snicholls::is_e164_country_code(61)));
    SUCCEED();
}

// ---- parsing: what is accepted ---------------------------------------------------------------------------

TEST(PhoneNumberParse, AcceptsTheWrittenFormsOfOneNumber)
{
    constexpr phone_number expected = must("+61412345678");
    static_assert(must("+61 412 345 678") == expected);
    static_assert(must("+61-412-345-678") == expected);
    static_assert(must("(+61) 412345678") == expected);
    static_assert(must("00 61 412 345 678") == expected);
    static_assert(must("0061412345678") == expected);
    static_assert(must("  +61 412 345 678  ") == expected);
    static_assert(must("+61.412.345.678") == expected);
    static_assert(must("+61/412/345/678") == expected);
    static_assert(must("+ 61 (412) 345-678") == expected);
    static_assert(must("+61 (412) 345 678") == expected);
    EXPECT_EQ(snicholls::to_text(expected).view(), "+61412345678");
}

TEST(PhoneNumberParse, SplitsCountryCodeAndNationalNumber)
{
    constexpr phone_number au = must("+61 412 345 678");
    static_assert(au.country_code() == 61 && au.country_code_length() == 2);
    static_assert(au.national_number().view() == "412345678" && au.national_number_length() == 9);
    static_assert(au.digit_count() == 11);
    constexpr phone_number us = must("+1 (650) 253-0000");
    static_assert(us.country_code() == 1 && us.national_number().view() == "6502530000");
    constexpr phone_number ie = must("+353 1 234 5678");
    static_assert(ie.country_code() == 353 && ie.country_code_length() == 3);
    constexpr phone_number intl = must("+800 1234 5678");   // a non-geographic code (freephone)
    static_assert(intl.country_code() == 800 && intl.national_number().view() == "12345678");
    EXPECT_EQ(std::string(au.national_number().c_str()), "412345678");
}

TEST(PhoneNumberParse, SharedZonesKeepTheirOneCode)
{
    static_assert(must("+1 416 555 0123").country_code() == 1);   // Canada, in the North American plan
    static_assert(must("+1 876 555 0123").country_code() == 1);   // Jamaica
    static_assert(must("+7 495 123 45 67").country_code() == 7);   // Russia
    static_assert(must("+7 701 123 4567").country_code() == 7);   // Kazakhstan
    SUCCEED();
}

TEST(PhoneNumberParse, KeepsALeadingZeroThatBelongsToTheNumber)
{
    constexpr phone_number it = must("+39 06 6988 4444");   // Italy: the 0 is part of the number
    static_assert(it.national_number().view() == "0669884444");
    static_assert(must("+225 01 23 45 67 89").national_number().view() == "0123456789");   // Côte d'Ivoire
    EXPECT_EQ(snicholls::to_text(it).view(), "+390669884444");
}

TEST(PhoneNumberParse, KeepsOtherTrunkPrefixesAsWritten)
{
    // E.164 structure only: "+1 1..." is not a valid North American number, but it is not a trunk 0
    static_assert(must("+1 1650 253 000").national_number().view() == "1650253000");
    static_assert(must("+7 8495 123 456").national_number().view() == "8495123456");
    SUCCEED();
}

TEST(PhoneNumberParse, AcceptsFifteenDigitsAndTheShortestNumber)
{
    constexpr phone_number longest = must("+881 234 567 890 123");
    static_assert(longest.digit_count() == 15 && longest.country_code() == 881);
    static_assert(must("+123456789012345").digit_count() == 15);
    constexpr phone_number shortest = must("+1 23");
    static_assert(shortest.digit_count() == 3 && shortest.national_number().view() == "23");
    EXPECT_EQ(snicholls::to_text(longest).view(), "+881234567890123");
    EXPECT_EQ(snicholls::to_text(longest).size(), snicholls::phone_number_max_text_size);
}

// ---- parsing: every error --------------------------------------------------------------------------------

TEST(PhoneNumberParse, RefusesNothingButSeparatorsAsEmpty)
{
    static_assert(error_of("") == phone_errc::empty);
    static_assert(error_of("   ") == phone_errc::empty);
    static_assert(error_of("( - . / )") == phone_errc::empty);
    SUCCEED();
}

TEST(PhoneNumberParse, NeverGuessesARegion)
{
    static_assert(error_of("0412 345 678") == phone_errc::missing_country_code);
    static_assert(error_of("61412345678") == phone_errc::missing_country_code);
    static_assert(error_of("011 61 412 345 678") == phone_errc::missing_country_code);   // only 00 is the prefix
    static_assert(error_of("0 0 61 412 345 678") == phone_errc::missing_country_code);   // 00 is two adjacent zeros
    static_assert(error_of("0") == phone_errc::missing_country_code);
    SUCCEED();
}

TEST(PhoneNumberParse, RefusesLettersExtensionsAndUris)
{
    static_assert(error_of("+1-800-FLOWERS") == phone_errc::invalid_character);
    static_assert(error_of("+61 412 345 678 ext. 9") == phone_errc::invalid_character);
    static_assert(error_of("+61 412 345 678 x9") == phone_errc::invalid_character);
    static_assert(error_of("tel:+61-412-345-678") == phone_errc::invalid_character);
    static_assert(error_of("+61 412 345 678;ext=9") == phone_errc::invalid_character);
    static_assert(error_of("+61\t412345678") == phone_errc::invalid_character);
    static_assert(error_of("+61++412345678") == phone_errc::invalid_character);
    static_assert(error_of("00+61412345678") == phone_errc::invalid_character);
    static_assert(error_of("+61 412 345 678+") == phone_errc::invalid_character);
    EXPECT_EQ(error_of("+61 412\0 345 678"sv), phone_errc::invalid_character);
    EXPECT_EQ(error_of("+61 \xE2\x80\x93 412 345 678"), phone_errc::invalid_character);   // an en dash
}

TEST(PhoneNumberParse, RefusesUnbalancedParentheses)
{
    static_assert(error_of("(+61 412 345 678") == phone_errc::unbalanced_parentheses);
    static_assert(error_of("+61) 412 345 678") == phone_errc::unbalanced_parentheses);
    static_assert(error_of("((+61)) 412 345 678") == phone_errc::unbalanced_parentheses);
    static_assert(error_of("+61 (412)) 345 678") == phone_errc::unbalanced_parentheses);
    SUCCEED();
}

TEST(PhoneNumberParse, RefusesUnknownCountryCodes)
{
    static_assert(error_of("+999 1234 5678") == phone_errc::unknown_country_code);
    static_assert(error_of("+28 1234 5678") == phone_errc::unknown_country_code);   // 28x is spare
    static_assert(error_of("+0 123 4567") == phone_errc::unknown_country_code);
    static_assert(error_of("+007 495 123 4567") == phone_errc::unknown_country_code);   // never read as +7
    static_assert(error_of("+00 61 412") == phone_errc::unknown_country_code);
    static_assert(error_of("+99") == phone_errc::unknown_country_code);
    SUCCEED();
}

TEST(PhoneNumberParse, RefusesATrunkZeroAfterTheCountryCode)
{
    static_assert(error_of("+61 0412 345 678") == phone_errc::trunk_prefix);
    static_assert(error_of("+44 (0)20 7946 0000") == phone_errc::trunk_prefix);
    static_assert(error_of("+49 030 123456") == phone_errc::trunk_prefix);
    static_assert(error_of("+353 01 234 5678") == phone_errc::trunk_prefix);
    static_assert(error_of("0044 020 7946 0000") == phone_errc::trunk_prefix);
    SUCCEED();
}

TEST(PhoneNumberParse, RefusesFewerThanTwoDigitsAfterTheCode)
{
    static_assert(error_of("+") == phone_errc::too_short);
    static_assert(error_of("00") == phone_errc::too_short);
    static_assert(error_of("+1") == phone_errc::too_short);
    static_assert(error_of("+12") == phone_errc::too_short);
    static_assert(error_of("+61 4") == phone_errc::too_short);
    static_assert(error_of("+353 1") == phone_errc::too_short);
    static_assert(error_of("+61 0") == phone_errc::too_short);   // short before it is a trunk 0
    SUCCEED();
}

TEST(PhoneNumberParse, RefusesMoreThanFifteenDigits)
{
    static_assert(error_of("+1234567890123456") == phone_errc::too_long);
    static_assert(error_of("+61 412 345 678 901 234") == phone_errc::too_long);
    static_assert(error_of("+0000000000000000000000") == phone_errc::too_long);
    // a letter is reported before the length, wherever it stands
    static_assert(error_of("+1234567890123456 x") == phone_errc::invalid_character);
    EXPECT_EQ(error_of(std::string(1000, '1').insert(0, "+")), phone_errc::too_long);
}

// ---- text out --------------------------------------------------------------------------------------------

TEST(PhoneNumberText, ToCharsWritesTheCanonicalFormIntoAnExactBuffer)
{
    constexpr phone_number p = "+44 20 7946 0000"_e164;
    std::array<char, 13> buf{};
    const auto r = snicholls::to_chars(buf.data(), buf.data() + buf.size(), p);
    ASSERT_EQ(r.ec, std::errc{});
    EXPECT_EQ(std::string_view(buf.data(), static_cast<std::size_t>(r.ptr - buf.data())), "+442079460000");
    EXPECT_EQ(r.ptr, buf.data() + buf.size());
}

TEST(PhoneNumberText, ToCharsRefusesABufferOneTooSmall)
{
    constexpr phone_number p = "+61412345678"_e164;
    std::array<char, 11> buf{};
    const auto r = snicholls::to_chars(buf.data(), buf.data() + buf.size(), p);
    EXPECT_EQ(r.ec, std::errc::value_too_large);
    EXPECT_EQ(r.ptr, buf.data() + buf.size());
}

TEST(PhoneNumberText, IsConstexprAndRoundTrips)
{
    static_assert(snicholls::to_text("+61 412 345 678"_e164).view() == "+61412345678");
    static_assert(std::string_view(snicholls::to_text("+390669884444"_e164)) == "+390669884444");
    static_assert(must(snicholls::to_text("00 1 650 253 0000"_e164).view()) == "+16502530000"_e164);
    constexpr phone_number p = "+44 20 7946 0000"_e164;
    static_assert(p.digit(0) == '4' && p.digit(1) == '4' && p.digit(2) == '2' && p.digit(12) == '0');
    EXPECT_EQ(snicholls::to_string(p), "+442079460000");
    EXPECT_EQ(std::string(snicholls::to_text(p).c_str()), "+442079460000");
}

TEST(PhoneNumberText, PrintsItselfThroughAcoutAndPlainStreams)
{
    std::ostringstream os;
    "+61412345678"_e164.print(os);
    EXPECT_EQ(os.str(), "+61412345678");
    using snicholls::print_operators::operator<<;
    std::ostringstream os2;
    os2 << "+1 650 253 0000"_e164;
    EXPECT_EQ(os2.str(), "+16502530000");
    std::ostringstream os3;
    snicholls::print_to(os3, std::vector<phone_number>{"+61412345678"_e164, "+16502530000"_e164});
    EXPECT_EQ(os3.str(), "[+61412345678, +16502530000]");
}

// ---- ordering and hashing --------------------------------------------------------------------------------

TEST(PhoneNumberOrder, IsTheOrderOfTheCanonicalText)
{
    static_assert("+16502530000"_e164 < "+442079460000"_e164);
    static_assert("+442079460000"_e164 < "+61412345678"_e164);
    static_assert("+6141"_e164 < "+61410"_e164);   // a prefix sorts first
    static_assert("+61410"_e164 < "+6142"_e164);
    static_assert(("+61412345678"_e164 <=> "+61 412 345 678"_e164) == std::strong_ordering::equal);
    static_assert(phone_number{} < "+123"_e164);
    std::vector<std::string> texts{"+61412345678", "+16502530000", "+442079460000", "+390669884444", "+6141",
                                   "+61410", "+8001234567", "+79161234567", "+88123456789012"};
    std::set<phone_number> numbers;
    for (const auto& t : texts)
        numbers.insert(must(t));
    std::set<std::string> sorted(texts.begin(), texts.end());
    std::vector<std::string> by_number;
    for (const auto& p : numbers)
        by_number.push_back(snicholls::to_string(p));
    EXPECT_EQ(by_number, std::vector<std::string>(sorted.begin(), sorted.end()));
}

TEST(PhoneNumberHash, AgreesForEqualNumbersAndSpreadsOthers)
{
    static_assert(snicholls::hash_value("+61 412 345 678"_e164) == snicholls::hash_value("0061412345678"_e164));
    static_assert(std::hash<phone_number>{}("+61412345678"_e164) == snicholls::hash_value("+61412345678"_e164));
    std::unordered_set<std::size_t> hashes;
    std::unordered_set<phone_number> numbers;
    for (int i = 0; i < 1000; ++i) {
        const phone_number p = must("+61 4" + std::to_string(10000000 + i));
        hashes.insert(std::hash<phone_number>{}(p));
        numbers.insert(p);
    }
    EXPECT_EQ(hashes.size(), 1000U);
    EXPECT_EQ(numbers.count(must("+61 410 000 042")), 1U);
}

// ---- the country code table ------------------------------------------------------------------------------

TEST(PhoneNumberCountryCode, KnowsTheAssignedCodes)
{
    static_assert(snicholls::is_e164_country_code(1) && snicholls::is_e164_country_code(7));
    static_assert(snicholls::is_e164_country_code(44) && snicholls::is_e164_country_code(61));
    static_assert(snicholls::is_e164_country_code(800) && snicholls::is_e164_country_code(998));
    static_assert(!snicholls::is_e164_country_code(0) && !snicholls::is_e164_country_code(28));
    static_assert(!snicholls::is_e164_country_code(999) && !snicholls::is_e164_country_code(1000));
    static_assert(!snicholls::is_e164_country_code(~0U));
    int assigned = 0;
    for (unsigned c = 0; c < 1000; ++c)
        assigned += snicholls::is_e164_country_code(c) ? 1 : 0;
    EXPECT_EQ(assigned, static_cast<int>(std::size(snicholls::detail::e164::country_codes)));
}

// ---- the literal -----------------------------------------------------------------------------------------

TEST(PhoneNumberLiteral, IsCheckedAtCompileTime)
{
    // An invalid literal does not compile: tests/compile_fail/phone_number_literal_invalid.cpp
    constexpr phone_number p = "+61 412 345 678"_e164;
    static_assert(p == must("+61412345678"));
    static_assert("00 44 20 7946 0000"_e164.country_code() == 44);
    SUCCEED();
}

// ---- agreement with phonenumbers -------------------------------------------------------------------------

struct oracle_row
{
    std::size_t line;
    std::string kind, input, oracle, country_code, national_number, trunk_zero;
};

std::vector<oracle_row> read_oracle()
{
    const std::string path = std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/e164/parse.tsv";
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("e164 fixture not found: " + path);
    std::vector<oracle_row> rows;
    std::string text;
    std::size_t line = 0;
    while (std::getline(in, text)) {
        ++line;
        if (!text.empty() && text.back() == '\r') text.pop_back();
        if (text.empty() || text.front() == '#') continue;
        std::vector<std::string> f;
        std::size_t start = 0;
        for (;;) {
            const std::size_t tab = text.find('\t', start);
            f.push_back(text.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
            if (tab == std::string::npos) break;
            start = tab + 1;
        }
        if (f.size() != 6) throw std::runtime_error("e164 fixture line " + std::to_string(line) + ": 6 fields expected");
        rows.push_back({line, f[0], f[1], f[2], f[3], f[4], f[5]});
    }
    if (rows.empty()) throw std::runtime_error("e164 fixture is empty: " + path);
    return rows;
}

// The digits of an input as E.164 reads them: every ASCII digit, less an international prefix 00.
std::string digits_of(const std::string& input)
{
    std::string d;
    for (const char c : input)
        if (c >= '0' && c <= '9') d += c;
    const std::size_t first = input.find_first_not_of(" ()-./");
    if (first != std::string::npos && input.compare(first, 2, "00") == 0) d.erase(0, 2);
    return d;
}

bool has_other_character(const std::string& input)
{
    return input.find_first_not_of("0123456789 ()-./+") != std::string::npos;
}

TEST(PhoneNumberOracle, AgreesWithPhonenumbersOnEveryFixture)
{
    // How core's result follows from phonenumbers' (parse with no region, format E164), row by row:
    //   phonenumbers refuses              -> core refuses (with the matching phone_errc where there is one)
    //   a letter or another character     -> invalid_character (phonenumbers reads letters and extensions)
    //   more than 15 digits written       -> too_long (phonenumbers allows a 17-digit national number)
    //   a 0 after a trunk-zero code       -> trunk_prefix, whether phonenumbers removed it or (when the rest would
    //                                        be too short for the region) kept it
    //   phonenumbers keeps every digit    -> core gives the same E.164 text, country code and national number
    //   phonenumbers removed a prefix     -> another trunk prefix or carrier code (1, 8, 06, ...): core keeps the
    //                                        digits as written, since E.164 structure cannot tell them apart
    std::size_t refused = 0, letters = 0, long_ones = 0, same = 0, trunk = 0, kept = 0;
    for (const auto& row : read_oracle()) {
        SCOPED_TRACE("tests/fixtures/e164/parse.tsv:" + std::to_string(row.line) + " " + row.kind + " \"" +
                     row.input + "\" phonenumbers " + row.oracle);
        const auto r = parse_phone_number(row.input);
        const std::string digits = digits_of(row.input);
        if (row.oracle.front() != '+') {
            ++refused;
            ASSERT_FALSE(r.is_ok()) << snicholls::to_string(*r);
            if (has_other_character(row.input)) continue;
            if (row.oracle == "INVALID_COUNTRY_CODE") {
                EXPECT_TRUE(r.err() == phone_errc::missing_country_code || r.err() == phone_errc::unknown_country_code);
            }
            if (row.oracle == "TOO_SHORT_NSN") {
                EXPECT_EQ(r.err(), phone_errc::too_short);
            }
            if (row.oracle == "TOO_LONG") {
                EXPECT_EQ(r.err(), phone_errc::too_long);
            }
            continue;
        }
        if (has_other_character(row.input)) {
            ++letters;
            ASSERT_FALSE(r.is_ok());
            EXPECT_EQ(r.err(), phone_errc::invalid_character);
            continue;
        }
        if (digits.size() > 15) {
            ++long_ones;
            ASSERT_FALSE(r.is_ok());
            EXPECT_EQ(r.err(), phone_errc::too_long);
            continue;
        }
        const std::string oracle_digits = row.oracle.substr(1);
        ASSERT_EQ(digits.compare(0, row.country_code.size(), row.country_code), 0);
        if (row.trunk_zero == "1" && digits.size() > row.country_code.size() && digits[row.country_code.size()] == '0') {
            ++trunk;
            ASSERT_FALSE(r.is_ok());
            EXPECT_EQ(r.err(), phone_errc::trunk_prefix);
            continue;
        }
        if (digits == oracle_digits) {
            ++same;
            ASSERT_TRUE(r.is_ok()) << static_cast<int>(r.unwrap_err_unchecked());
            EXPECT_EQ(snicholls::to_text(*r).view(), row.oracle);
            EXPECT_EQ(std::to_string(r->country_code()), row.country_code);
            EXPECT_EQ(r->national_number().view(), row.national_number);
            continue;
        }
        // phonenumbers removed digits between the country code and its national number
        ASSERT_GT(digits.size(), oracle_digits.size());
        ++kept;
        ASSERT_TRUE(r.is_ok()) << static_cast<int>(r.unwrap_err_unchecked());
        EXPECT_EQ(snicholls::to_text(*r).view(), "+" + digits);
        EXPECT_EQ(std::to_string(r->country_code()), row.country_code);
    }
    // every rule is exercised, so a fixture that stopped covering one fails here
    EXPECT_GT(refused, 200U);
    EXPECT_GT(letters, 50U);
    EXPECT_GT(long_ones, 50U);
    EXPECT_GT(same, 1500U);
    EXPECT_GT(trunk, 200U);
    EXPECT_GE(kept, 5U);
}

} // namespace
