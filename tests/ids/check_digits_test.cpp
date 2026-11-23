//
//  check_digits_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ids/check_digits.hpp>: every scheme against python-stdnum 2.2 on the fixture
//  tests/fixtures/ids/check_digits.txt (scripts/generate_check_digit_fixtures.py, never core's own output), the
//  errors at compile time, and the places core is stricter than stdnum.
//
//  How core's answer follows from a row (scheme, input, stdnum's verdict, stdnum's check digits):
//  - core accepts the input exactly when stdnum does, except:
//    - an ISIN whose country stdnum does not know (InvalidComponent): core does not check the country, so it
//      accepts when the check digit matches;
//    - a LEI of a length other than 20, or without digits in its last two places: stdnum checks MOD 97-10 alone;
//    - lower case: stdnum upper-cases its input first, core refuses it (invalid_character);
//    - an IBAN's verdict is stdnum's MOD 97-10 check alone; core also refuses check digits 00, 01 and 99;
//  - whenever both compute check digits for the input's body, they are the same.
//

#include <snicholls/core/ids/check_digits.hpp>

#include "../support/fixture_file.hpp"

#include <gtest/gtest.h>

#include <array>
#include <map>
#include <string>
#include <utility>
#include <string_view>

namespace {

using namespace snicholls;

// compile time: the textbook examples
static_assert(validate_luhn("79927398713") && !validate_luhn("79927398710"));
static_assert(*luhn_check_digit("7992739871") == '3' && *luhn_check_digit("") == '0');
static_assert(validate_isin("US0378331005") && *isin_check_digit("US037833100") == '5');
static_assert(validate_isin("AU0000XVGZA3") && *isin_check_digit("AU0000XVGZA") == '3');
static_assert(validate_iban("GB82WEST12345698765432"));
static_assert(*iban_check_digits("GB", "WEST12345698765432") == std::array<char, 2>{'8', '2'});
static_assert(validate_lei("5493001KJTIIGC8Y1R12") && *lei_check_digits("5493001KJTIIGC8Y1R") == std::array<char, 2>{'1', '2'});
static_assert(validate_cusip("037833100") && *cusip_check_digit("38259P50") == '8');
static_assert(validate_sedol("0263494") && validate_sedol("B0YBKJ7") && *sedol_check_digit("B0YBKJ") == '7');

// the errors
constexpr identifier_errc error_of(result<void, identifier_errc> r) { return r.unwrap_err_unchecked(); }
static_assert(error_of(validate_luhn("")) == identifier_errc::wrong_length);
static_assert(error_of(validate_luhn("7992 7398 713")) == identifier_errc::invalid_character);
static_assert(error_of(validate_isin("US037833100")) == identifier_errc::wrong_length);
static_assert(error_of(validate_isin("us0378331005")) == identifier_errc::invalid_character);
static_assert(error_of(validate_isin("120378331005")) == identifier_errc::invalid_format);
static_assert(error_of(validate_isin("US037833100X")) == identifier_errc::invalid_format);
static_assert(error_of(validate_isin("US0378331006")) == identifier_errc::invalid_check_digit);
static_assert(error_of(validate_iban("GB82")) == identifier_errc::wrong_length);
static_assert(error_of(validate_iban("GB82WEST1234569876543212345678901234")) == identifier_errc::wrong_length);
static_assert(error_of(validate_iban("GB82 WEST12345698765432")) == identifier_errc::invalid_character);
static_assert(error_of(validate_iban("G182WEST12345698765432")) == identifier_errc::invalid_format);
static_assert(error_of(validate_iban("GBX2WEST12345698765432")) == identifier_errc::invalid_format);
static_assert(error_of(validate_iban("GB83WEST12345698765432")) == identifier_errc::invalid_check_digit);
static_assert(error_of(validate_iban("GB99WEST12345698765432")) == identifier_errc::invalid_check_digit);   // = 02 mod 97
static_assert(error_of(validate_lei("5493001KJTIIGC8Y1R1")) == identifier_errc::wrong_length);
static_assert(error_of(validate_lei("5493001KJTIIGC8Y1R1A")) == identifier_errc::invalid_format);
static_assert(error_of(validate_lei("5493001KJTIIGC8Y1R13")) == identifier_errc::invalid_check_digit);
static_assert(error_of(validate_cusip("03783310")) == identifier_errc::wrong_length);
static_assert(error_of(validate_cusip("03783310!")) == identifier_errc::invalid_character);
static_assert(error_of(validate_cusip("03783310A")) == identifier_errc::invalid_format);
static_assert(error_of(validate_cusip("037833101")) == identifier_errc::invalid_check_digit);
static_assert(error_of(validate_sedol("026349")) == identifier_errc::wrong_length);
static_assert(error_of(validate_sedol("A263494")) == identifier_errc::invalid_character);   // a vowel
static_assert(error_of(validate_sedol("0B63494")) == identifier_errc::invalid_format);      // a letter after a digit
static_assert(error_of(validate_sedol("026349B")) == identifier_errc::invalid_format);
static_assert(error_of(validate_sedol("0263495")) == identifier_errc::invalid_check_digit);
static_assert(iban_check_digits("G1", "WEST").unwrap_err_unchecked() == identifier_errc::invalid_format);
static_assert(iban_check_digits("GB", "").unwrap_err_unchecked() == identifier_errc::wrong_length);
static_assert(lei_check_digits("5493001KJTIIGC8Y1r").unwrap_err_unchecked() == identifier_errc::invalid_character);
static_assert(luhn_check_digit("12a").unwrap_err_unchecked() == identifier_errc::invalid_character);
static_assert(noexcept(validate_iban(std::declval<std::string_view>())) && noexcept(sedol_check_digit(std::declval<std::string_view>())));

bool strict(std::string_view s)
{
    for (const char c : s)
        if (c >= 'a' && c <= 'z')
            return false;
    return true;
}

// core's verdict and its check digits for the input's body, as text ("-" when it computes none)
std::pair<bool, std::string> core(const std::string& scheme, const std::string& s)
{
    const auto one = [](const result<char, identifier_errc>& r) { return r ? std::string(1, *r) : std::string("-"); };
    const auto two = [](const result<std::array<char, 2>, identifier_errc>& r) {
        return r ? std::string(r->data(), 2) : std::string("-");
    };
    if (scheme == "luhn")
        return {validate_luhn(s).is_ok(), s.empty() ? "-" : one(luhn_check_digit(std::string_view(s).substr(0, s.size() - 1)))};
    if (scheme == "isin")
        return {validate_isin(s).is_ok(), one(isin_check_digit(std::string_view(s).substr(0, 11)))};
    if (scheme == "iban")
        return {validate_iban(s).is_ok(),
                s.size() < 4 ? "-" : two(iban_check_digits(std::string_view(s).substr(0, 2), std::string_view(s).substr(4)))};
    if (scheme == "lei")
        return {validate_lei(s).is_ok(), two(lei_check_digits(std::string_view(s).substr(0, 18)))};
    if (scheme == "cusip")
        return {validate_cusip(s).is_ok(), one(cusip_check_digit(std::string_view(s).substr(0, 8)))};
    return {validate_sedol(s).is_ok(), one(sedol_check_digit(std::string_view(s).substr(0, 6)))};
}

bool expected_valid(const std::string& scheme, const std::string& s, const std::string& verdict, const std::string& check)
{
    if (!strict(s))
        return false;
    if (scheme == "isin" && verdict == "InvalidComponent")
        return check.size() == 1 && check[0] == s.back();
    if (scheme == "lei")
        return verdict == "valid" && s.size() == 20 && s[18] >= '0' && s[18] <= '9' && s[19] >= '0' && s[19] <= '9';
    if (scheme == "iban") {
        const std::string digits = s.size() >= 4 ? s.substr(2, 2) : "";
        return verdict == "valid" && s.size() >= 5 && s.size() <= 34 && digits != "00" && digits != "01" && digits != "99" &&
               s[0] >= 'A' && s[0] <= 'Z' && s[1] >= 'A' && s[1] <= 'Z';
    }
    return verdict == "valid";
}

TEST(IdsCheckDigits, AgreesWithPythonStdnum)
{
    const auto rows = snicholls::test::read_fixture("ids/check_digits.txt");
    std::map<std::string, int> valid, invalid, compared;
    for (const auto& row : rows) {
        ASSERT_GE(row.fields.size(), 4U) << "line " << row.line;
        const std::string& scheme = row.fields[0];
        const std::string& s = row.fields[1];
        SCOPED_TRACE("tests/fixtures/ids/check_digits.txt:" + std::to_string(row.line) + " " + scheme + " " + s);
        const auto [ok, check] = core(scheme, s);
        const bool expected = expected_valid(scheme, s, row.fields[2], row.fields[3]);
        EXPECT_EQ(ok, expected) << "stdnum: " << row.fields[2];
        ++(expected ? valid : invalid)[scheme];
        if (check != "-" && row.fields[3] != "-") {
            EXPECT_EQ(check, row.fields[3]);
            ++compared[scheme];
        }
    }
    for (const char* scheme : {"luhn", "isin", "iban", "lei", "cusip", "sedol"}) {
        EXPECT_GT(valid[scheme], 30) << scheme;
        EXPECT_GT(invalid[scheme], 60) << scheme;
        EXPECT_GT(compared[scheme], 40) << scheme;
    }
}

TEST(IdsCheckDigits, ComputedCheckDigitsValidate)
{
    // every body of a pseudo-random sequence, completed with core's check digits, validates; one change does not
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    const auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    constexpr std::string_view alnum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    for (int i = 0; i < 2000; ++i) {
        std::string body = "GB";
        for (int k = 0; k < 9; ++k) body += alnum[next() % alnum.size()];
        const std::string isin = body + *isin_check_digit(body);
        EXPECT_TRUE(validate_isin(isin)) << isin;
        std::string lei;
        for (int k = 0; k < 18; ++k) lei += alnum[next() % alnum.size()];
        const auto ck = *lei_check_digits(lei);
        lei += std::string(ck.data(), 2);
        EXPECT_TRUE(validate_lei(lei)) << lei;
        // MOD 97-10 catches every substitution of a digit by a digit or a letter by a letter: the number changes
        // by d * 10^k, which 97, a prime, never divides
        std::string broken = lei;
        char& c = broken[next() % 18];
        c = c >= 'A' ? static_cast<char>(c == 'Z' ? 'A' : c + 1) : static_cast<char>(c == '9' ? '0' : c + 1);
        EXPECT_FALSE(validate_lei(broken)) << broken;
    }
}

} // namespace
