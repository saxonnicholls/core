//
//  check_digits_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ids/check_digits.hpp>: each scheme validated and its check digits computed, at compile
//  time and at run time, with the typed error of a mistyped identifier. Exits non-zero if a verdict is wrong.
//

#include <snicholls/core/ids/check_digits.hpp>

#include <array>
#include <cstdio>
#include <string>
#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
const char* why(snicholls::identifier_errc e)
{
    switch (e) {
    case snicholls::identifier_errc::wrong_length: return "wrong length";
    case snicholls::identifier_errc::invalid_character: return "invalid character";
    case snicholls::identifier_errc::invalid_format: return "invalid format";
    case snicholls::identifier_errc::invalid_check_digit: return "invalid check digit";
    }
    return "?";
}
} // namespace

int main()
{
    std::printf("snicholls-core check_digits demo\n");
    using namespace snicholls;

    //! [check_digits]
    static_assert(validate_isin("US0378331005"));                    // Apple
    static_assert(validate_iban("GB82WEST12345698765432"));
    static_assert(validate_lei("5493001KJTIIGC8Y1R12"));
    static_assert(validate_cusip("037833100") && validate_sedol("B0YBKJ7") && validate_luhn("4111111111111111"));

    // a mistyped identifier is refused with the reason
    const auto r = validate_isin("US0378331006");
    // check digits for a new identifier's body
    constexpr auto iban = iban_check_digits("DE", "370400440532013000");   // "89"
    //! [check_digits]

    if (!r)
        std::printf("  US0378331006: %s\n", why(r.unwrap_err_unchecked()));
    for (const std::string_view s : {"us0378331005", "US037833100", "1S0378331005"}) {
        const auto e = validate_isin(s);
        std::printf("  %-14.*s %s\n", static_cast<int>(s.size()), s.data(), e ? "valid" : why(e.unwrap_err_unchecked()));
    }
    check(!r && r.unwrap_err_unchecked() == identifier_errc::invalid_check_digit, "a wrong ISIN check digit");
    check(iban && (*iban)[0] == '8' && (*iban)[1] == '9', "IBAN check digits of a German BBAN");
    check(*isin_check_digit("AU0000XVGZA") == '3' && *cusip_check_digit("38259P50") == '8' && *sedol_check_digit("026349") == '4',
          "single check digits");
    check(*luhn_check_digit("411111111111111") == '1', "Luhn");
    const auto lei = lei_check_digits("5493001KJTIIGC8Y1R");
    check(lei && std::string(lei->data(), 2) == "12", "LEI check digits");
    check(validate_iban("GB82WEST1234569876543").unwrap_err_unchecked() == identifier_errc::invalid_check_digit,
          "a digit dropped from an IBAN");
    check(validate_sedol("0B63494").unwrap_err_unchecked() == identifier_errc::invalid_format, "a SEDOL mixing styles");
    check(validate_lei("5493001KJTIIGC8Y1R1").unwrap_err_unchecked() == identifier_errc::wrong_length, "a short LEI");
    check(validate_cusip("03783310!").unwrap_err_unchecked() == identifier_errc::invalid_character, "a CUSIP with '!'");
    return failures == 0 ? 0 : 1;
}
