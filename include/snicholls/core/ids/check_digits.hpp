//
//  check_digits.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Check digits of financial identifiers, validated and computed at compile time: Luhn, ISIN (ISO 6166),
/// IBAN (ISO 13616, MOD 97-10), LEI (ISO 17442, MOD 97-10), CUSIP and SEDOL.
/// \ingroup core
///
/// **Overview.** An identifier typed, pasted or received is checked before it is trusted: one wrong character
/// and the check digits no longer match. For each scheme there is a `validate_<scheme>(text)`, which checks the
/// shape and the check digits and returns `result<void, identifier_errc>`, and a `<scheme>_check_digit(s)`, which
/// computes them for the identifier's body:
/// - Luhn (ISO/IEC 7812-1): a string of decimal digits whose last is the Luhn check digit (card numbers, IMEIs);
/// - ISIN: two letters, nine letters or digits, one Luhn check digit over the letters spelled as numbers (A = 10);
/// - IBAN: two letters (the country), two check digits, a basic bank account number (BBAN) of 1 to 30 letters or
///   digits; the whole read as a number with the first four characters moved to the end is 1 modulo 97;
/// - LEI: eighteen letters or digits and two check digits, the whole 1 modulo 97 (ISO 7064 MOD 97-10);
/// - CUSIP: eight letters, digits, `*`, `@` or `#` and one check digit (doubling every second value);
/// - SEDOL: six digits or consonants and one check digit (weights 1, 3, 1, 7, 3, 9); a SEDOL starting with a digit
///   is digits throughout.
///
/// Text is strict: upper case only, no spaces or separators. Strip and upper-case before validating.
///
/// \snippet ids/check_digits_demo.cpp check_digits
///
/// **Relationships.** `result.hpp`. `ids/` holds identifiers and their checks; `uuid.hpp` and `phone_number.hpp`
/// are identifiers with types of their own. Group: \ref core.
///
/// **Implementations.** One per scheme.
///
/// **Guarantees.** constexpr, `noexcept`, O(n), no allocation; MOD 97 is computed digit by digit, so no
/// identifier is too long for the arithmetic.
///
/// **Errors.** `identifier_errc`:
/// - `wrong_length`: the scheme's length is not met;
/// - `invalid_character`: a character the scheme never uses (a lower-case letter, a space, a vowel in a SEDOL);
/// - `invalid_format`: a usable character in the wrong place (a digit in an ISIN's or IBAN's country, a letter in
///   a check digit, a letter after a SEDOL's leading digit);
/// - `invalid_check_digit`: the check digits do not match, or an IBAN's are 00, 01 or 99, which ISO 13616 never
///   issues.
///
/// **Performance.** `benchmarks/ids/check_digits_benchmark.cpp`: 4,096 identifiers, a quarter of them wrong, Release,
/// Apple clang 17, Xeon W-3245, 2026-11-23: `validate_isin` 265 us against 287 us for the same checks written by
/// hand, `validate_iban` (22 characters) 424 us against 434 us (about 65 and 104 ns an identifier).
///
/// **Tests and demos.** `tests/ids/check_digits_test.cpp`, against fixtures from python-stdnum 2.2
/// (`scripts/generate_check_digit_fixtures.py`, `tests/fixtures/ids/check_digits.txt`);
/// `demos/ids/check_digits_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstdint>`, `<string_view>`, `result.hpp`.
///
/// **Known issues.** Structure is checked, not registration:
/// - an ISIN's first two letters are not checked against ISO 3166 (python-stdnum does check them);
/// - an IBAN's BBAN length and format per country are not checked: they are data (the SWIFT IBAN Registry,
///   revised several times a year), which by core's rule stays with the caller;
/// - a LEI's issuer prefix is not checked.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_ids_check_digits_hpp
#define snicholls_core_ids_check_digits_hpp

#include <snicholls/core/result.hpp>

#include <array>
#include <cstdint>
#include <string_view>

namespace snicholls {

    /// \brief Why an identifier was refused.
    /// \ingroup core
    enum class identifier_errc : std::uint8_t {
        wrong_length = 1,      ///< the scheme's length is not met
        invalid_character,     ///< a character the scheme never uses
        invalid_format,        ///< a usable character in the wrong place
        invalid_check_digit,   ///< the check digits do not match
    };

    namespace detail::check_digits {
        constexpr bool digit(char c) noexcept { return c >= '0' && c <= '9'; }
        constexpr bool upper(char c) noexcept { return c >= 'A' && c <= 'Z'; }
        // 0-9 for a digit, 10-35 for A-Z
        constexpr int value(char c) noexcept { return digit(c) ? c - '0' : c - 'A' + 10; }

        constexpr bool all_digits(std::string_view s) noexcept
        {
            for (const char c : s)
                if (!digit(c))
                    return false;
            return true;
        }
        constexpr bool all_alphanumeric(std::string_view s) noexcept
        {
            for (const char c : s)
                if (!digit(c) && !upper(c))
                    return false;
            return true;
        }

        // Luhn over the digits of `s` read right to left (a letter A-Z as its two digits), doubling the rightmost
        // when `double_first`. The sum modulo 10.
        constexpr int luhn_sum(std::string_view s, bool double_first) noexcept
        {
            int sum = 0;
            bool twice = double_first;
            const auto add = [&](int d) noexcept {
                if (twice)
                    d = d * 2 > 9 ? d * 2 - 9 : d * 2;
                sum += d;
                twice = !twice;
            };
            for (std::size_t i = s.size(); i-- > 0;) {
                const int v = value(s[i]);
                add(v % 10);
                if (v > 9)
                    add(v / 10);
            }
            return sum % 10;
        }

        // ISO 7064 MOD 97-10 remainder of the number spelled by `s` (a letter A-Z as its two digits), from `r`.
        constexpr int mod97(std::string_view s, int r = 0) noexcept
        {
            for (const char c : s) {
                const int v = value(c);
                r = v > 9 ? (r * 100 + v) % 97 : (r * 10 + v) % 97;
            }
            return r;
        }

        constexpr std::array<char, 2> two_digits(int n) noexcept
        {
            return {static_cast<char>('0' + n / 10), static_cast<char>('0' + n % 10)};
        }

        // The IBAN remainder: BBAN, then country, then check digits.
        constexpr int iban_mod97(std::string_view country, std::string_view check, std::string_view bban) noexcept
        {
            return mod97(check, mod97(country, mod97(bban)));
        }

        // The CUSIP value of one character: digits, A-Z, then *, @, #; -1 for anything else.
        constexpr int cusip_value(char c) noexcept
        {
            return digit(c) || upper(c) ? value(c) : c == '*' ? 36 : c == '@' ? 37 : c == '#' ? 38 : -1;
        }

        constexpr bool sedol_character(char c) noexcept
        {
            return digit(c) || (upper(c) && c != 'A' && c != 'E' && c != 'I' && c != 'O' && c != 'U');
        }
    } // namespace detail::check_digits

    // ---- Luhn -------------------------------------------------------------------------------------------------------

    /// \brief The Luhn check digit of a string of decimal digits.
    /// \ingroup core
    /// \param payload The digits, without the check digit; may be empty.
    /// \return The check digit, or `invalid_character`.
    [[nodiscard]] constexpr result<char, identifier_errc> luhn_check_digit(std::string_view payload) noexcept
    {
        if (!detail::check_digits::all_digits(payload))
            return err(identifier_errc::invalid_character);
        return ok(static_cast<char>('0' + (10 - detail::check_digits::luhn_sum(payload, true)) % 10));
    }

    /// \brief Check a string of decimal digits whose last is its Luhn check digit.
    /// \ingroup core
    /// \param text The digits, at least one.
    /// \return Success, or `wrong_length`, `invalid_character` or `invalid_check_digit`.
    [[nodiscard]] constexpr result<void, identifier_errc> validate_luhn(std::string_view text) noexcept
    {
        if (text.empty())
            return err(identifier_errc::wrong_length);
        if (!detail::check_digits::all_digits(text))
            return err(identifier_errc::invalid_character);
        if (detail::check_digits::luhn_sum(text, false) != 0)
            return err(identifier_errc::invalid_check_digit);
        return ok();
    }

    // ---- ISIN -------------------------------------------------------------------------------------------------------

    /// \brief The check digit of an ISIN's first eleven characters.
    /// \ingroup core
    /// \param payload Two letters and nine letters or digits: `"US037833100"`.
    /// \return The check digit, or `wrong_length`, `invalid_character` or `invalid_format`.
    [[nodiscard]] constexpr result<char, identifier_errc> isin_check_digit(std::string_view payload) noexcept
    {
        using namespace detail::check_digits;
        if (payload.size() != 11)
            return err(identifier_errc::wrong_length);
        if (!all_alphanumeric(payload))
            return err(identifier_errc::invalid_character);
        if (!upper(payload[0]) || !upper(payload[1]))
            return err(identifier_errc::invalid_format);
        return ok(static_cast<char>('0' + (10 - luhn_sum(payload, true)) % 10));
    }

    /// \brief Check an ISIN (ISO 6166): `"US0378331005"`.
    /// \ingroup core
    /// \param text Twelve characters.
    /// \return Success, or the first rule it breaks.
    [[nodiscard]] constexpr result<void, identifier_errc> validate_isin(std::string_view text) noexcept
    {
        using namespace detail::check_digits;
        if (text.size() != 12)
            return err(identifier_errc::wrong_length);
        if (!all_alphanumeric(text))
            return err(identifier_errc::invalid_character);
        if (!upper(text[0]) || !upper(text[1]) || !digit(text[11]))
            return err(identifier_errc::invalid_format);
        if (luhn_sum(text, false) != 0)
            return err(identifier_errc::invalid_check_digit);
        return ok();
    }

    // ---- IBAN -------------------------------------------------------------------------------------------------------

    /// \brief The two check digits of an IBAN from its country and BBAN.
    /// \ingroup core
    /// \param country The two-letter country code: `"GB"`.
    /// \param bban    The basic bank account number, 1 to 30 letters or digits.
    /// \return The check digits, 02 to 98, or `wrong_length`, `invalid_character` or `invalid_format`.
    [[nodiscard]] constexpr result<std::array<char, 2>, identifier_errc> iban_check_digits(std::string_view country,
                                                                                         std::string_view bban) noexcept
    {
        using namespace detail::check_digits;
        if (country.size() != 2 || bban.empty() || bban.size() > 30)
            return err(identifier_errc::wrong_length);
        if (!all_alphanumeric(country) || !all_alphanumeric(bban))
            return err(identifier_errc::invalid_character);
        if (!upper(country[0]) || !upper(country[1]))
            return err(identifier_errc::invalid_format);
        return ok(two_digits(98 - iban_mod97(country, "00", bban)));
    }

    /// \brief Check an IBAN (ISO 13616) in its electronic form: `"GB82WEST12345698765432"`.
    /// \ingroup core
    /// \param text 5 to 34 characters, no spaces.
    /// \return Success, or the first rule it breaks.
    [[nodiscard]] constexpr result<void, identifier_errc> validate_iban(std::string_view text) noexcept
    {
        using namespace detail::check_digits;
        if (text.size() < 5 || text.size() > 34)
            return err(identifier_errc::wrong_length);
        if (!all_alphanumeric(text))
            return err(identifier_errc::invalid_character);
        if (!upper(text[0]) || !upper(text[1]) || !digit(text[2]) || !digit(text[3]))
            return err(identifier_errc::invalid_format);
        const int check = value(text[2]) * 10 + value(text[3]);
        if (check < 2 || check > 98 || iban_mod97(text.substr(0, 2), text.substr(2, 2), text.substr(4)) != 1)
            return err(identifier_errc::invalid_check_digit);
        return ok();
    }

    // ---- LEI --------------------------------------------------------------------------------------------------------

    /// \brief The two check digits of a LEI's first eighteen characters.
    /// \ingroup core
    /// \param payload Eighteen letters or digits.
    /// \return The check digits, or `wrong_length` or `invalid_character`.
    [[nodiscard]] constexpr result<std::array<char, 2>, identifier_errc> lei_check_digits(std::string_view payload) noexcept
    {
        using namespace detail::check_digits;
        if (payload.size() != 18)
            return err(identifier_errc::wrong_length);
        if (!all_alphanumeric(payload))
            return err(identifier_errc::invalid_character);
        return ok(two_digits(98 - mod97("00", mod97(payload))));
    }

    /// \brief Check a LEI (ISO 17442): `"5493001KJTIIGC8Y1R12"`.
    /// \ingroup core
    /// \param text Twenty characters.
    /// \return Success, or the first rule it breaks.
    [[nodiscard]] constexpr result<void, identifier_errc> validate_lei(std::string_view text) noexcept
    {
        using namespace detail::check_digits;
        if (text.size() != 20)
            return err(identifier_errc::wrong_length);
        if (!all_alphanumeric(text))
            return err(identifier_errc::invalid_character);
        if (!digit(text[18]) || !digit(text[19]))
            return err(identifier_errc::invalid_format);
        if (mod97(text) != 1)
            return err(identifier_errc::invalid_check_digit);
        return ok();
    }

    // ---- CUSIP ------------------------------------------------------------------------------------------------------

    /// \brief The check digit of a CUSIP's first eight characters.
    /// \ingroup core
    /// \param payload Eight digits, letters, `*`, `@` or `#`.
    /// \return The check digit, or `wrong_length` or `invalid_character`.
    [[nodiscard]] constexpr result<char, identifier_errc> cusip_check_digit(std::string_view payload) noexcept
    {
        if (payload.size() != 8)
            return err(identifier_errc::wrong_length);
        int sum = 0;
        for (std::size_t i = 0; i < 8; ++i) {
            int v = detail::check_digits::cusip_value(payload[i]);
            if (v < 0)
                return err(identifier_errc::invalid_character);
            v *= i % 2 == 1 ? 2 : 1;
            sum += v / 10 + v % 10;
        }
        return ok(static_cast<char>('0' + (10 - sum % 10) % 10));
    }

    /// \brief Check a CUSIP: `"037833100"`.
    /// \ingroup core
    /// \param text Nine characters.
    /// \return Success, or the first rule it breaks.
    [[nodiscard]] constexpr result<void, identifier_errc> validate_cusip(std::string_view text) noexcept
    {
        if (text.size() != 9)
            return err(identifier_errc::wrong_length);
        const auto check = cusip_check_digit(text.substr(0, 8));
        if (!check)
            return err(check.unwrap_err_unchecked());
        if (detail::check_digits::cusip_value(text[8]) < 0)
            return err(identifier_errc::invalid_character);
        if (!detail::check_digits::digit(text[8]))
            return err(identifier_errc::invalid_format);
        if (*check != text[8])
            return err(identifier_errc::invalid_check_digit);
        return ok();
    }

    // ---- SEDOL ------------------------------------------------------------------------------------------------------

    /// \brief The check digit of a SEDOL's first six characters.
    /// \ingroup core
    /// \param payload Six digits or consonants; digits throughout when the first is a digit.
    /// \return The check digit, or `wrong_length`, `invalid_character` or `invalid_format`.
    [[nodiscard]] constexpr result<char, identifier_errc> sedol_check_digit(std::string_view payload) noexcept
    {
        using namespace detail::check_digits;
        if (payload.size() != 6)
            return err(identifier_errc::wrong_length);
        constexpr std::array<int, 6> weights{1, 3, 1, 7, 3, 9};
        int sum = 0;
        for (std::size_t i = 0; i < 6; ++i) {
            if (!sedol_character(payload[i]))
                return err(identifier_errc::invalid_character);
            sum += weights[i] * value(payload[i]);
        }
        if (digit(payload[0]) && !all_digits(payload))
            return err(identifier_errc::invalid_format);
        return ok(static_cast<char>('0' + (10 - sum % 10) % 10));
    }

    /// \brief Check a SEDOL: `"0263494"`, `"B0YBKJ7"`.
    /// \ingroup core
    /// \param text Seven characters.
    /// \return Success, or the first rule it breaks.
    [[nodiscard]] constexpr result<void, identifier_errc> validate_sedol(std::string_view text) noexcept
    {
        if (text.size() != 7)
            return err(identifier_errc::wrong_length);
        const auto check = sedol_check_digit(text.substr(0, 6));
        if (!check)
            return err(check.unwrap_err_unchecked());
        if (!detail::check_digits::sedol_character(text[6]))
            return err(identifier_errc::invalid_character);
        if (!detail::check_digits::digit(text[6]))
            return err(identifier_errc::invalid_format);
        if (*check != text[6])
            return err(identifier_errc::invalid_check_digit);
        return ok();
    }

} // namespace snicholls

#endif // snicholls_core_ids_check_digits_hpp
