//
//  phone_number.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::phone_number`: a normalised E.164 telephone number.
/// \ingroup core
///
/// snicholls::phone_number - an ITU-T E.164 number: a country calling code of 1 to 3 digits and the national
/// significant number after it, at most 15 digits in all, held in one 64-bit word.
///
/// \code
///   auto r = snicholls::parse_phone_number("+61 412 345 678");   // result<phone_number, phone_errc>, no exceptions
///   snicholls::parse_phone_number("00 61 412-345-678");          // the same number: 00 is the international prefix
///   snicholls::parse_phone_number("0412 345 678");               // err(phone_errc::missing_country_code): no region guessed
///   snicholls::to_text(*r).view();                               // "+61412345678", no allocation
///   r->country_code();  r->national_number().view();             // 61, "412345678"
///   using namespace snicholls::literals;  constexpr auto n = "+61412345678"_e164;   // checked at compile time
/// \endcode
///
/// Semantics:
/// - An aggregate holding one std::uint64_t: sizeof 8, trivially copyable, standard layout, structural (usable as a
///   template argument). Value-initialised is the empty number, which no parse produces.
/// - Every non-empty value has a country calling code assigned in E.164 (detail/e164_country_codes.hpp, generated
///   from libphonenumber's metadata) and at least 2, at most 15 - (code length) digits after it.
/// - == and <=> are constexpr and order the canonical text as strings ("+1..." < "+44..." < "+61..."); std::hash is
///   constexpr and noexcept (not portable, like every std::hash).
/// - Text is the canonical "+" and digits form (E.164 in its E.123 written form without spaces). Input accepts the
///   written forms people use: spaces, hyphens, dots, slashes and one level of parentheses between the digits, and
///   the international prefix 00 in place of "+".

#ifndef snicholls_core_phone_number_hpp
#define snicholls_core_phone_number_hpp

#include <array>
#include <charconv>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>
#include <system_error>

#include <snicholls/core/detail/e164_country_codes.hpp>
#include <snicholls/core/hash/hash_combine.hpp>
#include <snicholls/core/result.hpp>

namespace snicholls {

/// \brief Why a text is not an E.164 number (the error of parse_phone_number).
/// \ingroup core
enum class phone_errc : std::uint8_t
{
    empty = 1,   ///< nothing but separators
    missing_country_code,   ///< no leading "+" or "00": a national number, and no region is ever guessed
    invalid_character,   ///< a letter or a character that is neither a digit nor a separator, or a second "+"
    unbalanced_parentheses,   ///< a ")" without its "(", a "(" left open, or a "(" inside another
    unknown_country_code,   ///< the leading digits are not an assigned country calling code
    trunk_prefix,   ///< the number after the country code starts with 0, the trunk prefix of that country
    too_short,   ///< fewer than 2 digits after the country code
    too_long,   ///< more than 15 digits in all
};

/// \brief The canonical text of a phone number, or of its national number, in a small value (no allocation).
/// \ingroup core
struct phone_number_text
{
    /// \brief The characters, NUL-terminated.
    std::array<char, 17> chars{};
    /// \brief The number of characters before the NUL.
    std::uint8_t length = 0;

    /// \brief The text.
    /// \return the text, without the NUL.
    [[nodiscard]] constexpr std::string_view view() const noexcept { return {chars.data(), length}; }
    /// \brief The text, NUL-terminated.
    /// \return a pointer to the first character.
    [[nodiscard]] constexpr const char* c_str() const noexcept { return chars.data(); }
    /// \brief The number of characters.
    /// \return the number of characters.
    [[nodiscard]] constexpr std::size_t size() const noexcept { return length; }
    /// \brief The text as a view.
    /// \return the view.
    constexpr operator std::string_view() const noexcept { return view(); }
};

namespace detail {

inline constexpr std::array<std::uint64_t, 16> e164_pow10 = [] {
    std::array<std::uint64_t, 16> t{};
    std::uint64_t p = 1;
    for (auto& v : t) {
        v = p;
        p *= 10;
    }
    return t;
}();

// Per code 0..999: bit 0 assigned, bit 1 a leading 0 after it is the trunk prefix
inline constexpr std::array<std::uint8_t, 1000> e164_code_flags = [] {
    std::array<std::uint8_t, 1000> t{};
    for (const auto& e : e164::country_codes)
        t[e.code] = static_cast<std::uint8_t>(1U | (e.trunk_zero ? 2U : 0U));
    return t;
}();

// The representation: the digits as an integer scaled to 15 digits (so integer order is string order), then the
// digit count (4 bits) and the code length (2 bits). 10^15 < 2^50: 56 bits in all.
constexpr std::uint64_t e164_aligned(std::uint64_t packed) noexcept { return packed >> 6; }

} // namespace detail

/// \brief An E.164 telephone number: a country calling code and the national significant number, at most 15 digits.
/// \ingroup core
///
/// \overview
/// The vocabulary type for a phone number used as an identity (a login, an SMS destination, a directory key): one
/// canonical value per number, so two spellings of the same number compare equal and hash alike. It checks the
/// structure E.164 defines - a "+", an assigned country calling code, no trunk prefix, at most 15 digits - and
/// nothing that depends on the country's own numbering plan. Use libphonenumber where a number must be valid for
/// its region (lengths, mobile or fixed line, carrier); use this where the canonical E.164 form is the contract.
///
/// \relations
/// - `snicholls::parse_phone_number` (text in, `snicholls::result`), `snicholls::to_chars` / `snicholls::to_text`
///   (canonical text out), `operator""_e164` (a literal checked at compile time).
/// - The country calling codes: detail/e164_country_codes.hpp, generated by scripts/generate_e164_country_codes.py.
/// - The same shape as `snicholls::uuid`: a structural value with allocation-free text.
/// - Group: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - sizeof 8, alignof 8, trivially copyable, standard layout, structural; every member function is constexpr and
///   noexcept except print.
/// - \invariant empty(), or: country_code() is assigned in E.164, 2 <= national_number_length(), digit_count() <= 15.
///   Only `packed` breaks it, by assignment; it is public because a structural type's members must be.
/// - Ordering is that of the canonical text; hashing is O(1).
///
/// \errors
/// None: parse_phone_number returns its errors as values (`snicholls::phone_errc`).
///
/// \examples
/// \snippet phone_number_demo.cpp demo
///
/// \perf
/// Release, macOS, Intel Xeon W-3245, Apple clang 17, 2026-11-23, median of 9, 4,096 numbers in five written
/// forms, on a loaded machine: `parse_phone_number` 44.6 ns a number against 43.2 ns for the same scan written by
/// hand into a digit array; `to_chars` 18.0 ns against 2.8 ns for copying stored digit characters (the price of 8
/// bytes in place of 17). benchmarks/phone_number_benchmark.cpp; not yet in BENCHMARKS.md.
///
/// \testing
/// - Tests: tests/phone_number_test.cpp (against fixtures from the Python phonenumbers package,
///   tests/fixtures/e164/), tests/detail/e164_country_codes_test.cpp.
/// - Demo: demos/phone_number_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only, and result.hpp.
///
/// \gotchas
/// - Region-specific validity is out of scope: the lengths each country allows, mobile against fixed line, and
///   which ranges are in service are not checked. "+61 12" parses; libphonenumber would call it invalid.
/// - A leading 0 after the country code is refused only where 0 is that country's trunk prefix ("+44 (0)20 ...",
///   written with the trunk 0, is refused with phone_errc::trunk_prefix, not repaired); where the 0 belongs to the
///   number (Italy's "+39 06 ...") it is kept. Other trunk prefixes (1 in the North American plan, 8 in Russia)
///   are digits of the number as far as E.164 is concerned and are kept as written.
/// - Only 00 is taken as the international prefix (the ITU-T E.164 recommendation); 011 and the other national
///   choices of prefix are read as a missing country code.
/// - Codes the ITU holds as spare, reserved or assigned to trials without a numbering plan are unknown here, since
///   the table is libphonenumber's.
/// - Extensions ("ext. 12"), vanity letters ("1-800-FLOWERS") and "tel:" URIs are refused as invalid characters.
///
/// \status
/// Experimental, new in 0.22.0.
struct phone_number
{
    /// \brief The representation (public only because a structural type's members must be): use the accessors.
    std::uint64_t packed = 0;

    /// \brief The most digits an E.164 number has (ITU-T E.164 §6.1).
    static constexpr std::size_t max_digits = 15;

    /// \brief Whether this is the empty number (value-initialised; never the result of a parse).
    /// \return `true` if empty.
    [[nodiscard]] constexpr bool empty() const noexcept { return packed == 0; }

    /// \brief The number of digits, country calling code included (0 when empty).
    /// \return 0, or 3 to 15.
    [[nodiscard]] constexpr std::size_t digit_count() const noexcept { return (packed >> 2) & 0xFU; }

    /// \brief The number of digits in the country calling code (0 when empty).
    /// \return 0 to 3.
    [[nodiscard]] constexpr std::size_t country_code_length() const noexcept { return packed & 0x3U; }

    /// \brief The country calling code (0 when empty).
    /// \return 1 to 999 (61 for "+61412345678"), or 0.
    [[nodiscard]] constexpr std::uint16_t country_code() const noexcept
    {
        return static_cast<std::uint16_t>(detail::e164_aligned(packed) /
                                          detail::e164_pow10[max_digits - country_code_length()]);
    }

    /// \brief The number of digits after the country calling code.
    /// \return 0, or 2 to 14.
    [[nodiscard]] constexpr std::size_t national_number_length() const noexcept
    {
        return digit_count() - country_code_length();
    }

    /// \brief One digit of the number, country calling code first.
    /// \pre i < digit_count().
    /// \param i the position, from 0.
    /// \return '0' to '9'.
    [[nodiscard]] constexpr char digit(std::size_t i) const noexcept
    {
        return static_cast<char>('0' + detail::e164_aligned(packed) / detail::e164_pow10[max_digits - 1 - i] % 10);
    }

    /// \brief The national significant number: the digits after the country calling code, leading zeros kept.
    /// \return "412345678" for "+61412345678" (empty when the number is).
    [[nodiscard]] constexpr phone_number_text national_number() const noexcept
    {
        phone_number_text t;
        const std::size_t c = country_code_length();
        for (std::size_t i = c; i < digit_count(); ++i)
            t.chars[i - c] = digit(i);
        t.length = static_cast<std::uint8_t>(national_number_length());
        return t;
    }

    /// \brief Writes the canonical text ("+61412345678") to a stream; nothing when empty.
    /// \param os the stream.
    void print(std::ostream& os) const;

    /// \brief Equality: the same number.
    /// \param a the left operand.
    /// \param b the right operand.
    /// \return `true` if equal.
    friend constexpr bool operator==(const phone_number& a, const phone_number& b) noexcept = default;

    /// \brief Three-way comparison, in the order of the canonical text.
    /// \param a the left operand.
    /// \param b the right operand.
    /// \return the ordering.
    friend constexpr std::strong_ordering operator<=>(const phone_number& a, const phone_number& b) noexcept = default;
};

/// \brief Characters in the longest canonical text: "+" and 15 digits.
/// \ingroup core
inline constexpr std::size_t phone_number_max_text_size = 16;

/// \brief Writes the canonical text ("+" and the digits, no terminator) to [first, last).
///
/// The empty number writes nothing. errc::value_too_large (and ptr == last) if it does not fit;
/// phone_number_max_text_size always does.
/// \ingroup core
/// \param first the first character of the buffer.
/// \param last one past its last.
/// \param p the number.
/// \return one past the last character written, or the error.
constexpr std::to_chars_result to_chars(char* first, char* last, const phone_number& p) noexcept
{
    const std::size_t n = p.digit_count();
    if (n == 0) return {first, std::errc{}};
    if (last - first < static_cast<std::ptrdiff_t>(n + 1)) return {last, std::errc::value_too_large};
    // All 15 places by constant division (a multiply), then the first n: no division by a variable power of 10
    std::array<char, phone_number::max_digits> all{};
    std::uint64_t v = detail::e164_aligned(p.packed);
    for (std::size_t k = all.size(); k > 0; --k) {
        all[k - 1] = static_cast<char>('0' + v % 10);
        v /= 10;
    }
    first[0] = '+';
    for (std::size_t k = 0; k < n; ++k)
        first[k + 1] = all[k];
    return {first + n + 1, std::errc{}};
}

/// \brief The canonical text in a small value (no allocation).
/// \ingroup core
/// \param p the number.
/// \return "+61412345678", or empty text for the empty number.
[[nodiscard]] constexpr phone_number_text to_text(const phone_number& p) noexcept
{
    phone_number_text t;
    const auto r = to_chars(t.chars.data(), t.chars.data() + phone_number_max_text_size, p);
    t.length = static_cast<std::uint8_t>(r.ptr - t.chars.data());
    return t;
}

/// \brief The canonical text, allocated.
/// \ingroup core
/// \param p the number.
/// \return "+61412345678".
/// \throws std::bad_alloc if the string cannot allocate.
[[nodiscard]] inline std::string to_string(const phone_number& p) { return std::string(to_text(p).view()); }

inline void phone_number::print(std::ostream& os) const
{
    const phone_number_text t = to_text(*this);
    os.write(t.c_str(), static_cast<std::streamsize>(t.size()));
}

/// \brief A well-mixed hash of the number (what std::hash\<snicholls::phone_number> returns).
///
/// Constexpr. Not portable across builds or architectures (size_t), like every std::hash.
/// \ingroup core
/// \param p the number.
/// \return the hash.
[[nodiscard]] constexpr std::size_t hash_value(const phone_number& p) noexcept
{
    return hash_combine(0, static_cast<std::size_t>(p.packed));
}

/// \brief Whether a country calling code is assigned in E.164 (by the table core carries).
/// \ingroup core
/// \param code the code, without "+".
/// \return `true` for 1, 44, 61, 800, ...; `false` for 0, 28, 1000, ...
[[nodiscard]] constexpr bool is_e164_country_code(unsigned code) noexcept
{
    return code < detail::e164_code_flags.size() && (detail::e164_code_flags[code] & 1U) != 0;
}

/// \brief Parses an E.164 number from the ways people write one.
///
/// A "+" or the international prefix "00", then the country calling code and the national significant number.
/// Spaces, hyphens, dots and slashes may stand anywhere before, between and after the digits, and one level of
/// parentheses anywhere ("(+61) 412 345 678", "+1 (650) 253-0000"); "00" must be two adjacent zeros. Nothing else
/// is accepted: not a letter, an extension, a second "+", or a national number without its country code.
/// \ingroup core
/// \param text the text.
/// \return the number, or why the text is not one (`snicholls::phone_errc`).
[[nodiscard]] constexpr result<phone_number, phone_errc> parse_phone_number(std::string_view text) noexcept
{
    std::array<std::uint8_t, phone_number::max_digits> d{};
    std::size_t n = 0;
    bool international = false;
    bool open = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c >= '0' && c <= '9') {
            if (!international) {
                if (c != '0' || i + 1 == text.size() || text[i + 1] != '0') return err(phone_errc::missing_country_code);
                international = true;
                ++i;
                continue;
            }
            if (n < d.size()) d[n] = static_cast<std::uint8_t>(c - '0');
            ++n;
        } else if (c == '+') {
            if (international) return err(phone_errc::invalid_character);
            international = true;
        } else if (c == '(') {
            if (open) return err(phone_errc::unbalanced_parentheses);
            open = true;
        } else if (c == ')') {
            if (!open) return err(phone_errc::unbalanced_parentheses);
            open = false;
        } else if (c != ' ' && c != '-' && c != '.' && c != '/') {
            return err(phone_errc::invalid_character);
        }
    }
    if (open) return err(phone_errc::unbalanced_parentheses);
    if (!international) return err(phone_errc::empty);
    if (n > phone_number::max_digits) return err(phone_errc::too_long);
    if (n == 0) return err(phone_errc::too_short);
    if (d[0] == 0) return err(phone_errc::unknown_country_code);
    std::size_t cc_length = 0;
    unsigned code = 0;
    for (std::size_t k = 0; k < 3 && k < n && cc_length == 0; ++k) {
        code = code * 10 + d[k];
        if ((detail::e164_code_flags[code] & 1U) != 0) cc_length = k + 1;
    }
    if (cc_length == 0) return err(phone_errc::unknown_country_code);
    if (n - cc_length < 2) return err(phone_errc::too_short);
    if (d[cc_length] == 0 && (detail::e164_code_flags[code] & 2U) != 0) return err(phone_errc::trunk_prefix);
    std::uint64_t v = 0;
    for (std::size_t k = 0; k < n; ++k)
        v = v * 10 + d[k];
    v *= detail::e164_pow10[phone_number::max_digits - n];
    return ok(phone_number{(v << 6) | (static_cast<std::uint64_t>(n) << 2) | cc_length});
}

namespace literals {

/// \brief "+61412345678"_e164: a snicholls::phone_number checked at compile time.
///
/// Any spelling parse_phone_number accepts; anything else does not compile.
/// \ingroup core
/// \param text the text.
/// \param length its length.
/// \return the number.
consteval phone_number operator""_e164(const char* text, std::size_t length)
{
    const auto r = parse_phone_number(std::string_view(text, length));
    if (!r) throw "snicholls::operator\"\"_e164: not an E.164 number";
    return *r;
}

} // namespace literals

} // namespace snicholls

/// \cond
template <>
struct std::hash<snicholls::phone_number>
{
    constexpr std::size_t operator()(const snicholls::phone_number& p) const noexcept { return snicholls::hash_value(p); }
};
/// \endcond

#endif /* snicholls_core_phone_number_hpp */
