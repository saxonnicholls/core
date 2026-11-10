//
//  decimal.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::decimal`: exact decimals (strikes, ticks, multipliers) with one canonical text.
/// \ingroup core

#ifndef snicholls_core_decimal_hpp
#define snicholls_core_decimal_hpp

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace snicholls {

/// \brief How strictly `snicholls::decimal::parse` reads text.
/// \ingroup core
enum class decimal_syntax : std::uint8_t {
    canonical,   ///< Only the canonical spelling: no trailing fractional zeros, no "+", no "-0" ("72.5", "6930").
    plain,       ///< Also trailing fractional zeros and a leading "+" ("72.50", "+6930.0"); still no exponent.
};

/// \brief Fixed-capacity text of a decimal (no allocation): `data`, `size`, `view()`.
/// \ingroup core
struct decimal_text
{
    std::array<char, 40> data{};   ///< The characters (not NUL-terminated).
    std::size_t size = 0;          ///< How many are used.
    /// \brief The text.
    /// \return a view of `data[0, size)`.
    [[nodiscard]] constexpr std::string_view view() const noexcept { return {data.data(), size}; }
};

/// \brief An exact decimal `mantissa x 10^exponent`, normalised so that equal numbers are equal objects.
/// \ingroup core
///
/// \overview
/// Strikes, tick sizes, lot sizes and multipliers must round-trip exactly through text (`CME:LOZ26 P72.5`, a tick of
/// 0.0000001); `double` cannot promise that. A decimal holds up to 18 significant digits and an exponent in
/// [-18, 18]. It is always normalised (the mantissa has no factor of ten; zero is `0 x 10^0`), so the defaulted `==`
/// is numeric equality and the canonical text is unique: `"6930"`, `"72.5"`, `"0.0000001"`, `"-5.25"`, `"0"` - no
/// exponent notation, no trailing fractional zeros, no "+", no "-0". It is not an arithmetic type: S1 only needs exact
/// identity, ordering and text.
///
/// \relations
/// - Used for option strikes and instrument definitions in symbology.
/// - Group: \ref core.
///
/// \impls
/// None: a value type.
///
/// \contracts
/// - Trivially copyable, 16 bytes, every member function `constexpr` and `noexcept`.
/// - `parse(to_text(d)) == d` for every decimal, and `to_text(*parse(t)) == t` for every canonical `t`.
/// - `<=>` is exact (no conversion through `double`).
///
/// \errors
/// `from_parts` and `parse` return `std::nullopt` for out-of-range or malformed input; nothing throws.
///
/// \examples
/// \snippet decimal_demo.cpp decimal
///
/// \perf
/// Parse and format are a single pass over at most 40 characters; not benchmarked in core.
///
/// \testing
/// - Tests: tests/decimal_test.cpp.
/// - Demo: demos/decimal_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// - `to_double()` is for display and maths, not identity: 0.1 has no exact `double`.
///
/// \status
/// Experimental, pre-1.0.
class decimal
{
public:
    static constexpr int max_digits = 18;      ///< Significant digits a decimal can hold.
    static constexpr int min_exponent = -18;   ///< Smallest exponent (0.000000000000000001).
    static constexpr int max_exponent = 18;    ///< Largest exponent.

    /// \brief Zero.
    constexpr decimal() noexcept = default;

    /// \brief The decimal `mantissa x 10^exponent`, normalised.
    /// \param mantissa the digits (|mantissa| < 10^18 after removing factors of ten).
    /// \param exponent the power of ten.
    /// \return the decimal, or empty when it does not fit 18 digits and an exponent in [-18, 18].
    [[nodiscard]] static constexpr std::optional<decimal> from_parts(std::int64_t mantissa, int exponent) noexcept
    {
        if (mantissa == 0) return decimal{};
        if (mantissa == INT64_MIN) return std::nullopt;
        while (mantissa % 10 == 0) {
            mantissa /= 10;
            ++exponent;
        }
        const std::int64_t magnitude = mantissa < 0 ? -mantissa : mantissa;
        if (magnitude >= limit) return std::nullopt;
        if (exponent < min_exponent || exponent > max_exponent) return std::nullopt;
        decimal d;
        d.mantissa_ = mantissa;
        d.exponent_ = static_cast<std::int8_t>(exponent);
        return d;
    }

    /// \brief An integer as a decimal.
    /// \param value the integer (|value| < 10^18 once trailing zeros are removed).
    /// \return the decimal, or empty when it does not fit.
    [[nodiscard]] static constexpr std::optional<decimal> from_integer(std::int64_t value) noexcept { return from_parts(value, 0); }

    /// \brief Read a decimal from text.
    /// \param text e.g. "72.5", "-5.25", "6930" (canonical) or also "72.50", "+6930.0" (plain).
    /// \param syntax how strict to be.
    /// \return the decimal, or empty for malformed or out-of-range text.
    [[nodiscard]] static constexpr std::optional<decimal> parse(std::string_view text,
                                                                decimal_syntax syntax = decimal_syntax::canonical) noexcept
    {
        bool negative = false;
        if (!text.empty() && (text.front() == '-' || (text.front() == '+' && syntax == decimal_syntax::plain))) {
            negative = text.front() == '-';
            text.remove_prefix(1);
        }
        const std::size_t point = text.find('.');
        const std::string_view whole = text.substr(0, point);
        const std::string_view frac = point == std::string_view::npos ? std::string_view{} : text.substr(point + 1);
        if (whole.empty() || (point != std::string_view::npos && frac.empty())) return std::nullopt;
        if (!all_digits(whole) || !all_digits(frac)) return std::nullopt;
        if (whole.size() > 1 && whole.front() == '0') return std::nullopt;   // no leading zeros
        if (syntax == decimal_syntax::canonical && !frac.empty() && frac.back() == '0') return std::nullopt;
        // Significant digits: strip leading zeros of whole+frac and trailing zeros of frac.
        std::int64_t mantissa = 0;
        int digits = 0;
        int exponent = 0;
        int pending_zeros = 0;   // zeros of the whole part not yet multiplied in (trailing zeros of an integer)
        for (const char c : whole) {
            if (c == '0' && digits == 0) continue;
            if (c == '0') {
                ++pending_zeros;
                continue;
            }
            for (; pending_zeros > 0; --pending_zeros) {
                if (++digits > max_digits) return std::nullopt;
                mantissa *= 10;
            }
            if (++digits > max_digits) return std::nullopt;
            mantissa = mantissa * 10 + (c - '0');
        }
        std::size_t frac_len = frac.size();
        while (frac_len > 0 && frac[frac_len - 1] == '0') --frac_len;   // plain: trailing fractional zeros
        if (frac_len > 0) {
            for (; pending_zeros > 0; --pending_zeros) {
                if (++digits > max_digits) return std::nullopt;
                mantissa *= 10;
            }
            for (std::size_t i = 0; i < frac_len; ++i) {
                const char c = frac[i];
                --exponent;
                if (c == '0' && digits == 0) continue;
                if (++digits > max_digits) return std::nullopt;
                mantissa = mantissa * 10 + (c - '0');
            }
        } else {
            exponent += pending_zeros;
        }
        if (mantissa == 0) {
            if (negative && syntax == decimal_syntax::canonical) return std::nullopt;   // "-0"
            return decimal{};
        }
        return from_parts(negative ? -mantissa : mantissa, exponent);
    }

    /// \brief The normalised mantissa (no factor of ten; 0 for zero).
    /// \return the mantissa.
    [[nodiscard]] constexpr std::int64_t mantissa() const noexcept { return mantissa_; }
    /// \brief The exponent (0 for zero).
    /// \return the exponent.
    [[nodiscard]] constexpr int exponent() const noexcept { return exponent_; }
    /// \brief Whether the value is zero.
    /// \return `mantissa() == 0`.
    [[nodiscard]] constexpr bool is_zero() const noexcept { return mantissa_ == 0; }
    /// \brief Whether the value is below zero.
    /// \return `mantissa() < 0`.
    [[nodiscard]] constexpr bool is_negative() const noexcept { return mantissa_ < 0; }

    /// \brief The canonical text, without allocating.
    /// \return e.g. "72.5".
    [[nodiscard]] constexpr decimal_text text() const noexcept
    {
        decimal_text t;
        t.size = write(t.data);
        return t;
    }

    /// \brief The canonical text as a `std::string`.
    /// \return e.g. "72.5".
    /// \throws std::bad_alloc.
    [[nodiscard]] std::string to_string() const { return std::string(text().view()); }

    /// \brief Write the canonical text.
    /// \param out the destination.
    /// \return the number of characters written, or empty when `out` is too small (nothing useful written).
    [[nodiscard]] constexpr std::optional<std::size_t> format(std::span<char> out) const noexcept
    {
        const decimal_text t = text();
        if (out.size() < t.size) return std::nullopt;
        for (std::size_t i = 0; i < t.size; ++i) out[i] = t.data[i];
        return t.size;
    }

    /// \brief The nearest `double` (for display and maths only).
    /// \return the value.
    [[nodiscard]] constexpr double to_double() const noexcept
    {
        double v = static_cast<double>(mantissa_);
        for (int e = exponent_; e > 0; --e) v *= 10.0;
        for (int e = exponent_; e < 0; ++e) v /= 10.0;
        return v;
    }

    /// \brief Numeric equality (decimals are normalised).
    friend constexpr bool operator==(const decimal&, const decimal&) noexcept = default;

    /// \brief Exact numeric ordering.
    /// \param a,b the decimals.
    /// \return their order.
    friend constexpr std::strong_ordering operator<=>(const decimal& a, const decimal& b) noexcept
    {
        if (a.mantissa_ == b.mantissa_ && a.exponent_ == b.exponent_) return std::strong_ordering::equal;
        const int sa = sign(a.mantissa_), sb = sign(b.mantissa_);
        if (sa != sb) return sa <=> sb;
        const std::strong_ordering magnitude = compare_magnitude(a, b);
        return sa >= 0 ? magnitude : 0 <=> magnitude;
    }

private:
    static constexpr std::int64_t limit = 1'000'000'000'000'000'000;   // 10^18

    static constexpr bool all_digits(std::string_view s) noexcept
    {
        for (const char c : s)
            if (c < '0' || c > '9') return false;
        return true;
    }
    static constexpr int sign(std::int64_t m) noexcept { return m < 0 ? -1 : (m > 0 ? 1 : 0); }
    static constexpr int magnitude_digits(std::int64_t m) noexcept
    {
        std::uint64_t u = m < 0 ? static_cast<std::uint64_t>(-m) : static_cast<std::uint64_t>(m);
        int n = 1;
        while (u >= 10) {
            u /= 10;
            ++n;
        }
        return n;
    }
    static constexpr std::strong_ordering compare_magnitude(const decimal& a, const decimal& b) noexcept
    {
        std::int64_t ma = a.mantissa_ < 0 ? -a.mantissa_ : a.mantissa_;
        std::int64_t mb = b.mantissa_ < 0 ? -b.mantissa_ : b.mantissa_;
        const int da = magnitude_digits(ma), db = magnitude_digits(mb);
        const int ta = a.exponent_ + da, tb = b.exponent_ + db;   // position of the leading digit
        if (ta != tb) return ta <=> tb;
        for (int i = da; i < db; ++i) ma *= 10;   // same leading position: align digit counts (both <= 18)
        for (int i = db; i < da; ++i) mb *= 10;
        return ma <=> mb;
    }

    constexpr std::size_t write(std::array<char, 40>& out) const noexcept
    {
        std::size_t n = 0;
        if (mantissa_ == 0) {
            out[n++] = '0';
            return n;
        }
        if (mantissa_ < 0) out[n++] = '-';
        std::array<char, 20> digits{};
        int count = 0;
        for (std::uint64_t u = mantissa_ < 0 ? static_cast<std::uint64_t>(-mantissa_) : static_cast<std::uint64_t>(mantissa_);
             u > 0; u /= 10)
            digits[static_cast<std::size_t>(count++)] = static_cast<char>('0' + u % 10);
        // digits[] is reversed. value = digits x 10^exponent_.
        if (exponent_ >= 0) {
            for (int i = count - 1; i >= 0; --i) out[n++] = digits[static_cast<std::size_t>(i)];
            for (int i = 0; i < exponent_; ++i) out[n++] = '0';
            return n;
        }
        const int frac = -exponent_;
        if (count > frac) {
            for (int i = count - 1; i >= frac; --i) out[n++] = digits[static_cast<std::size_t>(i)];
            out[n++] = '.';
            for (int i = frac - 1; i >= 0; --i) out[n++] = digits[static_cast<std::size_t>(i)];
        } else {
            out[n++] = '0';
            out[n++] = '.';
            for (int i = 0; i < frac - count; ++i) out[n++] = '0';
            for (int i = count - 1; i >= 0; --i) out[n++] = digits[static_cast<std::size_t>(i)];
        }
        return n;
    }

    std::int64_t mantissa_ = 0;
    std::int8_t exponent_ = 0;
};

} // namespace snicholls

// Hash of a decimal (of its normalised parts; not portable across builds, like every `std::hash`).
/// \cond
template <>
struct std::hash<snicholls::decimal>
{
    /// \brief The hash.
    /// \param d the decimal.
    /// \return a hash of mantissa and exponent.
    std::size_t operator()(const snicholls::decimal& d) const noexcept
    {
        return std::hash<std::int64_t>{}(d.mantissa()) ^ (std::hash<int>{}(d.exponent()) * 0x9e3779b97f4a7c15ULL);
    }
};
/// \endcond

#endif /* snicholls_core_decimal_hpp */
