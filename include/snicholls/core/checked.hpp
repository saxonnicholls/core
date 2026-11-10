//
//  checked.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Checked and saturating integer arithmetic, `numeric_cast` that refuses a lossy conversion, and `parse_number`.
/// \ingroup core
///
/// **Overview.** A money path must not wrap silently. `checked_add`, `checked_sub` and `checked_mul` return
/// `std::nullopt` instead of overflowing; `saturating_*` clamp to the type's range; `numeric_cast<To>` gives
/// `std::nullopt` unless the value survives the conversion exactly; `parse_number<T>` reads a whole string
/// with `std::from_chars` and returns the reason it could not. Integers use the compiler's overflow builtins,
/// one instruction and a flag, and a portable form during constant evaluation and on MSVC.
///
/// \snippet checked_demo.cpp checked
///
/// **Relationships.** `uint256.hpp` for 256-bit amounts, which checks the same way; `result.hpp`; 
/// `meta/assignable.hpp` for refusing narrowing at compile time. Group: \ref core.
///
/// **Guarantees.** `constexpr` and `noexcept`, but `parse_number`, which is `noexcept` and not `constexpr`
/// for floating point.
///
/// **Errors.** `std::nullopt`, or a `std::errc` in the `result`: `invalid_argument` for text that is not a
/// number or has trailing characters, `result_out_of_range` for one that does not fit.
///
/// **Performance.** Not benchmarked: a builtin and a branch.
///
/// **Tests and demos.** `tests/checked_test.cpp`, `demos/checked_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cerrno>`, `<charconv>`, `<cstdlib>`, `<concepts>`, `<limits>`, `<optional>`, `<string_view>`, `<system_error>`,
/// `<type_traits>`, `<utility>`, `result.hpp`.
///
/// **Known issues.** Integers only for the arithmetic; floating point has its own infinities. Where the standard
/// library lacks floating-point `std::from_chars` (Xcode 16's libc++ and older), `parse_number` of a floating-point
/// type uses the C library: the same syntax, but the decimal point follows `LC_NUMERIC` if the program changes it,
/// and text over 255 characters is refused.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_checked_hpp
#define snicholls_core_checked_hpp

#include <snicholls/core/result.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <concepts>
#include <limits>
#include <optional>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief `a + b`, or none if it overflows `T`.
    /// \ingroup core
    /// \tparam T An integer type.
    /// \param a One.
    /// \param b The other.
    /// \return The sum.
    template <std::integral T>
    [[nodiscard]] constexpr std::optional<T> checked_add(T a, T b) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        T r{};
        if (!__builtin_add_overflow(a, b, &r))
            return r;
        return std::nullopt;
#else
        if (b > 0 ? a > std::numeric_limits<T>::max() - b : a < std::numeric_limits<T>::min() - b)
            return std::nullopt;
        return static_cast<T>(a + b);
#endif
    }

    /// \brief `a - b`, or none if it overflows `T`.
    /// \ingroup core
    /// \tparam T An integer type.
    /// \param a One.
    /// \param b The other.
    /// \return The difference.
    template <std::integral T>
    [[nodiscard]] constexpr std::optional<T> checked_sub(T a, T b) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        T r{};
        if (!__builtin_sub_overflow(a, b, &r))
            return r;
        return std::nullopt;
#else
        if (b > 0 ? a < std::numeric_limits<T>::min() + b : a > std::numeric_limits<T>::max() + b)
            return std::nullopt;
        return static_cast<T>(a - b);
#endif
    }

    /// \brief `a * b`, or none if it overflows `T`.
    /// \ingroup core
    /// \tparam T An integer type.
    /// \param a One.
    /// \param b The other.
    /// \return The product.
    template <std::integral T>
    [[nodiscard]] constexpr std::optional<T> checked_mul(T a, T b) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        T r{};
        if (!__builtin_mul_overflow(a, b, &r))
            return r;
        return std::nullopt;
#else
        if (a == 0 || b == 0)
            return T{0};
        const T r = static_cast<T>(static_cast<std::make_unsigned_t<T>>(a) * static_cast<std::make_unsigned_t<T>>(b));
        if (r / b != a || (std::is_signed_v<T> && a == -1 && b == std::numeric_limits<T>::min()) ||
            (std::is_signed_v<T> && b == -1 && a == std::numeric_limits<T>::min()))
            return std::nullopt;
        return r;
#endif
    }

    /// \brief `a + b`, clamped to `T`'s range.
    /// \ingroup core
    /// \tparam T An integer type.
    /// \param a One.
    /// \param b The other.
    /// \return The sum, or the bound it passed.
    template <std::integral T>
    [[nodiscard]] constexpr T saturating_add(T a, T b) noexcept
    {
        return checked_add(a, b).value_or(b > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min());
    }

    /// \brief `a - b`, clamped to `T`'s range.
    /// \ingroup core
    /// \tparam T An integer type.
    /// \param a One.
    /// \param b The other.
    /// \return The difference, or the bound it passed.
    template <std::integral T>
    [[nodiscard]] constexpr T saturating_sub(T a, T b) noexcept
    {
        return checked_sub(a, b).value_or(b < 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min());
    }

    /// \brief `a * b`, clamped to `T`'s range.
    /// \ingroup core
    /// \tparam T An integer type.
    /// \param a One.
    /// \param b The other.
    /// \return The product, or the bound it passed.
    template <std::integral T>
    [[nodiscard]] constexpr T saturating_mul(T a, T b) noexcept
    {
        return checked_mul(a, b).value_or((a < 0) != (b < 0) ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max());
    }

    /// \brief The value as a `To`, or none if the conversion would change it: out of range, or a fraction lost.
    /// \ingroup core
    /// \tparam To   The target arithmetic type.
    /// \tparam From The source arithmetic type.
    /// \param v The value.
    /// \return It, converted exactly.
    template <typename To, typename From>
        requires std::is_arithmetic_v<To> && std::is_arithmetic_v<From>
    [[nodiscard]] constexpr std::optional<To> numeric_cast(From v) noexcept
    {
        if constexpr (std::integral<To> && std::integral<From>) {
            return std::in_range<To>(v) ? std::optional<To>(static_cast<To>(v)) : std::nullopt;
        } else {
            if (v != v)   // NaN survives only into a floating type
                return std::floating_point<To> ? std::optional<To>(static_cast<To>(v)) : std::nullopt;
            if (std::integral<To> && (v < static_cast<From>(std::numeric_limits<To>::lowest()) ||
                                      v >= static_cast<From>(std::numeric_limits<To>::max()) + static_cast<From>(1)))
                return std::nullopt;
            const To t = static_cast<To>(v);
            return static_cast<From>(t) == v ? std::optional<To>(t) : std::nullopt;
        }
    }

    namespace detail::parse {
        // Whether the standard library converts T with std::from_chars: every one does integers; Xcode 16's libc++
        // and older have no floating-point overloads (a double would reach the deleted bool overload).
        template <typename T>
        concept has_from_chars = requires(const char* p, T& v) {
            { std::from_chars(p, p, v) } -> std::same_as<std::from_chars_result>;
        };

        // The text std::from_chars accepts for a floating-point number in its general format: an optional minus, then
        // digits with an optional point and exponent, or inf, infinity, nan, nan(chars), in any case. No leading
        // space, no plus, no hexadecimal.
        constexpr bool float_syntax(std::string_view t) noexcept
        {
            const auto lower = [](char c) { return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c); };
            const auto starts = [&](std::string_view w) {
                if (t.size() < w.size())
                    return false;
                for (std::size_t i = 0; i < w.size(); ++i)
                    if (lower(t[i]) != w[i])
                        return false;
                return true;
            };
            const auto digit = [](char c) { return c >= '0' && c <= '9'; };
            if (!t.empty() && t.front() == '-')
                t.remove_prefix(1);
            if (starts("infinity"))
                return t.size() == 8;
            if (starts("inf"))
                return t.size() == 3;
            if (starts("nan")) {
                t.remove_prefix(3);
                if (t.empty())
                    return true;
                if (t.front() != '(' || t.back() != ')')
                    return false;
                for (const char c : t.substr(1, t.size() - 2))
                    if (!(digit(c) || (lower(c) >= 'a' && lower(c) <= 'z') || c == '_'))
                        return false;
                return true;
            }
            std::size_t i = 0, mantissa = 0;
            while (i < t.size() && digit(t[i]))
                ++i, ++mantissa;
            if (i < t.size() && t[i] == '.') {
                ++i;
                while (i < t.size() && digit(t[i]))
                    ++i, ++mantissa;
            }
            if (mantissa == 0)
                return false;
            if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
                ++i;
                if (i < t.size() && (t[i] == '+' || t[i] == '-'))
                    ++i;
                const std::size_t exponent = i;
                while (i < t.size() && digit(t[i]))
                    ++i;
                if (i == exponent)
                    return false;
            }
            return i == t.size();
        }

        inline float c_parse(const char* b, char** e, float*) noexcept { return std::strtof(b, e); }
        inline double c_parse(const char* b, char** e, double*) noexcept { return std::strtod(b, e); }
        inline long double c_parse(const char* b, char** e, long double*) noexcept { return std::strtold(b, e); }

        // The fallback where std::from_chars has no floating-point overload: the same syntax, checked first, then the
        // C library's correctly rounded conversion. The C library reads the decimal point from LC_NUMERIC, which is
        // "." unless the program calls setlocale; up to 255 characters.
        template <std::floating_point T>
        result<T, std::errc> parse_float(std::string_view text) noexcept
        {
            if (!float_syntax(text) || text.size() > 255)
                return err(std::errc::invalid_argument);
            std::array<char, 256> buffer{};
            std::copy(text.begin(), text.end(), buffer.begin());
            char* end = nullptr;
            errno = 0;
            const T value = c_parse(buffer.data(), &end, static_cast<T*>(nullptr));
            if (end != buffer.data() + text.size())
                return err(std::errc::invalid_argument);
            if (errno == ERANGE && (value > T(1) || value < T(-1)))   // overflow; an underflow is the nearest value
                return err(std::errc::result_out_of_range);
            return ok(value);
        }
    } // namespace detail::parse

    /// \brief A whole string as a number, by `std::from_chars`: no leading space, no trailing characters.
    /// \ingroup core
    ///
    /// Where the standard library has no floating-point `std::from_chars` (Xcode 16's libc++ and older), a floating-point
    /// type is parsed with the same syntax by the C library instead; see Known issues.
    /// \tparam T An integer or floating-point type.
    /// \param text The text.
    /// \return The number, or `std::errc::invalid_argument` / `std::errc::result_out_of_range`.
    template <typename T>
        requires std::is_arithmetic_v<T> && (!std::same_as<T, bool>) && detail::parse::has_from_chars<T>
    [[nodiscard]] result<T, std::errc> parse_number(std::string_view text) noexcept
    {
        T value{};
        const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (ec != std::errc{})
            return err(ec);
        if (end != text.data() + text.size())
            return err(std::errc::invalid_argument);
        return ok(value);
    }

    /// \brief A whole string as a floating-point number where `std::from_chars` cannot parse one: the same syntax, by the C library.
    /// \ingroup core
    /// \tparam T A floating-point type.
    /// \param text The text.
    /// \return The number, or `std::errc::invalid_argument` / `std::errc::result_out_of_range`.
    template <std::floating_point T>
        requires(!detail::parse::has_from_chars<T>)
    [[nodiscard]] result<T, std::errc> parse_number(std::string_view text) noexcept
    {
        return detail::parse::parse_float<T>(text);
    }

} // namespace snicholls

#endif /* snicholls_core_checked_hpp */
