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
/// **Dependencies.** `<charconv>`, `<concepts>`, `<limits>`, `<optional>`, `<string_view>`, `<system_error>`,
/// `<type_traits>`, `<utility>`, `result.hpp`.
///
/// **Known issues.** Integers only for the arithmetic; floating point has its own infinities.
///
/// **Status.** New in 0.3.0.

#ifndef snicholls_core_checked_hpp
#define snicholls_core_checked_hpp

#include <snicholls/core/result.hpp>

#include <charconv>
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

    /// \brief A whole string as a number, by `std::from_chars`: no leading space, no trailing characters.
    /// \ingroup core
    /// \tparam T An integer or floating-point type.
    /// \param text The text.
    /// \return The number, or `std::errc::invalid_argument` / `std::errc::result_out_of_range`.
    template <typename T>
        requires std::is_arithmetic_v<T> && (!std::same_as<T, bool>)
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

} // namespace snicholls

#endif /* snicholls_core_checked_hpp */
