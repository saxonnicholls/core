//
//  asset.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::asset`: what an amount is of, a code and a scale, as a type or as a 26-byte value.
/// \ingroup core
///
/// **Overview.** An amount of money is a count of minor units of one asset: cents of `AUD`, wei of `ETH`, the
/// smallest unit of a token. The asset is its code and its scale, the number of decimal places a minor unit
/// stands for (`AUD` 2, `ETH` 18, `USDC` 6). Both are known when the program is written and never change, so they
/// are compile-time first:
///
/// - **as a type.** Every ISO 4217 currency is a type (`snicholls::currency::AUD`, generated from the SIX lists,
///   `money/iso4217.hpp`), every native chain asset too (`snicholls::native::ETH`, `money/native.hpp`), and a
///   product declares its own tokens the same way: a struct with `static constexpr` member functions `code()` and
///   `scale()` (and, for a currency, `name()` and `numeric()`). The concepts are `snicholls::asset_like` and
///   `snicholls::currency_like` (`concepts/asset.hpp`);
/// - **as a value.** `snicholls::asset`, 26 bytes, trivially copyable and structural, for data that names its
///   asset at run time (a ledger row, a message). `asset_v<T>` is a type's value, checked by the compiler;
///   `make_asset("USDC:56", 18)` builds one at run time and refuses a malformed code.
///
/// A code is 1 to 24 characters: capitals, digits, and `:` `.` `-` `_` (`AUD`, `ETH`, `USDC:56`, `EEAU2029`). A
/// code is unique within a book: a token that exists on several chains is told apart by its code (`USDC:1`,
/// `USDC:56`); the registry that assigns those codes is the server's, not core's. The scale is at most
/// `max_scale` (36), which leaves room in 256 bits for a rate's product.
///
/// Use it for money, balances and prices; use `decimal` for a bounded rate or quantity within 18 digits, and
/// `uint256` for a raw on-chain integer that has no asset.
///
/// \snippet money/asset_demo.cpp assets
///
/// **Relationships.** `concepts/asset.hpp` (the concepts), `interfaces/asset.hpp` (the virtual interface and its
/// adapters), `money/iso4217.hpp`, `money/native.hpp`, `money/money.hpp`. Group: \ref core.
///
/// **Implementations.** The value `asset` is the capability's reference implementation; every currency and native
/// type is another, and `interfaces/asset.hpp` adapts any of them to `asset_interface`.
///
/// **Guarantees.** constexpr throughout, `noexcept`, no allocation. `asset` is trivially copyable, standard layout
/// and structural (usable as a template argument); equality compares code and scale.
///
/// **Errors.** `make_asset` returns `std::nullopt` for an empty, over-long or malformed code, or a scale above
/// `max_scale`. `asset_v<T>` of a type with a bad code or scale is a compile error.
///
/// **Performance.** Equality is a 26-byte compare. Not benchmarked separately: `money/money.hpp`'s benchmarks
/// include it.
///
/// **Tests and demos.** `tests/money/asset_test.cpp`, `demos/money/asset_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<string_view>`, `concepts/asset.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_money_asset_hpp
#define snicholls_core_money_asset_hpp

#include <snicholls/core/concepts/asset.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief The largest scale an asset may have: 36 decimal places.
    inline constexpr unsigned max_scale = 36;

    /// \brief The longest asset code: 24 characters, room for a 13-character symbol, `:` and a 10-digit chain id.
    inline constexpr std::size_t max_asset_code = 24;

    /// \brief Whether `c` may appear in an asset code: a capital, a digit, `:`, `.`, `-` or `_`.
    /// \param c The character.
    /// \return Whether it is allowed.
    [[nodiscard]] constexpr bool is_asset_code_char(char c) noexcept
    {
        return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == ':' || c == '.' || c == '-' || c == '_';
    }

    /// \brief Whether `code` is a valid asset code: 1 to 24 allowed characters.
    /// \param code The code.
    /// \return Whether it is valid.
    [[nodiscard]] constexpr bool is_asset_code(std::string_view code) noexcept
    {
        if (code.empty() || code.size() > max_asset_code)
            return false;
        for (const char c : code)
            if (!is_asset_code_char(c))
                return false;
        return true;
    }

    /// \brief An asset as a value: its code and its scale. 26 bytes, trivially copyable, structural.
    /// \ingroup core
    struct asset {
        /// \brief The code's characters, padded with NULs. Public so that the type is structural; use `code()`.
        std::array<char, max_asset_code> chars{};
        /// \brief The code's length. Public so that the type is structural.
        std::uint8_t length = 0;
        /// \brief The scale. Public so that the type is structural; use `scale()`.
        std::uint8_t places = 0;

        /// \brief The code.
        /// \return A view of the code, valid while this asset lives.
        [[nodiscard]] constexpr std::string_view code() const noexcept { return {chars.data(), length}; }

        /// \brief The scale: the decimal places a minor unit stands for.
        /// \return The scale.
        [[nodiscard]] constexpr std::uint8_t scale() const noexcept { return places; }

        /// \brief Equality: the same code and the same scale.
        friend constexpr bool operator==(const asset&, const asset&) noexcept = default;
    };

    /// \brief The asset with this code and scale, if both are valid.
    /// \param code  The code: 1 to 24 characters, capitals, digits, `:`, `.`, `-`, `_`.
    /// \param scale The scale, at most `max_scale`.
    /// \return The asset, or `std::nullopt`.
    [[nodiscard]] constexpr std::optional<asset> make_asset(std::string_view code, unsigned scale) noexcept
    {
        if (!is_asset_code(code) || scale > max_scale)
            return std::nullopt;
        asset a;
        for (std::size_t i = 0; i < code.size(); ++i)
            a.chars[i] = code[i];
        a.length = static_cast<std::uint8_t>(code.size());
        a.places = static_cast<std::uint8_t>(scale);
        return a;
    }

    namespace detail {
        template <asset_like T>
        consteval asset asset_of()
        {
            const auto a = make_asset(T::code(), T::scale());
            if (!a)
                throw "snicholls::asset_v: the type's code or scale is not a valid asset";   // a compile error
            return *a;
        }
    } // namespace detail

    /// \brief The value of an asset type, checked by the compiler: `asset_v<currency::AUD>` is `AUD` at scale 2.
    /// \tparam T A type modelling `asset_like` whose `code()` and `scale()` are constant expressions.
    template <asset_like T>
    inline constexpr asset asset_v = detail::asset_of<T>();

    /// \brief An asset type with a scale given by its user: `with_scale<currency::XAU, 18>` is gold at 18 places.
    ///
    /// ISO 4217 gives the precious metals, the SDR and the funds no minor units, and SIX's historic list gives none
    /// for withdrawn codes, so their types have no `scale()` and are not `asset_like` until one is chosen.
    /// \tparam T     A type with `static constexpr std::string_view code()`.
    /// \tparam Scale The scale, at most `max_scale`.
    template <typename T, unsigned Scale>
        requires(Scale <= max_scale)
    struct with_scale : T {
        /// \brief The scale chosen.
        /// \return `Scale`.
        static constexpr std::uint8_t scale() noexcept { return static_cast<std::uint8_t>(Scale); }
    };

} // namespace snicholls

#endif // snicholls_core_money_asset_hpp
