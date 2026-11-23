//
//  asset_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_asset` and `snicholls::check_currency`: the conformance kit of the asset capability.
/// \ingroup core
///
/// **Overview.** Drives any `asset_like` (or `currency_like`) implementation, a type, a value or an object behind
/// the interface, through the contract and says which check failed:
/// - the code is a valid asset code and the scale is at most `max_scale`;
/// - both are the same on every call;
/// - for a currency: the name is not empty, and the numeric code is between 1 and 999.
///
/// No test framework is included; a test asserts that the result is empty.
///
/// \snippet interfaces/asset_conformance_demo.cpp conformance
///
/// **Relationships.** `concepts/asset.hpp`, `interfaces/asset.hpp`, `interfaces/named_conformance.hpp`. Group:
/// \ref core.
///
/// **Implementations.** Not applicable.
///
/// **Guarantees.** constexpr, `noexcept`.
///
/// **Errors.** The returned message names the failed check.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/interfaces/asset_conformance_test.cpp`, `demos/interfaces/asset_conformance_demo.cpp`.
///
/// **Dependencies.** `<optional>`, `<string_view>`, `concepts/asset.hpp`, `interfaces/named_conformance.hpp`,
/// `money/asset.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_interfaces_asset_conformance_hpp
#define snicholls_core_interfaces_asset_conformance_hpp

#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/interfaces/named_conformance.hpp>
#include <snicholls/core/money/asset.hpp>

#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief Check `t` against the asset contract.
    /// \param t The implementation.
    /// \return `std::nullopt` if it conforms, or the failed check.
    template <asset_like T>
    [[nodiscard]] constexpr std::optional<std::string_view> check_asset(const T& t) noexcept
    {
        const std::string_view code = t.code();
        if (!is_asset_code(code))
            return "asset: code() is not 1 to 24 of A-Z 0-9 : . - _";
        if (static_cast<unsigned>(t.scale()) > max_scale)
            return "asset: scale() is above max_scale";
        if (std::string_view{t.code()} != code || t.scale() != t.scale())
            return "asset: code() or scale() changes between calls";
        return std::nullopt;
    }

    /// \brief Check `t` against the currency contract: the asset contract, `has_name`, and a numeric code.
    /// \param t The implementation.
    /// \return `std::nullopt` if it conforms, or the failed check.
    template <currency_like T>
    [[nodiscard]] constexpr std::optional<std::string_view> check_currency(const T& t) noexcept
    {
        if (const auto failed = check_asset(t))
            return failed;
        if (const auto failed = check_named(t))
            return failed;
        if (t.numeric() < 1 || t.numeric() > 999)
            return "currency: numeric() is not between 1 and 999";
        return std::nullopt;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_asset_conformance_hpp
