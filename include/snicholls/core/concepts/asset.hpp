//
//  asset.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::asset_like` and `snicholls::currency_like`: what an amount can be of, as concepts.
/// \ingroup core
///
/// **Overview.** An asset answers `code()` (a short string) and `scale()` (the decimal places of its minor unit);
/// a currency is an asset that is also `has_name` and has an ISO 4217 `numeric()` code. The concepts are written over
/// object expressions, so they are modelled by compile-time types with `static constexpr` member functions
/// (`snicholls::currency::AUD`, `snicholls::native::ETH`, a product's own token struct), by the value
/// `snicholls::asset`, and by any object behind the virtual twins `asset_interface` and `currency_interface`
/// (`interfaces/asset.hpp`).
///
/// \snippet concepts/asset_demo.cpp concepts
///
/// **Relationships.** `concepts/named.hpp`, `interfaces/asset.hpp`, `interfaces/asset_conformance.hpp`,
/// `money/asset.hpp`, `money/iso4217.hpp`, `money/native.hpp`. Group: \ref core.
///
/// **Implementations.** `snicholls::asset` (the reference), the generated ISO 4217 types, the native asset types,
/// `with_scale<T, S>`, and the adapters of `interfaces/asset.hpp`.
///
/// **Guarantees.** Concepts: no code, no cost.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/asset_test.cpp`, `demos/concepts/asset_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstdint>`, `<string_view>`, `concepts/named.hpp`.
///
/// **Known issues.** The ISO 4217 codes with no minor units (XAU, XAG, XPT, XPD, XDR, the funds) and the historic
/// codes do not model `asset_like` until a scale is given with `with_scale`.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_concepts_asset_hpp
#define snicholls_core_concepts_asset_hpp

#include <snicholls/core/concepts/named.hpp>

#include <concepts>
#include <cstdint>
#include <string_view>

namespace snicholls {

    /// \brief A type whose objects answer `code()` with a string view and `scale()` with an unsigned integer.
    template <typename T>
    concept asset_like = requires(const T& t) {
        { t.code() } -> std::convertible_to<std::string_view>;
        { t.scale() } -> std::convertible_to<std::uint8_t>;
    };

    /// \brief An `asset_like`, `has_name` type with an ISO 4217 `numeric()` code.
    template <typename T>
    concept currency_like = asset_like<T> && has_name<T> && requires(const T& t) {
        { t.numeric() } -> std::convertible_to<std::uint16_t>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_asset_hpp
