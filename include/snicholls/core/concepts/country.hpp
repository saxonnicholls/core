//
//  country.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::country_like`: an ISO 3166-1 country as a concept.
/// \ingroup core
///
/// **Overview.** A country answers:
/// - `name()`;
/// - `alpha2()` and `alpha3()`, its ISO 3166-1 codes, and `numeric()`;
/// - `currencies()`, the ISO 4217 codes in use, as a span of string views;
/// - `calling_code()`, its E.164 country calling code, 0 where none is assigned.
///
/// The concept is written over object expressions, so the compile-time types of `country.hpp`
/// (`snicholls::country::AU`) and any object behind the virtual twin `country_interface` (`interfaces/country.hpp`)
/// model it, as would a caller's own record of a country.
///
/// \snippet concepts/country_demo.cpp concepts
///
/// **Relationships.** `concepts/named.hpp`, `interfaces/country.hpp`, `interfaces/country_conformance.hpp`,
/// `country.hpp`. Group: \ref core.
///
/// **Implementations.** The country types of `country.hpp`, `country_model<T>`, `country_interface`.
///
/// **Guarantees.** A concept: no code, no cost.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/country_test.cpp`, `demos/concepts/country_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstdint>`, `<span>`, `<string_view>`, `concepts/named.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_concepts_country_hpp
#define snicholls_core_concepts_country_hpp

#include <snicholls/core/concepts/named.hpp>

#include <concepts>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls {

    /// \brief A `has_name` type whose objects answer `alpha2()`, `alpha3()`, `numeric()`, `currencies()` and
    /// `calling_code()`.
    template <typename T>
    concept country_like = has_name<T> && requires(const T& t) {
        { t.alpha2() } -> std::convertible_to<std::string_view>;
        { t.alpha3() } -> std::convertible_to<std::string_view>;
        { t.numeric() } -> std::convertible_to<std::uint16_t>;
        { t.currencies() } -> std::convertible_to<std::span<const std::string_view>>;
        { t.calling_code() } -> std::convertible_to<std::uint16_t>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_country_hpp
