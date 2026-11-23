//
//  country_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_country`: the conformance kit of the country capability.
/// \ingroup core
///
/// **Overview.** Drives any `country_like` implementation, a type or an object behind `country_interface`, through
/// the contract and says which check failed:
/// - the name is not empty and does not change;
/// - `alpha2()` is two capitals, `alpha3()` three, `numeric()` 1 to 999;
/// - a code ISO 3166-1 assigns agrees with ISO: the alpha-3 and numeric of its alpha-2 (a user-assigned code, such
///   as XK, is checked for form only);
/// - every currency is three capitals;
/// - `calling_code()` is at most 999.
///
/// No test framework is included.
///
/// \snippet interfaces/country_conformance_demo.cpp conformance
///
/// **Relationships.** `concepts/country.hpp`, `interfaces/country.hpp`, `interfaces/named_conformance.hpp`,
/// `country.hpp`. Group: \ref core.
///
/// **Implementations.** Not applicable.
///
/// **Guarantees.** constexpr, `noexcept`.
///
/// **Errors.** The returned message names the failed check.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/interfaces/country_conformance_test.cpp`,
/// `demos/interfaces/country_conformance_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<optional>`, `<string_view>`, `concepts/country.hpp`, `country.hpp` (the ISO
/// table it checks against), `interfaces/named_conformance.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_country_conformance_hpp
#define snicholls_core_interfaces_country_conformance_hpp

#include <snicholls/core/concepts/country.hpp>
#include <snicholls/core/country.hpp>
#include <snicholls/core/interfaces/named_conformance.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls {

    namespace detail {
        constexpr bool country_capitals(std::string_view s, std::size_t n) noexcept
        {
            if (s.size() != n)
                return false;
            for (const char c : s)
                if (c < 'A' || c > 'Z')
                    return false;
            return true;
        }
    } // namespace detail

    /// \brief Check `t` against the country contract.
    /// \param t The implementation.
    /// \return `std::nullopt` if it conforms, or the failed check.
    template <country_like T>
    [[nodiscard]] constexpr std::optional<std::string_view> check_country(const T& t) noexcept
    {
        if (const auto failed = check_named(t))
            return failed;
        const std::string_view alpha2 = t.alpha2(), alpha3 = t.alpha3();
        if (!detail::country_capitals(alpha2, 2))
            return "country: alpha2() is not two capitals";
        if (!detail::country_capitals(alpha3, 3))
            return "country: alpha3() is not three capitals";
        const auto numeric = static_cast<std::uint16_t>(t.numeric());
        if (numeric < 1 || numeric > 999)
            return "country: numeric() is not 1 to 999";
        if (const std::size_t i = detail::iso3166_index(alpha2); i < detail::iso3166_rows.size()) {
            if (detail::iso3166_rows[i].alpha3 != alpha3 || detail::iso3166_rows[i].numeric != numeric)
                return "country: alpha3() or numeric() disagrees with ISO 3166-1";
        }
        for (const std::string_view c : t.currencies())
            if (!detail::country_capitals(c, 3))
                return "country: a currency is not three capitals";
        if (static_cast<std::uint16_t>(t.calling_code()) > 999)
            return "country: calling_code() is above 999";
        return std::nullopt;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_country_conformance_hpp
