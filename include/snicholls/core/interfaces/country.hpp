//
//  country.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::country_interface`: the virtual twin of `country_like`, and the adapter between them.
/// \ingroup core
///
/// **Overview.** For code that chooses a country at run time, from a form or a customer record, without templates.
/// `country_model<T>` adapts any `country_like` type (one of `country.hpp`'s types, or a caller's own) to the
/// interface; the interface models `country_like`, so it goes back through any template written over the concept.
///
/// \snippet interfaces/country_demo.cpp interface
///
/// **Relationships.** `concepts/country.hpp`, `interfaces/named.hpp`, `interfaces/country_conformance.hpp`,
/// `country.hpp`. Group: \ref core.
///
/// **Implementations.** `country_model<T>`.
///
/// **Guarantees.** Every member is `noexcept`; the adapter is usable in constant expressions when `T` is. Views
/// returned are valid while the object lives (for the generated types, for ever).
///
/// **Errors.** None.
///
/// **Performance.** One virtual call per member; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/country_test.cpp`, `demos/interfaces/country_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<span>`, `<string_view>`, `concepts/country.hpp`, `interfaces/named.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_country_hpp
#define snicholls_core_interfaces_country_hpp

#include <snicholls/core/concepts/country.hpp>
#include <snicholls/core/interfaces/named.hpp>

#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls {

    /// \brief A country chosen at run time.
    class country_interface : public named_interface {
    public:
        /// \brief The ISO 3166-1 alpha-2 code.
        /// \return Two capitals.
        [[nodiscard]] constexpr virtual std::string_view alpha2() const noexcept = 0;
        /// \brief The ISO 3166-1 alpha-3 code.
        /// \return Three capitals.
        [[nodiscard]] constexpr virtual std::string_view alpha3() const noexcept = 0;
        /// \brief The ISO 3166-1 numeric code.
        /// \return 1 to 999.
        [[nodiscard]] constexpr virtual std::uint16_t numeric() const noexcept = 0;
        /// \brief The ISO 4217 currencies in use.
        /// \return The codes.
        [[nodiscard]] constexpr virtual std::span<const std::string_view> currencies() const noexcept = 0;
        /// \brief The E.164 country calling code.
        /// \return The code, or 0 where none is assigned.
        [[nodiscard]] constexpr virtual std::uint16_t calling_code() const noexcept = 0;
    };

    /// \brief Any `country_like` type behind `country_interface`.
    /// \tparam T The type, held by value.
    template <country_like T>
    class country_model final : public country_interface {
    public:
        /// \brief Adapt a default-constructed `T` (a compile-time type with static members).
        constexpr country_model() = default;
        /// \brief Adapt this value.
        /// \param t The value.
        constexpr explicit country_model(T t) : t_(t) {}
        /// \brief The name.
        /// \return The adapted country's name.
        [[nodiscard]] constexpr std::string_view name() const noexcept override { return t_.name(); }
        /// \brief The alpha-2 code.
        /// \return The adapted country's alpha-2 code.
        [[nodiscard]] constexpr std::string_view alpha2() const noexcept override { return t_.alpha2(); }
        /// \brief The alpha-3 code.
        /// \return The adapted country's alpha-3 code.
        [[nodiscard]] constexpr std::string_view alpha3() const noexcept override { return t_.alpha3(); }
        /// \brief The numeric code.
        /// \return The adapted country's numeric code.
        [[nodiscard]] constexpr std::uint16_t numeric() const noexcept override { return t_.numeric(); }
        /// \brief The currencies in use.
        /// \return The adapted country's currencies.
        [[nodiscard]] constexpr std::span<const std::string_view> currencies() const noexcept override
        {
            return t_.currencies();
        }
        /// \brief The calling code.
        /// \return The adapted country's calling code, or 0.
        [[nodiscard]] constexpr std::uint16_t calling_code() const noexcept override { return t_.calling_code(); }

    private:
        T t_{};
    };

    static_assert(country_like<country_interface>, "snicholls::country_interface must model snicholls::country_like");

} // namespace snicholls

#endif // snicholls_core_interfaces_country_hpp
