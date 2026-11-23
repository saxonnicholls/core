//
//  asset.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::asset_interface` and `snicholls::currency_interface`: the virtual twins of `asset_like` and
/// `currency_like`, and the adapters between them.
/// \ingroup core
///
/// **Overview.** For code that chooses an asset at run time from many types, without templates: a list of every
/// currency, a plug-in. `asset_model<T>` and `currency_model<T>` adapt any type modelling the concept; the
/// interfaces model the concepts, so either form goes wherever the other is expected. `to_asset(x)` turns either
/// into the value `snicholls::asset`.
///
/// \snippet interfaces/asset_demo.cpp interface
///
/// **Relationships.** `concepts/asset.hpp`, `interfaces/named.hpp`, `interfaces/asset_conformance.hpp`,
/// `money/asset.hpp`. Group: \ref core.
///
/// **Implementations.** `asset_model<T>`, `currency_model<T>`.
///
/// **Guarantees.** Every member is `noexcept`; the adapters are usable in constant expressions when `T` is.
///
/// **Errors.** `to_asset` returns `std::nullopt` for an implementation whose code or scale is not a valid asset.
///
/// **Performance.** One virtual call per member; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/asset_test.cpp`, `demos/interfaces/asset_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<optional>`, `<string_view>`, `concepts/asset.hpp`, `interfaces/named.hpp`,
/// `money/asset.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_interfaces_asset_hpp
#define snicholls_core_interfaces_asset_hpp

#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/interfaces/named.hpp>
#include <snicholls/core/money/asset.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief An asset chosen at run time.
    class asset_interface {
    public:
        /// \brief Virtual, so an implementation can be deleted through the interface.
        constexpr virtual ~asset_interface() = default;
        /// \brief The code.
        /// \return A view valid while the object lives.
        [[nodiscard]] constexpr virtual std::string_view code() const noexcept = 0;
        /// \brief The scale.
        /// \return The decimal places of a minor unit.
        [[nodiscard]] constexpr virtual std::uint8_t scale() const noexcept = 0;
    };

    /// \brief A currency chosen at run time.
    class currency_interface : public asset_interface, public named_interface {
    public:
        /// \brief The ISO 4217 numeric code.
        /// \return The code.
        [[nodiscard]] constexpr virtual std::uint16_t numeric() const noexcept = 0;
    };

    /// \brief Any `asset_like` type behind `asset_interface`.
    /// \tparam T The type, held by value.
    template <asset_like T>
    class asset_model final : public asset_interface {
    public:
        /// \brief Adapt a default-constructed `T`.
        constexpr asset_model() = default;
        /// \brief Adapt this value.
        /// \param t The value.
        constexpr explicit asset_model(T t) : t_(t) {}
        /// \brief The code.
        /// \return The adapted value's code.
        [[nodiscard]] constexpr std::string_view code() const noexcept override { return t_.code(); }
        /// \brief The scale.
        /// \return The adapted value's scale.
        [[nodiscard]] constexpr std::uint8_t scale() const noexcept override { return t_.scale(); }

    private:
        T t_{};
    };

    /// \brief Any `currency_like` type behind `currency_interface`.
    /// \tparam T The type, held by value.
    template <currency_like T>
    class currency_model final : public currency_interface {
    public:
        /// \brief Adapt a default-constructed `T`.
        constexpr currency_model() = default;
        /// \brief The code.
        /// \return The adapted value's code.
        [[nodiscard]] constexpr std::string_view code() const noexcept override { return T{}.code(); }
        /// \brief The scale.
        /// \return The adapted value's scale.
        [[nodiscard]] constexpr std::uint8_t scale() const noexcept override { return T{}.scale(); }
        /// \brief The name.
        /// \return The adapted value's name.
        [[nodiscard]] constexpr std::string_view name() const noexcept override { return T{}.name(); }
        /// \brief The ISO 4217 numeric code.
        /// \return The adapted value's numeric code.
        [[nodiscard]] constexpr std::uint16_t numeric() const noexcept override { return T{}.numeric(); }
    };

    /// \brief The value of any asset, a type or an object behind the interface.
    /// \param x The asset.
    /// \return Its value, or `std::nullopt` if its code or scale is not a valid asset.
    template <asset_like T>
    [[nodiscard]] constexpr std::optional<asset> to_asset(const T& x) noexcept
    {
        return make_asset(x.code(), x.scale());
    }

    static_assert(asset_like<asset_interface>, "snicholls::asset_interface must model snicholls::asset_like");
    static_assert(currency_like<currency_interface>, "snicholls::currency_interface must model snicholls::currency_like");

} // namespace snicholls

#endif // snicholls_core_interfaces_asset_hpp
