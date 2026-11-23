//
//  chain.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::chain_interface`: the virtual twin of `chain_like`, and the adapters between them.
/// \ingroup core
///
/// **Overview.** For code that chooses a chain at run time, a wallet's list of networks or a config file, without
/// templates. `chain_model<T>` adapts any `chain_like` type. The interface models `chain_like`:
/// - `native_asset()` returns the value `snicholls::asset`;
/// - `chain_id()` returns `std::optional<std::uint64_t>`, empty for a chain with no EIP-155 identifier.
///
/// \snippet interfaces/chain_demo.cpp interface
///
/// **Relationships.** `concepts/chain.hpp`, `interfaces/named.hpp`, `interfaces/chain_conformance.hpp`,
/// `chain/chains.hpp`, `money/asset.hpp`. Group: \ref core.
///
/// **Implementations.** `chain_model<T>`.
///
/// **Guarantees.** Every member is `noexcept`; the adapter is usable in constant expressions when `T` is.
///
/// **Errors.** None.
///
/// **Performance.** One virtual call per member; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/chain_test.cpp`, `demos/interfaces/chain_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<optional>`, `<string_view>`, `concepts/chain.hpp`, `interfaces/named.hpp`,
/// `money/asset.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_interfaces_chain_hpp
#define snicholls_core_interfaces_chain_hpp

#include <snicholls/core/concepts/chain.hpp>
#include <snicholls/core/interfaces/named.hpp>
#include <snicholls/core/money/asset.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief A chain chosen at run time.
    class chain_interface : public named_interface {
    public:
        /// \brief The chain's family.
        /// \return The kind.
        [[nodiscard]] constexpr virtual chain_kind kind() const noexcept = 0;
        /// \brief What its fees are paid in.
        /// \return The native asset.
        [[nodiscard]] constexpr virtual asset native_asset() const noexcept = 0;
        /// \brief Its EIP-155 identifier.
        /// \return The identifier, or empty for a chain that has none.
        [[nodiscard]] constexpr virtual std::optional<std::uint64_t> chain_id() const noexcept = 0;
    };

    namespace detail {
        template <evm_chain_like T>
        constexpr std::optional<std::uint64_t> chain_id_of(const T& t) noexcept { return t.chain_id(); }
        template <chain_like T>
            requires(!evm_chain_like<T>)
        constexpr std::optional<std::uint64_t> chain_id_of(const T& t) noexcept
        {
            if constexpr (requires { { t.chain_id() } -> std::same_as<std::optional<std::uint64_t>>; })
                return t.chain_id();   // chain_interface: already optional
            else
                return std::nullopt;
        }
    } // namespace detail

    /// \brief Any `chain_like` type behind `chain_interface`.
    /// \tparam T The type, default-constructed.
    template <chain_like T>
    class chain_model final : public chain_interface {
    public:
        /// \brief The name.
        /// \return The adapted chain's name.
        [[nodiscard]] constexpr std::string_view name() const noexcept override { return T{}.name(); }
        /// \brief The kind.
        /// \return The adapted chain's kind.
        [[nodiscard]] constexpr chain_kind kind() const noexcept override { return T{}.kind(); }
        /// \brief The native asset.
        /// \return The adapted chain's native asset, as a value.
        [[nodiscard]] constexpr asset native_asset() const noexcept override
        {
            using native = decltype(T{}.native_asset());
            return asset_v<native>;
        }
        /// \brief The EIP-155 identifier.
        /// \return The adapted chain's identifier, or empty.
        [[nodiscard]] constexpr std::optional<std::uint64_t> chain_id() const noexcept override
        {
            return detail::chain_id_of(T{});
        }
    };

    static_assert(chain_like<chain_interface>, "snicholls::chain_interface must model snicholls::chain_like");

} // namespace snicholls

#endif // snicholls_core_interfaces_chain_hpp
