//
//  token.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::token_interface`: the virtual twin of `token_like`, and the adapters between them.
/// \ingroup core
///
/// **Overview.** For code that chooses a token at run time without templates: a wallet's list of balances, a
/// deposit monitor fed from a registry, a plug-in. The adapters go both ways:
/// - `token_model<T>` puts any `token_like` type behind the interface;
/// - the interface models `token_like`, so an object behind it goes wherever a template over the concept is
///   expected, `check_token` and `to_asset` among them.
///
/// `chain()` returns the chain behind `chain_interface` (a `chain_model` held by the adapter), so its kind and
/// identifier are there at run time too.
///
/// \snippet interfaces/token_demo.cpp interface
///
/// **Relationships.** `concepts/token.hpp`, `interfaces/asset.hpp` (the base), `interfaces/chain.hpp`,
/// `interfaces/token_conformance.hpp`, `token/basic_token.hpp`, `token/tokens.hpp`. Group: \ref core.
///
/// **Implementations.** `token_model<T>`. A run-time registry (servers') implements the interface over its rows.
///
/// **Guarantees.** Every member is `noexcept`; the adapter is usable in constant expressions when `T` is. Single
/// inheritance (`token_interface` derives only from `asset_interface`), so no constexpr virtual needs a thunk.
///
/// **Errors.** None.
///
/// **Performance.** One virtual call per member; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/token_test.cpp`, `demos/interfaces/token_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<string_view>`, `concepts/token.hpp`, `interfaces/asset.hpp`,
/// `interfaces/chain.hpp`.
///
/// **Known issues.** `token_model<T>` default-constructs `T` and its chain type, as the curated types allow.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_token_hpp
#define snicholls_core_interfaces_token_hpp

#include <snicholls/core/concepts/token.hpp>
#include <snicholls/core/interfaces/asset.hpp>
#include <snicholls/core/interfaces/chain.hpp>

#include <cstdint>
#include <string_view>

namespace snicholls {

    /// \brief A token chosen at run time.
    class token_interface : public asset_interface {
    public:
        /// \brief The chain it is deployed on.
        /// \return The chain, valid while this object lives.
        [[nodiscard]] constexpr virtual const chain_interface& chain() const noexcept = 0;
        /// \brief The contract address, in the chain's canonical text.
        /// \return A view valid while this object lives.
        [[nodiscard]] constexpr virtual std::string_view contract() const noexcept = 0;
        /// \brief The issuer's symbol.
        /// \return A view valid while this object lives.
        [[nodiscard]] constexpr virtual std::string_view symbol() const noexcept = 0;
    };

    /// \brief Any `token_like` type behind `token_interface`.
    /// \tparam T The type, default-constructed; its chain type is held as a `chain_model`.
    template <token_like T>
    class token_model final : public token_interface {
    public:
        /// \brief The asset code.
        /// \return The adapted token's code.
        [[nodiscard]] constexpr std::string_view code() const noexcept override { return T{}.code(); }
        /// \brief The decimals.
        /// \return The adapted token's scale.
        [[nodiscard]] constexpr std::uint8_t scale() const noexcept override { return T{}.scale(); }
        /// \brief The chain.
        /// \return The adapted token's chain, behind `chain_interface`.
        [[nodiscard]] constexpr const chain_interface& chain() const noexcept override { return chain_; }
        /// \brief The contract address.
        /// \return The adapted token's contract.
        [[nodiscard]] constexpr std::string_view contract() const noexcept override { return T{}.contract(); }
        /// \brief The symbol.
        /// \return The adapted token's symbol.
        [[nodiscard]] constexpr std::string_view symbol() const noexcept override { return T{}.symbol(); }

    private:
        chain_model<decltype(T{}.chain())> chain_{};
    };

    static_assert(token_like<token_interface>, "snicholls::token_interface must model snicholls::token_like");

} // namespace snicholls

#endif // snicholls_core_interfaces_token_hpp
