//
//  chain_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_chain`: the conformance kit of the chain capability.
/// \ingroup core
///
/// **Overview.** Drives any `chain_like` implementation, a type or an object behind `chain_interface`, through the
/// contract and says which check failed:
/// - the name is not empty;
/// - the native asset conforms to the asset contract;
/// - an EVM chain's identifier is not zero.
///
/// No test framework is included.
///
/// \snippet interfaces/chain_conformance_demo.cpp conformance
///
/// **Relationships.** `concepts/chain.hpp`, `interfaces/chain.hpp`, `interfaces/asset_conformance.hpp`. Group:
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
/// **Tests and demos.** `tests/interfaces/chain_conformance_test.cpp`, `demos/interfaces/chain_conformance_demo.cpp`.
///
/// **Dependencies.** `<optional>`, `<string_view>`, `concepts/chain.hpp`, `interfaces/asset_conformance.hpp`,
/// `interfaces/named_conformance.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_interfaces_chain_conformance_hpp
#define snicholls_core_interfaces_chain_conformance_hpp

#include <snicholls/core/concepts/chain.hpp>
#include <snicholls/core/interfaces/asset_conformance.hpp>
#include <snicholls/core/interfaces/chain.hpp>
#include <snicholls/core/interfaces/named_conformance.hpp>

#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief Check `t` against the chain contract.
    /// \param t The implementation.
    /// \return `std::nullopt` if it conforms, or the failed check.
    template <chain_like T>
    [[nodiscard]] constexpr std::optional<std::string_view> check_chain(const T& t) noexcept
    {
        if (const auto failed = check_named(t))
            return failed;
        if (const auto failed = check_asset(t.native_asset()))
            return failed;
        if (const auto id = detail::chain_id_of(t); id && *id == 0)
            return "chain: chain_id() is zero";
        return std::nullopt;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_chain_conformance_hpp
