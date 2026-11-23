//
//  chain.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::chain_like` and `snicholls::evm_chain_like`: a blockchain as a concept.
/// \ingroup core
///
/// **Overview.** A chain answers:
/// - `name()`;
/// - `kind()`, its family: EVM, Bitcoin, Solana, TRON;
/// - `native_asset()`, an `asset_like` value of what its fees are paid in.
///
/// An EVM chain also answers `chain_id()`, its EIP-155 identifier. Chains change rarely and a recompile is cheap,
/// so they are compile-time types (`snicholls::chain::ethereum`, `chain::base`, `chain/chains.hpp`). The concepts are
/// written over object expressions, so the types, and any object behind the virtual twin `chain_interface`
/// (`interfaces/chain.hpp`), model them.
///
/// \snippet concepts/chain_demo.cpp concepts
///
/// **Relationships.** `concepts/named.hpp`, `concepts/asset.hpp`, `interfaces/chain.hpp`,
/// `interfaces/chain_conformance.hpp`, `chain/chains.hpp`. Group: \ref core.
///
/// **Implementations.** The chain types of `chain/chains.hpp`, `chain_model<T>`.
///
/// **Guarantees.** Concepts and an enumeration: no code, no cost.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/chain_test.cpp`, `demos/concepts/chain_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstdint>`, `concepts/asset.hpp`, `concepts/named.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_concepts_chain_hpp
#define snicholls_core_concepts_chain_hpp

#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/concepts/named.hpp>

#include <concepts>
#include <cstdint>

namespace snicholls {

    /// \brief A chain's family: what its addresses, transactions and signatures look like.
    enum class chain_kind : std::uint8_t {
        evm,       ///< Ethereum and the chains that run its virtual machine
        bitcoin,   ///< Bitcoin and its UTXO descendants
        solana,    ///< Solana
        tron,      ///< TRON: an EVM-like machine with its own addresses and transactions
        other,     ///< anything else
    };

    /// \brief A type whose objects answer `name()`, `kind()` and `native_asset()`.
    template <typename T>
    concept chain_like = has_name<T> && requires(const T& t) {
        { t.kind() } -> std::convertible_to<chain_kind>;
        { t.native_asset() } -> asset_like;
    };

    /// \brief A `chain_like` type that also answers `chain_id()`, its EIP-155 identifier.
    template <typename T>
    concept evm_chain_like = chain_like<T> && requires(const T& t) {
        { t.chain_id() } -> std::convertible_to<std::uint64_t>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_chain_hpp
