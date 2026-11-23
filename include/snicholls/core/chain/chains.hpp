//
//  chains.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The major blockchains as types: `snicholls::chain::ethereum`, `base`, `arbitrum`, `bsc`, `tron`, ...
/// \ingroup core
///
/// **Overview.** A chain's identity changes rarely, and a recompile is cheap, so each chain is a type. Each answers:
/// - `name()` and `kind()`;
/// - `native_asset()`, a value of the native asset's type (`native::ETH`, `native::BTC`, ...);
/// - for an EVM chain, `chain_id()`, its EIP-155 identifier, the one wallets and RPC nodes report.
///
/// The types model `chain_like` (and `evm_chain_like`). A wallet, a token registry or a deposit monitor is written
/// once, as a template over the concept, or once over `chain_interface` when the choice is made at run time.
/// Testnets are here too, so that a test network is a type and never a magic number.
///
/// \snippet chain/chains_demo.cpp chains
///
/// **Relationships.** `concepts/chain.hpp`, `interfaces/chain.hpp`, `money/native.hpp`. Group: \ref core.
///
/// **Implementations.** One type per chain.
///
/// **Guarantees.** constexpr, `noexcept`, no state.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/chain/chains_test.cpp` checks every EIP-155 identifier and native asset against
/// the chain registry (ethereum-lists/chains); demo `demos/chain/chains_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<string_view>`, `concepts/chain.hpp`, `money/native.hpp`.
///
/// **Known issues.**
/// - TRON reports the EIP-155-style identifier 728126428 (0x2b6653dc) through its JSON-RPC, but its kind is `tron`,
///   since its addresses and transactions are not Ethereum's.
/// - Bitcoin and Solana have no EIP-155 identifier.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_chain_chains_hpp
#define snicholls_core_chain_chains_hpp

#include <snicholls/core/concepts/chain.hpp>
#include <snicholls/core/money/native.hpp>

#include <cstdint>
#include <string_view>

namespace snicholls::chain {

    namespace detail {
        /// An EVM chain: its identifier, name and native asset.
        template <std::uint64_t Id, typename Native, chain_kind Kind = chain_kind::evm>
        struct evm_chain {
            /// \brief The family. \return `Kind`.
            static constexpr chain_kind kind() noexcept { return Kind; }
            /// \brief The EIP-155 identifier. \return `Id`.
            static constexpr std::uint64_t chain_id() noexcept { return Id; }
            /// \brief The native asset. \return A value of its type.
            static constexpr Native native_asset() noexcept { return {}; }
        };
        /// A chain with no EIP-155 identifier.
        template <typename Native, chain_kind Kind>
        struct plain_chain {
            /// \brief The family. \return `Kind`.
            static constexpr chain_kind kind() noexcept { return Kind; }
            /// \brief The native asset. \return A value of its type.
            static constexpr Native native_asset() noexcept { return {}; }
        };
    } // namespace detail

    /// \brief Ethereum mainnet: chain id 1, ETH.
    struct ethereum : detail::evm_chain<1, native::ETH> {
        /// \brief The name. \return `"Ethereum"`.
        static constexpr std::string_view name() noexcept { return "Ethereum"; }
    };
    /// \brief OP Mainnet (Optimism): chain id 10, ETH.
    struct optimism : detail::evm_chain<10, native::ETH> {
        /// \brief The name. \return `"OP Mainnet"`.
        static constexpr std::string_view name() noexcept { return "OP Mainnet"; }
    };
    /// \brief BNB Smart Chain: chain id 56, BNB.
    struct bsc : detail::evm_chain<56, native::BNB> {
        /// \brief The name. \return `"BNB Smart Chain"`.
        static constexpr std::string_view name() noexcept { return "BNB Smart Chain"; }
    };
    /// \brief Polygon PoS: chain id 137, POL.
    struct polygon : detail::evm_chain<137, native::POL> {
        /// \brief The name. \return `"Polygon"`.
        static constexpr std::string_view name() noexcept { return "Polygon"; }
    };
    /// \brief Base: chain id 8453, ETH.
    struct base : detail::evm_chain<8453, native::ETH> {
        /// \brief The name. \return `"Base"`.
        static constexpr std::string_view name() noexcept { return "Base"; }
    };
    /// \brief Arbitrum One: chain id 42161, ETH.
    struct arbitrum : detail::evm_chain<42161, native::ETH> {
        /// \brief The name. \return `"Arbitrum One"`.
        static constexpr std::string_view name() noexcept { return "Arbitrum One"; }
    };
    /// \brief Avalanche C-Chain: chain id 43114, AVAX.
    struct avalanche : detail::evm_chain<43114, native::AVAX> {
        /// \brief The name. \return `"Avalanche C-Chain"`.
        static constexpr std::string_view name() noexcept { return "Avalanche C-Chain"; }
    };
    /// \brief Base Sepolia, Base's testnet: chain id 84532, ETH.
    struct base_sepolia : detail::evm_chain<84532, native::ETH> {
        /// \brief The name. \return `"Base Sepolia"`.
        static constexpr std::string_view name() noexcept { return "Base Sepolia"; }
    };
    /// \brief Arbitrum Sepolia, Arbitrum's testnet: chain id 421614, ETH.
    struct arbitrum_sepolia : detail::evm_chain<421614, native::ETH> {
        /// \brief The name. \return `"Arbitrum Sepolia"`.
        static constexpr std::string_view name() noexcept { return "Arbitrum Sepolia"; }
    };
    /// \brief Sepolia, Ethereum's testnet: chain id 11155111, ETH.
    struct sepolia : detail::evm_chain<11155111, native::ETH> {
        /// \brief The name. \return `"Sepolia"`.
        static constexpr std::string_view name() noexcept { return "Sepolia"; }
    };
    /// \brief TRON mainnet: identifier 728126428 on its JSON-RPC, TRX; kind `tron`.
    struct tron : detail::evm_chain<728126428, native::TRX, chain_kind::tron> {
        /// \brief The name. \return `"TRON"`.
        static constexpr std::string_view name() noexcept { return "TRON"; }
    };
    /// \brief Bitcoin: BTC, no EIP-155 identifier.
    struct bitcoin : detail::plain_chain<native::BTC, chain_kind::bitcoin> {
        /// \brief The name. \return `"Bitcoin"`.
        static constexpr std::string_view name() noexcept { return "Bitcoin"; }
    };
    /// \brief Solana mainnet: SOL, no EIP-155 identifier.
    struct solana : detail::plain_chain<native::SOL, chain_kind::solana> {
        /// \brief The name. \return `"Solana"`.
        static constexpr std::string_view name() noexcept { return "Solana"; }
    };

} // namespace snicholls::chain

#endif // snicholls_core_chain_chains_hpp
