//
//  native.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The native assets of the major chains as types: `snicholls::native::ETH`, `BTC`, `SOL`, `TRX`, ...
/// \ingroup core
///
/// **Overview.** A chain's native asset is what its fees are paid in. Its symbol and decimals are fixed by the
/// chain's protocol and never change, so like ISO 4217 codes they are compile-time types. Each answers:
/// - `code()`, `name()` and `scale()`, the decimals of its smallest unit (wei, satoshi, lamport, sun);
/// - `slip44()`, its SLIP-44 coin type, the number BIP-44 derivation paths use.
///
/// They model `asset_like` and `has_name`, so `asset_v<native::ETH>` and `money_of<native::ETH>` work. Tokens (ERC-20,
/// TRC-20, SPL) are not here: they are deployed contracts, not protocol constants, and they live in the
/// application's token registry, which gives each a unique code (`USDC:56`). A product declares its own tokens as
/// types of the same shape.
///
/// \snippet money/native_demo.cpp native
///
/// **Relationships.** `money/asset.hpp`, `chain/chains.hpp` (each chain names its native asset), `money/money.hpp`.
/// Group: \ref core.
///
/// **Implementations.** One type per asset.
///
/// **Guarantees.** constexpr, `noexcept`, no state.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/money/native_test.cpp`: the decimals are checked against each protocol's
/// specification, and the coin types against the SLIP-44 registry. Demo: `demos/money/native_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<string_view>`.
///
/// **Known issues.**
/// - SLIP-44 numbers BNB 714 (its original Beacon chain); BNB Smart Chain wallets derive with Ethereum's 60. The type
///   gives 714, the registered value.
/// - POL keeps MATIC's coin type, 966.
///
/// **Status.** New in 0.22.0.

#ifndef snicholls_core_money_native_hpp
#define snicholls_core_money_native_hpp

#include <cstdint>
#include <string_view>

namespace snicholls::native {

    /// \brief Bitcoin: 8 decimals (satoshi), coin type 0.
    struct BTC {
        /// \brief The symbol.
        /// \return `"BTC"`.
        static constexpr std::string_view code() noexcept { return "BTC"; }
        /// \brief The asset's name.
        /// \return `"Bitcoin"`.
        static constexpr std::string_view name() noexcept { return "Bitcoin"; }
        /// \brief The decimals of its smallest unit.
        /// \return 8.
        static constexpr std::uint8_t scale() noexcept { return 8; }
        /// \brief Its SLIP-44 coin type.
        /// \return 0.
        static constexpr std::uint32_t slip44() noexcept { return 0; }
    };

    /// \brief Litecoin: 8 decimals, coin type 2.
    struct LTC {
        /// \brief The symbol.
        /// \return `"LTC"`.
        static constexpr std::string_view code() noexcept { return "LTC"; }
        /// \brief The asset's name.
        /// \return `"Litecoin"`.
        static constexpr std::string_view name() noexcept { return "Litecoin"; }
        /// \brief The decimals of its smallest unit.
        /// \return 8.
        static constexpr std::uint8_t scale() noexcept { return 8; }
        /// \brief Its SLIP-44 coin type.
        /// \return 2.
        static constexpr std::uint32_t slip44() noexcept { return 2; }
    };

    /// \brief Dogecoin: 8 decimals, coin type 3.
    struct DOGE {
        /// \brief The symbol.
        /// \return `"DOGE"`.
        static constexpr std::string_view code() noexcept { return "DOGE"; }
        /// \brief The asset's name.
        /// \return `"Dogecoin"`.
        static constexpr std::string_view name() noexcept { return "Dogecoin"; }
        /// \brief The decimals of its smallest unit.
        /// \return 8.
        static constexpr std::uint8_t scale() noexcept { return 8; }
        /// \brief Its SLIP-44 coin type.
        /// \return 3.
        static constexpr std::uint32_t slip44() noexcept { return 3; }
    };

    /// \brief Ether: 18 decimals (wei), coin type 60. The native asset of Ethereum and of its rollups (Base, Arbitrum, Optimism).
    struct ETH {
        /// \brief The symbol.
        /// \return `"ETH"`.
        static constexpr std::string_view code() noexcept { return "ETH"; }
        /// \brief The asset's name.
        /// \return `"Ether"`.
        static constexpr std::string_view name() noexcept { return "Ether"; }
        /// \brief The decimals of its smallest unit.
        /// \return 18.
        static constexpr std::uint8_t scale() noexcept { return 18; }
        /// \brief Its SLIP-44 coin type.
        /// \return 60.
        static constexpr std::uint32_t slip44() noexcept { return 60; }
    };

    /// \brief XRP: 6 decimals (drops), coin type 144.
    struct XRP {
        /// \brief The symbol.
        /// \return `"XRP"`.
        static constexpr std::string_view code() noexcept { return "XRP"; }
        /// \brief The asset's name.
        /// \return `"XRP"`.
        static constexpr std::string_view name() noexcept { return "XRP"; }
        /// \brief The decimals of its smallest unit.
        /// \return 6.
        static constexpr std::uint8_t scale() noexcept { return 6; }
        /// \brief Its SLIP-44 coin type.
        /// \return 144.
        static constexpr std::uint32_t slip44() noexcept { return 144; }
    };

    /// \brief TRON's TRX: 6 decimals (sun), coin type 195.
    struct TRX {
        /// \brief The symbol.
        /// \return `"TRX"`.
        static constexpr std::string_view code() noexcept { return "TRX"; }
        /// \brief The asset's name.
        /// \return `"TRON"`.
        static constexpr std::string_view name() noexcept { return "TRON"; }
        /// \brief The decimals of its smallest unit.
        /// \return 6.
        static constexpr std::uint8_t scale() noexcept { return 6; }
        /// \brief Its SLIP-44 coin type.
        /// \return 195.
        static constexpr std::uint32_t slip44() noexcept { return 195; }
    };

    /// \brief Solana's SOL: 9 decimals (lamports), coin type 501.
    struct SOL {
        /// \brief The symbol.
        /// \return `"SOL"`.
        static constexpr std::string_view code() noexcept { return "SOL"; }
        /// \brief The asset's name.
        /// \return `"Solana"`.
        static constexpr std::string_view name() noexcept { return "Solana"; }
        /// \brief The decimals of its smallest unit.
        /// \return 9.
        static constexpr std::uint8_t scale() noexcept { return 9; }
        /// \brief Its SLIP-44 coin type.
        /// \return 501.
        static constexpr std::uint32_t slip44() noexcept { return 501; }
    };

    /// \brief Toncoin: 9 decimals (nanotons), coin type 607.
    struct TON {
        /// \brief The symbol.
        /// \return `"TON"`.
        static constexpr std::string_view code() noexcept { return "TON"; }
        /// \brief The asset's name.
        /// \return `"Toncoin"`.
        static constexpr std::string_view name() noexcept { return "Toncoin"; }
        /// \brief The decimals of its smallest unit.
        /// \return 9.
        static constexpr std::uint8_t scale() noexcept { return 9; }
        /// \brief Its SLIP-44 coin type.
        /// \return 607.
        static constexpr std::uint32_t slip44() noexcept { return 607; }
    };

    /// \brief BNB: 18 decimals on BNB Smart Chain, coin type 714.
    struct BNB {
        /// \brief The symbol.
        /// \return `"BNB"`.
        static constexpr std::string_view code() noexcept { return "BNB"; }
        /// \brief The asset's name.
        /// \return `"BNB"`.
        static constexpr std::string_view name() noexcept { return "BNB"; }
        /// \brief The decimals of its smallest unit.
        /// \return 18.
        static constexpr std::uint8_t scale() noexcept { return 18; }
        /// \brief Its SLIP-44 coin type.
        /// \return 714.
        static constexpr std::uint32_t slip44() noexcept { return 714; }
    };

    /// \brief Polygon's POL: 18 decimals, coin type 966 (MATIC's).
    struct POL {
        /// \brief The symbol.
        /// \return `"POL"`.
        static constexpr std::string_view code() noexcept { return "POL"; }
        /// \brief The asset's name.
        /// \return `"Polygon"`.
        static constexpr std::string_view name() noexcept { return "Polygon"; }
        /// \brief The decimals of its smallest unit.
        /// \return 18.
        static constexpr std::uint8_t scale() noexcept { return 18; }
        /// \brief Its SLIP-44 coin type.
        /// \return 966.
        static constexpr std::uint32_t slip44() noexcept { return 966; }
    };

    /// \brief Cardano's ADA: 6 decimals (lovelace), coin type 1815.
    struct ADA {
        /// \brief The symbol.
        /// \return `"ADA"`.
        static constexpr std::string_view code() noexcept { return "ADA"; }
        /// \brief The asset's name.
        /// \return `"Cardano"`.
        static constexpr std::string_view name() noexcept { return "Cardano"; }
        /// \brief The decimals of its smallest unit.
        /// \return 6.
        static constexpr std::uint8_t scale() noexcept { return 6; }
        /// \brief Its SLIP-44 coin type.
        /// \return 1815.
        static constexpr std::uint32_t slip44() noexcept { return 1815; }
    };

    /// \brief Avalanche's AVAX: 18 decimals on the C-Chain, coin type 9000.
    struct AVAX {
        /// \brief The symbol.
        /// \return `"AVAX"`.
        static constexpr std::string_view code() noexcept { return "AVAX"; }
        /// \brief The asset's name.
        /// \return `"Avalanche"`.
        static constexpr std::string_view name() noexcept { return "Avalanche"; }
        /// \brief The decimals of its smallest unit.
        /// \return 18.
        static constexpr std::uint8_t scale() noexcept { return 18; }
        /// \brief Its SLIP-44 coin type.
        /// \return 9000.
        static constexpr std::uint32_t slip44() noexcept { return 9000; }
    };

} // namespace snicholls::native

#endif // snicholls_core_money_native_hpp
