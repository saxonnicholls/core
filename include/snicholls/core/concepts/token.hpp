//
//  token.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::token_like`: a token deployed on a chain, as a concept.
/// \ingroup core
///
/// **Overview.** A token is an asset that a contract on a chain issues: USDC on Ethereum, USDT on TRON. Beyond
/// `asset_like`'s `code()` and `scale()` (its decimals) it answers:
/// - `chain()`, a `chain_like` value: the chain it is deployed on;
/// - `contract()`, the contract's address in its chain's canonical text: EIP-55 checksummed hex for an EVM chain,
///   base58check (`T...`) for TRON;
/// - `symbol()`, the symbol the issuer gives it (`USDC`).
///
/// Its asset code is `SYMBOL:chainid` (`USDC:1`, `USDC:56`, `USDT:728126428`), so that one symbol on two chains is
/// two assets in one book; an issuer's single-supply token may keep its bare symbol. The concept is written over object
/// expressions, so it is modelled by compile-time types (`snicholls::token::ethereum::USDC`, a product's own
/// `basic_token<...>`) and by any object behind the virtual twin `token_interface` (`interfaces/token.hpp`).
///
/// \snippet concepts/token_demo.cpp concepts
///
/// **Relationships.** `concepts/asset.hpp`, `concepts/chain.hpp`, `interfaces/token.hpp`,
/// `interfaces/token_conformance.hpp`, `token/basic_token.hpp`, `token/tokens.hpp`. Group: \ref core.
///
/// **Implementations.** `basic_token<C, Address, Decimals, Symbol>`, the curated types of `token/tokens.hpp`,
/// `token_model<T>`.
///
/// **Guarantees.** A concept: no code, no cost.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/token_test.cpp`, `demos/concepts/token_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<string_view>`, `concepts/asset.hpp`, `concepts/chain.hpp`.
///
/// **Known issues.** The concept does not check the address's form or the code's shape; the conformance kit
/// `check_token` does.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_concepts_token_hpp
#define snicholls_core_concepts_token_hpp

#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/concepts/chain.hpp>

#include <concepts>
#include <string_view>

namespace snicholls {

    /// \brief An `asset_like` type whose objects also answer `chain()` (a `chain_like` value), `contract()` and
    /// `symbol()`.
    template <typename T>
    concept token_like = asset_like<T> && requires(const T& t) {
        { t.chain() } -> chain_like;
        { t.contract() } -> std::convertible_to<std::string_view>;
        { t.symbol() } -> std::convertible_to<std::string_view>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_token_hpp
