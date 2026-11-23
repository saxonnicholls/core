//
//  basic_token.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::basic_token<Chain, Address, Decimals, Symbol>`: a token as a type, declared in one line and
/// checked by the compiler.
/// \ingroup core
///
/// **Overview.** A deployed token's chain, contract, decimals and symbol never change, so a token is a type. A
/// product declares its own in one line,
///
///     struct SAX : snicholls::basic_token<snicholls::chain::base, "0x...", 18, "SAX"> {};
///
/// and gets every member `token_like` asks for, all `static constexpr`:
/// - `chain()`, a value of the chain type; `contract()`, the address as written; `symbol()`;
/// - `scale()`, the decimals;
/// - `code()`, `SYMBOL:chainid` (`SAX:8453`), built at compile time, so that the same symbol on two chains is two
///   assets.
///
/// The compiler checks the declaration: the address must be in its chain's canonical form (EIP-55 mixed-case hex
/// for an EVM chain, verified with core's Keccak-256; base58check `T...` for TRON, verified with SHA-256), the
/// decimals at most `max_scale`, and the code a valid asset code (so the symbol is capitals, digits and `.` `-` `_`).
/// A mistyped address is a compile error, never a silent loss of funds.
///
/// An issuer's single-supply token, one asset across its chains, keeps its bare symbol: the deriving struct hides
/// `code()` with its own (`static constexpr std::string_view code() noexcept { return "SAX"; }`).
///
/// \snippet token/basic_token_demo.cpp declare
///
/// **Relationships.** `concepts/token.hpp` (the concept it models), `interfaces/token.hpp`,
/// `interfaces/token_conformance.hpp`, `token/tokens.hpp` (the major tokens, declared with it), `chain/chains.hpp`,
/// `money/asset.hpp` (`asset_v<T>` works on any token type). Group: \ref core.
///
/// **Implementations.** One template; each token is a struct deriving from it, so that its name is its own in
/// diagnostics and documentation.
///
/// **Guarantees.** constexpr, `noexcept`, no state, no allocation; the code is a static array built once at compile
/// time. \pre `Chain` models `evm_chain_like` (an identifier is needed for the code: TRON's is 728126428).
///
/// **Errors.** Compile errors (`static_assert`) for a malformed or wrongly checksummed address, decimals above
/// `max_scale`, or a symbol that makes an invalid asset code.
///
/// **Performance.** Nothing at run time; the address check is one Keccak-256 (or two SHA-256) per token type at
/// compile time. Not benchmarked.
///
/// **Tests and demos.** `tests/token/basic_token_test.cpp`, `demos/token/basic_token_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<string_view>`, `fixed_string.hpp`, `concepts/chain.hpp`,
/// `concepts/token.hpp`, `money/asset.hpp`, `detail/token_checks.hpp` (which includes `hash/sha2.hpp` and
/// `hash/sha3.hpp`).
///
/// **Known issues.**
/// - A chain with no EIP-155 identifier (Bitcoin, Solana) is not accepted: an SPL token's code has no rule yet.
/// - The symbol must be an asset code's characters: `USDC.e` is written `USDC.E`, `USD₮0` as `USDT0`.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_token_basic_token_hpp
#define snicholls_core_token_basic_token_hpp

#include <snicholls/core/concepts/chain.hpp>
#include <snicholls/core/concepts/token.hpp>
#include <snicholls/core/detail/token_checks.hpp>
#include <snicholls/core/fixed_string.hpp>
#include <snicholls/core/money/asset.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace snicholls {

    namespace detail {
        // SYMBOL ':' chain id, as characters, built at compile time.
        template <basic_fixed_string Symbol, std::uint64_t Id>
        inline constexpr auto token_code = [] {
            std::array<char, Symbol.size() + 1 + token_digits(Id)> a{};
            std::size_t i = 0;
            for (; i < Symbol.size(); ++i)
                a[i] = Symbol[i];
            a[i] = ':';
            std::uint64_t n = Id;
            for (std::size_t j = a.size(); j-- > i + 1; n /= 10)
                a[j] = static_cast<char>('0' + n % 10);
            return a;
        }();
    } // namespace detail

    /// \brief A token deployed on `Chain` at `Address`, with `Decimals` decimals and the symbol `Symbol`.
    /// \tparam Chain    The chain type (`chain::ethereum`, `chain::tron`, ...); it must have an EIP-155 identifier.
    /// \tparam Address  The contract address in the chain's canonical text, checked at compile time.
    /// \tparam Decimals The decimals of the token's smallest unit, at most `max_scale`.
    /// \tparam Symbol   The issuer's symbol: capitals, digits, `.`, `-`, `_`.
    template <evm_chain_like Chain, basic_fixed_string Address, unsigned Decimals, basic_fixed_string Symbol>
    struct basic_token {
        static_assert(!detail::check_token_address(Chain::kind(), Address.view()),
                      "snicholls::basic_token: the address is not in its chain's canonical form "
                      "(EIP-55 checksummed 0x hex for an EVM chain, base58check T... for TRON)");
        static_assert(Decimals <= max_scale, "snicholls::basic_token: the decimals are above snicholls::max_scale");
        static_assert(Symbol.size() > 0, "snicholls::basic_token: the symbol is empty");

        /// \brief The chain it is deployed on.
        /// \return A value of the chain type.
        static constexpr Chain chain() noexcept { return {}; }
        /// \brief The contract address, in the chain's canonical text.
        /// \return `Address`.
        static constexpr std::string_view contract() noexcept { return Address.view(); }
        /// \brief The issuer's symbol.
        /// \return `Symbol`.
        static constexpr std::string_view symbol() noexcept { return Symbol.view(); }
        /// \brief The asset code: the symbol, `:` and the chain's identifier.
        /// \return `"SYMBOL:chainid"`.
        static constexpr std::string_view code() noexcept
        {
            constexpr const auto& c = detail::token_code<Symbol, Chain::chain_id()>;
            return {c.data(), c.size()};
        }
        /// \brief The decimals of the smallest unit.
        /// \return `Decimals`.
        static constexpr std::uint8_t scale() noexcept { return static_cast<std::uint8_t>(Decimals); }

        static_assert(is_asset_code(std::string_view{detail::token_code<Symbol, Chain::chain_id()>.data(),
                                                     detail::token_code<Symbol, Chain::chain_id()>.size()}),
                      "snicholls::basic_token: SYMBOL:chainid is not a valid asset code (capitals, digits, . - _; "
                      "at most 24 characters)");
    };

} // namespace snicholls

#endif // snicholls_core_token_basic_token_hpp
