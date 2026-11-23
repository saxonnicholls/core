//
//  token_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_token`: the conformance kit of the token capability.
/// \ingroup core
///
/// **Overview.** Drives any `token_like` implementation, a type or an object behind `token_interface` (a run-time
/// registry's row), through the contract and says which check failed:
/// - the asset contract (`check_asset`): a valid code, decimals at most `max_scale`;
/// - the chain contract (`check_chain`) for `chain()`;
/// - the symbol is not empty, and the code is `symbol():chain_id` (`USDC:1`) or, for a single-supply token, the bare
///   symbol;
/// - the contract address is in its chain's canonical form: for an EVM chain `0x` and 40 hex digits whose case is
///   the EIP-55 checksum (Keccak-256 of the lower-case digits); for TRON 34 base58 characters beginning `T` whose
///   decoded 25 bytes are `0x41`, the address and a correct double-SHA-256 checksum; for any other kind, not empty;
/// - every answer is the same on every call.
///
/// No test framework is included; a test asserts that the result is empty.
///
/// \snippet interfaces/token_conformance_demo.cpp conformance
///
/// **Relationships.** `concepts/token.hpp`, `interfaces/token.hpp`, `interfaces/asset_conformance.hpp`,
/// `interfaces/chain_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** Not applicable.
///
/// **Guarantees.** constexpr, `noexcept`, no allocation. The same address checks run at compile time in
/// `basic_token`.
///
/// **Errors.** The returned message names the failed check.
///
/// **Performance.** One Keccak-256 (EVM) or two SHA-256 (TRON) per call. Not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/token_conformance_test.cpp`, `demos/interfaces/token_conformance_demo.cpp`.
///
/// **Dependencies.** `<optional>`, `<string_view>`, `concepts/token.hpp`, `interfaces/asset_conformance.hpp`,
/// `interfaces/chain_conformance.hpp`, `detail/token_checks.hpp` (which includes `hash/sha2.hpp` and
/// `hash/sha3.hpp`).
///
/// **Known issues.** An all-lower-case EVM address, which EIP-55 calls unchecksummed, is rejected: the canonical
/// text is the checksummed one.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_token_conformance_hpp
#define snicholls_core_interfaces_token_conformance_hpp

#include <snicholls/core/concepts/token.hpp>
#include <snicholls/core/detail/token_checks.hpp>
#include <snicholls/core/interfaces/asset_conformance.hpp>
#include <snicholls/core/interfaces/chain_conformance.hpp>

#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief Check `t` against the token contract.
    /// \param t The implementation.
    /// \return `std::nullopt` if it conforms, or the failed check.
    template <token_like T>
    [[nodiscard]] constexpr std::optional<std::string_view> check_token(const T& t) noexcept
    {
        if (const auto failed = check_asset(t))
            return failed;
        if (const auto failed = check_chain(t.chain()))
            return failed;
        const std::string_view symbol = t.symbol();
        const std::string_view contract = t.contract();
        if (symbol.empty())
            return "token: symbol() is empty";
        if (!detail::is_token_code(t.code(), symbol, detail::chain_id_of(t.chain())))
            return "token: code() is neither symbol() nor symbol():chain_id";
        if (const auto failed = detail::check_token_address(t.chain().kind(), contract))
            return failed;
        if (std::string_view{t.symbol()} != symbol || std::string_view{t.contract()} != contract)
            return "token: symbol() or contract() changes between calls";
        return std::nullopt;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_token_conformance_hpp
