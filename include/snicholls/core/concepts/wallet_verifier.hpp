//
//  wallet_verifier.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::wallet_verifier`: did this wallet sign this sign-in message (EIP-191 `personal_sign`)?
/// \ingroup core
///
/// **Overview.** The seam between Sign-In with Ethereum (EIP-4361) and the two ways a wallet proves a signature:
/// - an externally owned account: secp256k1 `ecrecover` of the EIP-191 hash (`personal_sign_hash`, which is
///   `keccak256("\x19Ethereum Signed Message:\n" + decimal length + message)`), compared with the address; high-s
///   refused (EIP-2), `v` 27 or 28 (0 or 1 mapped);
/// - a smart-contract wallet: ERC-1271 `isValidSignature(hash, signature)` returning `0x1626ba7e` on the chain the
///   message names, and ERC-6492 for a wallet not yet deployed (the counterfactual wrapper, magic suffix
///   `0x6492...6492`).
///
/// This is the concept half of a pair; the pure virtual half is `wallet_verifier_interface`
/// (`interfaces/wallet_verifier.hpp`), with `wallet_verifier_adapter<V>` and `wallet_verifier_ref`.
///
/// **The contract** (`interfaces/wallet_verifier_conformance.hpp` checks it):
/// - `ok()` exactly when `address` signed `message`; any other signer, a tampered message, or a malformed or empty
///   signature is `capability_errc::signature_invalid`;
/// - `address` is lowercase `0x` and 40 hex digits (`is_lowercase_evm_address`); anything else is
///   `capability_errc::invalid_argument`;
/// - a chain the verifier cannot reach for ERC-1271 is `capability_errc::unavailable` (retryable) or
///   `capability_errc::unsupported`;
/// - errors follow the redaction rule of `capability_error.hpp`: no message, signature, address or node response in
///   them.
///
/// \snippet concepts/wallet_verifier_demo.cpp concept
///
/// **Relationships.** `hash/sha3.hpp` (`keccak256`), `capability_error.hpp`, `capability_ids.hpp` (`wallet_address`);
/// the twin `interfaces/wallet_verifier.hpp`, the reference `interfaces/wallet_verifier_reference.hpp`, the kit
/// `interfaces/wallet_verifier_conformance.hpp`. Group: \ref core.
///
/// **Implementations.**
/// - `memory_wallet_verifier` (`interfaces/wallet_verifier_reference.hpp`): test wallets whose "signature" is keyed
///   HMAC-SHA256 over `personal_sign_hash`, standard library only, **for tests only**.
/// - Above core: servers' `crypto_wallet_verifier` (`auth/crypto_verifiers.hpp`): an EOA through crypto's
///   `secp256k1::recover_ethereum(hash, rsv, high_s::reject)`, the address the last 20 bytes of core's
///   `keccak256(key.xy())`; ERC-1271 and ERC-6492 through an `eth_call` on the message's chain (servers' EVM client).
///   secp256k1 is crypto's, never core's.
///
/// **Guarantees.** `is_lowercase_evm_address` and `personal_sign_hash` are `constexpr`, `noexcept` and allocation-free.
/// `wallet_signature_request` borrows everything for the call. Thread safety is the implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Not benchmarked: one per sign-in.
///
/// **Tests and demos.** `tests/concepts/wallet_verifier_test.cpp` (`personal_sign_hash` against published vectors,
/// checked with an independent Keccak), `demos/concepts/wallet_verifier_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`,
/// `capability_error.hpp`, `hash/digest.hpp`, `hash/sha3.hpp`.
///
/// **Known issues.** The address is compared as text: a checksummed (EIP-55) address must be lowercased first.
///
/// **Status.** New in 0.27.0, lifted from servers. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_wallet_verifier_hpp
#define snicholls_core_concepts_wallet_verifier_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/hash/digest.hpp>
#include <snicholls/core/hash/sha3.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls {

    /// \brief One wallet signature to check; everything is borrowed for the call.
    /// \ingroup core
    struct wallet_signature_request {
        std::uint64_t chain_id{1};             ///< the EIP-155 chain the message names (for ERC-1271)
        std::string_view address;              ///< lowercase `0x` and 40 hex digits
        std::string_view message;              ///< the exact text the wallet signed
        std::span<const std::byte> signature;  ///< 65 bytes r||s||v for an EOA; any length for ERC-1271 or ERC-6492
    };

    /// \brief Whether `text` is an EVM address in the contract's form: `0x` and 40 lowercase hex digits (stricter than
    /// `is_evm_address`, `concepts/onramp_provider.hpp`, which accepts either case).
    /// \ingroup core
    /// \param text The text.
    /// \return `true` when it is.
    [[nodiscard]] constexpr bool is_lowercase_evm_address(std::string_view text) noexcept
    {
        if (text.size() != 42 || text[0] != '0' || text[1] != 'x')
            return false;
        for (const char c : text.substr(2))
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                return false;
        return true;
    }

    /// \brief The EIP-191 (version `0x45`) hash a wallet signs for `personal_sign`:
    /// `keccak256("\x19Ethereum Signed Message:\n" + decimal length + message)`.
    /// \ingroup core
    /// \param message The text signed.
    /// \return The 32-byte hash.
    [[nodiscard]] constexpr digest<32> personal_sign_hash(std::string_view message) noexcept
    {
        std::array<char, 20> digits{};
        std::size_t n = digits.size();
        std::size_t length = message.size();
        do {
            digits[--n] = static_cast<char>('0' + length % 10);
            length /= 10;
        } while (length != 0);
        keccak256 h;
        h.update(std::string_view{"\x19" "Ethereum Signed Message:\n"});
        h.update(std::string_view{digits.data() + n, digits.size() - n});
        h.update(message);
        return h.finish();
    }

    /// \brief A wallet verifier: `v.verify(request)`, on a `const` verifier.
    /// \ingroup core
    template <typename V>
    concept wallet_verifier = requires(const V& v, const wallet_signature_request& r) {
        { v.verify(r) } -> std::same_as<capability_result<void>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_wallet_verifier_hpp
