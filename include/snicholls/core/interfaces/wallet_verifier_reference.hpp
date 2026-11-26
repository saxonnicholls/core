//
//  wallet_verifier_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_wallet_verifier`: the standard-library reference `wallet_verifier`, test wallets with a
/// deterministic keyed "signature", for tests only.
/// \ingroup core
///
/// **Overview.** `add_wallet(address, key)` registers a test wallet, `sign(address, message)` makes its 65-byte test
/// signature, and `verify` checks one in constant time. The scheme is **not** ECDSA: with `h =
/// personal_sign_hash(message)`, the signature is `r = HMAC-SHA256(key, h)`, `s = HMAC-SHA256(key, r)` and `v = 27`
/// (`0x1b`), so it has an EOA signature's length and shape and binds the wallet, the message and its EIP-191 framing,
/// but proves nothing about a real wallet. A real one needs secp256k1, which is crypto's (servers'
/// `crypto_wallet_verifier`). `fail_next(code, response)` makes the next `verify` fail as though a node had answered
/// `response`. **Never configure it in production.**
///
/// \snippet interfaces/wallet_verifier_reference_demo.cpp reference
///
/// **Relationships.** Models `wallet_verifier` (`concepts/wallet_verifier.hpp`); passes its kit in every form. Group:
/// \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** `verify` is `const`; it is safe to call concurrently once the wallets are added and no failure is
/// armed. Keys are held as `snicholls::secret`, and no operation returns them. Errors carry fixed messages only.
///
/// **Errors.** `invalid_argument` (an address not in the contract's form), `signature_invalid` (an unknown wallet, a
/// signature not 65 bytes, a wrong one), and whatever `fail_next` armed.
///
/// **Performance.** One Keccak-256 and two HMAC-SHA256.
///
/// **Tests and demos.** `tests/interfaces/wallet_verifier_reference_test.cpp`,
/// `demos/interfaces/wallet_verifier_reference_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<functional>`, `<map>`, `<optional>`, `<span>`, `<string>`,
/// `<string_view>`, `<vector>`, `hash/hmac.hpp`, `hash/secure.hpp`, `hash/sha2.hpp`, `secret.hpp`,
/// `interfaces/wallet_verifier.hpp`.
///
/// **Known issues.** ERC-1271 and ERC-6492 are not modelled: every wallet is an EOA.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_wallet_verifier_reference_hpp
#define snicholls_core_interfaces_wallet_verifier_reference_hpp

#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/interfaces/wallet_verifier.hpp>
#include <snicholls/core/secret.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief The reference wallet verifier: test wallets with a keyed, deterministic test signature.
    /// \ingroup core
    class memory_wallet_verifier {
    public:
        /// \brief The length of a test signature: an EOA signature's, r||s||v.
        static constexpr std::size_t signature_bytes = 65;

        /// \brief Registers (or replaces) a test wallet.
        /// \param address Lowercase `0x` and 40 hex digits.
        /// \param key     Its test key (any bytes).
        /// \return Success, or `invalid_argument` for an address not in the contract's form.
        capability_result<void> add_wallet(std::string_view address, std::string_view key)
        {
            if (!is_lowercase_evm_address(address))
                return fail(capability_errc::invalid_argument, {}, "not a lowercase EVM address");
            wallets_.insert_or_assign(std::string(address), secret<std::string>{std::string(key)});
            return ok();
        }

        /// \brief The test signature of `message` by `address`.
        /// \param address A registered wallet.
        /// \param message The text.
        /// \return 65 bytes, or none for an unknown wallet.
        [[nodiscard]] std::vector<std::byte> sign(std::string_view address, std::string_view message) const
        {
            const auto it = wallets_.find(address);
            if (it == wallets_.end())
                return {};
            const auto s = test_signature(it->second.reveal(), message);
            return {s.begin(), s.end()};
        }

        /// \brief Checks a test signature.
        /// \param r The request.
        /// \return Success, or `invalid_argument`, `signature_invalid`, or the armed failure.
        [[nodiscard]] capability_result<void> verify(const wallet_signature_request& r) const
        {
            if (injected_) {
                const capability_error e = *injected_;
                injected_.reset();
                return err(e);
            }
            if (!is_lowercase_evm_address(r.address))
                return fail(capability_errc::invalid_argument, {}, "not a lowercase EVM address");
            const auto it = wallets_.find(r.address);
            if (it == wallets_.end() || r.signature.size() != signature_bytes)
                return fail(capability_errc::signature_invalid);
            const auto expected = test_signature(it->second.reveal(), r.message);
            if (!constant_time_equal(std::span<const std::byte>(expected), r.signature))
                return fail(capability_errc::signature_invalid);
            return ok();
        }

        /// \brief Makes the next `verify` fail (once) as though a node had answered `response`.
        /// \param code     The error's code.
        /// \param response The response: its first word becomes the provider code when it is one; nothing else is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        static std::array<std::byte, signature_bytes> test_signature(std::string_view key, std::string_view message) noexcept
        {
            const auto h = personal_sign_hash(message);
            const auto r = hmac<sha256>::mac(key, h);
            const auto s = hmac<sha256>::mac(key, r);
            std::array<std::byte, signature_bytes> out{};
            for (std::size_t i = 0; i < 32; ++i) {
                out[i] = r.bytes[i];
                out[32 + i] = s.bytes[i];
            }
            out[64] = std::byte{0x1b};
            return out;
        }

        std::map<std::string, secret<std::string>, std::less<>> wallets_;
        mutable std::optional<capability_error> injected_;
    };

    static_assert(wallet_verifier<memory_wallet_verifier>, "snicholls::memory_wallet_verifier must model snicholls::wallet_verifier");

} // namespace snicholls

#endif // snicholls_core_interfaces_wallet_verifier_reference_hpp
