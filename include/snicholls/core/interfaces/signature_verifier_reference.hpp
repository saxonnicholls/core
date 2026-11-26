//
//  signature_verifier_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_signature_verifier`: the standard-library reference `signature_verifier`, HS256 under a
/// `memory_key_store`'s keys, for tests only.
/// \ingroup core
///
/// **Overview.** Borrows a `memory_key_store` (`interfaces/key_store_reference.hpp`), which must outlive it, and checks
/// HS256 tags with core's `hmac<sha256>` and `constant_time_equal` for every version the store still holds: retired
/// versions verify, as rotation requires. Every asymmetric algorithm (EdDSA, ES256, ES256K, RS256) is `unsupported`:
/// those need constant-time elliptic-curve and RSA code, which is crypto's (servers' `crypto_signature_verifier`).
/// `fail_next(code, response)` makes the next `verify` fail as though the store had answered `response`.
/// **Never configure it in production**: a production JWT policy refuses HS256 before it reaches a verifier.
///
/// \snippet interfaces/signature_verifier_reference_demo.cpp reference
///
/// **Relationships.** Models `signature_verifier` (`concepts/signature_verifier.hpp`); passes its kit in every form;
/// verifies what `memory_signer` (`interfaces/signer_reference.hpp`) signs. Group: \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** `verify` is `const` and reads the store; it is as thread-safe as the store, which is not, while a
/// failure is armed. Errors carry fixed messages only.
///
/// **Errors.** `unsupported` (not HS256), `invalid_argument` (no version), `not_found` (no such key or version),
/// `signature_invalid`, and whatever `fail_next` armed.
///
/// **Performance.** One `hmac<sha256>` and a constant-time compare
/// (`benchmarks/interfaces/key_store_reference_benchmark.cpp` measures the same MAC).
///
/// **Tests and demos.** `tests/interfaces/signature_verifier_reference_test.cpp`,
/// `demos/interfaces/signature_verifier_reference_demo.cpp`.
///
/// **Dependencies.** `<optional>`, `<string_view>`, `<vector>`, `interfaces/key_store_reference.hpp`,
/// `interfaces/signature_verifier.hpp`.
///
/// **Known issues.** HS256 only, by design.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_signature_verifier_reference_hpp
#define snicholls_core_interfaces_signature_verifier_reference_hpp

#include <snicholls/core/interfaces/key_store_reference.hpp>
#include <snicholls/core/interfaces/signature_verifier.hpp>

#include <optional>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief The reference signature verifier: HS256 under a `memory_key_store`'s keys.
    /// \ingroup core
    class memory_signature_verifier {
    public:
        /// \brief Verifies under the keys of `store`, which must outlive this verifier.
        /// \param store The store.
        explicit memory_signature_verifier(const memory_key_store& store) noexcept : store_(&store) {}

        /// \brief Checks an HS256 tag.
        /// \param r The request.
        /// \return Success, or `unsupported`, `invalid_argument`, `not_found`, `signature_invalid`.
        [[nodiscard]] capability_result<void> verify(const verify_request& r) const
        {
            if (injected_) {
                const capability_error e = *injected_;
                injected_.reset();
                return err(e);
            }
            if (r.algorithm != signature_algorithm::hs256)
                return fail(capability_errc::unsupported, {}, "the reference verifies HS256 only");
            if (r.key_version == 0)
                return fail(capability_errc::invalid_argument, {}, "an HS256 key needs its version");
            const auto id = provider_id::make(r.key_id);
            const auto current = store_->info(r.key_id);
            if (!id || current.is_err() || r.key_version > current->version)
                return fail(capability_errc::not_found, {}, "no such key version");
            const signature s{signature_algorithm::hs256, *id, r.key_version,
                              std::vector<std::byte>(r.signature.begin(), r.signature.end())};
            if (!store_->verify(s, r.message))
                return fail(capability_errc::signature_invalid);
            return ok();
        }

        /// \brief Makes the next `verify` fail (once) as though the key store had answered `response`.
        /// \param code     The error's code.
        /// \param response The response: its first word becomes the provider code when it is one; nothing else is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        const memory_key_store* store_;
        mutable std::optional<capability_error> injected_;
    };

    static_assert(signature_verifier<memory_signature_verifier>,
                  "snicholls::memory_signature_verifier must model snicholls::signature_verifier");

} // namespace snicholls

#endif // snicholls_core_interfaces_signature_verifier_reference_hpp
