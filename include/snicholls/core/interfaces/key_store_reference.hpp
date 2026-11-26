//
//  key_store_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_key_store`: the standard-library reference `key_store` and `signer` in one, HS256 keys in
/// memory, for tests only.
/// \ingroup core
///
/// **Overview.** Holds HMAC-SHA256 keys (core's `hmac<sha256>`), so a product's tests can create, rotate, retire and
/// destroy keys and sign with them without a KMS. HS256 is the only algorithm the standard library can give: ES256,
/// EdDSA and ES256K need constant-time elliptic-curve code above core, so `create` refuses them with `unsupported`.
/// **Never use it for production tokens.**
///
/// Key material comes from the caller's `random_source` (the operating system's in a program, a fixed pattern in a
/// test: `deterministic()`), so the reference reads nothing from the system. Keys are held as `snicholls::secret`,
/// and no operation returns them. `verify` is the test-side check of a signature (a constant-time compare).
/// `fail_next(code, response)` makes the next mutating call fail as though the KMS had answered `response`.
///
/// \snippet interfaces/key_store_reference_demo.cpp reference
///
/// **Relationships.** Models `key_store` (`concepts/key_store.hpp`) and `signer` (`concepts/signer.hpp`); passes both
/// kits in every form; `interfaces/signer_reference.hpp` names it `memory_signer`. Group: \ref core.
///
/// **Implementations / variants.** One.
/// `basic_memory_key_store<Clock>` reads any `utc_clock` (`time/utc_clock.hpp`), owned by value and reached through
/// `clock()`; `memory_key_store` is `basic_memory_key_store<manual_utc_clock>`.
///
/// **Guarantees.** Not thread-safe. Signing is deterministic. Key ids are `key-1`, `key-2`, ... and never reused.
/// Errors follow the redaction rule: no key material, payload or response in them.
///
/// **Errors.** As the two contracts, and whatever `fail_next` armed.
///
/// **Performance.** `benchmarks/interfaces/key_store_reference_benchmark.cpp`: `sign` against `hmac<sha256>::mac`
/// written by hand. Not yet quoted from a quiet machine: `make
/// benchmarks` records them in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/interfaces/key_store_reference_test.cpp`,
/// `demos/interfaces/key_store_reference_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<functional>`, `<map>`, `<memory>`, `<optional>`, `<span>`,
/// `<string>`, `<string_view>`, `<utility>`, `<vector>`, `hash/hmac.hpp`, `hash/secure.hpp`, `hash/sha2.hpp`,
/// `secret.hpp`, `interfaces/key_store.hpp`, `interfaces/signer.hpp`, `time/utc_clock.hpp`.
///
/// **Known issues.** HS256 only, by design.
///
/// **Status.** New in 0.23.0; the clock a `utc_clock` template parameter, owned, since 0.27.0 (it was a
/// `std::function<utc_s()>`).

#ifndef snicholls_core_interfaces_key_store_reference_hpp
#define snicholls_core_interfaces_key_store_reference_hpp

#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/interfaces/key_store.hpp>
#include <snicholls/core/interfaces/signer.hpp>
#include <snicholls/core/secret.hpp>
#include <snicholls/core/time/utc_clock.hpp>

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The reference key store and signer (HS256, in memory).
    /// \ingroup core
    /// \tparam Clock A `utc_clock` (`time/utc_clock.hpp`), owned by value; `manual_utc_clock` by default, moved by
    ///               `clock().advance(d)`.
    template <utc_clock Clock = manual_utc_clock>
    class basic_memory_key_store {
    public:
        /// \brief Fills a buffer with key material.
        using random_source = std::function<void(std::span<std::byte>)>;
        /// \brief The clock.
        using clock_type = Clock;

        /// \brief The key length: 32 bytes, SHA-256's output (RFC 7518 section 3.2 requires at least that).
        static constexpr std::size_t key_bytes = 32;

        /// \brief A store drawing key material from `random` and the time from `clock`.
        /// \param random The key material source.
        /// \param clock  The clock.
        basic_memory_key_store(random_source random, Clock clock) : random_(std::move(random)), clock_(std::move(clock)) {}

        /// \brief The clock, to move a manual one: `clock().advance(d)`.
        /// \return The clock.
        [[nodiscard]] Clock& clock() noexcept { return clock_; }
        /// \brief The clock.
        /// \return The clock.
        [[nodiscard]] const Clock& clock() const noexcept { return clock_; }

        /// \brief A store for tests: key material is the byte pattern 0, 1, 2, ... (so every run is the same) and the
        /// clock is fixed at `now`. Never for anything but tests.
        /// \param now The fixed time.
        /// \return The store.
        [[nodiscard]] static basic_memory_key_store deterministic(utc_s now = utc_s::from_unix_seconds(1'800'000'000))
            requires std::same_as<Clock, manual_utc_clock>
        {
            auto counter = std::make_shared<std::uint8_t>(0);
            return basic_memory_key_store{[counter](std::span<std::byte> out) {
                                              for (auto& b : out)
                                                  b = static_cast<std::byte>((*counter)++);
                                          },
                                          manual_utc_clock{now}};
        }

        /// \brief Creates a key (HS256 only).
        /// \param spec What to create.
        /// \return Version 1, active, or `unsupported` or `invalid_argument`.
        capability_result<key_info> create(const key_spec& spec)
        {
            if (auto e = take_injected())
                return err(*e);
            if (spec.algorithm != signature_algorithm::hs256)
                return fail(capability_errc::unsupported, {}, "the reference holds HS256 keys only");
            auto label = make_id<short_text>(spec.label);
            if (!label)
                return err(label.unwrap_err());
            const auto id = *provider_id::make("key-" + std::to_string(++next_id_));
            entry e{spec.algorithm, *label, {}};
            e.versions.push_back(make_version(1));
            const auto [it, inserted] = keys_.emplace(std::string(id.view()), std::move(e));
            static_cast<void>(inserted);
            return ok(describe(it->first, it->second, it->second.versions.back()));
        }

        /// \brief Makes the next version active and retires the previous one.
        /// \param id The key.
        /// \return The new version, or `not_found` or `conflict`.
        capability_result<key_info> rotate(std::string_view id)
        {
            if (auto e = take_injected())
                return err(*e);
            const auto it = keys_.find(id);
            if (it == keys_.end())
                return fail(capability_errc::not_found);
            auto& versions = it->second.versions;
            if (versions.back().state != key_state::active)
                return fail(capability_errc::conflict, {}, "a retired key does not rotate");
            versions.push_back(make_version(static_cast<std::uint32_t>(versions.size() + 1)));
            versions[versions.size() - 2].state = key_state::retired;
            return ok(describe(it->first, it->second, versions.back()));
        }

        /// \brief Retires every version.
        /// \param id The key.
        /// \return Success, or `not_found`.
        capability_result<void> retire(std::string_view id)
        {
            if (auto e = take_injected())
                return err(*e);
            const auto it = keys_.find(id);
            if (it == keys_.end())
                return fail(capability_errc::not_found);
            for (auto& v : it->second.versions)
                v.state = key_state::retired;
            return ok();
        }

        /// \brief Destroys every version; the id is never reused.
        /// \param id The key.
        /// \return Success, or `not_found`.
        capability_result<void> destroy(std::string_view id)
        {
            if (auto e = take_injected())
                return err(*e);
            const auto it = keys_.find(id);
            if (it == keys_.end())
                return fail(capability_errc::not_found);
            keys_.erase(it);
            return ok();
        }

        /// \brief The current version.
        /// \param id The key.
        /// \return Its description, or `not_found`.
        [[nodiscard]] capability_result<key_info> info(std::string_view id) const
        {
            const auto it = keys_.find(id);
            if (it == keys_.end())
                return fail(capability_errc::not_found);
            return ok(describe(it->first, it->second, it->second.versions.back()));
        }

        /// \brief Every version not destroyed, by id then version.
        /// \return The versions.
        [[nodiscard]] capability_result<std::vector<key_info>> list() const
        {
            std::vector<key_info> out;
            for (const auto& [id, e] : keys_)
                for (const auto& v : e.versions)
                    out.push_back(describe(id, e, v));
            return ok(std::move(out));
        }

        /// \brief Signs with the key's current version: HMAC-SHA256 over the message.
        /// \param r The request.
        /// \return The 32-byte tag, or `unsupported` (a digest), `not_found` or `conflict`.
        capability_result<signature> sign(const sign_request& r)
        {
            if (auto e = take_injected())
                return err(*e);
            if (r.kind == payload_kind::digest)
                return fail(capability_errc::unsupported, {}, "HS256 hashes for itself: no digests");
            const auto it = keys_.find(r.key_id);
            if (it == keys_.end())
                return fail(capability_errc::not_found);
            const version& v = it->second.versions.back();
            if (v.state != key_state::active)
                return fail(capability_errc::conflict, {}, "the key is retired");
            return ok(signature{signature_algorithm::hs256, *provider_id::make(it->first), v.number, mac(v, r.payload)});
        }

        /// \brief HS256 has no public key.
        /// \param k The key and version.
        /// \return `not_found` for an unknown key or version, otherwise `unsupported`.
        [[nodiscard]] capability_result<public_key> public_key_of(const key_ref& k) const
        {
            const auto it = keys_.find(k.key_id);
            if (it == keys_.end() || k.version > it->second.versions.size())
                return fail(capability_errc::not_found);
            return fail(capability_errc::unsupported, {}, "HS256 has no public key");
        }

        /// \brief The test-side check: whether `s` is this store's tag over `payload`, in constant time.
        /// \param s       The signature.
        /// \param payload The signed bytes.
        /// \return `true` when it verifies under a version not destroyed.
        [[nodiscard]] bool verify(const signature& s, std::span<const std::byte> payload) const
        {
            const auto it = keys_.find(s.key_id.view());
            if (it == keys_.end() || s.algorithm != signature_algorithm::hs256 || s.key_version == 0 ||
                s.key_version > it->second.versions.size())
                return false;
            const auto expected = mac(it->second.versions[s.key_version - 1], payload);
            return constant_time_equal(std::span<const std::byte>(expected), std::span<const std::byte>(s.bytes));
        }

        /// \brief Makes the next `create`, `rotate`, `retire`, `destroy` or `sign` fail (once) as though the KMS
        /// answered `response`.
        /// \param code     The error's code.
        /// \param response The KMS's response: its first word becomes the provider code when it is one, and nothing
        ///                 else of it is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        // The clock's time, in the seconds every record of this reference carries.
        [[nodiscard]] utc_s stamp() const noexcept(noexcept(std::declval<const Clock&>().now()))
        {
            return detail::clock_now<std::chrono::seconds>(clock_);
        }
        struct version {
            std::uint32_t number{0};
            key_state state{key_state::active};
            utc_s created_at{};
            secret<std::string> material;
        };
        struct entry {
            signature_algorithm algorithm{signature_algorithm::hs256};
            short_text label;
            std::vector<version> versions;
        };

        std::optional<capability_error> take_injected() noexcept
        {
            auto e = injected_;
            injected_.reset();
            return e;
        }

        version make_version(std::uint32_t number)
        {
            std::string bytes(key_bytes, '\0');
            random_(std::as_writable_bytes(std::span<char>(bytes.data(), bytes.size())));
            return version{number, key_state::active, stamp(), secret<std::string>{std::move(bytes)}};
        }

        static std::vector<std::byte> mac(const version& v, std::span<const std::byte> payload)
        {
            const auto tag = hmac<sha256>::mac(v.material.reveal(), payload);
            return {tag.bytes.begin(), tag.bytes.end()};
        }

        static key_info describe(const std::string& id, const entry& e, const version& v)
        {
            return key_info{*provider_id::make(id), v.number, e.algorithm, v.state, e.label, v.created_at};
        }

        random_source random_;
        Clock clock_;
        std::map<std::string, entry, std::less<>> keys_;
        std::uint64_t next_id_{0};
        std::optional<capability_error> injected_;
    };

    /// \brief The reference with a `manual_utc_clock`: what tests use.
    /// \ingroup core
    using memory_key_store = basic_memory_key_store<>;

    static_assert(key_store<memory_key_store>, "snicholls::memory_key_store must model snicholls::key_store");
    static_assert(signer<memory_key_store>, "snicholls::memory_key_store must model snicholls::signer");

} // namespace snicholls

#endif // snicholls_core_interfaces_key_store_reference_hpp
