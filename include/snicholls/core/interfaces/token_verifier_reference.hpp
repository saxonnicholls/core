//
//  token_verifier_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_token_verifier`: the standard-library reference `token_verifier`, HS256-tagged tokens it
/// issues itself, for tests only.
/// \ingroup core
///
/// **Overview.** `issue(claims)` returns a token `mt1.<serial>.<tag>`, where the tag is the lowercase hexadecimal
/// HMAC-SHA256 (core's `hmac<sha256>`) of `mt1.<serial>` under the verifier's key, and remembers the claims.
/// `verify` recomputes the tag and compares it with the presented one by `constant_time_equal`, as a real verifier
/// must, then answers the claims; a token whose `expires_at` is at or before the verifier's clock, or that `expire`
/// marked, is `stale`; anything else is `unauthorized`. `fail_next(code, response)` makes the next `verify` fail as
/// though the store it consults had answered `response`.
///
/// It is not a JWT: base64url, JSON and the JWKS are servers' (`auth/jwt.hpp`), and the EdDSA, ES256 and RS256
/// checks behind it are crypto's. It needs neither, so a product's route tests run with the standard library alone.
/// **Never use it for production tokens.**
///
/// \snippet interfaces/token_verifier_reference_demo.cpp reference
///
/// **Relationships.** Models `token_verifier` (`concepts/token_verifier.hpp`); passes its kit in every form. Group:
/// \ref core.
///
/// **Implementations / variants.** One.
/// `basic_memory_token_verifier<Clock>` reads any `utc_clock` (`time/utc_clock.hpp`), owned by value and reached through
/// `clock()`; `memory_token_verifier` is `basic_memory_token_verifier<manual_utc_clock>`.
///
/// **Guarantees.** `verify` is `const`; it is safe to call concurrently once the tokens are issued and no failure is
/// armed (`issue`, `expire` and `fail_next` are not). Errors carry fixed messages only: never the token, never the key.
/// The key is held as a `snicholls::secret`.
///
/// **Errors.** `unauthorized` ("the token does not verify"), `stale` ("the token has expired"), and
/// whatever `fail_next` armed.
///
/// **Performance.** `benchmarks/interfaces/token_verifier_reference_benchmark.cpp`: `verify` against the same HMAC and
/// compare written by hand. Not yet quoted from a quiet machine: `make benchmarks` records them in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/interfaces/token_verifier_reference_test.cpp`,
/// `demos/interfaces/token_verifier_reference_demo.cpp`.
///
/// **Dependencies.** `<charconv>`, `<chrono>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<string>`,
/// `<string_view>`, `<system_error>`, `<utility>`, `<vector>`, `hash/hmac.hpp`, `hash/secure.hpp`, `hash/sha2.hpp`,
/// `secret.hpp`, `interfaces/token_verifier.hpp`, `time/utc_clock.hpp`.
///
/// **Known issues.** Claims are held in memory for the verifier's life; it is a test fixture.
///
/// **Status.** New in 0.27.0; the clock a `utc_clock` template parameter, owned, since 0.27.0 (it was a
/// `std::function<utc_s()>`).

#ifndef snicholls_core_interfaces_token_verifier_reference_hpp
#define snicholls_core_interfaces_token_verifier_reference_hpp

#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/interfaces/token_verifier.hpp>
#include <snicholls/core/secret.hpp>
#include <snicholls/core/time/utc_clock.hpp>

#include <charconv>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The reference token verifier: tokens it issues, tagged with HMAC-SHA256.
    /// \ingroup core
    /// \tparam Clock A `utc_clock` (`time/utc_clock.hpp`), owned by value; `manual_utc_clock` by default, moved by
    ///               `clock().advance(d)`.
    template <utc_clock Clock = manual_utc_clock>
    class basic_memory_token_verifier {
    public:
        /// \brief The clock.
        using clock_type = Clock;

        /// \brief The prefix of every token it issues.
        static constexpr std::string_view prefix = "mt1.";

        /// \brief A verifier tagging with `key` and reading the time from `clock`.
        /// \param key   The HMAC key (any bytes; 32 or more, as RFC 7518 requires of HS256).
        /// \param clock The clock.
        basic_memory_token_verifier(secret<std::string> key, Clock clock) : key_(std::move(key)), clock_(std::move(clock)) {}

        /// \brief The clock, to move a manual one: `clock().advance(d)`.
        /// \return The clock.
        [[nodiscard]] Clock& clock() noexcept { return clock_; }
        /// \brief The clock.
        /// \return The clock.
        [[nodiscard]] const Clock& clock() const noexcept { return clock_; }

        /// \brief A verifier for tests: a fixed key and a clock fixed at `now`. Never for anything but tests.
        /// \param now The fixed time.
        /// \return The verifier.
        [[nodiscard]] static basic_memory_token_verifier deterministic(utc_s now = utc_s::from_unix_seconds(1'800'000'000))
            requires std::same_as<Clock, manual_utc_clock>
        {
            return basic_memory_token_verifier{secret<std::string>{"memory_token_verifier: a test-only HS256 key"}, manual_utc_clock{now}};
        }

        /// \brief Issues a token for `claims`.
        /// \param claims The caller the token stands for.
        /// \return The token.
        [[nodiscard]] std::string issue(token_claims claims)
        {
            const std::string body = std::string(prefix) + std::to_string(issued_.size());
            issued_.push_back({std::move(claims), false});
            return body + "." + std::string(tag(body).view());
        }

        /// \brief Treats `token` as expired from now on; a token it did not issue is ignored.
        /// \param token The token.
        void expire(std::string_view token)
        {
            if (const auto i = index_of(token))
                issued_[*i].expired = true;
        }

        /// \brief Verifies a token.
        /// \param token The token.
        /// \return Its claims, or `unauthorized`, `stale`, or the armed failure.
        [[nodiscard]] capability_result<token_claims> verify(std::string_view token) const
        {
            if (injected_) {
                const capability_error e = *injected_;
                injected_.reset();
                return err(e);
            }
            const auto i = index_of(token);
            if (!i)
                return fail(capability_errc::unauthorized, {}, "the token does not verify");
            const auto& entry = issued_[*i];
            if (entry.expired || (entry.claims.expires_at && *entry.claims.expires_at <= stamp()))
                return fail(capability_errc::stale, {}, "the token has expired");
            return ok(entry.claims);
        }

        /// \brief Makes the next `verify` fail (once) as though the store it consults had answered `response`.
        /// \param code     The error's code.
        /// \param response The response: its first word becomes the provider code when it is one; nothing else is kept.
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
        struct entry {
            token_claims claims;
            bool expired;
        };

        [[nodiscard]] digest_hex<32> tag(std::string_view body) const noexcept
        {
            return hmac<sha256>::mac(key_.reveal(), body).hex();
        }

        // The serial of a token this verifier issued, after a constant-time check of its tag.
        [[nodiscard]] std::optional<std::size_t> index_of(std::string_view token) const noexcept
        {
            if (!token.starts_with(prefix))
                return std::nullopt;
            const auto dot = token.find('.', prefix.size());
            if (dot == std::string_view::npos)
                return std::nullopt;
            const auto body = token.substr(0, dot);
            const auto presented = token.substr(dot + 1);
            if (!constant_time_equal(tag(body).view(), presented))
                return std::nullopt;
            std::size_t serial = 0;
            const auto digits = body.substr(prefix.size());
            const auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), serial);
            if (ec != std::errc{} || end != digits.data() + digits.size() || serial >= issued_.size())
                return std::nullopt;
            return serial;
        }

        secret<std::string> key_;
        Clock clock_;
        std::vector<entry> issued_;
        mutable std::optional<capability_error> injected_;
    };

    /// \brief The reference with a `manual_utc_clock`: what tests use.
    /// \ingroup core
    using memory_token_verifier = basic_memory_token_verifier<>;

    static_assert(token_verifier<memory_token_verifier>, "snicholls::memory_token_verifier must model snicholls::token_verifier");

} // namespace snicholls

#endif // snicholls_core_interfaces_token_verifier_reference_hpp
