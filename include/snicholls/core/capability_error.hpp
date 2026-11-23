//
//  capability_error.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::capability_error`: the one error of every capability contract, its code `capability_errc`,
/// `is_retryable`, and `capability_result<V>`.
/// \ingroup core
///
/// **Overview.** The capability contracts (`concepts/message_sender.hpp`, `concepts/signer.hpp`,
/// `concepts/key_store.hpp`, `concepts/payment_provider.hpp`, `concepts/onramp_provider.hpp`) fail the same way, so
/// a caller writes its retry and alerting logic once:
/// - **one enumeration**, `capability_errc`, for every capability. A caller branches on it: retry, tell the user,
///   or page someone. Only `unavailable`, `timeout` and `rate_limited` are retryable (`is_retryable`). Its names
///   print through core's enum reflection (`enum_name(capability_errc::stale)` is `"stale"`); there is no hand table;
/// - **`capability_error`** is the code, the provider's own short code for the failure (`provider_code`, at most 32
///   characters of `A-Z a-z 0-9 _ - . :`: a Stripe `decline_code`, an OKX `sCode`, an HTTP status) and a fixed
///   diagnostic (`capability_message`, at most 64 characters, from a string literal only). Neither holds a
///   `std::string`: the error is trivially copyable, so returning one never allocates;
/// - **`capability_result<V>`** is `result<V, capability_error>`; `fail(code, provider_code, message)` makes one.
///
/// Human detail belongs at the boundary, through `snicholls::traced<capability_error>` (`traced.hpp`), where a
/// route or a worker adds its context and logs it.
///
/// \snippet capability_error_demo.cpp capability_error
///
/// **Relationships.** `result.hpp` (`result`, `err`), `containers/inplace_string.hpp` (the storage),
/// `meta/enums.hpp` (the names), `capability_ids.hpp` (the identifiers the contracts use),
/// `interfaces/capability_conformance.hpp` (the kits' report and the redaction check). Group: \ref core.
///
/// **Implementations / variants.** One error for every capability; a capability that needs a finer reason says it in
/// `provider_code` or `message`, never with an enumeration of its own.
///
/// **Guarantees.**
/// - Everything is `constexpr` and `noexcept`; `capability_error` is trivially copyable and standard layout
///   (`static_assert`ed), 99 bytes.
/// - **The redaction rule.** `message` holds fixed diagnostic text only: it can be made from a string literal and
///   nothing else, so a provider's raw response body, user data (an address, a phone number, a message body, a
///   one-time code) or a secret cannot reach it. `provider_code` is the provider's short error code only
///   (`"card_declined"`): `provider_code::make` refuses text longer than 32 characters or with any character outside
///   `A-Z a-z 0-9 _ - . :` (a space, a quote, a brace: free text and JSON), and `provider_error` drops such text
///   rather than keep it. Every reference implementation honours the rule, and every conformance kit checks it (a
///   provider failure whose response carries a marker; the marker must not appear in either field).
/// - Refused, never truncated: a code that does not fit is not a shorter code.
///
/// **Errors.** `provider_code::make` returns `capability_errc::invalid_argument` for text that is not a provider
/// code. A literal that is not one is a compile error.
///
/// **Performance.** Not benchmarked: the error path. No allocation by construction.
///
/// **Tests and demos.** `tests/capability_error_test.cpp`, `demos/capability_error_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<string_view>`, `<type_traits>`, `containers/inplace_string.hpp`,
/// `result.hpp`. The names come from `meta/enums.hpp`, which a caller that prints them includes.
///
/// **Known issues.**
/// - The redaction rule is enforced for `message` by construction and for `provider_code` by its character set; a
///   provider code made of a user's data in those characters (an implementation that passes an email's local part
///   as a code) is still the implementation's defect, which the kits look for but cannot rule out.
/// - The enumeration is shared: a code that is meaningless for one capability (`insufficient_funds` from a message
///   sender) is never returned by it, but the type does not forbid it.
///
/// **Status.** New in 0.23.0. Experimental, with the contracts.

#ifndef snicholls_core_capability_error_hpp
#define snicholls_core_capability_error_hpp

#include <snicholls/core/containers/inplace_string.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace snicholls {

    /// \brief What went wrong, as a caller branches on it: one enumeration for every capability.
    /// \ingroup core
    enum class capability_errc : std::uint8_t {
        unavailable = 1,         ///< the provider is down or unreachable; retryable
        timeout = 2,             ///< no answer in time; retryable, and the outcome is unknown until checked
        rate_limited = 3,        ///< slow down; retryable
        rejected = 4,            ///< the provider or a policy declined (a card decline, a compliance hold, a wrong code)
        invalid_argument = 5,    ///< the request is malformed; fix it, do not retry
        conflict = 6,            ///< an idempotency key reused for another request, or a state that forbids the call
        not_found = 7,           ///< no such payment, key, order, withdrawal or route
        unauthorized = 8,        ///< missing, wrong or insufficient credentials
        internal = 9,            ///< a defect; page someone
        insufficient_funds = 10, ///< not enough balance
        signature_invalid = 11,  ///< a webhook, token or message failed authentication
        stale = 12,              ///< outside its time window: an expired quote, token, code or webhook
        unsupported = 13,        ///< this implementation does not do that
    };

    /// \brief Whether the same request may be sent again unchanged (with the same idempotency key).
    /// \ingroup core
    /// \param code The code.
    /// \return `true` only for `unavailable`, `timeout` and `rate_limited`.
    [[nodiscard]] constexpr bool is_retryable(capability_errc code) noexcept
    {
        return code == capability_errc::unavailable || code == capability_errc::timeout ||
               code == capability_errc::rate_limited;
    }

    /// \brief The provider's own short code for a failure: at most 32 characters of `A-Z a-z 0-9 _ - . :`.
    /// \ingroup core
    ///
    /// Made from a literal checked by the compiler (`provider_code{"card_declined"}`) or at run time by `make`, which
    /// refuses text that is too long or has any other character, so free text, JSON or a response body cannot pass
    /// for a code.
    class provider_code {
    public:
        /// \brief The capacity, in characters.
        static constexpr std::size_t capacity = 32;

        /// \brief Whether `text` is a provider code: 1 to 32 characters of `A-Z a-z 0-9 _ - . :`.
        /// \param text The text.
        /// \return `true` when it is.
        [[nodiscard]] static constexpr bool is_code(std::string_view text) noexcept
        {
            if (text.empty() || text.size() > capacity)
                return false;
            for (const char c : text)
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                      c == '-' || c == '.' || c == ':'))
                    return false;
            return true;
        }

        /// \brief No code.
        constexpr provider_code() noexcept = default;

        /// \brief A literal code, checked by the compiler.
        /// \tparam M The literal's size with its terminator.
        /// \param literal The code.
        template <std::size_t M>
            requires(M >= 2 && M <= capacity + 1)
        consteval provider_code(const char (&literal)[M])   // NOLINT(google-explicit-constructor)
            : text_(literal)
        {
            if (!is_code(text_.view()))
                throw "snicholls::provider_code: not a provider code (1 to 32 of A-Z a-z 0-9 _ - . :)";
        }

        /// \brief The code `text`, or `invalid_argument` when it is not one (see `is_code`).
        /// \param text The provider's code.
        /// \return The code, or the error.
        [[nodiscard]] static constexpr result<provider_code, capability_errc> make(std::string_view text) noexcept
        {
            if (!is_code(text))
                return err(capability_errc::invalid_argument);
            provider_code out;
            out.text_ = *inplace_string<capacity>::make(text);
            return ok(out);
        }

        /// \brief The code.
        /// \return A view valid while this object lives.
        [[nodiscard]] constexpr std::string_view view() const noexcept { return text_.view(); }
        /// \brief Whether there is no code.
        /// \return `true` when empty.
        [[nodiscard]] constexpr bool empty() const noexcept { return text_.empty(); }

        /// \brief Equality of the text.
        friend constexpr bool operator==(const provider_code&, const provider_code&) noexcept = default;
        /// \brief Equality with text.
        /// \param a The code.
        /// \param b The text.
        /// \return Whether they are equal.
        friend constexpr bool operator==(const provider_code& a, std::string_view b) noexcept { return a.view() == b; }

    private:
        inplace_string<capacity> text_{};
    };

    /// \brief Fixed diagnostic text for a `capability_error`: at most 64 characters, from a string literal only.
    /// \ingroup core
    ///
    /// There is no run-time constructor, so nothing received (a response body, user data, a secret) can become one:
    /// the redaction rule by construction.
    class capability_message {
    public:
        /// \brief The capacity, in characters.
        static constexpr std::size_t capacity = 64;

        /// \brief No message.
        constexpr capability_message() noexcept = default;

        /// \brief A literal message, checked by the compiler to fit.
        /// \tparam M The literal's size with its terminator.
        /// \param literal The message.
        template <std::size_t M>
            requires(M <= capacity + 1)
        consteval capability_message(const char (&literal)[M]) noexcept   // NOLINT(google-explicit-constructor)
            : text_(literal)
        {
        }

        /// \brief The message.
        /// \return A view valid while this object lives.
        [[nodiscard]] constexpr std::string_view view() const noexcept { return text_.view(); }
        /// \brief Whether there is no message.
        /// \return `true` when empty.
        [[nodiscard]] constexpr bool empty() const noexcept { return text_.empty(); }

        /// \brief Equality of the text.
        friend constexpr bool operator==(const capability_message&, const capability_message&) noexcept = default;
        /// \brief Equality with text.
        /// \param a The message.
        /// \param b The text.
        /// \return Whether they are equal.
        friend constexpr bool operator==(const capability_message& a, std::string_view b) noexcept { return a.view() == b; }

    private:
        inplace_string<capacity> text_{};
    };

    /// \brief A capability's error: the code a caller branches on, the provider's own code, a fixed diagnostic.
    /// \ingroup core
    struct capability_error {
        capability_errc code{capability_errc::internal};  ///< what a caller branches on
        snicholls::provider_code provider_code{};          ///< the provider's short code, if any; never free text
        capability_message message{};                      ///< fixed diagnostic text, if any; never received data

        /// \brief Whether the request may be retried unchanged.
        /// \return `is_retryable(code)`.
        [[nodiscard]] constexpr bool retryable() const noexcept { return is_retryable(code); }

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const capability_error&, const capability_error&) noexcept = default;
    };

    static_assert(std::is_trivially_copyable_v<capability_error>, "snicholls::capability_error never allocates");
    static_assert(std::is_standard_layout_v<capability_error>);
    static_assert(sizeof(capability_error) == 1 + 33 + 65, "snicholls::capability_error: a code, 32 and 64 characters, two lengths");

    /// \brief The result of every capability operation.
    /// \ingroup core
    /// \tparam V The value.
    template <typename V>
    using capability_result = result<V, capability_error>;

    /// \brief An error result: `return fail(capability_errc::not_found);`, optionally with the provider's code and a
    /// fixed diagnostic.
    /// \ingroup core
    /// \param code     The code.
    /// \param provider The provider's code, or none.
    /// \param message  A fixed diagnostic, or none.
    /// \return The error, ready to convert into any `capability_result<V>`.
    [[nodiscard]] constexpr auto fail(capability_errc code, provider_code provider = {}, capability_message message = {}) noexcept
    {
        return err(capability_error{code, provider, message});
    }

    /// \brief The error for a provider's failure: `code`, and `provider_text` as the provider code when it is one;
    /// any other text (a response body, free text) is dropped, never kept.
    /// \ingroup core
    /// \param code          The code.
    /// \param provider_text What the provider gave as its code.
    /// \return The error.
    [[nodiscard]] constexpr capability_error provider_error(capability_errc code, std::string_view provider_text) noexcept
    {
        capability_error e{code, {}, {}};
        if (auto c = provider_code::make(provider_text); c)
            e.provider_code = *c;
        return e;
    }

} // namespace snicholls

#endif // snicholls_core_capability_error_hpp
