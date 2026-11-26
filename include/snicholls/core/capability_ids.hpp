//
//  capability_ids.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The identifiers of the capability contracts: bounded `inplace_string`s, one alias per kind of id, and
/// `make_id`, which refuses an id that does not fit.
/// \ingroup core
///
/// **Overview.** A contract owns no `std::string`: an identifier a provider returns or a ledger books is an
/// `inplace_string<N>` (trivially copyable, no allocation, refused rather than truncated), and a free-form payload (a
/// message body, a webhook body, a request) is borrowed for the call as `std::string_view` or
/// `std::span<const std::byte>`. Time is `utc_s`, core's `utc<std::chrono::seconds>` (`time/utc.hpp`); a fact's dates
/// are one `fact_dates` (trade, settlement, when known), validated by `validate(fact_dates)`. The bounds:
///
/// | Alias | Capacity | For | Why that bound |
/// |---|---|---|---|
/// | `provider_id` | 96 | an id a provider assigns: a payment, order, withdrawal, quote, message, key, cursor, transaction hash | an EVM transaction hash is 66 characters, a Stripe Checkout session id about 66, an SES message id about 60, a UUID 36, OKX and Twilio ids under 35 |
/// | `fact_id` | 128 | what a ledger books once | `<chain id>:<transaction hash>:<log index>` is at most 98 with a 20-digit chain id, plus a suffix such as `:reversed` or `:dispute_loss` |
/// | `event_id` | 64 | a webhook delivery, the dedup key | Stripe `evt_...` is about 28, Wise's `X-Delivery-Id` a UUID |
/// | `provider_account` | 64 | the opaque id of our account at a provider | Stripe `acct_...` is 21, a Wise profile id a number |
/// | `client_id` | 64 | an id we choose: an idempotency key, a client order id, a client reference | OKX `clOrdId` allows 32; ours are UUIDs (36) or shorter |
/// | `provider_name` | 32 | which provider: `"stripe"`, `"wise"`, `"okx"`, `"chain"` | a name |
/// | `short_text` | 255 | a short field a provider returns: a label, a name, a payment reference, an address | SEPA's remittance text is 140 characters, an account name 70 |
/// | `role_id` | 64 | a role: `"owner"`, `"admin"`, `"auditor"` | a name |
/// | `permission_id` | 64 | a permission or scope, `resource.action` | `"payments.refund"`, an OAuth scope |
/// | `tenant_id` | 64 | an organisation or workspace that roles are held in, or `"platform"` for staff | a UUID (36) or a slug |
/// | `wallet_address` | 42 | an EVM address, lowercase `0x` and 40 hex digits | exactly 42 |
///
/// Stripe documents its ids as up to 255 characters, but none it issues today comes near 96; an implementation that
/// meets a longer one answers `capability_errc::internal` rather than cut it (an id cut short is another id).
///
/// \snippet capability_ids_demo.cpp ids
///
/// \snippet capability_ids_demo.cpp dates
///
/// **Relationships.** `containers/inplace_string.hpp`, `capability_error.hpp`, `time/utc.hpp`; every contract under
/// `concepts/` that names an id. Group: \ref core.
///
/// **Implementations / variants.** Aliases; `make_id<Id>` for any of them.
///
/// **Guarantees.** Every alias is trivially copyable, standard layout and structural. `make_id` is `constexpr` and
/// `noexcept`, and never truncates.
///
/// **Errors.** `make_id` returns `capability_errc::invalid_argument` ("identifier too long") when the text does not fit.
///
/// **Performance.** Not benchmarked: a bounded copy (`containers/inplace_string.hpp` has the numbers).
///
/// **Tests and demos.** `tests/capability_ids_test.cpp`, `demos/capability_ids_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<string_view>`, `capability_error.hpp`, `containers/inplace_string.hpp`,
/// `time/utc.hpp`.
///
/// **Known issues.** A URL (a hosted checkout's) is a payload, not an id: it stays a `std::string`
/// (`concepts/payment_provider.hpp`).
///
/// **Status.** New in 0.23.0. Experimental, with the contracts.

#ifndef snicholls_core_capability_ids_hpp
#define snicholls_core_capability_ids_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/containers/inplace_string.hpp>
#include <snicholls/core/time/utc.hpp>

#include <chrono>
#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief An id a provider assigns (a payment, order, withdrawal, quote, message, key, cursor, transaction hash).
    /// \ingroup core
    using provider_id = inplace_string<96>;
    /// \brief What a ledger books once (Stripe `txn_...`, Wise `referenceNumber`, `<chain>:<tx>:<log>`).
    /// \ingroup core
    using fact_id = inplace_string<128>;
    /// \brief A webhook delivery, stable across redeliveries: the dedup key.
    /// \ingroup core
    using event_id = inplace_string<64>;
    /// \brief The opaque id of our account at a provider.
    /// \ingroup core
    using provider_account = inplace_string<64>;
    /// \brief An id we choose: an idempotency key, a client order id, a client reference.
    /// \ingroup core
    using client_id = inplace_string<64>;
    /// \brief Which provider (`"stripe"`, `"wise"`, `"okx"`, `"chain"`, `"memory"`).
    /// \ingroup core
    using provider_name = inplace_string<32>;
    /// \brief A short field a provider returns: a label, a name, a payment reference, an address.
    /// \ingroup core
    using short_text = inplace_string<255>;
    /// \brief A role (`"owner"`, `"admin"`, `"auditor"`).
    /// \ingroup core
    using role_id = inplace_string<64>;
    /// \brief A permission or scope, `resource.action` (`"payments.refund"`).
    /// \ingroup core
    using permission_id = inplace_string<64>;
    /// \brief An organisation or workspace roles are held in, or `"platform"` for staff.
    /// \ingroup core
    using tenant_id = inplace_string<64>;
    /// \brief An EVM address: lowercase `0x` and 40 hex digits.
    /// \ingroup core
    using wallet_address = inplace_string<42>;

    /// \brief `text` as the identifier `Id`, or `invalid_argument` when it does not fit; never truncated.
    /// \ingroup core
    /// \tparam Id One of the aliases above (any `inplace_string<N>`).
    /// \param text The text.
    /// \return The id, or the error.
    template <typename Id>
        requires requires(std::string_view t) { Id::make(t); }
    [[nodiscard]] constexpr capability_result<Id> make_id(std::string_view text) noexcept
    {
        auto r = Id::make(text);
        if (!r)
            return fail(capability_errc::invalid_argument, {}, "identifier too long");
        return ok(*r);
    }

    /// \brief A fact's dates, the same on every fact a ledger books (payments, on-ramp, securities): when the economic
    /// event happened, when it settles, and when we learnt of it.
    /// \ingroup core
    ///
    /// The trade and the settlement are kept apart because the gap between them is where cash is short: a card payment
    /// is available days later, a counter payment when the agent's file settles, a share trade at T+2. `settles_on` is a
    /// calendar date in the settlement account's own calendar (its time zone is the account's, not the fact's), present
    /// once known; a provider that gives none leaves it to the ledger's intake, which estimates it with
    /// `finance/business_day.hpp` and marks it `settles_on_estimated`.
    ///
    /// **Provenance, not content:** `known_at` (when we recorded the delivery) is never part of a fact's content: two
    /// deliveries of one fact are known at different times and are the same fact. A content comparison leaves it out.
    struct fact_dates {
        utc_s trade_at{};                                 ///< when the economic event happened (the provider's time)
        utc_s known_at{};                                 ///< when we recorded it: provenance, not content
        std::optional<std::chrono::sys_days> settles_on;  ///< the value date, in the settlement account's calendar
        bool settles_on_estimated{false};                 ///< `settles_on` computed by business-day rules, not given
        std::optional<utc_s> settled_at;                  ///< when it settled, once it has
        std::optional<utc_s> available_at;                ///< an instant the provider gives (Stripe's `available_on`)

        /// \brief Equality, of every field (provenance included; compare content with `same_content`).
        friend constexpr bool operator==(const fact_dates&, const fact_dates&) noexcept = default;

        /// \brief Equality of content: every field but `known_at`.
        /// \param other The other dates.
        /// \return `true` when the dates describe the same event.
        [[nodiscard]] constexpr bool same_content(const fact_dates& other) const noexcept
        {
            return trade_at == other.trade_at && settles_on == other.settles_on &&
                   settles_on_estimated == other.settles_on_estimated && settled_at == other.settled_at &&
                   available_at == other.available_at;
        }
    };

    /// \brief The checks a fact's dates pass: it was known no earlier than it happened, it settled no earlier than it
    /// happened, and an estimate is of a date that exists. A `settles_on` in the future is allowed: a pending settlement.
    /// \ingroup core
    /// \param d The dates.
    /// \return Success, or `invalid_argument` with a fixed message.
    [[nodiscard]] constexpr capability_result<void> validate(const fact_dates& d) noexcept
    {
        if (d.known_at < d.trade_at)
            return fail(capability_errc::invalid_argument, {}, "a fact is known before it happened");
        if (d.settled_at && *d.settled_at < d.trade_at)
            return fail(capability_errc::invalid_argument, {}, "a fact settled before it happened");
        if (d.settles_on_estimated && !d.settles_on)
            return fail(capability_errc::invalid_argument, {}, "an estimate without a settlement date");
        return ok();
    }

} // namespace snicholls

#endif // snicholls_core_capability_ids_hpp
