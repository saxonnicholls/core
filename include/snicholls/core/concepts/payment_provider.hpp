//
//  payment_provider.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::payment_provider`: anything that moves fiat money in and out (collections, payouts, refunds),
/// verifies its webhooks and gives a ledger authoritative facts.
/// \ingroup core
///
/// **Overview.** Three movements cover every provider:
///
/// | Movement | A pull provider (Stripe) | A push provider (Wise) |
/// |---|---|---|
/// | **collection**, money in from a payer | a checkout session or payment intent: we pull, the payer confirms | an inbound bank transfer: the payer pushes to our account details with a reference, and a webhook says it arrived |
/// | **payout**, money out to a beneficiary | a payout to our bank | a transfer to a registered recipient, optionally on a quote |
/// | **refund**, money back to a payer | a refund of a charge | unsupported (a payout to the payer instead) |
///
/// `collect` returns the `payment` and the payer's `next_step`: a redirect (a hosted checkout), a client confirmation
/// (a secret for the browser SDK), or bank details with the reference to quote. Status then arrives by webhook
/// (`verify_webhook`, which authenticates and parses without I/O) or by asking (`status`). A push provider's credits
/// are listed by `credits` for reconciliation. This is the concept half of a pair; the pure virtual half is
/// `payment_provider_interface` (`interfaces/payment_provider.hpp`), with `payment_provider_adapter<P>` and
/// `payment_provider_ref`.
///
/// **The contract** (`interfaces/payment_provider_conformance.hpp` checks it):
/// - Every amount is an exact `money` (`money/money.hpp`), never a `double` or an `int64`. A collection, payout or
///   refund amount is positive; zero or negative is `invalid_argument`.
/// - Every request that moves money carries an `idempotency_key` (at most 64 characters, `client_id`; so is
///   `our_reference`). The same key with the same request returns the same result and moves money once; with a
///   different request it is `conflict`.
/// - What a provider does not do (a method, a currency, payouts, quotes, refunds, credits) is `unsupported`, and
///   `capabilities()` says so in advance.
/// - `verify_webhook` fails closed: no signature, or a wrong one, is `signature_invalid`; a delivery outside the
///   provider's time window is `stale`. It never contacts the network and is safe on any input. This holds in every
///   mode, tests included: there is no unsigned path. The caller deduplicates deliveries on `dedup_key()` and books
///   money once per `fact_id`.
/// - An event that names a payment but carries no detail has `needs_lookup` set: the caller asks `status` or `credits`.
/// - A refund never exceeds what was received less what was refunded (`conflict` otherwise).
/// - `next_step::client_secret` and every credential are `secret`s: they print as `********`.
/// - **The webhook flow.** The route calls `verify_webhook` first; then inserts `dedup_key()` with the raw event
///   atomically and answers 2xx; then a durable worker calls `resolve`, which fetches the authoritative data (the
///   provider's balance transaction or statement line) and returns one `payment_fact` per line. **Amounts never come
///   from the webhook body**, which is why `webhook_event` carries none. An event that books nothing resolves to no
///   facts.
/// - Stripe's events to facts: `payment_intent.succeeded`, `checkout.session.completed` and `async_payment_succeeded`
///   to `payment` (one `txn_`); `charge.refunded` to `refund`; `refund.updated` (failed) to `refund_failed`;
///   `charge.dispute.created` and `funds_withdrawn` to `dispute_hold`; `funds_reinstated` or `closed` (won) to
///   `dispute_release`; `closed` (lost) to `dispute_loss`; `payout.paid` to `payout`; `payout.failed` to
///   `payout_failed`. Every other event is recorded and produces no fact.
/// - `statement` reads the provider's lines for the reconcilers. Every fact passes `validate`: `gross == net + fee`.
///   **Every statement line maps to exactly one fact**, with `fact_id` the provider's line id: lines that are not
///   payments are `fee`, `conversion`, `transfer`, `adjustment` (reviewed, never auto-posted), `card_spend` or `other`
///   (reviewed), never skipped. Facts are oldest first within a page; the reconciler matches by `fact_id` as a set.
/// - `dispute_loss` books no new money movement: its `fact_id` is `<withdrawal txn>:dispute_loss`, `gross` the
///   disputed amount as withdrawn, `fee` zero, `net == gross`.
/// - The map from provider account to book is the product's; a provider knows only the opaque account id.
/// - Errors follow the redaction rule of `capability_error.hpp`: a decline's `provider_code` is the provider's code
///   (`card_declined`), never its message, and never card or payer data.
///
/// \snippet concepts/payment_provider_demo.cpp concept
///
/// **Relationships.** `money/money.hpp`, `money/rate.hpp`, `http_header.hpp`, `secret.hpp`, `uuid.hpp`,
/// `capability_error.hpp`, `capability_ids.hpp`, `time/utc.hpp`; the twin `interfaces/payment_provider.hpp`, the
/// reference `interfaces/payment_provider_reference.hpp`, the kit `interfaces/payment_provider_conformance.hpp`. Group:
/// \ref core.
///
/// **Implementations.** `memory_payment_provider` (`interfaces/payment_provider_reference.hpp`, both shapes, for
/// tests). Above core: Stripe and Wise.
///
/// **Guarantees.** `is_terminal`, `balances` and `validate` are `constexpr` and `noexcept`; `dedup_key` is
/// allocation-free. Thread safety is the implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Webhook verification is the hot path: `benchmarks/interfaces/payment_provider_reference_benchmark.cpp`.
///
/// **Tests and demos.** `tests/concepts/payment_provider_test.cpp`, `demos/concepts/payment_provider_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<span>`, `<string>`,
/// `<string_view>`, `<utility>`, `<vector>`, `capability_error.hpp`, `capability_ids.hpp`, `http_header.hpp`,
/// `money/money.hpp`, `money/rate.hpp`, `secret.hpp`, `time/utc.hpp`, `uuid.hpp`.
///
/// **Known issues.** `fx_quote` holds a `rate`, which has no default value, so a quote is made with its rate given. A
/// hosted checkout's URL is a `std::string` (a payload of unbounded length), and so is the client secret inside its
/// `secret`. Mandates (BECS, SEPA debit) and subscriptions are not in the contract.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_payment_provider_hpp
#define snicholls_core_concepts_payment_provider_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/http_header.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/money/rate.hpp>
#include <snicholls/core/secret.hpp>
#include <snicholls/core/time/utc.hpp>
#include <snicholls/core/uuid.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief How the payer pays.
    /// \ingroup core
    enum class payment_method : std::uint8_t {
        card = 1,           ///< a card, pulled
        wallet_pay = 2,     ///< Apple Pay or Google Pay, pulled
        bank_debit = 3,     ///< a direct debit under a mandate, pulled
        bank_transfer = 4,  ///< the payer pushes to our account details with a reference
    };

    /// \brief Which way money moves.
    /// \ingroup core
    enum class movement : std::uint8_t {
        collection = 1,  ///< in, from a payer
        payout = 2,      ///< out, to a beneficiary
        refund = 3,      ///< back to a payer
    };

    /// \brief Where a movement is.
    /// \ingroup core
    enum class payment_status : std::uint8_t {
        pending = 1,             ///< created; waiting for the payer (or the bank transfer)
        requires_action = 2,     ///< the payer must act (a redirect, 3-D Secure)
        processing = 3,          ///< in flight
        succeeded = 4,           ///< done: the money moved
        failed = 5,              ///< it will not move
        cancelled = 6,           ///< withdrawn before it moved
        refunded = 7,            ///< a collection refunded in full
        partially_refunded = 8,  ///< a collection refunded in part
    };

    /// \brief Whether no further change is expected without a new request.
    /// \ingroup core
    /// \param s The status.
    /// \return `true` for succeeded, failed, cancelled and refunded.
    [[nodiscard]] constexpr bool is_terminal(payment_status s) noexcept
    {
        return s == payment_status::succeeded || s == payment_status::failed || s == payment_status::cancelled ||
               s == payment_status::refunded;
    }

    /// \brief Bank details a payer pushes to.
    /// \ingroup core
    struct bank_details {
        short_text account_name;    ///< the beneficiary
        short_text bank_name;       ///< the bank
        short_text account_number;  ///< the local account number
        short_text routing_code;    ///< a BSB, an ABA routing number, a sort code
        short_text swift;           ///< the BIC, for international wires
        short_text iban;            ///< for EUR
        short_text reference;       ///< the payer must quote it: it is how the credit is matched

        /// \brief Equality.
        friend constexpr bool operator==(const bank_details&, const bank_details&) noexcept = default;
    };

    /// \brief What the payer does next.
    /// \ingroup core
    enum class next_step_kind : std::uint8_t {
        none = 0,            ///< nothing
        redirect = 1,        ///< open `url` (a hosted checkout)
        client_confirm = 2,  ///< confirm in the browser or app with `client_secret`
        bank_transfer = 3,   ///< push to `bank`, quoting its reference
    };

    /// \brief The payer's next step after `collect`.
    /// \ingroup core
    struct next_step {
        next_step_kind kind{next_step_kind::none};  ///< which step
        std::string url;                            ///< for `redirect`
        secret<std::string> client_secret;          ///< for `client_confirm`; never logged
        std::optional<bank_details> bank;           ///< for `bank_transfer`

        /// \brief Equality.
        friend bool operator==(const next_step&, const next_step&) = default;
    };

    /// \brief A request to collect money from a payer (borrowed for the call).
    /// \ingroup core
    struct collection_request {
        std::string_view idempotency_key;  ///< required, at most 64 characters
        std::string_view our_reference;    ///< our id (an order, a session), at most 64 characters; returned on the payment
        money amount;                      ///< requested (for a bank transfer, expected: the payer may send another amount)
        payment_method method{payment_method::card};  ///< how
        std::string_view payer_reference;  ///< for a bank transfer: the stable reference the payer quotes
        std::string_view return_url;       ///< for a hosted checkout
        std::string_view cancel_url;       ///< for a hosted checkout
        std::span<const std::pair<std::string_view, std::string_view>> metadata;  ///< round-tripped where supported
    };

    /// \brief A request to pay a registered beneficiary (borrowed for the call).
    /// \ingroup core
    struct payout_request {
        std::string_view idempotency_key;          ///< required, at most 64 characters
        std::string_view our_reference;            ///< our id, at most 64 characters
        money amount;                              ///< what the beneficiary receives, in its currency
        std::string_view beneficiary_id;           ///< registered at the provider in advance (never raw bank details)
        std::string_view reference;                ///< shown to the beneficiary
        std::optional<std::string_view> quote_id;  ///< an FX quote to honour

        /// \brief Equality.
        friend constexpr bool operator==(const payout_request&, const payout_request&) noexcept = default;
    };

    /// \brief A request to refund a collection (borrowed for the call).
    /// \ingroup core
    struct refund_request {
        std::string_view idempotency_key;  ///< required, at most 64 characters
        std::string_view payment_id;       ///< the collection
        std::optional<money> amount;       ///< in part; the whole remaining amount when empty
        std::string_view reason;           ///< for people

        /// \brief Equality.
        friend constexpr bool operator==(const refund_request&, const refund_request&) noexcept = default;
    };

    /// \brief A request for an FX quote: send `amount` (the source side) or receive it (the target side).
    /// \ingroup core
    struct quote_request {
        money amount;                 ///< the fixed side
        asset other;                  ///< the other currency
        bool amount_is_source{true};  ///< `true`: `amount` is sent; `false`: `amount` is received

        /// \brief Equality.
        friend constexpr bool operator==(const quote_request&, const quote_request&) noexcept = default;
    };

    /// \brief An FX quote.
    /// \ingroup core
    struct fx_quote {
        snicholls::provider_id quote_id;  ///< to pass to `payout`
        money source;                     ///< sent
        money target;                     ///< received
        money fee;                        ///< in the source currency, included in `source`
        snicholls::rate rate;             ///< exact: source to target, before the fee
        utc_s expires_at{};               ///< after this it is `stale`

        /// \brief Equality.
        friend constexpr bool operator==(const fx_quote&, const fx_quote&) noexcept = default;
    };

    /// \brief One money movement, as the provider reports it.
    /// \ingroup core
    struct payment {
        snicholls::provider_id payment_id;    ///< the provider's id; the key for `status`
        snicholls::client_id our_reference;   ///< as requested
        movement kind{movement::collection};  ///< which way
        payment_method method{payment_method::card};     ///< how (collections)
        payment_status status{payment_status::pending};  ///< where it is
        money amount;                         ///< requested or expected
        money received;                       ///< settled gross so far (zero until then)
        money fee;                            ///< the provider's fee
        money refunded;                       ///< refunded so far
        snicholls::provider_code failure_code;  ///< the provider's code when failed (`card_declined`)
        utc_s updated_at{};                   ///< the provider's time of the last change

        /// \brief Equality.
        friend constexpr bool operator==(const payment&, const payment&) noexcept = default;
    };

    /// \brief What `collect` returns.
    /// \ingroup core
    struct collection {
        payment pay;     ///< the movement
        next_step next;  ///< what the payer does

        /// \brief Equality.
        friend bool operator==(const collection&, const collection&) = default;
    };

    /// \brief One credit into our account (a push provider's inbound transfer).
    /// \ingroup core
    struct credit {
        snicholls::provider_id credit_id;  ///< the provider's id: the once-only key
        money net;                         ///< credited
        money fee;                         ///< the provider's fee on it
        short_text reference;              ///< what the payer quoted
        short_text sender;                 ///< the payer's name as the bank gives it (personal data: never logged)
        utc_s occurred_at{};               ///< when it arrived

        /// \brief Equality.
        friend constexpr bool operator==(const credit&, const credit&) noexcept = default;
    };

    /// \brief Which credits to list.
    /// \ingroup core
    struct credit_query {
        asset currency;  ///< the account's currency
        utc_s from{};    ///< inclusive
        utc_s to{};      ///< exclusive

        /// \brief Equality.
        friend constexpr bool operator==(const credit_query&, const credit_query&) noexcept = default;
    };

    /// \brief What a webhook says happened.
    /// \ingroup core
    enum class event_kind : std::uint8_t {
        payment_updated = 1,  ///< a collection changed
        credit_received = 2,  ///< money arrived in an account
        payout_updated = 3,   ///< a payout changed
        refund_updated = 4,   ///< a refund changed
        other = 5,            ///< anything else; ignore it safely
    };

    /// \brief One webhook delivery, exactly as received: the raw body, not a parsed one (borrowed for the call).
    /// \ingroup core
    struct webhook_delivery {
        std::span<const http_header> headers;  ///< every header, any case
        std::string_view body;                 ///< the raw body bytes the signature covers
        utc_s received_at{};                   ///< our clock, for the time window
    };

    /// \brief A webhook delivery that owns its bytes, as a test or a queue keeps it.
    /// \ingroup core
    struct owned_delivery {
        std::vector<http_header> headers;  ///< the headers
        std::string body;                  ///< the raw body
        utc_s received_at{};               ///< the receiver's time

        /// \brief A view for `verify_webhook`, valid while this object lives and is unchanged.
        /// \return The delivery.
        [[nodiscard]] webhook_delivery view() const noexcept { return {headers, body, received_at}; }
    };

    /// \brief A verified webhook event. It carries no amount: amounts come from `resolve` and `statement` only.
    /// \ingroup core
    struct webhook_event {
        provider_name source;                          ///< the provider: `"stripe"`, `"wise"`, `"memory"`
        snicholls::provider_account provider_account;  ///< the opaque id of our account at the provider
        snicholls::event_id event_id;      ///< the delivery, stable across redeliveries; never a hash of the body
        snicholls::fact_id fact_id;        ///< the fact it points at, when it names one; a hint only
        event_kind kind{event_kind::other};  ///< what happened
        short_text type;                   ///< the provider's own name (`checkout.session.completed`)
        snicholls::provider_id payment_id; ///< the movement, when named
        snicholls::client_id our_reference;  ///< ours, when the provider round-trips it
        short_text reference;              ///< a credit's payer reference, when carried
        utc_s occurred_at{};               ///< the provider's time
        bool needs_lookup{false};          ///< the event carries no detail: ask `status` or `credits`

        /// \brief The key the webhook route inserts atomically with the raw event before answering 2xx.
        /// \return `<source>:evt:<event_id>` (`stripe:evt:evt_1...`), at most 101 characters, never allocated.
        [[nodiscard]] constexpr inplace_string<101> dedup_key() const noexcept
        {
            std::array<char, 101> text{};
            std::size_t n = 0;
            for (const std::string_view part : {source.view(), std::string_view{":evt:"}, event_id.view()})
                for (const char c : part)
                    text[n++] = c;
            return *inplace_string<101>::make({text.data(), n});
        }

        /// \brief Equality.
        friend constexpr bool operator==(const webhook_event&, const webhook_event&) noexcept = default;
    };

    /// \brief What a fact books (the seam agreed with the ledger).
    /// \ingroup core
    enum class fact_kind : std::uint8_t {
        payment = 1,          ///< money collected
        refund = 2,           ///< money returned
        refund_failed = 3,    ///< a refund that bounced
        dispute_hold = 4,     ///< funds withdrawn for a dispute
        dispute_release = 5,  ///< funds reinstated (or a dispute won)
        dispute_loss = 6,     ///< a dispute lost
        payout = 7,           ///< paid out to our bank
        payout_failed = 8,    ///< a payout that failed
        bank_credit = 9,      ///< a bank transfer arrived
        fee = 10,             ///< a provider fee not tied to a payment
        conversion = 11,      ///< a currency conversion inside the account
        transfer = 12,        ///< between our own accounts or balances
        adjustment = 13,      ///< a provider correction: reviewed, never auto-posted
        card_spend = 14,      ///< a card spend
        other = 15,           ///< anything else on a statement: reviewed
        agent_collection = 16,  ///< a payment an agent collected (cash or card at a post office counter), settled later
                                ///< in a lump: it debits the agent's collections due, and the agent's settlement, a
                                ///< `bank_credit`, clears it
    };

    /// \brief What settled in our account's currency: `gross == net + fee`, all one asset.
    /// \ingroup core
    struct settlement {
        money gross;  ///< before the provider's fee (negative for money out)
        money fee;    ///< the provider's fee
        money net;    ///< what moved in the balance

        /// \brief Equality.
        friend constexpr bool operator==(const settlement&, const settlement&) noexcept = default;
    };

    /// \brief Whether a settlement balances: one asset and `gross == net + fee` exactly.
    /// \ingroup core
    /// \param s The settlement.
    /// \return `true` when it does.
    [[nodiscard]] constexpr bool balances(const settlement& s) noexcept
    {
        const auto sum = add(s.net, s.fee);
        return sum && *sum == s.gross;
    }

    /// \brief The provider's ids a fact relates to (empty where none).
    /// \ingroup core
    struct fact_refs {
        snicholls::provider_id payment_intent;  ///< Stripe `pi_...`
        snicholls::provider_id charge;          ///< Stripe `ch_...`
        snicholls::provider_id refund;          ///< Stripe `re_...`
        snicholls::provider_id dispute;         ///< Stripe `dp_...`
        snicholls::provider_id payout;          ///< Stripe `po_...`, a Wise transfer id

        /// \brief Equality.
        friend constexpr bool operator==(const fact_refs&, const fact_refs&) noexcept = default;
    };

    /// \brief One fact for the ledger: authoritative, from the provider's balance transaction or statement line, never
    /// from a webhook body. Booked once, by `fact_id`.
    /// \ingroup core
    struct payment_fact {
        provider_name source;                          ///< `"stripe"`, `"wise"`
        snicholls::provider_account provider_account;  ///< opaque; the product maps it to a book
        snicholls::event_id event_id;                  ///< the delivery that led here; empty when read from a statement
        snicholls::fact_id fact_id;                    ///< the provider's line id (`txn_...`, a `referenceNumber`)
        fact_kind kind{fact_kind::payment};            ///< what it books
        utc_s occurred_at{};                           ///< the provider's time
        settlement settled;                            ///< in the account's currency; `balances(settled)` holds
        std::optional<money> presentment;              ///< what the payer was charged, when it differs from `settled.gross`
        std::optional<snicholls::rate> exchange_rate;  ///< exact: presentment to settlement, when they differ
        std::optional<uuid> party;                     ///< our customer's id from the provider metadata; never an email
        fact_refs refs;                                ///< related provider ids

        /// \brief Equality.
        friend constexpr bool operator==(const payment_fact&, const payment_fact&) noexcept = default;
    };

    /// \brief The checks every fact passes before it leaves a provider.
    /// \ingroup core
    /// \param f The fact.
    /// \return Success, or `invalid_argument` with a fixed message.
    [[nodiscard]] constexpr capability_result<void> validate(const payment_fact& f) noexcept
    {
        if (f.source.empty() || f.fact_id.empty())
            return fail(capability_errc::invalid_argument, {}, "a fact needs its source and fact id");
        if (!balances(f.settled))
            return fail(capability_errc::invalid_argument, {}, "gross is not net plus fee in one asset");
        if (f.presentment && *f.presentment == f.settled.gross)
            return fail(capability_errc::invalid_argument, {}, "a presentment only when it differs");
        if (f.presentment.has_value() != f.exchange_rate.has_value())
            return fail(capability_errc::invalid_argument, {}, "a presentment comes with its exchange rate");
        return ok();
    }

    /// \brief Which statement lines to read: a page from `cursor` (empty for the first), optionally one payout's.
    /// \ingroup core
    struct statement_query {
        asset currency;              ///< the balance's currency
        std::string_view cursor;     ///< from the previous page's `next_cursor`
        std::string_view payout_id;  ///< only the lines a payout settled (Stripe `?payout=po_...`)
        std::size_t limit{100};      ///< at most this many lines

        /// \brief Equality.
        friend constexpr bool operator==(const statement_query&, const statement_query&) noexcept = default;
    };

    /// \brief One page of statement lines, each a fact.
    /// \ingroup core
    struct statement_page {
        std::vector<payment_fact> facts;     ///< oldest first within the page
        snicholls::provider_id next_cursor;  ///< empty on the last page

        /// \brief Equality.
        friend bool operator==(const statement_page&, const statement_page&) = default;
    };

    /// \brief What a provider can do.
    /// \ingroup core
    struct provider_capabilities {
        std::vector<payment_method> methods;           ///< the collection methods
        std::vector<asset> currencies;                 ///< the currencies
        bool payouts{false};                           ///< `payout`
        bool quotes{false};                            ///< `quote`
        bool refunds{false};                           ///< `refund`
        bool credits{false};                           ///< `credits`
        snicholls::provider_account provider_account;  ///< the opaque id of our account at the provider

        /// \brief Equality.
        friend bool operator==(const provider_capabilities&, const provider_capabilities&) = default;
    };

    /// \brief A payment provider: `capabilities`, `collect`, `payout`, `refund`, `quote`, `status`, `balances`,
    /// `credits`, `verify_webhook`, `resolve`, `statement`.
    /// \ingroup core
    template <typename P>
    concept payment_provider = requires(P& p, const P& cp, const collection_request& c, const payout_request& o,
                                        const refund_request& r, const quote_request& q, std::string_view id,
                                        const credit_query& cq, const webhook_delivery& w, const webhook_event& e,
                                        const statement_query& sq) {
        { cp.capabilities() } -> std::same_as<provider_capabilities>;
        { p.collect(c) } -> std::same_as<capability_result<collection>>;
        { p.payout(o) } -> std::same_as<capability_result<payment>>;
        { p.refund(r) } -> std::same_as<capability_result<payment>>;
        { p.quote(q) } -> std::same_as<capability_result<fx_quote>>;
        { p.status(id) } -> std::same_as<capability_result<payment>>;
        { p.balances() } -> std::same_as<capability_result<std::vector<money>>>;
        { p.credits(cq) } -> std::same_as<capability_result<std::vector<credit>>>;
        { cp.verify_webhook(w) } -> std::same_as<capability_result<webhook_event>>;
        { p.resolve(e) } -> std::same_as<capability_result<std::vector<payment_fact>>>;
        { p.statement(sq) } -> std::same_as<capability_result<statement_page>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_payment_provider_hpp
