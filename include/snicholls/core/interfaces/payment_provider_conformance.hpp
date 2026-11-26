//
//  payment_provider_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_payment_provider_conformance`: the conformance kit of the `payment_provider` contract.
/// \ingroup core
///
/// **Overview.** Drives any `payment_provider`, in either form, through the contract of
/// `concepts/payment_provider.hpp` and reports which check failed (`interfaces/capability_conformance.hpp`). The probe
/// is a default-constructible type with
/// - `auto& provider()`: the implementation (a provider, a `payment_provider_interface&`, a `payment_provider_ref`);
/// - `money amount()`: a positive amount in a supported currency; `asset unsupported_currency()`: one it does not hold;
/// - `void payer_pays(const collection& c)`: the payer completes `c` (a card, or a bank transfer by its reference);
/// - `owned_delivery deliver(const webhook_event& e)`: `e` as the provider would sign and send it, received now;
///   `owned_delivery deliver_late(const webhook_event& e)`: the same, received outside the provider's time window;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `collect` fails with `code`, as
///   though the provider had answered `response`;
///
/// and, optionally, `webhook_event webhook_sample()` (the event the webhook check round-trips, for a provider whose
/// events cannot carry the kit's default) and `std::size_t add_other_lines()` (puts lines that are not payments on the
/// statement and returns how many).
///
/// The checks: `capabilities name a method and a currency`, `a request without a key or with a non-positive amount is
/// invalid`, `an unsupported currency is unsupported`, `collect is idempotent`, `a card collection succeeds once the
/// payer pays`, `a bank transfer gives details and a reference and succeeds on the credit`, `a refund never exceeds
/// what remains`, `an unknown payment is not found`, `a webhook verifies and carries its event`, `a webhook fails
/// closed`, `an event resolves to authoritative facts that balance` (a test delivery to none), `a statement pages
/// through every line once`,
/// `a provider failure never leaks its response`.
///
/// \snippet interfaces/payment_provider_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/payment_provider.hpp`, `interfaces/payment_provider.hpp`,
/// `interfaces/capability_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form; checks that need a method the provider lacks pass without
/// running it.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/payment_provider_conformance_test.cpp`,
/// `demos/interfaces/payment_provider_conformance_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cstddef>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `<vector>`, `concepts/payment_provider.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_payment_provider_conformance_hpp
#define snicholls_core_interfaces_payment_provider_conformance_hpp

#include <snicholls/core/concepts/payment_provider.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept payment_provider_probe = std::is_default_constructible_v<Probe> &&
            payment_provider<std::remove_reference_t<decltype(std::declval<Probe&>().provider())>> &&
            requires(Probe& p, const collection& c, const webhook_event& e, capability_errc code, std::string_view r) {
                { p.amount() } -> std::same_as<money>;
                { p.unsupported_currency() } -> std::same_as<asset>;
                p.payer_pays(c);
                { p.deliver(e) } -> std::same_as<owned_delivery>;
                { p.deliver_late(e) } -> std::same_as<owned_delivery>;
                p.provider_fails(code, r);
            };

        template <typename Probe>
        bool kit_supports(Probe& p, payment_method m)
        {
            const auto caps = p.provider().capabilities();
            return std::find(caps.methods.begin(), caps.methods.end(), m) != caps.methods.end();
        }

        template <typename Probe>
        payment_method kit_method(Probe& p)
        {
            return kit_supports(p, payment_method::card) ? payment_method::card : payment_method::bank_transfer;
        }

        template <typename Probe>
        collection_request kit_request(Probe& p, std::string_view key, payment_method m)
        {
            return {key, "order-1", p.amount(), m, "CUST-7", "https://example.test/ok", "https://example.test/cancel", {}};
        }

        inline webhook_event kit_event()
        {
            webhook_event e;
            e.source = "test";
            e.event_id = "evt_1";
            e.fact_id = "txn_1";
            e.kind = event_kind::payment_updated;
            e.type = "payment.succeeded";
            e.payment_id = "pay_1";
            e.our_reference = "order-1";
            e.occurred_at = utc_s::from_unix_seconds(1'800'000'000);
            return e;
        }

        template <typename Probe>
        webhook_event kit_sample_event(Probe& p)
        {
            if constexpr (requires { p.webhook_sample(); })
                return p.webhook_sample();
            else
                return kit_event();
        }
    } // namespace detail

    /// \brief Runs every check of the `payment_provider` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::payment_provider_probe<Probe>
    [[nodiscard]] capability_conformance_report run_payment_provider_conformance()
    {
        capability_conformance_report report;
        using detail::kit_method;
        using detail::kit_request;
        using detail::kit_supports;
        const auto code_is = [](const auto& r, capability_errc c) { return r.is_err() && r.unwrap_err().code == c; };

        detail::capability_check<Probe>(report, "capabilities name a method and a currency", [](Probe& p, const auto& expect) {
            const auto caps = p.provider().capabilities();
            expect(!caps.methods.empty(), "at least one method");
            expect(!caps.currencies.empty(), "at least one currency");
        });

        detail::capability_check<Probe>(report, "a request without a key or with a non-positive amount is invalid", [&](Probe& p, const auto& expect) {
            auto& pp = p.provider();
            const payment_method m = kit_method(p);
            expect(code_is(pp.collect(kit_request(p, "", m)), capability_errc::invalid_argument), "no key is invalid_argument");
            const std::string long_key(client_id::capacity() + 1, 'k');
            expect(code_is(pp.collect(kit_request(p, long_key, m)), capability_errc::invalid_argument),
                   "a key longer than 64 characters is invalid_argument");
            auto zero = kit_request(p, "k-zero", m);
            zero.amount = money::zero(zero.amount.asset_of());
            expect(code_is(pp.collect(zero), capability_errc::invalid_argument), "zero is invalid_argument");
            auto negative = kit_request(p, "k-neg", m);
            negative.amount = -negative.amount;
            expect(code_is(pp.collect(negative), capability_errc::invalid_argument), "a negative amount is invalid_argument");
        });

        detail::capability_check<Probe>(report, "an unsupported currency is unsupported", [&](Probe& p, const auto& expect) {
            auto r = kit_request(p, "k-ccy", kit_method(p));
            r.amount = money::from_units(p.unsupported_currency(), 100);
            expect(code_is(p.provider().collect(r), capability_errc::unsupported), "collect is unsupported");
        });

        detail::capability_check<Probe>(report, "collect is idempotent", [&](Probe& p, const auto& expect) {
            auto& pp = p.provider();
            const payment_method m = kit_method(p);
            const auto first = pp.collect(kit_request(p, "k-1", m));
            const auto again = pp.collect(kit_request(p, "k-1", m));
            expect(first.is_ok() && again.is_ok(), "both answer");
            if (first.is_ok() && again.is_ok())
                expect(first->pay.payment_id == again->pay.payment_id, "a repeat returns the first payment");
            auto changed = kit_request(p, "k-1", m);
            changed.our_reference = "order-2";
            expect(code_is(pp.collect(changed), capability_errc::conflict), "the key with another request is a conflict");
        });

        detail::capability_check<Probe>(report, "a card collection succeeds once the payer pays", [](Probe& p, const auto& expect) {
            if (!kit_supports(p, payment_method::card))
                return;
            auto& pp = p.provider();
            const auto c = pp.collect(kit_request(p, "k-card", payment_method::card));
            expect(c.is_ok(), "a card collection starts");
            if (!c.is_ok())
                return;
            expect(c->next.kind == next_step_kind::redirect || c->next.kind == next_step_kind::client_confirm,
                   "the payer is redirected or confirms");
            expect(!is_terminal(c->pay.status), "it waits for the payer");
            p.payer_pays(*c);
            const auto s = pp.status(c->pay.payment_id.view());
            expect(s.is_ok() && s->status == payment_status::succeeded, "it succeeds");
            expect(s.is_ok() && s->received == p.amount(), "the amount is received");
        });

        detail::capability_check<Probe>(report, "a bank transfer gives details and a reference and succeeds on the credit", [](Probe& p, const auto& expect) {
            if (!kit_supports(p, payment_method::bank_transfer))
                return;
            auto& pp = p.provider();
            const auto c = pp.collect(kit_request(p, "k-bank", payment_method::bank_transfer));
            expect(c.is_ok() && c->next.kind == next_step_kind::bank_transfer && c->next.bank.has_value(),
                   "bank details are the next step");
            if (!c.is_ok() || !c->next.bank)
                return;
            expect(!c->next.bank->reference.empty(), "there is a reference to quote");
            p.payer_pays(*c);
            const auto s = pp.status(c->pay.payment_id.view());
            expect(s.is_ok() && s->status == payment_status::succeeded, "the credit completes it");
        });

        detail::capability_check<Probe>(report, "a refund never exceeds what remains", [&](Probe& p, const auto& expect) {
            if (!p.provider().capabilities().refunds || !kit_supports(p, payment_method::card))
                return;
            auto& pp = p.provider();
            const auto c = pp.collect(kit_request(p, "k-refund", payment_method::card));
            expect(c.is_ok(), "a card collection starts");
            if (!c.is_ok())
                return;
            p.payer_pays(*c);
            const money whole = p.amount();
            const money part = money::from_units(whole.asset_of(), 1);
            const std::string_view id = c->pay.payment_id.view();
            expect(pp.refund({"r-1", id, part, "partial"}).is_ok(), "a partial refund succeeds");
            const auto after = pp.status(id);
            expect(after.is_ok() && after->status == payment_status::partially_refunded, "the collection is partially refunded");
            expect(code_is(pp.refund({"r-2", id, whole, "too much"}), capability_errc::conflict), "more than remains is a conflict");
            expect(pp.refund({"r-3", id, std::nullopt, "the rest"}).is_ok(), "the rest is refunded");
            const auto done = pp.status(id);
            expect(done.is_ok() && done->status == payment_status::refunded, "the collection is refunded");
        });

        detail::capability_check<Probe>(report, "an unknown payment is not found", [&](Probe& p, const auto& expect) {
            expect(code_is(p.provider().status("no-such-payment"), capability_errc::not_found), "status is not_found");
        });

        detail::capability_check<Probe>(report, "a webhook verifies and carries its event", [](Probe& p, const auto& expect) {
            const webhook_event e = detail::kit_sample_event(p);
            const owned_delivery d = p.deliver(e);
            const auto v = p.provider().verify_webhook(d.view());
            expect(v.is_ok(), "a signed delivery verifies");
            if (!v.is_ok())
                return;
            expect(v->event_id == e.event_id && v->fact_id == e.fact_id && v->kind == e.kind && v->payment_id == e.payment_id,
                   "the event round-trips");
            const auto again = p.provider().verify_webhook(d.view());
            expect(again.is_ok() && *again == *v, "the event id is stable across deliveries");
        });

        detail::capability_check<Probe>(report, "a webhook fails closed", [&](Probe& p, const auto& expect) {
            auto& pp = p.provider();
            owned_delivery tampered = p.deliver(detail::kit_event());
            tampered.body[tampered.body.size() / 2] = static_cast<char>(tampered.body[tampered.body.size() / 2] ^ 1);
            expect(code_is(pp.verify_webhook(tampered.view()), capability_errc::signature_invalid), "a tampered body");
            owned_delivery unsigned_one = p.deliver(detail::kit_event());
            unsigned_one.headers.clear();
            expect(code_is(pp.verify_webhook(unsigned_one.view()), capability_errc::signature_invalid), "no signature");
            const owned_delivery late = p.deliver_late(detail::kit_event());
            expect(code_is(pp.verify_webhook(late.view()), capability_errc::stale), "outside the window is stale");
            const webhook_delivery garbage{{}, "\x01\x02 not a webhook", {}};
            expect(code_is(pp.verify_webhook(garbage), capability_errc::signature_invalid), "garbage");
        });

        detail::capability_check<Probe>(report, "an event resolves to authoritative facts that balance", [](Probe& p, const auto& expect) {
            if (!kit_supports(p, payment_method::card))
                return;
            auto& pp = p.provider();
            const auto c = pp.collect(kit_request(p, "k-fact", payment_method::card));
            expect(c.is_ok(), "a card collection starts");
            if (!c.is_ok())
                return;
            p.payer_pays(*c);
            webhook_event e = detail::kit_event();
            e.payment_id = c->pay.payment_id;
            const auto first = pp.resolve(e);
            expect(first.is_ok() && first->size() == 1, "the payment resolves to one fact");
            if (!first.is_ok() || first->size() != 1)
                return;
            const payment_fact& f = first->front();
            expect(f.kind == fact_kind::payment, "the fact is a payment");
            expect(validate(f).is_ok(), "the fact validates");
            expect(f.settled.gross == p.amount(), "gross is the amount paid");
            expect(!f.fact_id.empty(), "the fact has an id");
            e.event_id = "evt_redelivered";
            const auto again = pp.resolve(e);
            expect(again.is_ok() && !again->empty() && again->front().fact_id == f.fact_id, "a redelivery points at the same fact");
            webhook_event nothing = detail::kit_event();
            nothing.kind = event_kind::other;
            const auto none = pp.resolve(nothing);
            expect(none.is_ok() && none->empty(), "an event that books nothing resolves to no facts");
            webhook_event test = detail::kit_event();
            test.payment_id = c->pay.payment_id;
            test.event_id = "evt_test";
            test.is_test = true;
            const auto booked = pp.resolve(test);
            expect(booked.is_ok() && booked->empty(), "a test delivery resolves to no facts");
            expect(f.dates.known_at >= f.dates.trade_at, "a fact is known no earlier than it happened");
        });

        detail::capability_check<Probe>(report, "a statement pages through every line once", [](Probe& p, const auto& expect) {
            auto& pp = p.provider();
            const payment_method m = kit_method(p);
            for (const std::string_view key : {"k-s1", "k-s2", "k-s3"}) {
                const auto c = pp.collect(kit_request(p, key, m));
                expect(c.is_ok(), "a collection starts");
                if (c.is_ok())
                    p.payer_pays(*c);
            }
            std::size_t others = 0;
            if constexpr (requires { p.add_other_lines(); })
                others = p.add_other_lines();
            std::size_t not_payments = 0;
            std::vector<std::string> seen;
            statement_query q{p.amount().asset_of(), "", "", 2};
            std::string cursor;
            for (int guard = 0; guard < 50; ++guard) {
                const auto page = pp.statement(q);
                expect(page.is_ok(), "every page answers");
                if (!page.is_ok())
                    return;
                for (const payment_fact& f : page->facts) {
                    expect(validate(f).is_ok(), "every fact validates");
                    seen.emplace_back(f.fact_id.view());
                    if (f.kind != fact_kind::payment && f.kind != fact_kind::bank_credit)
                        ++not_payments;
                }
                if (page->next_cursor.empty())
                    break;
                cursor = std::string(page->next_cursor.view());
                q.cursor = cursor;
            }
            expect(seen.size() >= 3 + others, "every line is a fact");
            expect(not_payments >= others, "lines that are not payments are facts too, never skipped");
            std::sort(seen.begin(), seen.end());
            expect(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "no line twice");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [&](Probe& p, const auto& expect) {
            p.provider_fails(capability_errc::rejected, capability_provider_response);
            const auto r = p.provider().collect(kit_request(p, "k-leak", kit_method(p)));
            expect(code_is(r, capability_errc::rejected), "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_payment_provider_conformance_hpp
