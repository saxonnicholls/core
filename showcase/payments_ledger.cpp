//
//  payments_ledger.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase: a shop's payments for one day, written against core's capability contracts and run on their reference
//  implementations. Each checkout gets a time-ordered uuid v7 (the first one also made by the compiler), goes to the
//  payment provider on a result railway (SNICHOLLS_TRY_ASSIGN, no exceptions), is retried with jittered backoff and
//  the same idempotency key when the provider times out, walks the payment lifecycle, and is journalled in a
//  key-value store. A webhook is verified (and a tampered copy refused), the provider's statement is reconciled to
//  its balance to the cent, and the expected cash is laid out on a liquidity ladder that finds the day the account
//  would go overdrawn. Exits non-zero if any figure is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/backoff.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/finance/cash_ladder.hpp>
#include <snicholls/core/fsm/lifecycle.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/interfaces/uuid_generator_reference.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/secret.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/time/business_days.hpp>
#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/utc.hpp>
#include <snicholls/core/uuid.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace shop {

using namespace snicholls;
namespace chr = std::chrono;

constexpr asset aud = checked_asset("AUD", 2);
constexpr money dollars(std::string_view text) { return *parse_money(aud, text); }

// Signed amounts read as cash: green in, red out, grey when nothing moves
struct cash {
    money amount;
    void print(acout& out) const
    {
        const int sign = amount.signum();
        const std::string text = to_string(amount, money_digits::all) + ' ' + std::string(amount.asset_of().code());   // every minor unit
        out << styled(sign > 0 ? colours::green : sign < 0 ? colours::red : colours::grey, text);
    }
};

// The day: Thursday 26 November 2026, 09:00 in UTC; the identifiers' clock and the provider's clock agree.
constexpr chr::sys_days today = chr::year{2026} / chr::November / 26;
constexpr utc_ms opening = utc_ms{today} + chr::hours{9};

// The compiler composes the day's first identifier: the same seed and clock give the same uuid at run time.
constexpr uuid first_id = [] {
    auto ids = make_reference_uuid_generator(2026, opening);
    return *ids.v7();
}();
static_assert(first_id.version() == 7 && first_id.timestamp()->time_since_epoch() == opening.time_since_epoch());

enum class outcome : std::uint8_t { pays, declines, transfers, times_out_once };

struct checkout {
    std::string_view order;
    std::string_view amount;
    payment_method method;
    outcome then;
};

struct receipt {
    uuid id;
    payment pay;
    std::uint32_t attempts;
};

// The shop's till: one clock for the identifiers and the provider, the provider, the journal, the jitter.
struct till {
    reference_uuid_generator ids = make_reference_uuid_generator(2026, opening);
    utc_s now = chr::time_point_cast<chr::seconds>(opening.sys_time());
    memory_payment_provider provider{{aud}, secret<std::string>{"whsec_showcase_only"}, [this] { return now; }};
    reference_key_value_store journal;
    stats::wyrand jitter{2026};

    till() = default;
    till(const till&) = delete;   // the provider's clock reads this till
    till& operator=(const till&) = delete;

    void advance(chr::milliseconds d)
    {
        ids.clock().advance(d);
        now = chr::time_point_cast<chr::seconds>(ids.clock().now().sys_time());
    }
};

// The railway: every step returns a result, and the first error leaves the function with its reason.
capability_result<receipt> take_payment(till& t, const checkout& c)
{
    SNICHOLLS_TRY_ASSIGN(const uuid id, t.ids.v7());
    const std::string key = to_string(id);   // the idempotency key: a retry is the same request
    const collection_request request{key, c.order, dollars(c.amount), c.method, c.method == payment_method::bank_transfer ? "CUST-7" : "", "", "", {}};
    if (c.then == outcome::times_out_once)
        t.provider.fail_next(capability_errc::timeout);

    constexpr backoff_policy policy{.min = chr::milliseconds{200}, .max = chr::seconds{5}, .session_grace = chr::seconds{30}};
    std::uint32_t attempt = 0;
    auto started = t.provider.collect(request);
    while (started.is_err() && started.unwrap_err().retryable() && attempt < 3) {
        const auto pause = backoff_delay(policy, attempt++, t.jitter);   // the caller owns the clock and the sleep
        acout{} << "    " << c.order << ": " << started.unwrap_err().code << ", retrying in " << pause.count() << " ms with the same key\n";
        t.advance(pause);
        started = t.provider.collect(request);
    }
    SNICHOLLS_TRY_ASSIGN(const collection opened, std::move(started));
    const std::string_view pid = opened.pay.payment_id.view();

    SNICHOLLS_TRY_ASSIGN(const payment done, c.then == outcome::declines    ? t.provider.decline(pid, "card_declined")
                                             : c.then == outcome::transfers ? t.provider.receive_credit(dollars(c.amount), "CUST-7", "A Customer")
                                                                                 .and_then([&](const credit&) { return t.provider.status(pid); })
                                                                            : t.provider.complete(pid));

    // Journal the outcome; the store's own error is translated into the contract's.
    SNICHOLLS_TRY_VOID(kv_put(t.journal, "payment/" + key, std::string(c.order) + ' ' + to_string(done.amount) + ' ' + std::string(pid))
                           .map_err([](kv_errc) { return capability_error{capability_errc::unavailable}; }));
    t.advance(chr::minutes{7});
    return ok(receipt{id, done, attempt + 1});
}

} // namespace shop

int main()
{
    using namespace shop;
    namespace fsm = snicholls::fsm;
    namespace fin = snicholls::finance;
    bool ok = true;
    const auto expect = [&](bool good, std::string_view what) {
        acout{} << "  " << styled(good ? colours::green : colours::red, good ? "ok  " : "FAIL") << ' ' << what << '\n';
        ok &= good;
    };

    till t;
    t.provider.set_fee_bps(175);   // 1.75% on cards
    t.provider.add_beneficiary("supplier-acme");

    println(colours::cyan, "A shop's payments, {}", to_string(chr::year_month_day{today}));
    println("\n1. Checkouts: uuid v7 keys, the provider's answers, the lifecycle");
    const std::array<checkout, 5> day{{{"ORD-1001", "120.00", payment_method::card, outcome::pays},
                                       {"ORD-1002", "89.90", payment_method::card, outcome::declines},
                                       {"ORD-1003", "1500.00", payment_method::bank_transfer, outcome::transfers},
                                       {"ORD-1004", "45.50", payment_method::card, outcome::pays},
                                       {"ORD-1005", "310.00", payment_method::card, outcome::times_out_once}}};
    std::vector<receipt> receipts;
    std::vector<fsm::sm<fsm::lifecycle>> lifecycles(day.size());
    for (std::size_t i = 0; i < day.size(); ++i) {
        auto& life = lifecycles[i];
        life.process_event(fsm::lifecycle_event::submit{});
        const auto r = take_payment(t, day[i]);
        if (r.is_err()) {
            acout{} << "  " << day[i].order << " refused: " << r.unwrap_err().code << '\n';
            ok = false;
            continue;
        }
        r->pay.status == payment_status::succeeded ? life.process_event(fsm::lifecycle_event::confirm{}) : life.process_event(fsm::lifecycle_event::fail{});
        acout{} << "  " << day[i].order << "  " << r->id << "  " << r->pay.method << "  " << cash{r->pay.amount} << "  fee " << cash{-r->pay.fee} << "  "
                << life.current() << '\n';
        receipts.push_back(*r);
    }
    expect(receipts.size() == 5 && receipts[0].id == first_id, "the first key is the one the compiler made");
    expect(receipts[0].id < receipts[1].id && receipts[3].id < receipts[4].id, "uuid v7 keys sort in the order they were made");
    expect(lifecycles[1].current() == fsm::lifecycle_state::failed && receipts[1].pay.failure_code.view() == "card_declined",
           "ORD-1002 declined, with the provider's code only");
    expect(receipts[4].attempts == 2 && receipts[4].pay.fee == dollars("5.43"), "ORD-1005 timed out once, then went through; 1.75% rounds up");
    const auto replay = t.provider.collect({to_string(receipts[0].id), "ORD-1001", dollars("120.00"), payment_method::card, "", "", "", {}});
    expect(replay && replay->pay.payment_id == receipts[0].pay.payment_id, "replaying a key returns the original payment, not a second one");

    println("\n2. A refund, a supplier payout, and a webhook");
    const auto refund = t.provider.refund({"refund-ORD-1004", receipts[3].pay.payment_id.view(), std::nullopt, "customer changed their mind"});
    lifecycles[3].process_event(fsm::lifecycle_event::settle{});
    lifecycles[3].process_event(fsm::lifecycle_event::reverse{});
    const auto payout = t.provider.payout({"payout-acme-1126", "INV-77", dollars("1000.00"), "supplier-acme", "November stock", std::nullopt});
    acout{} << "  refund " << cash{-refund->amount} << " (ORD-1004 " << lifecycles[3].current() << "), payout " << cash{-payout->amount}
            << " to supplier-acme\n";
    expect(refund && payout && lifecycles[3].current() == fsm::lifecycle_state::reversed, "refunded in full and paid out");

    webhook_event event;
    event.source = "memory";
    event.event_id = "evt_1005";
    event.kind = event_kind::payment_updated;
    event.payment_id = receipts[4].pay.payment_id;
    auto delivery = t.provider.signed_webhook(event);
    const auto verified = t.provider.verify_webhook(delivery.view());
    const auto facts = verified.and_then([&](const webhook_event& e) { return t.provider.resolve(e); });
    delivery.body[0] ^= 1;   // one bit flipped in transit
    const auto tampered = t.provider.verify_webhook(delivery.view());
    acout{} << "  webhook evt_1005: " << (verified ? "verified" : "refused") << ", " << facts->size() << " fact, net " << cash{(*facts)[0].settled.net}
            << "; flipped one bit: " << tampered.unwrap_err().code << '\n';
    expect(facts && facts->size() == 1 && (*facts)[0].settled.net == dollars("304.57"), "the webhook resolves to the payment's fact, net of fee");
    expect(tampered.is_err() && tampered.unwrap_err().code == capability_errc::signature_invalid, "a tampered delivery fails closed");

    println("\n3. Reconciliation: the statement, line by line, against the balance");
    const auto statement = t.provider.statement({aud, "", "", 100});
    money booked = money::zero(aud);
    for (const auto& f : statement->facts) {
        booked = *add(booked, f.settled.net);
        acout{} << "  " << f.fact_id.view() << "  " << f.kind << "  gross " << cash{f.settled.gross} << "  fee " << cash{-f.settled.fee} << "  net "
                << cash{f.settled.net} << '\n';
    }
    const money balance = (*t.provider.balances())[0];
    acout{} << "  booked " << cash{booked} << ", provider balance " << cash{balance} << '\n';
    expect(booked == balance && balance == dollars("921.67"), "the statement reconciles to the balance, to the cent");
    std::size_t journalled = 0;
    for (const auto& r : receipts)
        journalled += kv_get(t.journal, "payment/" + to_string(r.id))->has_value();
    expect(journalled == 5, "every checkout is in the journal under its uuid");

    println("\n4. Liquidity: cards settle two business days later; is there cash for Friday's payroll?");
    std::vector<fin::cash_flow> flows;
    for (const auto& f : statement->facts)
        flows.push_back({f.kind == fact_kind::payment ? add_business_days(today, 2) : today, f.settled.net});
    const chr::sys_days friday = today + chr::days{1};
    flows.push_back({friday, dollars("-2400.00")});                         // payroll
    flows.push_back({today + chr::days{5}, dollars("1850.00")});            // Tuesday: the monthly subscriptions
    std::vector<fin::cash_rung> rungs(flows.size() + 1);
    const auto ladder = fin::make_cash_ladder(flows, {today, dollars("1200.00"), dollars("0.00"), fin::overdue_policy::prudent()}, rungs);
    for (const auto& rung : ladder->rungs)
        acout{} << "  " << to_string(chr::year_month_day{rung.on}) << "  in " << cash{rung.inflow} << "  out " << cash{-rung.outflow}
                << "  balance " << cash{rung.cumulative} << '\n';
    acout{} << "  first below zero: " << to_string(chr::year_month_day{*ladder->first_below}) << ", lowest " << cash{ladder->lowest} << '\n';
    expect(ladder->first_below == friday && ladder->lowest == dollars("-745.50"), "payroll overdraws the account on Friday");
    expect(ladder->rungs[2].cumulative == dollars("-278.33") && ladder->closing == dollars("1571.67"),
           "Monday's card money is not enough; Tuesday's subscriptions restore it");

    acout{} << '\n';
    println(ok ? colours::green : colours::red, "payments ledger: {}", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
