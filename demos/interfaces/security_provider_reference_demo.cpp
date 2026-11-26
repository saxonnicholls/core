//
//  security_provider_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/security_provider_reference.hpp>: a buy filled and settled two days later, a
//  sale that fails to settle, a dividend and a demerger booked by hand, holdings by trade date and by settlement, a
//  statement read in pages, a signed webhook resolved to its facts, and an injected failure. Exits non-zero if a result
//  is not the one documented.
//

#include <snicholls/core/interfaces/security_provider_reference.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/money/asset.hpp>

#include <chrono>
#include <cstdio>
#include <string>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

constexpr asset aud = checked_asset("AUD", 2), bhp = checked_asset("BHP:ASX", 0), s32 = checked_asset("S32:ASX", 0);

void show(memory_security_provider& p, std::chrono::sys_days day, const char* when)
{
    for (const security_holding& h : *p.holdings(day))
        std::printf("  %s: %s traded %s, settled %s\n", when, std::string(h.instrument.code()).c_str(),
                    to_string(h.traded).c_str(), to_string(h.settled).c_str());
}
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/security_provider_reference.hpp demo\n");
    //! [reference]
    memory_security_provider broker{secret<std::string>{"whsec_demo_only"}, manual_utc_clock{utc_s::from_unix_seconds(1'800'000'000)}};
    const auto buy = broker.execute("ord-1", money::from_units(bhp, 100), -*parse_money(aud, "4512.50"),
                                    *parse_money(aud, "9.95"));          // T: a buy, due T+2
    const std::chrono::sys_days trade_day = broker.clock().now().days();
    broker.clock().advance(std::chrono::days{2});                         // the provider's own clock, moved by hand
    const auto settled = broker.settle("ord-1");                         // T+2: settled
    //! [reference]
    const utc_s now = std::chrono::floor<std::chrono::seconds>(broker.clock().now().sys_time());
    check(buy.is_ok() && buy->kind == security_fact_kind::buy && *buy->dates.settles_on == trade_day + std::chrono::days{2},
          "a buy, due to settle two days later");
    check(settled.is_ok() && settled->kind == security_fact_kind::settle, "then a settle fact");
    show(broker, trade_day, "T  ");
    show(broker, now.days(), "T+2");
    const auto at_trade = broker.holdings(trade_day);
    check(at_trade->size() == 1 && at_trade->front().settled.is_zero(), "on the trade date, nothing settled yet");

    check(broker.execute("ord-2", -money::from_units(bhp, 40), *parse_money(aud, "1805.00")).is_ok(), "a sale of 40");
    check(broker.fail_settlement("ord-2").is_ok(), "which fails to settle");
    check(broker.settle("ord-2").unwrap_err().code == capability_errc::conflict, "and cannot then settle");

    security_fact dividend;
    dividend.kind = security_fact_kind::dividend_paid;
    dividend.instrument = bhp;
    dividend.quantity = money::zero(bhp);
    dividend.consideration = *parse_money(aud, "72.00");
    dividend.fees = money::zero(aud);
    dividend.taxes.franking_credit = *parse_money(aud, "30.86");
    dividend.taxes.franked = finance::ratio{1, 1};
    dividend.record_date = now.days();
    dividend.pay_date = now.days() + std::chrono::days{21};
    dividend.dates.trade_at = now;
    check(broker.book(dividend).is_ok(), "a fully franked dividend booked by hand");

    security_fact spin_off;
    spin_off.kind = security_fact_kind::demerger;
    spin_off.instrument = bhp;
    spin_off.quantity = money::zero(bhp);
    spin_off.other_instrument = s32;
    spin_off.other_quantity = money::from_units(s32, 100);
    spin_off.cost_apportionment = *finance::ratio::parse("13.24%");
    spin_off.record_date = now.days();
    spin_off.dates.trade_at = now;
    spin_off.dates.settled_at = now;
    check(broker.book(spin_off).is_ok(), "a demerger: S32 arrives with 13.24% of the cost base");
    spin_off.cost_apportionment.reset();
    check(broker.book(spin_off).is_err(), "a demerger without its apportionment is refused");
    show(broker, now.days(), "now");

    std::size_t lines = 0, pages = 0;
    std::string cursor;
    for (;;) {
        const auto page = broker.statement({cursor, std::nullopt, 2});
        ++pages;
        for (const security_fact& f : page->facts) {
            ++lines;
            std::printf("  %s %s\n", std::string(f.fact_id.view()).c_str(), std::string(enum_name(f.kind)).c_str());
        }
        if (page->next_cursor.empty())
            break;
        cursor = std::string(page->next_cursor.view());
    }
    check(lines == 6 && pages == 3, "six facts in three pages of two");

    webhook_event e;
    e.source = "memory";
    e.event_id = "evt_1";
    e.kind = event_kind::security_updated;
    e.payment_id = "ord-1";
    const auto verified = broker.verify_webhook(broker.signed_webhook(e).view());
    const auto facts = verified ? broker.resolve(*verified) : capability_result<std::vector<security_fact>>(fail(capability_errc::internal));
    check(facts.is_ok() && facts->size() == 2, "the order's event resolves to its buy and its settlement");

    broker.fail_next(capability_errc::unavailable, "maintenance window");
    check(broker.statement({}).unwrap_err().code == capability_errc::unavailable, "an injected outage, once");
    check(broker.statement({}).is_ok(), "and then it answers");
    return failures == 0 ? 0 : 1;
}
