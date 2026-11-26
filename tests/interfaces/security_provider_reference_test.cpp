//
//  security_provider_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/security_provider_reference.hpp>: trades that settle or fail later, facts
//  booked by hand (idempotent, validated, conflicting), resolve by fact id and by order, the statement's cursor and
//  filter, holdings by trade date and by settlement across both instruments of a demerger, and injected failures.
//

#include <snicholls/core/interfaces/security_provider_reference.hpp>

#include "security_provider_probe.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

using namespace snicholls;

namespace {

constexpr asset s32 = checked_asset("S32:ASX", 0);
const money q250 = money::from_units(test::bhp, 250);
const money paid = -*parse_money(test::aud, "11281.25");

const security_holding* find(const std::vector<security_holding>& hs, const asset& a)
{
    for (const auto& h : hs)
        if (h.instrument == a)
            return &h;
    return nullptr;
}

TEST(SecurityProviderReference, ATradeIsBookedOnItsTradeDateAndSettlesLater)
{
    auto p = test::make_security_provider();
    const auto filled = p.execute("ord-1", q250, paid, *parse_money(test::aud, "9.95"));
    ASSERT_TRUE(filled.is_ok());
    EXPECT_EQ(filled->kind, security_fact_kind::buy);
    EXPECT_EQ(filled->source, std::string_view{"memory"});
    EXPECT_EQ(filled->fact_id, std::string_view{"mem_sf_1"});
    EXPECT_EQ(*filled->dates.settles_on, test::probe_time.days() + std::chrono::days{2});
    EXPECT_TRUE(filled->dates.settles_on_estimated);
    EXPECT_FALSE(filled->dates.settled_at.has_value());
    EXPECT_EQ(*cash_movement(*filled), -*parse_money(test::aud, "11291.20"));

    const auto settled = p.settle("ord-1");
    ASSERT_TRUE(settled.is_ok());
    EXPECT_EQ(settled->kind, security_fact_kind::settle);
    EXPECT_EQ(settled->quantity, q250);
    EXPECT_EQ(settled->refs.order, filled->refs.order);
    EXPECT_NE(settled->fact_id, filled->fact_id);
    EXPECT_EQ(p.settle("ord-1").unwrap_err().code, capability_errc::conflict);
    EXPECT_EQ(p.fail_settlement("ord-1").unwrap_err().code, capability_errc::conflict);
    EXPECT_EQ(p.settle("ord-none").unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(p.settle("").unwrap_err().code, capability_errc::not_found);

    ASSERT_TRUE(p.execute("ord-2", -q250, -paid).is_ok());
    const auto failed = p.fail_settlement("ord-2");
    ASSERT_TRUE(failed.is_ok());
    EXPECT_EQ(failed->kind, security_fact_kind::settle_failed);
    const auto h = p.holdings(test::probe_time.days());
    ASSERT_TRUE(h.is_ok());
    const auto* bhp = find(*h, test::bhp);
    ASSERT_NE(bhp, nullptr);
    EXPECT_TRUE(bhp->traded.is_zero()) << "bought 250, sold 250 by trade date";
    EXPECT_EQ(bhp->settled, q250) << "the sale failed to settle";
}

TEST(SecurityProviderReference, ExecuteRefusesWhatIsNotAnOrder)
{
    auto p = test::make_security_provider();
    EXPECT_EQ(p.execute("", q250, paid).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(p.execute("ord-0", money::zero(test::bhp), paid).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(p.execute(std::string(97, 'o'), q250, paid).unwrap_err().code, capability_errc::invalid_argument);
    ASSERT_TRUE(p.execute("ord-1", q250, paid).is_ok());
    EXPECT_EQ(p.execute("ord-1", q250, paid).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(p.execute("ord-3", q250, -paid).unwrap_err().code, capability_errc::invalid_argument) << "a buy that receives";
}

TEST(SecurityProviderReference, BookedFactsAreValidatedAndBookedOnce)
{
    auto p = test::make_security_provider();
    security_fact spin_off;
    spin_off.kind = security_fact_kind::demerger;
    spin_off.fact_id = "action-77";
    spin_off.instrument = test::bhp;
    spin_off.quantity = money::zero(test::bhp);
    spin_off.other_instrument = s32;
    spin_off.other_quantity = money::from_units(s32, 250);
    spin_off.cost_apportionment = *finance::ratio::parse("13.24%");
    spin_off.record_date = test::probe_time.days();
    spin_off.dates.trade_at = test::probe_time;
    spin_off.dates.settled_at = test::probe_time;
    const auto booked = p.book(spin_off);
    ASSERT_TRUE(booked.is_ok()) << booked.unwrap_err().message.view();
    EXPECT_EQ(booked->provider_account, std::string_view{"mem_sec_1"});
    EXPECT_EQ(booked->dates.known_at, test::probe_time);
    EXPECT_TRUE(p.book(spin_off).is_ok()) << "the same fact again changes nothing";
    security_fact changed = spin_off;
    changed.other_quantity = money::from_units(s32, 251);
    changed.quantity = money::from_units(test::bhp, 1);
    EXPECT_EQ(p.book(changed).unwrap_err().code, capability_errc::conflict);
    security_fact invalid = spin_off;
    invalid.fact_id = "action-78";
    invalid.cost_apportionment.reset();
    EXPECT_EQ(p.book(invalid).unwrap_err().message.view(), std::string_view{"a demerger apportions the cost base"});

    const auto h = p.holdings(test::probe_time.days());
    ASSERT_TRUE(h.is_ok());
    ASSERT_EQ(h->size(), 1u) << "BHP's holding is unchanged at zero; S32's arrives";
    EXPECT_EQ(h->front().instrument, s32);
    EXPECT_EQ(h->front().settled, money::from_units(s32, 250));

    webhook_event e;
    e.kind = event_kind::security_updated;
    e.event_id = "evt_9";
    e.fact_id = "action-77";
    const auto facts = p.resolve(e);
    ASSERT_TRUE(facts.is_ok() && facts->size() == 1);
    EXPECT_EQ(facts->front().event_id, std::string_view{"evt_9"});
    EXPECT_EQ(p.statement({"", s32, 10})->facts.size(), 1u) << "the filter matches either instrument";
}

TEST(SecurityProviderReference, TheStatementPagesAndRefusesAStrangeCursor)
{
    auto p = test::make_security_provider();
    for (const std::string_view o : {"a", "b", "c"})
        ASSERT_TRUE(p.execute(o, q250, paid).is_ok());
    const auto first = p.statement({"", std::nullopt, 2});
    ASSERT_TRUE(first.is_ok());
    EXPECT_EQ(first->facts.size(), 2u);
    EXPECT_EQ(first->next_cursor, std::string_view{"2"});
    const auto last = p.statement({"2", std::nullopt, 2});
    EXPECT_EQ(last->facts.size(), 1u);
    EXPECT_TRUE(last->next_cursor.empty());
    EXPECT_EQ(p.statement({"x", std::nullopt, 2}).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(p.statement({"99", std::nullopt, 2}).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_TRUE(p.statement({"", s32, 2})->facts.empty());
}

TEST(SecurityProviderReference, FailNextFailsOnceAndKeepsOnlyTheProvidersCode)
{
    auto p = test::make_security_provider();
    p.fail_next(capability_errc::rate_limited, "too_many_requests {\"detail\":\"secret\"}");
    const auto r = p.holdings(test::probe_time.days());
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().code, capability_errc::rate_limited);
    EXPECT_EQ(r.unwrap_err().provider_code.view(), std::string_view{"too_many_requests"});
    EXPECT_TRUE(p.holdings(test::probe_time.days()).is_ok()) << "once";
    p.fail_next(capability_errc::unavailable);
    EXPECT_EQ(p.resolve({}).unwrap_err().code, capability_errc::unavailable);
}

TEST(SecurityProviderReference, WebhooksAreSignedAsThePaymentReferenceSignsThem)
{
    auto p = test::make_security_provider();
    EXPECT_EQ(memory_security_provider::signature_header, memory_payment_provider::signature_header);
    EXPECT_EQ(memory_security_provider::webhook_tolerance, memory_payment_provider::webhook_tolerance);
    webhook_event e;
    e.source = "memory";
    e.event_id = "evt_1";
    e.kind = event_kind::security_updated;
    e.payment_id = "ord-1";
    const auto d = p.signed_webhook(e);
    const auto payments = test::make_payment_provider();   // the same secret
    const auto v = payments.verify_webhook(d.view());
    ASSERT_TRUE(v.is_ok());
    EXPECT_EQ(*v, *p.verify_webhook(d.view()));
    EXPECT_EQ(v->kind, event_kind::security_updated);
}

} // namespace
