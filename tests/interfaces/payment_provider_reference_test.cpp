//
//  payment_provider_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/payment_provider_reference.hpp>: credits, fees, payouts, quotes, webhook signatures, facts and statements.
//

#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

using std::chrono::seconds;

constexpr asset aud = checked_asset("AUD", 2);
constexpr asset usd = checked_asset("USD", 2);
money a(std::string_view text) { return *parse_money(aud, text); }
money u(std::string_view text) { return *parse_money(usd, text); }

struct MemoryPaymentProvider : testing::Test {
    utc_s now = utc_s::from_unix_seconds(1'800'000'000);
    memory_payment_provider p{{aud, usd}, secret<std::string>{"whsec_test_only"}, [this] { return now; }};
};

TEST_F(MemoryPaymentProvider, AnUnmatchedCreditStaysForSuspenseAndCompletesNothing)
{
    const auto c = p.collect({"k1", "order-1", a("50"), payment_method::bank_transfer, "CUST-1", "", "", {}});
    ASSERT_TRUE(c.is_ok());
    ASSERT_TRUE(p.receive_credit(a("20"), "SOMEONE-ELSE", "A Stranger").is_ok());
    EXPECT_EQ(p.status(c->pay.payment_id.view())->status, payment_status::pending);
    const auto listed = p.credits({aud, now, now + seconds{1}});
    ASSERT_EQ(listed->size(), 1u);
    EXPECT_EQ((*listed)[0].reference, std::string_view{"SOMEONE-ELSE"});
    EXPECT_TRUE(p.credits({usd, now, now + seconds{1}})->empty());
    EXPECT_TRUE(p.credits({aud, now + seconds{1}, now + seconds{2}})->empty()) << "the window is [from, to)";
}

TEST_F(MemoryPaymentProvider, AMatchedCreditRecordsWhatArrivedNotWhatWasExpected)
{
    const auto c = p.collect({"k1", "order-1", a("50"), payment_method::bank_transfer, "CUST-1", "", "", {}});
    ASSERT_TRUE(p.receive_credit(a("49.99"), "CUST-1", "A Payer").is_ok());
    const auto s = p.status(c->pay.payment_id.view());
    EXPECT_EQ(s->status, payment_status::succeeded);
    EXPECT_EQ(s->received, a("49.99"));
    EXPECT_EQ(s->amount, a("50"));
}

TEST_F(MemoryPaymentProvider, CardFeesAreTakenFromTheBalanceExactly)
{
    p.set_fee_bps(175);   // 1.75%
    const auto c = p.collect({"k1", "order-1", a("100.01"), payment_method::card, "", "", "", {}});
    ASSERT_TRUE(p.complete(c->pay.payment_id.view()).is_ok());
    EXPECT_EQ(p.status(c->pay.payment_id.view())->fee, a("1.76"));   // 100.01 * 0.0175 = 1.750175, rounded up
    EXPECT_EQ((*p.balances())[0], a("98.25"));
    EXPECT_EQ(p.complete(c->pay.payment_id.view()).unwrap_err().code, capability_errc::conflict);
}

TEST_F(MemoryPaymentProvider, ADeclineKeepsTheProvidersCode)
{
    const auto c = p.collect({"k1", "order-1", a("10"), payment_method::card, "", "", "", {}});
    const auto d = p.decline(c->pay.payment_id.view(), "card_declined");
    ASSERT_TRUE(d.is_ok());
    EXPECT_EQ(d->status, payment_status::failed);
    EXPECT_EQ(d->failure_code, std::string_view{"card_declined"});
}

TEST_F(MemoryPaymentProvider, PayoutsNeedABeneficiaryAndTheBalance)
{
    const auto c = p.collect({"k1", "o", a("100"), payment_method::card, "", "", "", {}});
    ASSERT_TRUE(p.complete(c->pay.payment_id.view()).is_ok());
    EXPECT_EQ(p.payout({"p1", "o", a("10"), "nobody", "ref", std::nullopt}).unwrap_err().code, capability_errc::not_found);
    p.add_beneficiary("ben-1");
    EXPECT_EQ(p.payout({"p2", "o", a("100.01"), "ben-1", "ref", std::nullopt}).unwrap_err().code, capability_errc::insufficient_funds);
    EXPECT_EQ(p.payout({"p3", "o", u("1"), "ben-1", "ref", std::nullopt}).unwrap_err().code, capability_errc::insufficient_funds);
    const auto paid = p.payout({"p4", "o", a("60"), "ben-1", "ref", std::nullopt});
    ASSERT_TRUE(paid.is_ok());
    EXPECT_EQ(paid->kind, movement::payout);
    EXPECT_EQ((*p.balances())[0], a("40"));
    EXPECT_EQ(p.payout({"p4", "o", a("60"), "ben-1", "ref", std::nullopt})->payment_id, paid->payment_id);
    EXPECT_EQ((*p.balances())[0], a("40")) << "an idempotent retry moves no money";
}

TEST_F(MemoryPaymentProvider, QuotesAreExactAndExpire)
{
    p.set_rate(*rate::parse(aud, usd, "0.66"));   // 1 AUD = 0.66 USD
    p.set_fee_bps(50);
    const auto send = p.quote({a("1000"), usd, true});
    ASSERT_TRUE(send.is_ok());
    EXPECT_EQ(send->fee, a("5"));
    EXPECT_EQ(send->target, u("656.7"));   // 995 * 0.66
    const auto receive = p.quote({u("66"), aud, false});
    ASSERT_TRUE(receive.is_ok());
    EXPECT_EQ(receive->target, u("66"));
    EXPECT_EQ(receive->source, a("100.5"));   // 100 + 0.50 fee
    EXPECT_EQ(p.quote({u("1"), checked_asset("EUR", 2), true}).unwrap_err().code, capability_errc::unsupported);
    p.add_beneficiary("ben-1");
    static_cast<void>(p.receive_credit(u("100"), "x", "y"));
    now += seconds{61};
    EXPECT_EQ(p.payout({"p1", "o", u("66"), "ben-1", "r", receive->quote_id.view()}).unwrap_err().code, capability_errc::stale);
}

TEST_F(MemoryPaymentProvider, AWebhookSignedWithAnotherSecretIsRefused)
{
    memory_payment_provider other{{aud}, secret<std::string>{"whsec_someone_else"}, [this] { return now; }};
    webhook_event e;
    e.event_id = "evt_9";
    e.kind = event_kind::credit_received;
    e.reference = "CUST-1";
    e.needs_lookup = true;
    const auto d = other.signed_webhook(e);
    EXPECT_EQ(p.verify_webhook(d.view()).unwrap_err().code, capability_errc::signature_invalid);
    const auto mine = p.verify_webhook(p.signed_webhook(e).view());
    ASSERT_TRUE(mine.is_ok());
    EXPECT_EQ(mine->reference, std::string_view{"CUST-1"});
    EXPECT_TRUE(mine->needs_lookup);
}

TEST_F(MemoryPaymentProvider, ARotatedSecretAcceptsEitherTag)
{
    webhook_event e;
    e.event_id = "evt_1";
    e.kind = event_kind::other;
    auto d = p.signed_webhook(e);
    std::string& v = d.headers[0].value;
    v = v.substr(0, v.find(',')) + ",v1=" + std::string(64, '0') + v.substr(v.find(','));
    EXPECT_TRUE(p.verify_webhook(d.view()).is_ok());
}

TEST_F(MemoryPaymentProvider, AMalformedButSignedBodyIsInvalid)
{
    webhook_event e;
    e.event_id = "evt_1";
    e.kind = event_kind::other;
    const auto d = p.signed_webhook(e);
    owned_delivery bad = d;
    bad.body += "surprise=1\n";   // a field this provider never sends, signed properly: strict parsing refuses it
    const std::string t = d.headers[0].value.substr(2, d.headers[0].value.find(',') - 2);
    hmac<sha256> m{std::string("whsec_test_only")};
    m.update(t);
    m.update(std::string_view{"."});
    m.update(bad.body);
    bad.headers[0].value = "t=" + t + ",v1=" + std::string(m.finish().hex().view());
    EXPECT_EQ(p.verify_webhook(bad.view()).unwrap_err().code, capability_errc::invalid_argument);
}

TEST_F(MemoryPaymentProvider, ABankCreditResolvesByItsReferenceNumberAndRefundsBookNegative)
{
    static_cast<void>(p.collect({"k1", "order-1", a("50"), payment_method::bank_transfer, "CUST-1", "", "", {}}));
    const auto cr = p.receive_credit(a("50"), "CUST-1", "A Payer");
    webhook_event e;
    e.source = "memory";
    e.event_id = "delivery-1";
    e.kind = event_kind::credit_received;
    e.fact_id = *fact_id::make(cr->credit_id.view());
    e.needs_lookup = true;
    const auto facts = p.resolve(e);
    ASSERT_EQ(facts->size(), 1u);
    EXPECT_EQ((*facts)[0].kind, fact_kind::bank_credit);
    EXPECT_EQ((*facts)[0].fact_id, cr->credit_id);
    EXPECT_EQ((*facts)[0].event_id, std::string_view{"delivery-1"});
    EXPECT_EQ((*facts)[0].settled.net, a("50"));

    const auto card = p.collect({"k2", "order-2", a("80"), payment_method::card, "", "", "", {}});
    static_cast<void>(p.complete(card->pay.payment_id.view()));
    const auto r = p.refund({"r1", card->pay.payment_id.view(), a("30"), "partial"});
    webhook_event re;
    re.source = "memory";
    re.event_id = "evt_r";
    re.kind = event_kind::refund_updated;
    re.payment_id = r->payment_id;
    const auto refund_facts = p.resolve(re);
    ASSERT_EQ(refund_facts->size(), 1u);
    EXPECT_EQ((*refund_facts)[0].kind, fact_kind::refund);
    EXPECT_EQ((*refund_facts)[0].settled.gross, a("-30"));
    EXPECT_TRUE(validate((*refund_facts)[0]).is_ok());
}

TEST_F(MemoryPaymentProvider, AStatementFiltersByPayoutAndRefusesABadCursor)
{
    const auto c = p.collect({"k1", "o", a("100"), payment_method::card, "", "", "", {}});
    static_cast<void>(p.complete(c->pay.payment_id.view()));
    p.add_beneficiary("ben-1");
    const auto paid = p.payout({"p1", "o", a("60"), "ben-1", "ref", std::nullopt});
    const auto only = p.statement({aud, "", paid->payment_id.view(), 100});
    ASSERT_EQ(only->facts.size(), 1u);
    EXPECT_EQ(only->facts[0].kind, fact_kind::payout);
    EXPECT_EQ(p.statement({aud, "x", "", 100}).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_TRUE(p.statement({usd, "", "", 100})->facts.empty());
}

TEST_F(MemoryPaymentProvider, FailNextFailsTheNextMovementAndKeepsOnlyTheCode)
{
    p.fail_next(capability_errc::rejected, "card_declined Your card was declined for 4242 4242 4242 4242");
    const auto r = p.collect({"k1", "o", a("10"), payment_method::card, "", "", "", {}});
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().provider_code, std::string_view{"card_declined"});
    EXPECT_TRUE(r.unwrap_err().message.empty());
    EXPECT_TRUE(p.collect({"k1", "o", a("10"), payment_method::card, "", "", "", {}}).is_ok());
}

} // namespace
