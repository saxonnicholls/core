//
//  payment_provider_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/payment_provider.hpp>: the models and the non-models, terminal states, settlements, fact validation and the dedup key.
//

#include <snicholls/core/concepts/payment_provider.hpp>
#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/money/asset.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

static_assert(payment_provider<memory_payment_provider>);
static_assert(payment_provider<payment_provider_interface>);
static_assert(payment_provider<payment_provider_adapter<memory_payment_provider>>);
static_assert(payment_provider<payment_provider_adapter<memory_payment_provider&>>);
static_assert(payment_provider<payment_provider_ref>);

struct no_webhooks {
    provider_capabilities capabilities() const;
    capability_result<collection> collect(const collection_request&);
    capability_result<payment> payout(const payout_request&);
    capability_result<payment> refund(const refund_request&);
    capability_result<fx_quote> quote(const quote_request&);
    capability_result<payment> status(std::string_view);
    capability_result<std::vector<money>> balances();
    capability_result<std::vector<credit>> credits(const credit_query&);
    capability_result<std::vector<payment_fact>> resolve(const webhook_event&);
    capability_result<statement_page> statement(const statement_query&);
};
struct balances_as_doubles {
    provider_capabilities capabilities() const;
    capability_result<collection> collect(const collection_request&);
    capability_result<payment> payout(const payout_request&);
    capability_result<payment> refund(const refund_request&);
    capability_result<fx_quote> quote(const quote_request&);
    capability_result<payment> status(std::string_view);
    capability_result<std::vector<double>> balances();
    capability_result<std::vector<credit>> credits(const credit_query&);
    capability_result<webhook_event> verify_webhook(const webhook_delivery&) const;
    capability_result<std::vector<payment_fact>> resolve(const webhook_event&);
    capability_result<statement_page> statement(const statement_query&);
};
static_assert(!payment_provider<no_webhooks>);
static_assert(!payment_provider<balances_as_doubles>);
static_assert(!payment_provider<int>);

static_assert(is_terminal(payment_status::succeeded) && is_terminal(payment_status::failed) &&
              is_terminal(payment_status::cancelled) && is_terminal(payment_status::refunded));
static_assert(!is_terminal(payment_status::pending) && !is_terminal(payment_status::processing) &&
              !is_terminal(payment_status::partially_refunded) && !is_terminal(payment_status::requires_action));

constexpr asset aud = checked_asset("AUD", 2), usd = checked_asset("USD", 2);
constexpr money a(std::uint64_t cents, bool negative = false) { return money::from_units(aud, cents, negative); }
static_assert(noexcept(balances(settlement{})));
static_assert(balances({a(10000), a(175), a(9825)}));
static_assert(balances({a(2000, true), a(0), a(2000, true)}));
static_assert(!balances({a(10000), a(175), a(9826)}));
static_assert(!balances({a(10000), money::zero(usd), a(10000)}));

TEST(PaymentProviderConcept, AClientSecretNeverPrints)
{
    next_step n;
    n.kind = next_step_kind::client_confirm;
    n.client_secret = secret<std::string>{"pi_123_secret_456"};
    std::ostringstream out;
    out << n.client_secret;
    EXPECT_EQ(out.str().find("secret_456"), std::string::npos);
    EXPECT_EQ(n.client_secret.reveal(), "pi_123_secret_456");
}

TEST(PaymentProviderConcept, AnOwnedDeliveryViewsItsOwnBytes)
{
    const owned_delivery d{{{"Sig", "t=1"}}, "body", utc_s::from_unix_seconds(9)};
    const webhook_delivery v = d.view();
    EXPECT_EQ(v.body, "body");
    EXPECT_EQ(find_header(v.headers, "sig"), "t=1");
    EXPECT_EQ(v.received_at, d.received_at);
}

TEST(PaymentProviderConcept, ValidateRefusesFactsTheLedgerCouldNotBook)
{
    payment_fact f;
    f.source = "stripe";
    f.fact_id = "txn_1";
    f.kind = fact_kind::payment;
    f.settled = {a(15300), a(250), a(15050)};
    EXPECT_TRUE(validate(f).is_ok());
    payment_fact no_id = f;
    no_id.fact_id = {};
    EXPECT_TRUE(validate(no_id).is_err());
    payment_fact unbalanced = f;
    unbalanced.settled.net = a(15000);
    EXPECT_EQ(validate(unbalanced).unwrap_err().message, std::string_view{"gross is not net plus fee in one asset"});
    payment_fact fx = f;
    fx.presentment = money::from_units(usd, 10000);
    EXPECT_TRUE(validate(fx).is_err()) << "presentment without its exchange rate";
    fx.exchange_rate = *rate::parse(usd, aud, "1.53");
    EXPECT_TRUE(validate(fx).is_ok());
    payment_fact same = f;
    same.presentment = f.settled.gross;
    same.exchange_rate = *rate::parse(aud, aud, "1");
    EXPECT_TRUE(validate(same).is_err()) << "presentment only when it differs";
}

TEST(PaymentProviderConcept, TheDedupKeyIsSourceEvtEventId)
{
    webhook_event e;
    e.source = "stripe";
    e.event_id = "evt_1";
    EXPECT_EQ(e.dedup_key(), std::string_view{"stripe:evt:evt_1"});
    e.source = *provider_name::make(std::string(32, 's'));
    e.event_id = *event_id::make(std::string(64, 'e'));
    EXPECT_EQ(e.dedup_key().size(), 101u) << "the longest key fits";
    EXPECT_EQ(enum_name(fact_kind::dispute_release), "dispute_release");
    EXPECT_EQ(enum_name(fact_kind::agent_collection), "agent_collection");
    static_assert(static_cast<int>(fact_kind::other) == 15 && static_cast<int>(fact_kind::agent_collection) == 16,
                  "appended: the wire values of the existing kinds never move");
}

} // namespace
