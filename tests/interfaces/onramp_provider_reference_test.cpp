//
//  onramp_provider_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/onramp_provider_reference.hpp>: exact rounding in the venue's favour, fees, withdrawals, max_price, the statement and the injected failure.
//

#include <snicholls/core/interfaces/onramp_provider_reference.hpp>

#include <gtest/gtest.h>

#include <string_view>

using namespace snicholls;

namespace {

constexpr asset usdc = checked_asset("USDC", 6);
constexpr asset eth = checked_asset("ETH", 18);
constexpr asset aud = checked_asset("AUD", 2);
money of(const asset& a, std::string_view text) { return *parse_money(a, text); }
constexpr std::string_view address = "0x52908400098527886E0F7030069857D2E4169EE7";

memory_onramp_provider venue()
{
    memory_onramp_provider v{{{usdc, {8453}, true, "USD"}, {eth, {1}, false, {}}}, manual_utc_clock{utc_s::from_unix_seconds(1'800'000'000)}};
    v.set_price(*rate::parse(usdc, aud, "1.53"));
    v.set_price(*rate::parse(eth, aud, "5123.45"));
    v.deposit(of(aud, "10000"));
    return v;
}

TEST(MemoryOnrampProvider, ASpendBuysTowardZeroAndAReceiveCostsTheCeiling)
{
    auto v = venue();
    const auto spend = v.quote({usdc, aud, of(aud, "100")});
    ASSERT_TRUE(spend.is_ok());
    EXPECT_EQ(spend->receive, of(usdc, "65.359477"));   // 100 / 1.53 = 65.3594771..., truncated
    const auto receive = v.quote({usdc, aud, of(usdc, "65.359477")});
    EXPECT_EQ(receive->spend, of(aud, "100"));          // 65.359477 * 1.53 = 99.99999981, rounded up
    EXPECT_EQ(receive->expires_at - receive->observed_at, memory_onramp_provider::quote_lifetime);
}

TEST(MemoryOnrampProvider, EighteenDecimalAssetsStayExact)
{
    auto v = venue();
    const auto q = v.quote({eth, aud, of(aud, "5123.45")});
    ASSERT_TRUE(q.is_ok());
    EXPECT_EQ(q->receive, of(eth, "1"));
    const auto tiny = v.quote({eth, aud, of(eth, "0.000000000000000001")});
    ASSERT_TRUE(tiny.is_ok());
    EXPECT_EQ(tiny->spend, of(aud, "0.01"));   // one wei still costs a cent: the ceiling
}

TEST(MemoryOnrampProvider, FeesComeOutOfTheSpend)
{
    auto v = venue();
    v.set_fee_bps(10);
    const auto q = v.quote({usdc, aud, of(aud, "153")});
    EXPECT_EQ(q->fee, of(aud, "0.16"));            // 0.153 rounded up
    EXPECT_EQ(q->receive, of(usdc, "99.895424"));  // 152.84 / 1.53
}

TEST(MemoryOnrampProvider, AWithdrawalTakesItsNetworkFeeAndConfirms)
{
    auto v = venue();
    v.set_withdrawal_fee(of(usdc, "0.5"));
    ASSERT_TRUE(v.buy({"b1", usdc, aud, of(aud, "153"), std::nullopt}).is_ok());   // 100 USDC
    const auto w = v.withdraw({"w1", of(usdc, "99.5"), 8453, address, ""});
    ASSERT_TRUE(w.is_ok());
    EXPECT_EQ(w->status, withdrawal_status::broadcast);
    EXPECT_EQ(w->fee, of(usdc, "0.5"));
    EXPECT_EQ(w->transaction_hash.size(), 66u);
    EXPECT_EQ(v.withdraw({"w2", of(usdc, "0.000001"), 8453, address, ""}).unwrap_err().code, capability_errc::insufficient_funds);
    EXPECT_EQ(v.confirm(w->withdrawal_id.view())->status, withdrawal_status::confirmed);
    EXPECT_EQ(v.confirm("no-such").unwrap_err().code, capability_errc::not_found);
}

TEST(MemoryOnrampProvider, AMaxPriceAtThePriceFills)
{
    auto v = venue();
    EXPECT_TRUE(v.buy({"b1", usdc, aud, of(aud, "10"), *rate::make(usdc, aud, 153, 100)}).is_ok());
    EXPECT_TRUE(v.buy({"b2", usdc, aud, of(aud, "10"), *rate::make(usdc, aud, 306, 200)}).is_ok()) << "2/4 and 1/2 compare equal";
    EXPECT_EQ(v.buy({"b3", usdc, aud, of(aud, "10"), *rate::make(usdc, aud, 152, 100)}).unwrap_err().code, capability_errc::rejected);
}

TEST(MemoryOnrampProvider, FactsAreInTheVenuesAssetsAndAWithdrawalIsOneToOne)
{
    auto v = venue();
    ASSERT_TRUE(v.buy({"b1", usdc, aud, of(aud, "153"), std::nullopt}).is_ok());
    ASSERT_TRUE(v.withdraw({"w1", of(usdc, "10"), 8453, address, ""}).is_ok());
    const auto fills = v.statement({onramp_statement_source::fills, {}, 100});
    ASSERT_EQ(fills->facts.size(), 1u);
    EXPECT_EQ(fills->facts[0].credit.asset_of(), memory_onramp_provider::venue_asset(usdc));
    EXPECT_EQ(fills->facts[0].credit.asset_of().code(), "USDC:MEM");
    const auto out = v.statement({onramp_statement_source::withdrawals, {}, 100});
    ASSERT_EQ(out->facts.size(), 1u);
    EXPECT_TRUE(same_value(out->facts[0].debit, out->facts[0].credit));
    EXPECT_TRUE(validate(out->facts[0]).is_ok());
    EXPECT_EQ(v.statement({onramp_statement_source::fills, "x", 1}).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(v.statement({onramp_statement_source::fills, {}, 0}).unwrap_err().code, capability_errc::invalid_argument);
}

TEST(MemoryOnrampProvider, FailNextFailsOnceAndKeepsOnlyTheCode)
{
    auto v = venue();
    v.fail_next(capability_errc::rate_limited, "50011 Too Many Requests");
    const auto q = v.quote({usdc, aud, of(aud, "100")});
    ASSERT_TRUE(q.is_err());
    EXPECT_EQ(q.unwrap_err().provider_code, std::string_view{"50011"});
    EXPECT_TRUE(q.unwrap_err().retryable());
    EXPECT_TRUE(v.quote({usdc, aud, of(aud, "100")}).is_ok());
}

} // namespace
