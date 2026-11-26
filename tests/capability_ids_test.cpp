//
//  capability_ids_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/capability_ids.hpp>: the bounds and make_id, which refuses rather than truncates.
//

#include <snicholls/core/capability_ids.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(provider_id::capacity() == 96 && fact_id::capacity() == 128 && event_id::capacity() == 64);
static_assert(provider_account::capacity() == 64 && client_id::capacity() == 64 && provider_name::capacity() == 32);
static_assert(short_text::capacity() == 255);
static_assert(role_id::capacity() == 64 && permission_id::capacity() == 64 && tenant_id::capacity() == 64);
static_assert(wallet_address::capacity() == 42 && std::is_trivially_copyable_v<wallet_address>);
static_assert(std::is_trivially_copyable_v<provider_id> && std::is_trivially_copyable_v<short_text>);
static_assert(std::is_same_v<utc_s, utc<std::chrono::seconds>>);
static_assert(make_id<client_id>("order-1").is_ok());
static_assert(noexcept(make_id<event_id>(std::string_view{})));

TEST(CapabilityIds, ATransactionHashAndAStripeCheckoutIdFit)
{
    const std::string tx = "0x" + std::string(64, 'a');
    EXPECT_TRUE(make_id<provider_id>(tx).is_ok());
    EXPECT_TRUE(make_id<provider_id>("cs_test_a1YS1URlnyQCN5fUUduORoQ7Pw41PJqDWkIVQCpJPqkfIhd6tVY8XB1OLY").is_ok());
    const std::string chain_fact = "18446744073709551615:" + tx + ":4294967295:reversed";   // 107 characters
    EXPECT_TRUE(make_id<fact_id>(chain_fact).is_ok());
}

TEST(CapabilityIds, TooLongIsRefusedNeverTruncated)
{
    const auto r = make_id<client_id>(std::string(65, 'k'));
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(r.unwrap_err().message, std::string_view{"identifier too long"});
    EXPECT_EQ(make_id<client_id>(std::string(64, 'k'))->size(), 64u);
    EXPECT_TRUE(make_id<short_text>("").is_ok());
}

// ---- fact_dates: the trade, the settlement and when we learnt of it ---------------------------------------------------

constexpr fact_dates card_payment()
{
    fact_dates d;
    d.trade_at = utc_s::from_unix_seconds(1'800'000'000);       // the charge
    d.known_at = utc_s::from_unix_seconds(1'800'000'030);       // the webhook arrived 30 s later
    d.available_at = utc_s::from_unix_seconds(1'800'172'800);   // Stripe's available_on, two days on
    d.settles_on = d.available_at->days();
    return d;
}
static_assert(validate(card_payment()).is_ok(), "a pending settlement in the future is valid");
static_assert(std::is_trivially_copyable_v<fact_dates>);

TEST(FactDates, KnownBeforeItHappenedIsInvalid)
{
    fact_dates d = card_payment();
    d.known_at = utc_s::from_unix_seconds(1'799'999'999);
    ASSERT_TRUE(validate(d).is_err());
    EXPECT_EQ(validate(d).unwrap_err().message, std::string_view{"a fact is known before it happened"});
}

TEST(FactDates, SettledBeforeItHappenedIsInvalid)
{
    fact_dates d = card_payment();
    d.settled_at = utc_s::from_unix_seconds(1'799'000'000);
    EXPECT_EQ(validate(d).unwrap_err().message, std::string_view{"a fact settled before it happened"});
    d.settled_at = d.available_at;
    EXPECT_TRUE(validate(d).is_ok());
}

TEST(FactDates, AnEstimateNeedsADate)
{
    fact_dates d = card_payment();
    d.settles_on_estimated = true;
    EXPECT_TRUE(validate(d).is_ok());
    d.settles_on.reset();
    EXPECT_EQ(validate(d).unwrap_err().message, std::string_view{"an estimate without a settlement date"});
}

TEST(FactDates, KnownAtIsProvenanceNotContent)
{
    const fact_dates first = card_payment();
    fact_dates redelivered = first;
    redelivered.known_at = utc_s::from_unix_seconds(1'800'000'600);   // the same fact, learnt again later
    EXPECT_NE(first, redelivered);
    EXPECT_TRUE(first.same_content(redelivered));
    redelivered.settles_on = *first.settles_on + std::chrono::days{1};
    EXPECT_FALSE(first.same_content(redelivered));
}

} // namespace
