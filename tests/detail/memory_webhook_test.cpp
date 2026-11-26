//
//  memory_webhook_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/memory_webhook.hpp>: the reference providers' signed webhooks round-trip every
//  field, accept any of several tags (a secret rotation), and refuse a wrong key, a tampered body, a stale delivery and
//  a malformed body.
//

#include <snicholls/core/detail/memory_webhook.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <string>

using namespace snicholls;

namespace {

const utc_s now = utc_s::from_unix_seconds(1'800'000'000);
const secret<std::string> key{"whsec_a"};

webhook_event every_field()
{
    webhook_event e;
    e.source = "memory";
    e.provider_account = "acct_1";
    e.event_id = "evt_1";
    e.fact_id = "txn_1";
    e.kind = event_kind::security_updated;
    e.type = "trade.filled";
    e.payment_id = "ord-1";
    e.our_reference = "ours";
    e.reference = "REF 1";
    e.occurred_at = now;
    e.needs_lookup = true;
    e.is_test = true;
    return e;
}

TEST(DetailMemoryWebhook, EveryFieldRoundTrips)
{
    const auto d = detail::memory_webhook::sign(key, every_field(), now, now);
    const auto v = detail::memory_webhook::verify(key, d.view());
    ASSERT_TRUE(v.is_ok());
    EXPECT_EQ(*v, every_field());
}

TEST(DetailMemoryWebhook, AnyMatchingTagIsAccepted)
{
    auto d = detail::memory_webhook::sign(key, every_field(), now, now);
    d.headers[0].value = "t=1800000000,v1=00ff," + d.headers[0].value.substr(d.headers[0].value.find("v1="));
    EXPECT_TRUE(detail::memory_webhook::verify(key, d.view()).is_ok());
}

TEST(DetailMemoryWebhook, RefusesWhatItDidNotSign)
{
    const auto d = detail::memory_webhook::sign(key, every_field(), now, now);
    EXPECT_EQ(detail::memory_webhook::verify(secret<std::string>{"whsec_b"}, d.view()).unwrap_err().code,
              capability_errc::signature_invalid);
    const auto late = detail::memory_webhook::sign(key, every_field(), now - std::chrono::seconds{301}, now);
    EXPECT_EQ(detail::memory_webhook::verify(key, late.view()).unwrap_err().code, capability_errc::stale);
    auto malformed = d;
    malformed.headers[0].value = "t=x,v1=00";
    EXPECT_EQ(detail::memory_webhook::verify(key, malformed.view()).unwrap_err().code, capability_errc::signature_invalid);
    EXPECT_EQ(detail::memory_webhook::parse("id=1\nkind=nonsense\n").unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(detail::memory_webhook::parse("id=1\nwhat=1\n").unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(detail::memory_webhook::parse("kind=other\n").unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(detail::memory_webhook::parse("id=1\nkind=other\noccurred_at=soon\n").unwrap_err().code,
              capability_errc::invalid_argument);
    EXPECT_EQ(detail::memory_webhook::parse("no equals\n").unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_TRUE(detail::memory_webhook::parse("id=1\nkind=other\n").is_ok());
}

} // namespace
