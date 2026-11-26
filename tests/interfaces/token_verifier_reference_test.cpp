//
//  token_verifier_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/token_verifier_reference.hpp>: tokens it issues verify; forged, altered and
//  foreign ones do not; expiry by the clock and by hand; an armed failure is returned once; no error carries the token.
//

#include <snicholls/core/interfaces/token_verifier_reference.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <string>

using namespace snicholls;

namespace {

const uuid alice = uuid::from_u64(0xa11ce, 1);

TEST(MemoryTokenVerifier, ATokenItIssuedVerifiesWithItsClaims)
{
    auto v = memory_token_verifier::deterministic();
    token_claims c;
    c.subject = alice;
    c.roles.resize(1);   // then assigned: GCC 13 under ASan misreports copying c.roles = {"admin"} as out of bounds
    c.roles[0] = "admin";
    const std::string t = v.issue(c);
    EXPECT_TRUE(t.starts_with(memory_token_verifier::prefix));
    EXPECT_EQ(t.size(), memory_token_verifier::prefix.size() + 1 + 1 + 64) << t;
    const auto r = v.verify(t);
    ASSERT_TRUE(r.is_ok());
    EXPECT_EQ(*r, c);
    EXPECT_NE(v.issue(c), t) << "every token is a new serial";
}

TEST(MemoryTokenVerifier, AForgedOrForeignTokenIsUnauthorized)
{
    auto v = memory_token_verifier::deterministic();
    const std::string t = v.issue({});
    memory_token_verifier other{secret<std::string>{"another test-only key, 32 bytes or more"}, manual_utc_clock{}};
    EXPECT_EQ(other.verify(t).unwrap_err().code, capability_errc::unauthorized) << "another key's verifier refuses it";
    std::string forged = t;
    forged[memory_token_verifier::prefix.size()] = '7';   // another serial under the same tag
    EXPECT_EQ(v.verify(forged).unwrap_err().code, capability_errc::unauthorized);
    EXPECT_EQ(v.verify(t.substr(0, t.size() - 1)).unwrap_err().code, capability_errc::unauthorized);
    EXPECT_EQ(v.verify("mt1.0").unwrap_err().code, capability_errc::unauthorized);
    EXPECT_EQ(v.verify("mt1.").unwrap_err().code, capability_errc::unauthorized);
    EXPECT_EQ(v.verify("").unwrap_err().code, capability_errc::unauthorized);
    EXPECT_TRUE(v.verify(t).is_ok()) << "the original still verifies";
}

TEST(MemoryTokenVerifier, ATagOverASerialItNeverIssuedIsUnauthorized)
{
    const secret<std::string> key{"k-test-only"};
    memory_token_verifier v{key, manual_utc_clock{}};
    const std::string body = "mt1.5";
    const std::string tagged = body + "." + std::string(hmac<sha256>::mac(key.reveal(), body).hex().view());
    EXPECT_EQ(v.verify(tagged).unwrap_err().code, capability_errc::unauthorized) << "a valid tag, no such serial";
}

TEST(MemoryTokenVerifier, ExpiresByTheClockOrByHand)
{
    memory_token_verifier v{secret<std::string>{"k-test-only"}, manual_utc_clock{utc_s::from_unix_seconds(1000)}};
    token_claims c;
    c.subject = alice;
    c.expires_at = utc_s::from_unix_seconds(2000);
    const std::string t1 = v.issue(c);
    const std::string t2 = v.issue(c);
    EXPECT_TRUE(v.verify(t1).is_ok());
    v.expire(t2);
    v.expire("not a token");
    EXPECT_EQ(v.verify(t2).unwrap_err().code, capability_errc::stale);
    v.clock().advance(std::chrono::seconds{1000});   // the clock is the verifier's own
    EXPECT_EQ(v.verify(t1).unwrap_err().code, capability_errc::stale) << "expiry at the instant is stale";
    EXPECT_EQ(v.verify(t1).unwrap_err().message.view(), "the token has expired");
}

TEST(MemoryTokenVerifier, AnArmedFailureIsReturnedOnceWithOnlyTheProviderCode)
{
    auto v = memory_token_verifier::deterministic();
    const std::string t = v.issue({});
    v.fail_next(capability_errc::unavailable, "session_store_down and the body with a token");
    const auto r = v.verify(t);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().code, capability_errc::unavailable);
    EXPECT_EQ(r.unwrap_err().provider_code.view(), "session_store_down");
    EXPECT_TRUE(v.verify(t).is_ok());
    v.fail_next(capability_errc::timeout);
    EXPECT_TRUE(v.verify(t).unwrap_err().provider_code.empty());
}

TEST(MemoryTokenVerifier, NoErrorCarriesTheToken)
{
    auto v = memory_token_verifier::deterministic();
    const std::string t = v.issue({}) + "x";
    const auto e = v.verify(t).unwrap_err();
    EXPECT_EQ(e.message.view().find("mt1"), std::string_view::npos);
    EXPECT_TRUE(e.provider_code.empty());
}

} // namespace
