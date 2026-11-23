//
//  message_sender_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/message_sender_reference.hpp>: the outbox, refusals, the injected failure and the redaction rule.
//

#include <snicholls/core/interfaces/message_sender_reference.hpp>

#include <gtest/gtest.h>

#include <string_view>

using namespace snicholls;

namespace {

message email(std::string_view key = {}) { return {channel::email, "someone@example.test", "Hello", "Hello there.", "", key}; }
message sms(std::string_view key = {}) { return {channel::sms, "+61400000000", "", "Hello there.", "", key}; }

TEST(MemoryMessageSender, TheOutboxHoldsWhatWasAcceptedInOrder)
{
    memory_message_sender s;
    ASSERT_TRUE(s.send(email()).is_ok());
    ASSERT_TRUE(s.send(sms()).is_ok());
    ASSERT_EQ(s.outbox().size(), 2u);
    EXPECT_EQ(s.outbox()[0].receipt.message_id, std::string_view{"memory-email-1"});
    EXPECT_EQ(s.outbox()[1].receipt.message_id, std::string_view{"memory-sms-2"});
    EXPECT_EQ(s.outbox()[1].msg.to, "+61400000000");
    EXPECT_EQ(s.outbox()[1].msg.view(), sms());
}

TEST(MemoryMessageSender, ARejectedMessageIsNotInTheOutbox)
{
    memory_message_sender s;
    message bad = email();
    bad.to = "x";
    const auto r = s.send(bad);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().message, std::string_view{"recipient is not an email address"});
    EXPECT_TRUE(s.outbox().empty());
}

TEST(MemoryMessageSender, AChannelSwitchedOffIsUnsupported)
{
    memory_message_sender s{true, false};
    EXPECT_TRUE(s.supports(channel::email));
    EXPECT_FALSE(s.supports(channel::sms));
    EXPECT_EQ(s.send(sms()).unwrap_err().code, capability_errc::unsupported);
}

TEST(MemoryMessageSender, FailNextInjectsOneFailureThenRecovers)
{
    memory_message_sender s;
    s.fail_next(capability_errc::unavailable, "503 Service Unavailable: someone@example.test");
    const auto first = s.send(sms("retry-1"));
    ASSERT_TRUE(first.is_err());
    EXPECT_TRUE(first.unwrap_err().retryable());
    EXPECT_EQ(first.unwrap_err().provider_code, std::string_view{"503"});   // the code only, never the rest
    EXPECT_TRUE(s.outbox().empty());
    const auto second = s.send(sms("retry-1"));
    ASSERT_TRUE(second.is_ok());
    EXPECT_EQ(s.outbox().size(), 1u);
}

TEST(MemoryMessageSender, AResponseThatIsNotACodeLeavesNoProviderCode)
{
    memory_message_sender s;
    s.fail_next(capability_errc::rejected, "{\"error\":\"bad\"}");
    const auto r = s.send(email());
    ASSERT_TRUE(r.is_err());
    EXPECT_TRUE(r.unwrap_err().provider_code.empty());
}

} // namespace
