//
//  message_sender_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/message_sender.hpp>: the models and the non-models, the validators, at compile time.
//

#include <snicholls/core/concepts/message_sender.hpp>
#include <snicholls/core/interfaces/message_sender_reference.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>

using namespace snicholls;

namespace {

static_assert(message_sender<memory_message_sender>);
static_assert(message_sender<message_sender_interface>);
static_assert(message_sender<message_sender_adapter<memory_message_sender>>);
static_assert(message_sender<message_sender_adapter<memory_message_sender&>>);
static_assert(message_sender<message_sender_ref>);

struct no_supports {
    capability_result<send_receipt> send(const message&);
};
struct send_returns_bool {
    bool send(const message&);
    bool supports(channel) const;
};
struct supports_not_const {
    capability_result<send_receipt> send(const message&);
    bool supports(channel);
};
static_assert(!message_sender<no_supports>);
static_assert(!message_sender<send_returns_bool>);
static_assert(!message_sender<supports_not_const>);
static_assert(!message_sender<int>);

static_assert(is_email_address("someone@example.test"));
static_assert(!is_email_address("someone@example") && !is_email_address("@example.test") && !is_email_address("a@b@c.test"));
static_assert(!is_email_address("a b@example.test") && !is_email_address("a..b@example.test") && !is_email_address("a@-x.test"));
static_assert(!is_email_address("a@example.test\r\nBcc: x@y.test"));
static_assert(has_line_break("a\nb") && has_line_break("a\rb") && !has_line_break("ab"));
static_assert(is_canonical_e164("+61400000000"));
static_assert(!is_canonical_e164("+61 400 000 000") && !is_canonical_e164("0400000000") && !is_canonical_e164("+999123456"));

constexpr message sms{channel::sms, "+61400000000", "", "Your code is 482910.", "", ""};
static_assert(validate(sms).is_ok());
static_assert(validate(message{channel::email, "someone@example.test", "Hi", "", "<p>Hi</p>", ""}).is_ok());

TEST(MessageSenderConcept, ValidateNamesTheFieldNeverItsValue)
{
    message m = sms;
    m.subject = "subject";
    EXPECT_EQ(validate(m).unwrap_err().message, std::string_view{"an SMS has no subject and no HTML"});
    m = sms;
    m.text = "";
    EXPECT_EQ(validate(m).unwrap_err().code, capability_errc::invalid_argument);
    const std::string long_text(max_sms_length + 1, 'x');
    m.text = long_text;
    EXPECT_TRUE(validate(m).is_err());
    message email{channel::email, "someone@example.test", "", "body", "", ""};
    EXPECT_EQ(validate(email).unwrap_err().message, std::string_view{"subject is empty, too long or has a line break"});
    email.subject = "s";
    email.text = "";
    EXPECT_EQ(validate(email).unwrap_err().message, std::string_view{"email has no body"});
    const std::string key(65, 'k');
    m = sms;
    m.idempotency_key = key;
    EXPECT_EQ(validate(m).unwrap_err().message, std::string_view{"idempotency key longer than 64 characters"});
    m.via = static_cast<channel>(9);
    m.idempotency_key = {};
    EXPECT_EQ(validate(m).unwrap_err().message, std::string_view{"unknown channel"});
}

} // namespace
