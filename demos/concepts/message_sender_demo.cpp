//
//  message_sender_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/message_sender.hpp>: code written once against the concept, and the validators. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/message_sender.hpp>
#include <snicholls/core/interfaces/message_sender_reference.hpp>

#include <cstdio>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

// Generic code: any sender, chosen at compile time.
template <message_sender S>
capability_result<send_receipt> send_code(S& sender, std::string_view to)
{
    return sender.send({channel::sms, to, "", "Your code is 482910.", "", "otp-1"});
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/message_sender.hpp demo\n");
    //! [concept]
    memory_message_sender sender;
    const auto receipt = send_code(sender, "+61412345678");
    const auto refused = send_code(sender, "0412 345 678");   // not E.164: refused before anything is sent
    //! [concept]
    check(receipt.is_ok() && receipt->message_id.view() == "memory-sms-1", "an E.164 number is sent");
    check(refused.is_err() && refused.unwrap_err().code == capability_errc::invalid_argument, "a national number is invalid");
    check(is_email_address("ops@example.test") && !is_email_address("ops@example"), "is_email_address");
    check(is_canonical_e164("+61412345678") && !is_canonical_e164("+61 412 345 678"), "is_canonical_e164");
    check(has_line_break("a\r\nb"), "has_line_break");
    check(validate({channel::email, "ops@example.test", "Alert", "disk full", "", ""}).is_ok(), "validate");
    return failures == 0 ? 0 : 1;
}
