//
//  message_sender_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/message_sender_reference.hpp>: the reference sender: an outbox, idempotency and an injected failure. Exits non-zero if a result is not the one documented.
//

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
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/message_sender_reference.hpp demo\n");
    //! [reference]
    memory_message_sender sender;
    const message m{channel::email, "ops@example.test", "Alert", "disk full", "", "alert-7"};
    const auto first = sender.send(m);
    const auto again = sender.send(m);   // the same key: the same receipt, sent once
    sender.fail_next(capability_errc::unavailable, "503 upstream says no");
    const auto failed = sender.send({channel::sms, "+61412345678", "", "hi", "", ""});
    //! [reference]
    check(first.is_ok() && again.is_ok() && *first == *again && sender.outbox().size() == 1, "sent once");
    check(failed.is_err() && failed.unwrap_err().provider_code == std::string_view{"503"}, "the injected failure keeps only the code");
    check(sender.outbox()[0].msg.view() == m && sender.supports(channel::sms), "the outbox keeps a copy");
    check(memory_message_sender::kept_message::of(m).subject == "Alert", "kept_message::of copies");
    return failures == 0 ? 0 : 1;
}
