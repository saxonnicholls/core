//
//  message_sender_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/message_sender.hpp>: a sender chosen at run time through the virtual interface, the adapter and the handle. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/message_sender.hpp>
#include <snicholls/core/interfaces/message_sender_reference.hpp>

#include <cstdio>
#include <memory>

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
    std::printf("snicholls-core interfaces/message_sender.hpp demo\n");
    //! [virtual]
    std::unique_ptr<message_sender_interface> chosen = std::make_unique<message_sender_adapter<memory_message_sender>>(memory_message_sender{true, false});
    message_sender_ref handle{*chosen};   // copyable, models the concept
    const auto r = handle.send({channel::email, "ops@example.test", "Alert", "disk full", "", ""});
    //! [virtual]
    check(r.is_ok(), "sent through the interface");
    check(chosen->supports(channel::email) && !handle.supports(channel::sms), "supports answers through both");
    auto& adapter = static_cast<message_sender_adapter<memory_message_sender>&>(*chosen);
    check(adapter.get().outbox().size() == 1, "the adapter reaches the model");
    return failures == 0 ? 0 : 1;
}
