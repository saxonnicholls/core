//
//  payment_provider_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/payment_provider_reference.hpp>: signed webhooks, verified in constant time and failing closed. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <cstdio>
#include <string>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

constexpr asset aud = checked_asset("AUD", 2);
const utc_s now = utc_s::from_unix_seconds(1'800'000'000);

memory_payment_provider make_provider()
{
    return memory_payment_provider{{aud}, secret<std::string>{"whsec_demo_only"}, manual_utc_clock{now}};
}
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/payment_provider_reference.hpp demo\n");
    //! [reference]
    auto provider = make_provider();
    webhook_event e;
    e.source = "memory";
    e.event_id = "evt_42";
    e.kind = event_kind::credit_received;
    e.needs_lookup = true;
    owned_delivery d = provider.signed_webhook(e);   // Memory-Signature: t=...,v1=<HMAC-SHA256>
    const auto verified = provider.verify_webhook(d.view());
    d.body[0] = 'X';                                    // one byte changed
    const auto tampered = provider.verify_webhook(d.view());
    //! [reference]
    check(verified.is_ok() && verified->event_id == std::string_view{"evt_42"} && verified->needs_lookup, "a signed delivery verifies");
    check(tampered.unwrap_err().code == capability_errc::signature_invalid, "a tampered one fails closed");
    const auto late = provider.signed_webhook(e, now - std::chrono::minutes{10});
    check(provider.verify_webhook(late.view()).unwrap_err().code == capability_errc::stale, "an old one is stale");
    return failures == 0 ? 0 : 1;
}
