//
//  security_provider_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/security_provider.hpp>: a custodian chosen at run time behind the virtual
//  interface (an owning adapter), the same provider borrowed through a handle, and the webhook route and worker written
//  once against the interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/security_provider.hpp>
#include <snicholls/core/interfaces/security_provider_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

constexpr asset aud = checked_asset("AUD", 2), vas = checked_asset("VAS:ASX", 0);
const utc_s now = utc_s::from_unix_seconds(1'800'000'000);

// The inbox and the worker, written once against the interface: verify, then resolve to facts.
std::vector<security_fact> on_delivery(security_provider_interface& custodian, const webhook_delivery& d)
{
    const auto event = custodian.verify_webhook(d);
    if (!event)
        return {};
    const auto facts = custodian.resolve(*event);
    return facts ? *facts : std::vector<security_fact>{};
}
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/security_provider.hpp demo\n");
    //! [virtual]
    memory_security_provider reference{secret<std::string>{"whsec_demo_only"}, [] { return now; }};
    static_cast<void>(reference.execute("ord-7", money::from_units(vas, 40), -*parse_money(aud, "3916.00")));
    webhook_event filled;
    filled.source = "memory";
    filled.event_id = "evt_7";
    filled.kind = event_kind::security_updated;
    filled.payment_id = "ord-7";
    const owned_delivery delivery = reference.signed_webhook(filled);

    std::unique_ptr<security_provider_interface> custodian =
        std::make_unique<security_provider_adapter<memory_security_provider>>(std::move(reference));   // chosen at run time
    const std::vector<security_fact> facts = on_delivery(*custodian, delivery.view());
    //! [virtual]
    check(facts.size() == 1 && facts.front().kind == security_fact_kind::buy, "the route and worker book one buy");
    check(facts.size() == 1 && facts.front().event_id == filled.event_id, "the fact names the delivery");

    security_provider_ref handle{*custodian};   // copyable, non-owning; models the concept
    static_assert(security_provider<security_provider_ref>);
    const auto held = handle.holdings(now.days());
    check(held.is_ok() && held->size() == 1 && held->front().traded == money::from_units(vas, 40), "40 VAS traded");
    check(held.is_ok() && held->front().settled.is_zero(), "not yet settled");
    check(handle.capabilities().holdings, "the reference reports holdings");
    check(handle.statement({})->facts.size() == 1, "the statement has the one fact");
    owned_delivery forged = delivery;
    forged.body += "x";
    check(on_delivery(*custodian, forged.view()).empty(), "a forged delivery books nothing");
    return failures == 0 ? 0 : 1;
}
