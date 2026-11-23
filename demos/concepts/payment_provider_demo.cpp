//
//  payment_provider_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/payment_provider.hpp>: a collection through the concept, a settlement that balances and a fact that validates. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/payment_provider.hpp>
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
    return memory_payment_provider{{aud}, secret<std::string>{"whsec_demo_only"}, [] { return now; }};
}

template <payment_provider P>
capability_result<collection> checkout(P& provider, std::string_view order, const money& amount)
{
    return provider.collect({order, order, amount, payment_method::card, "", "https://shop.test/ok", "https://shop.test/no", {}});
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/payment_provider.hpp demo\n");
    //! [concept]
    auto provider = make_provider();
    const auto c = checkout(provider, "order-1", *parse_money(aud, "49.95"));
    // the payer follows c->next.url; the provider tells us by webhook, and resolve gives the ledger its fact
    static_cast<void>(provider.complete(c->pay.payment_id.view()));
    webhook_event e;
    e.source = "memory";
    e.event_id = "evt_1";
    e.kind = event_kind::payment_updated;
    e.payment_id = c->pay.payment_id;
    const auto facts = provider.resolve(e);
    //! [concept]
    check(c.is_ok() && c->next.kind == next_step_kind::redirect && !is_terminal(c->pay.status), "a redirect, waiting");
    check(facts.is_ok() && facts->size() == 1 && validate(facts->front()).is_ok() && balances(facts->front().settled),
          "one authoritative fact that balances");
    check(e.dedup_key() == std::string_view{"memory:evt:evt_1"}, "the dedup key");
    return failures == 0 ? 0 : 1;
}
