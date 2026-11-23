//
//  payment_provider_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/payment_provider_conformance.hpp>: the kit run over the reference provider. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/payment_provider_conformance.hpp>
#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <string_view>

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

struct probe {
    memory_payment_provider impl = make_provider();
    auto& provider() { return impl; }
    money amount() { return *parse_money(aud, "125.50"); }
    asset unsupported_currency() { return checked_asset("XYZ", 2); }
    void payer_pays(const collection& c)
    {
        if (c.next.kind == next_step_kind::bank_transfer)
            static_cast<void>(impl.receive_credit(c.pay.amount, c.next.bank->reference.view(), "A Payer"));
        else
            static_cast<void>(impl.complete(c.pay.payment_id.view()));
    }
    owned_delivery deliver(const webhook_event& e) { return impl.signed_webhook(e); }
    owned_delivery deliver_late(const webhook_event& e) { return impl.signed_webhook(e, now - std::chrono::minutes{6}); }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/payment_provider_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_payment_provider_conformance<probe>();
    //! [kit]
    check(report.ok() && report.checks == 13, "the reference passes all thirteen checks");
    return failures == 0 ? 0 : 1;
}
