//
//  payment_provider_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/payment_provider.hpp>: a provider chosen at run time behind the virtual interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/payment_provider.hpp>
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
    std::printf("snicholls-core interfaces/payment_provider.hpp demo\n");
    //! [virtual]
    payment_provider_adapter owned{make_provider()};
    payment_provider_interface& chosen = owned;
    payment_provider_ref handle{chosen};
    const auto c = handle.collect({"k-1", "order-1", *parse_money(aud, "20"), payment_method::bank_transfer, "CUST-9", "", "", {}});
    //! [virtual]
    check(c.is_ok() && c->next.bank && c->next.bank->reference.view() == "CUST-9", "bank details with the payer's reference");
    check(handle.capabilities().refunds && handle.status("nope").is_err(), "capabilities and status through the handle");
    static_cast<void>(owned.get().receive_credit(*parse_money(aud, "20"), "CUST-9", "A Payer"));
    check(handle.status(c->pay.payment_id.view())->status == payment_status::succeeded, "the credit completes it");
    return failures == 0 ? 0 : 1;
}
