//
//  onramp_provider_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/onramp_provider.hpp>: a venue chosen at run time behind the virtual interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/onramp_provider.hpp>
#include <snicholls/core/interfaces/onramp_provider_reference.hpp>
#include <snicholls/core/money/asset.hpp>

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

constexpr asset usdc = checked_asset("USDC", 6);
constexpr asset aud = checked_asset("AUD", 2);

memory_onramp_provider make_venue()
{
    memory_onramp_provider v{{onramp_asset{usdc, {8453}, true, "USD"}}, manual_utc_clock{utc_s::from_unix_seconds(1'800'000'000)}};
    v.set_price(*rate::parse(usdc, aud, "1.53"));   // 1.53 AUD per USDC, exactly
    v.deposit(*parse_money(aud, "1000"));
    return v;
}
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/onramp_provider.hpp demo\n");
    //! [virtual]
    onramp_provider_adapter owned{make_venue()};
    onramp_provider_interface& chosen = owned;
    onramp_provider_ref handle{chosen};
    const auto q = handle.quote({usdc, aud, *parse_money(aud, "15.30")});
    //! [virtual]
    check(q.is_ok() && q->receive == *parse_money(usdc, "10"), "a quote through the interface");
    check(handle.assets().size() == 1 && handle.balances()->size() == 1, "assets and balances");
    check(handle.order_status("nope").unwrap_err().code == capability_errc::not_found, "an unknown order");
    return failures == 0 ? 0 : 1;
}
