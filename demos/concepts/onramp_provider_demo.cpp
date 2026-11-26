//
//  onramp_provider_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/onramp_provider.hpp>: buying a stablecoin through the concept, and the fact checks. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/onramp_provider.hpp>
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

template <onramp_provider O>
capability_result<order> buy_usdc(O& venue, std::string_view client_order_id, const money& spend)
{
    return venue.buy({client_order_id, usdc, aud, spend, std::nullopt});
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/onramp_provider.hpp demo\n");
    //! [concept]
    auto venue = make_venue();
    const auto filled = buy_usdc(venue, "c-1", *parse_money(aud, "153"));   // 100 USDC, exactly
    const auto facts = venue.statement({onramp_statement_source::fills, {}, 100});
    //! [concept]
    check(filled.is_ok() && filled->filled == *parse_money(usdc, "100"), "153 AUD buys 100 USDC");
    check(facts.is_ok() && facts->facts.size() == 1 && validate(facts->facts[0]).is_ok(), "one fact that validates");
    check(is_evm_address("0x52908400098527886E0F7030069857D2E4169EE7") && !is_evm_address("0x12"), "is_evm_address");
    check(same_value(*parse_money(usdc, "1"), *parse_money(checked_asset("USDC:OKX", 18), "1")), "same_value across scales");
    return failures == 0 ? 0 : 1;
}
