//
//  onramp_provider_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/onramp_provider_reference.hpp>: the reference venue: a buy, a withdrawal, its confirmation and the facts. Exits non-zero if a result is not the one documented.
//

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
    memory_onramp_provider v{{onramp_asset{usdc, {8453}, true, "USD"}}, [] { return utc_s::from_unix_seconds(1'800'000'000); }};
    v.set_price(*rate::parse(usdc, aud, "1.53"));   // 1.53 AUD per USDC, exactly
    v.deposit(*parse_money(aud, "1000"));
    return v;
}
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/onramp_provider_reference.hpp demo\n");
    //! [reference]
    auto venue = make_venue();
    venue.set_withdrawal_fee(*parse_money(usdc, "0.5"));
    static_cast<void>(venue.buy({"c-1", usdc, aud, *parse_money(aud, "153"), std::nullopt}));
    const auto w = venue.withdraw({"w-1", *parse_money(usdc, "99.5"), 8453, "0x52908400098527886E0F7030069857D2E4169EE7", ""});
    const auto landed = venue.confirm(w->withdrawal_id.view());
    //! [reference]
    check(w.is_ok() && w->status == withdrawal_status::broadcast && w->transaction_hash.size() == 66, "broadcast with a hash");
    check(landed.is_ok() && landed->status == withdrawal_status::confirmed, "confirmed");
    const auto facts = venue.statement({onramp_statement_source::withdrawals, {}, 100});
    check(facts->facts.size() == 1 && facts->facts[0].debit.asset_of() == memory_onramp_provider::venue_asset(usdc),
          "the withdrawal fact debits the venue's own asset");
    return failures == 0 ? 0 : 1;
}
