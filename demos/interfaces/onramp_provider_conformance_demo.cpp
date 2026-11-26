//
//  onramp_provider_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/onramp_provider_conformance.hpp>: the kit run over the reference venue. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/onramp_provider_conformance.hpp>
#include <snicholls/core/interfaces/onramp_provider_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <cstdint>
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

constexpr asset usdc = checked_asset("USDC", 6);
constexpr asset aud = checked_asset("AUD", 2);

memory_onramp_provider make_venue()
{
    memory_onramp_provider v{{onramp_asset{usdc, {8453}, true, "USD"}}, manual_utc_clock{utc_s::from_unix_seconds(1'800'000'000)}};
    v.set_price(*rate::parse(usdc, aud, "1.53"));   // 1.53 AUD per USDC, exactly
    v.deposit(*parse_money(aud, "1000"));
    return v;
}

struct probe {
    memory_onramp_provider impl = make_venue();
    auto& venue() { return impl; }
    asset unit() { return usdc; }
    asset quote_currency() { return aud; }
    money ticket() { return *parse_money(aud, "153"); }
    money too_much() { return *parse_money(aud, "100000"); }
    std::uint64_t chain() { return 8453; }
    std::string address() { return "0x52908400098527886E0F7030069857D2E4169EE7"; }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/onramp_provider_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_onramp_provider_conformance<probe>();
    //! [kit]
    check(report.ok() && report.checks == 9, "the reference passes all nine checks");
    return failures == 0 ? 0 : 1;
}
