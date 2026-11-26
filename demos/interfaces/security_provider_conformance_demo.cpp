//
//  security_provider_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/security_provider_conformance.hpp>: the kit run over the reference provider,
//  and over a provider that settles every trade on its trade date, which the kit catches. Exits non-zero if a result
//  is not the one documented.
//

#include <snicholls/core/interfaces/security_provider_conformance.hpp>
#include <snicholls/core/interfaces/security_provider_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <string_view>
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

constexpr asset aud = checked_asset("AUD", 2), bhp = checked_asset("BHP:ASX", 0);
const utc_s now = utc_s::from_unix_seconds(1'800'000'000);

template <bool SettlesAtOnce>
struct probe {
    memory_security_provider impl{secret<std::string>{"whsec_demo_only"}, [] { return now; }};
    auto& provider() { return impl; }
    money quantity() { return money::from_units(bhp, 100); }
    money consideration() { return -*parse_money(aud, "4512.50"); }
    void trade(std::string_view order, const money& q, const money& c)
    {
        static_cast<void>(impl.execute(order, q, c));
        if (SettlesAtOnce)
            static_cast<void>(impl.settle(order));   // the defect: settled on the trade date
    }
    void settle(std::string_view order) { static_cast<void>(impl.settle(order)); }
    owned_delivery deliver(const webhook_event& e) { return impl.signed_webhook(e); }
    owned_delivery deliver_late(const webhook_event& e) { return impl.signed_webhook(e, now - std::chrono::minutes{6}); }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/security_provider_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_security_provider_conformance<probe<false>>();
    for (const auto& f : report.failures)
        std::printf("  failed: %.*s: %.*s\n", static_cast<int>(f.check.size()), f.check.data(),
                    static_cast<int>(f.expectation.size()), f.expectation.data());
    //! [kit]
    std::printf("  %zu checks\n", report.checks);
    check(report.ok() && report.checks == 8, "the reference passes all eight checks");

    const auto broken = run_security_provider_conformance<probe<true>>();
    check(broken.failed("settlement moves the settled holding"), "a provider that settles at once is caught");
    for (const auto& f : broken.failures)
        std::printf("  caught: %.*s: %.*s\n", static_cast<int>(f.check.size()), f.check.data(),
                    static_cast<int>(f.expectation.size()), f.expectation.data());
    return failures == 0 ? 0 : 1;
}
