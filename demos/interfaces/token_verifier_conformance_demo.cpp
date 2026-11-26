//
//  token_verifier_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/token_verifier_conformance.hpp>: the kit run over the reference verifier. Exits
//  non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/token_verifier_conformance.hpp>
#include <snicholls/core/interfaces/token_verifier_reference.hpp>

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

constexpr uuid party = uuid::from_u64(0xfeed, 0xbeef);

struct probe {
    memory_token_verifier impl = memory_token_verifier::deterministic(utc_s::from_unix_seconds(1'800'000'000));
    auto& verifier() { return impl; }
    std::string valid_token()
    {
        token_claims c;
        c.subject = party;
        return impl.issue(c);
    }
    std::string expired_token()
    {
        token_claims c;
        c.subject = party;
        c.expires_at = utc_s::from_unix_seconds(1'700'000'000);
        return impl.issue(c);
    }
    static uuid subject() { return party; }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/token_verifier_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_token_verifier_conformance<probe>();
    //! [kit]
    for (const auto& f : report.failures)
        std::printf("  %.*s: %.*s\n", static_cast<int>(f.check.size()), f.check.data(), static_cast<int>(f.expectation.size()),
                    f.expectation.data());
    check(report.ok() && report.checks == 6, "the reference passes all six checks");
    return failures == 0 ? 0 : 1;
}
