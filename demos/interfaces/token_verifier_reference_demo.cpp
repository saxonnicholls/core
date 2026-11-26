//
//  token_verifier_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/token_verifier_reference.hpp>: issue, verify, expire, and a failure armed for a
//  test. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/token_verifier_reference.hpp>

#include <chrono>
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
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/token_verifier_reference.hpp demo\n");
    //! [reference]
    utc_s now = utc_s::from_unix_seconds(1'800'000'000);
    memory_token_verifier v{secret<std::string>{"a test-only HS256 key of 32 bytes or more"}, [&] { return now; }};
    token_claims c;
    c.subject = uuid::from_u64(0xa11ce, 1);
    c.expires_at = now + std::chrono::minutes{15};
    const std::string token = v.issue(c);   // mt1.<serial>.<HMAC-SHA256 in hex>
    const auto fresh = v.verify(token);
    now += std::chrono::minutes{15};
    const auto later = v.verify(token);   // stale: the client refreshes
    //! [reference]
    std::printf("  token: %s\n", token.c_str());
    check(fresh.is_ok() && fresh->subject == c.subject, "a token it issued verifies");
    check(later.is_err() && later.unwrap_err().code == capability_errc::stale, "at its expiry it is stale");
    const std::string second = v.issue({});
    v.expire(second);
    check(v.verify(second).unwrap_err().code == capability_errc::stale, "expire makes it stale by hand");
    check(v.verify(token + "0").unwrap_err().code == capability_errc::unauthorized, "an altered token is unauthorized");
    v.fail_next(capability_errc::unavailable, "store_down {\"body\":\"never kept\"}");
    const auto down = v.verify(second);
    check(down.is_err() && down.unwrap_err().provider_code.view() == "store_down", "an armed failure keeps only the provider code");
    check(memory_token_verifier::deterministic().verify("").is_err(), "deterministic() is the fixture's verifier");
    return failures == 0 ? 0 : 1;
}
