//
//  token_verifier_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/token_verifier.hpp>: authenticating through the concept, and the claims'
//  lookups. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/token_verifier.hpp>
#include <snicholls/core/interfaces/token_verifier_reference.hpp>

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

//! [concept]
// A route's authenticate step: any verifier, a bearer token in, the party out.
template <token_verifier V>
capability_result<token_claims> authenticate(const V& v, std::string_view authorization)
{
    constexpr std::string_view bearer = "Bearer ";
    if (!authorization.starts_with(bearer))
        return fail(capability_errc::unauthorized, {}, "no bearer token");
    return v.verify(authorization.substr(bearer.size()));
}
//! [concept]
} // namespace

int main()
{
    std::printf("snicholls-core concepts/token_verifier.hpp demo\n");
    auto verifier = memory_token_verifier::deterministic();   // for tests only
    token_claims alice;
    alice.subject = uuid::from_u64(0xa11ce, 1);
    alice.roles = {"admin"};
    alice.scopes = {"payments.refund"};
    const std::string header = "Bearer " + verifier.issue(alice);

    const auto who = authenticate(verifier, header);
    check(who.is_ok() && who->subject == alice.subject, "the bearer token names its party");
    check(who.is_ok() && who->has_role("admin") && !who->has_role("owner"), "has_role");
    check(who.is_ok() && who->has_scope("payments.refund") && !who->has_scope("payments.capture"), "has_scope");
    check(authenticate(verifier, "Basic abc").unwrap_err().code == capability_errc::unauthorized, "no bearer: unauthorized");
    check(authenticate(verifier, "Bearer garbage").unwrap_err().code == capability_errc::unauthorized, "garbage: unauthorized");
    check(static_cast<int>(sign_in_method::api_key) == 5, "sign_in_method::api_key is the machine credential");
    return failures == 0 ? 0 : 1;
}
