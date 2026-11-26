//
//  role_store_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/role_store.hpp>: an authorisation check through the concept, with a grant's
//  window. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/role_store.hpp>
#include <snicholls/core/interfaces/role_store_reference.hpp>

#include <chrono>
#include <cstdio>
#include <optional>
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
// Re-read on every request: does the party hold `role` in `tenant` now? A store that cannot answer denies.
template <role_store S>
bool holds(const S& roles, const uuid& party, std::string_view tenant, std::string_view role, utc_s now)
{
    const auto grants = roles.grants_of(party, tenant, now);
    if (grants.is_err())
        return false;
    for (const auto& g : *grants)
        if (g.role == role)
            return true;
    return false;
}
//! [concept]
} // namespace

int main()
{
    std::printf("snicholls-core concepts/role_store.hpp demo\n");
    constexpr uuid auditor = uuid::from_u64(0xa0d1, 7);
    constexpr uuid owner = uuid::from_u64(0x0e, 1);
    const utc_s now = utc_s::from_unix_seconds(1'800'000'000);
    memory_role_store roles;
    const role_grant window{auditor, "acme", "auditor", owner, "quarter-end audit", now, now + std::chrono::hours{72}};
    check(window.active(now) && !window.active(now + std::chrono::hours{72}), "a grant's window");
    static_cast<void>(roles.grant(window, "Grant auditor to the auditor in acme for three days"));

    check(holds(roles, auditor, "acme", "auditor", now), "the auditor holds the role in acme");
    check(!holds(roles, auditor, platform_tenant, "auditor", now), "and not on the platform");
    check(!holds(roles, auditor, "acme", "auditor", now + std::chrono::hours{72}), "and not after the window");
    return failures == 0 ? 0 : 1;
}
