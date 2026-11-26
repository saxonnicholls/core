//
//  role_store_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/role_store_reference.hpp>: grants, a revoke and the history over a key-value
//  store, read back by a second store over the same backend. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/role_store_reference.hpp>

#include <cstdio>
#include <optional>

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
    std::printf("snicholls-core interfaces/role_store_reference.hpp demo\n");
    constexpr uuid alice = uuid::from_u64(0xa11ce, 1);
    constexpr uuid bob = uuid::from_u64(0xb0b, 2);
    const utc_s now = utc_s::from_unix_seconds(1'800'000'000);
    //! [reference]
    reference_key_value_store kv;   // or any key_value_store a product persists to
    kv_role_store<reference_key_value_store&> roles{kv};
    static_cast<void>(roles.grant({bob, "acme", "member", alice, "joined", now, std::nullopt}, ""));
    static_cast<void>(roles.grant({bob, "acme", "admin", alice, "promoted", now, std::nullopt}, "Grant admin to bob in acme"));
    static_cast<void>(roles.revoke(bob, "acme", "member", alice, "admin covers it", now));
    const kv_role_store<reference_key_value_store&> after_restart{kv};
    const auto history = after_restart.history("acme");   // grant, grant, revoke: oldest first
    //! [reference]
    check(history.is_ok() && history->size() == 3 && !history->back().granted, "the history, read back from the rows");
    check(after_restart.grants_of(bob, "acme", now)->size() == 1, "bob holds admin only");
    memory_role_store fresh;   // over its own reference_key_value_store
    fresh.fail_next(capability_errc::unavailable, "db_down");
    check(fresh.members("acme", now).unwrap_err().code == capability_errc::unavailable, "an armed failure is returned once");
    check(fresh.members("acme", now)->empty(), "then the store answers");
    check(fresh.backend().sequence().is_ok(), "the backend is the store's own");
    return failures == 0 ? 0 : 1;
}
