//
//  role_store_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/role_store.hpp>: a role store chosen at run time, behind the virtual interface.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/role_store.hpp>
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
    std::printf("snicholls-core interfaces/role_store.hpp demo\n");
    constexpr uuid alice = uuid::from_u64(0xa11ce, 1);
    const utc_s now = utc_s::from_unix_seconds(1'800'000'000);
    memory_role_store reference;
    //! [virtual]
    role_store_adapter<memory_role_store&> adapter{reference};   // borrows
    role_store_interface& chosen = adapter;
    role_store_ref roles{chosen};
    static_cast<void>(roles.grant({alice, "acme", "admin", alice, "bootstrap", now, std::nullopt}, ""));
    const auto members = roles.members("acme", now);
    //! [virtual]
    check(members.is_ok() && members->size() == 1, "granted and read through the interface and the handle");
    check(&adapter.get() == &reference, "the adapter borrows");
    role_store_adapter owned{memory_role_store{}};
    check(owned.revoke(alice, "acme", "admin", alice, "", now).unwrap_err().code == capability_errc::not_found,
          "an owning adapter forwards: nothing to revoke in a new store");
    return failures == 0 ? 0 : 1;
}
