//
//  key_store_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/key_store.hpp>: the life of a key through the concept. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/key_store.hpp>
#include <snicholls/core/interfaces/key_store_reference.hpp>

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

template <key_store K>
capability_result<key_info> rotate_jwt_key(K& store)
{
    SNICHOLLS_TRY_ASSIGN(const key_info made, store.create({signature_algorithm::hs256, "jwt-access"}));
    return store.rotate(made.key_id.view());
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/key_store.hpp demo\n");
    //! [concept]
    memory_key_store store = memory_key_store::deterministic();
    const auto v2 = rotate_jwt_key(store);
    const auto all = store.list();   // version 1 retired (still verifies), version 2 active
    //! [concept]
    check(v2.is_ok() && v2->version == 2 && v2->state == key_state::active, "rotate makes version 2 active");
    check(all.is_ok() && all->size() == 2 && (*all)[0].state == key_state::retired, "version 1 is retired");
    check(store.retire(v2->key_id.view()).is_ok() && store.info(v2->key_id.view())->state == key_state::retired, "retire");
    check(store.destroy(v2->key_id.view()).is_ok() && store.list()->empty(), "destroy removes every version");
    return failures == 0 ? 0 : 1;
}
