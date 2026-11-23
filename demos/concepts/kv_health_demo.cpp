//
//  kv_health_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/kv_health.hpp>: a readiness probe written against the concept, run on the
//  reference store while it works and while it is unreachable. Exits non-zero if a result is not the one
//  documented.
//

#include <snicholls/core/concepts/kv_health.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <cstdio>

using namespace snicholls;

//! [health]
// A probe for any store that offers health: ready, and what it holds, as plain data to print or export.
template <kv_health S>
bool probe(const S& store)
{
    if (const auto up = store.ping(); up.is_err()) {
        std::printf("  not ready: error %d\n", static_cast<int>(up.unwrap_err_unchecked()));
        return false;
    }
    const auto d = store.diagnostics();
    if (d.is_err())
        return false;
    std::printf("  ready: %.*s, sequence %llu, %llu keys, %llu bytes, %s\n", static_cast<int>(d->backend.size()),
                d->backend.data(), static_cast<unsigned long long>(d->sequence),
                static_cast<unsigned long long>(d->keys.value_or(0)), static_cast<unsigned long long>(d->bytes.value_or(0)),
                d->persistent ? "persistent" : "in memory");
    return true;
}
//! [health]

int main()
{
    std::printf("snicholls-core kv_health demo\n");
    bool ok = true;
    reference_key_value_store store;
    static_cast<void>(kv_put(store, "user/1", "ada"));
    static_cast<void>(kv_put(store, "user/2", "grace"));
    ok &= probe(store);
    ok &= store.diagnostics()->keys == 2u && store.diagnostics()->bytes == 20u && !store.diagnostics()->read_only;

    store.set_unavailable(true);   // as a lost connection would
    ok &= !probe(store);
    ok &= store.diagnostics().is_err();

    std::printf(ok ? "kv_health demo: ok\n" : "kv_health demo: FAILED\n");
    return ok ? 0 : 1;
}
