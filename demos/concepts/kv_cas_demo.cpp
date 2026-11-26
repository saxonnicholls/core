//
//  kv_cas_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/kv_cas.hpp>: a lease taken, renewed and released with compare-and-swap. Exits
//  non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/kv_cas.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

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
kv_bytes b(std::string_view s) { return {reinterpret_cast<const std::byte*>(s.data()), s.size()}; }

// Generic code: take the lease if it is free.
template <kv_cas S>
bool take(S& store, std::string_view owner)
{
    return kv_compare_and_swap(store, std::string_view{"lease/settlement"}, std::nullopt, b(owner)).unwrap_or(false);
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/kv_cas.hpp demo\n");
    //! [cas]
    reference_key_value_store store;
    const bool a = take(store, "worker-a");   // free: taken
    const bool b_took = take(store, "worker-b");   // held: refused, nothing changed
    const auto released = store.compare_and_swap(b("lease/settlement"), b("worker-a"), std::nullopt);   // only its owner releases it
    //! [cas]
    check(a && !b_took, "one worker holds the lease");
    check(released && *released && !*store.contains(b("lease/settlement")), "the owner released it");
    check(take(store, "worker-b"), "then the other takes it");
    return failures == 0 ? 0 : 1;
}
