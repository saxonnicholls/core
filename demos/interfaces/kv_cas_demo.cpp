//
//  kv_cas_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/kv_cas.hpp>: compare-and-swap through the virtual interface, and the kit. Exits
//  non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_cas.hpp>

#include <cstdio>
#include <memory>
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
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/kv_cas.hpp demo\n");
    reference_key_value_store store;
    //! [virtual]
    std::unique_ptr<kv_cas_interface> cas = std::make_unique<kv_cas_adapter<reference_key_value_store&>>(store);
    const auto first = cas->compare_and_swap(b("version"), std::nullopt, b("1"));
    const auto stale = cas->compare_and_swap(b("version"), b("0"), b("2"));   // an optimistic update from an old read
    //! [virtual]
    check(first && *first, "created through the interface");
    check(stale && !*stale, "a stale update is refused");
    check(check_kv_cas(store) == std::nullopt, "the reference passes check_kv_cas");
    return failures == 0 ? 0 : 1;
}
