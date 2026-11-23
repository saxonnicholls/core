//
//  key_store_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/key_store.hpp>: a key store behind the virtual interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/key_store.hpp>
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
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/key_store.hpp demo\n");
    //! [virtual]
    key_store_adapter owned{memory_key_store::deterministic()};   // owns the store
    key_store_interface& chosen = owned;
    key_store_ref handle{chosen};
    const auto made = handle.create({signature_algorithm::hs256, "jwt"});
    //! [virtual]
    check(made.is_ok() && made->key_id.view() == "key-1", "created through the interface");
    check(handle.rotate(made->key_id.view())->version == 2, "rotated");
    check(handle.info("nope").unwrap_err().code == capability_errc::not_found, "an unknown id is not found");
    check(owned.get().list()->size() == 2, "the adapter reaches the store");
    return failures == 0 ? 0 : 1;
}
