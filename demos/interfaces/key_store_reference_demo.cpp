//
//  key_store_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/key_store_reference.hpp>: the reference key store: HS256 keys from the caller's randomness and clock. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/key_store_reference.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <span>

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
    std::printf("snicholls-core interfaces/key_store_reference.hpp demo\n");
    //! [reference]
    memory_key_store store{[](std::span<std::byte> out) { std::fill(out.begin(), out.end(), std::byte{0x5a}); },   // the OS's, in a program
                           [] { return utc_s::from_unix_seconds(1'800'000'000); }};
    const auto made = store.create({signature_algorithm::hs256, "webhooks"});
    const auto refused = store.create({signature_algorithm::eddsa, "jwt"});   // needs elliptic curves: above core
    //! [reference]
    check(made.is_ok() && made->created_at.unix_seconds() == 1'800'000'000, "created at the caller's time");
    check(refused.unwrap_err().code == capability_errc::unsupported, "EdDSA is unsupported here");
    store.fail_next(capability_errc::timeout, "kms_timeout");
    check(store.rotate(made->key_id.view()).unwrap_err().code == capability_errc::timeout, "fail_next fails the next call");
    check(memory_key_store::key_bytes == 32, "32-byte keys");
    return failures == 0 ? 0 : 1;
}
