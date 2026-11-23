//
//  signer_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/signer.hpp>: a signer behind the virtual interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/signer.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>

#include <array>
#include <cstddef>
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
    std::printf("snicholls-core interfaces/signer.hpp demo\n");
    memory_signer keys = memory_signer::deterministic();
    const auto key = keys.create({signature_algorithm::hs256, "jwt"});
    //! [virtual]
    signer_adapter<memory_signer&> adapter{keys};   // borrows
    signer_interface& chosen = adapter;
    signer_ref handle{chosen};
    const std::array<std::byte, 3> payload{std::byte{1}, std::byte{2}, std::byte{3}};
    const auto sig = handle.sign({key->key_id.view(), payload, payload_kind::message});
    //! [virtual]
    check(sig.is_ok() && keys.verify(*sig, payload), "signed through the interface, verified by the store");
    check(handle.public_key_of({"missing", 0}).unwrap_err().code == capability_errc::not_found, "an unknown key is not found");
    check(&adapter.get() == &keys, "the adapter borrows");
    return failures == 0 ? 0 : 1;
}
