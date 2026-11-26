//
//  signature_verifier_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/signature_verifier.hpp>: a verifier chosen at run time, behind the virtual
//  interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/signature_verifier.hpp>
#include <snicholls/core/interfaces/signature_verifier_reference.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>

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
    std::printf("snicholls-core interfaces/signature_verifier.hpp demo\n");
    memory_signer keys = memory_signer::deterministic();
    const std::string kid{keys.create({signature_algorithm::hs256, "jwt"})->key_id.view()};
    const std::array<std::byte, 3> message{std::byte{1}, std::byte{2}, std::byte{3}};
    const auto tag = keys.sign({kid, message, payload_kind::message});
    memory_signature_verifier reference{keys};
    //! [virtual]
    signature_verifier_adapter<memory_signature_verifier&> adapter{reference};   // borrows
    const signature_verifier_interface& chosen = adapter;
    const signature_verifier_ref handle{chosen};
    const auto checked = handle.verify({signature_algorithm::hs256, kid, 1, {}, message, tag->bytes});
    //! [virtual]
    check(checked.is_ok(), "verified through the interface and the handle");
    check(&adapter.get() == &reference, "the adapter borrows");
    const signature_verifier_adapter owned{memory_signature_verifier{keys}};
    check(owned.verify({signature_algorithm::rs256, kid, 1, {}, message, tag->bytes}).unwrap_err().code ==
              capability_errc::unsupported,
          "an owning adapter forwards: RS256 is unsupported by the reference");
    return failures == 0 ? 0 : 1;
}
