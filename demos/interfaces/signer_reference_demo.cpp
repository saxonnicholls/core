//
//  signer_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/signer_reference.hpp>: memory_signer, the reference key store under its signer name. Exits non-zero if a result is not the one documented.
//

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
    std::printf("snicholls-core interfaces/signer_reference.hpp demo\n");
    //! [reference]
    memory_signer s = memory_signer::deterministic();
    const auto key = s.create({signature_algorithm::hs256, "siwe"});
    const std::array<std::byte, 2> payload{std::byte{'h'}, std::byte{'i'}};
    const auto sig = s.sign({key->key_id.view(), payload, payload_kind::message});
    //! [reference]
    check(sig.is_ok() && s.verify(*sig, payload), "it signs and verifies");
    return failures == 0 ? 0 : 1;
}
