//
//  signer_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/signer.hpp>: signing through the concept, and the JOSE names. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/signer.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <span>
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

template <signer S>
capability_result<signature> sign_token(S& s, std::string_view key, std::string_view signing_input)
{
    return s.sign({key, std::as_bytes(std::span<const char>(signing_input.data(), signing_input.size())), payload_kind::message});
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/signer.hpp demo\n");
    //! [concept]
    memory_signer keys = memory_signer::deterministic();   // HS256, for tests only
    const auto key = keys.create({signature_algorithm::hs256, "jwt-access"});
    const auto sig = sign_token(keys, key->key_id.view(), "eyJhbGciOiJIUzI1NiJ9.e30");
    //! [concept]
    check(sig.is_ok() && sig->bytes.size() == 32 && sig->key_version == 1, "an HS256 tag of 32 bytes");
    check(jose_name(sig->algorithm) == "HS256" && !is_asymmetric(sig->algorithm), "HS256 is symmetric");
    check(keys.public_key_of({key->key_id.view(), 0}).unwrap_err().code == capability_errc::unsupported, "no public key for HS256");
    return failures == 0 ? 0 : 1;
}
