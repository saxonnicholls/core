//
//  signature_verifier_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/signature_verifier.hpp>: checking a JWS signature through the concept. Exits
//  non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/signature_verifier.hpp>
#include <snicholls/core/interfaces/signature_verifier_reference.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <span>
#include <string>
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

std::span<const std::byte> bytes(std::string_view s) { return std::as_bytes(std::span<const char>(s.data(), s.size())); }

//! [concept]
// A JWT verifier's last step: any verifier, the signing input and the signature it carried.
template <signature_verifier V>
bool signature_holds(const V& v, signature_algorithm alg, std::string_view kid, std::uint32_t version,
                     std::string_view signing_input, std::span<const std::byte> sig)
{
    return v.verify({alg, kid, version, {}, bytes(signing_input), sig}).is_ok();
}
//! [concept]
} // namespace

int main()
{
    std::printf("snicholls-core concepts/signature_verifier.hpp demo\n");
    memory_signer keys = memory_signer::deterministic();   // HS256, for tests only
    const std::string kid{keys.create({signature_algorithm::hs256, "jwt"})->key_id.view()};
    const std::string_view input = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkZW1vIn0";
    const auto tag = keys.sign({kid, bytes(input), payload_kind::message});
    const memory_signature_verifier verifier{keys};

    check(signature_holds(verifier, signature_algorithm::hs256, kid, 1, input, tag->bytes), "the tag verifies");
    check(!signature_holds(verifier, signature_algorithm::hs256, kid, 1, "eyJ.tampered", tag->bytes), "a tampered input does not");
    check(!signature_holds(verifier, signature_algorithm::eddsa, kid, 1, input, tag->bytes), "EdDSA is crypto's: unsupported here");
    return failures == 0 ? 0 : 1;
}
