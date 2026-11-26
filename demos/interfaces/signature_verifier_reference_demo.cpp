//
//  signature_verifier_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/signature_verifier_reference.hpp>: HS256 under the reference key store,
//  through a rotation, and a failure armed for a test. Exits non-zero if a result is not the one documented.
//

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
    std::printf("snicholls-core interfaces/signature_verifier_reference.hpp demo\n");
    //! [reference]
    memory_signer keys = memory_signer::deterministic();   // HS256 keys, for tests only
    const std::string kid{keys.create({signature_algorithm::hs256, "jwt"})->key_id.view()};
    const std::array<std::byte, 2> message{std::byte{'h'}, std::byte{'i'}};
    const auto old_tag = keys.sign({kid, message, payload_kind::message});
    static_cast<void>(keys.rotate(kid));   // version 2 signs; version 1 still verifies
    memory_signature_verifier verifier{keys};
    const auto still_good = verifier.verify({signature_algorithm::hs256, kid, 1, {}, message, old_tag->bytes});
    //! [reference]
    check(still_good.is_ok(), "a retired version still verifies");
    check(verifier.verify({signature_algorithm::hs256, kid, 0, {}, message, old_tag->bytes}).unwrap_err().code ==
              capability_errc::invalid_argument,
          "HS256 without a version is invalid_argument");
    check(verifier.verify({signature_algorithm::hs256, "key-9", 1, {}, message, old_tag->bytes}).unwrap_err().code ==
              capability_errc::not_found,
          "an unknown key is not_found");
    verifier.fail_next(capability_errc::unavailable, "kms_down");
    check(verifier.verify({signature_algorithm::hs256, kid, 1, {}, message, old_tag->bytes}).unwrap_err().code ==
              capability_errc::unavailable,
          "an armed failure is returned once");
    return failures == 0 ? 0 : 1;
}
