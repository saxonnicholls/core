//
//  signature_verifier_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/signature_verifier_reference.hpp>: HS256 tags from memory_signer verify,
//  retired versions too; unknown keys and versions, a missing version, asymmetric algorithms and a destroyed key are
//  refused with the contract's codes; an armed failure is returned once.
//

#include <snicholls/core/interfaces/signature_verifier_reference.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

std::span<const std::byte> bytes(std::string_view s) { return std::as_bytes(std::span<const char>(s.data(), s.size())); }

constexpr std::string_view input = "eyJhbGciOiJIUzI1NiJ9.e30";

TEST(MemorySignatureVerifier, VerifiesWhatTheReferenceSignerSigned)
{
    memory_signer keys = memory_signer::deterministic();
    const std::string id{keys.create({signature_algorithm::hs256, "jwt"})->key_id.view()};
    const auto s = keys.sign({id, bytes(input), payload_kind::message});
    const memory_signature_verifier v{keys};
    EXPECT_TRUE(v.verify({signature_algorithm::hs256, id, s->key_version, {}, bytes(input), s->bytes}).is_ok());
    EXPECT_EQ(v.verify({signature_algorithm::hs256, id, s->key_version, {}, bytes("other"), s->bytes}).unwrap_err().code,
              capability_errc::signature_invalid);
}

TEST(MemorySignatureVerifier, ARetiredVersionStillVerifies)
{
    memory_signer keys = memory_signer::deterministic();
    const std::string id{keys.create({signature_algorithm::hs256, "jwt"})->key_id.view()};
    const auto v1 = keys.sign({id, bytes(input), payload_kind::message});
    ASSERT_TRUE(keys.rotate(id).is_ok());
    const auto v2 = keys.sign({id, bytes(input), payload_kind::message});
    const memory_signature_verifier v{keys};
    EXPECT_TRUE(v.verify({signature_algorithm::hs256, id, 1, {}, bytes(input), v1->bytes}).is_ok());
    EXPECT_TRUE(v.verify({signature_algorithm::hs256, id, 2, {}, bytes(input), v2->bytes}).is_ok());
    EXPECT_EQ(v.verify({signature_algorithm::hs256, id, 2, {}, bytes(input), v1->bytes}).unwrap_err().code,
              capability_errc::signature_invalid) << "a tag names its version";
    EXPECT_EQ(v.verify({signature_algorithm::hs256, id, 3, {}, bytes(input), v1->bytes}).unwrap_err().code,
              capability_errc::not_found);
}

TEST(MemorySignatureVerifier, RefusesWithTheContractsCodes)
{
    memory_signer keys = memory_signer::deterministic();
    const std::string id{keys.create({signature_algorithm::hs256, "jwt"})->key_id.view()};
    const auto s = keys.sign({id, bytes(input), payload_kind::message});
    const memory_signature_verifier v{keys};
    for (const auto a : {signature_algorithm::eddsa, signature_algorithm::es256, signature_algorithm::es256k, signature_algorithm::rs256})
        EXPECT_EQ(v.verify({a, id, 1, {}, bytes(input), s->bytes}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(v.verify({signature_algorithm::hs256, id, 0, {}, bytes(input), s->bytes}).unwrap_err().code,
              capability_errc::invalid_argument);
    EXPECT_EQ(v.verify({signature_algorithm::hs256, "key-99", 1, {}, bytes(input), s->bytes}).unwrap_err().code,
              capability_errc::not_found);
    EXPECT_EQ(v.verify({signature_algorithm::hs256, std::string(200, 'k'), 1, {}, bytes(input), s->bytes}).unwrap_err().code,
              capability_errc::not_found);
    ASSERT_TRUE(keys.destroy(id).is_ok());
    EXPECT_EQ(v.verify({signature_algorithm::hs256, id, 1, {}, bytes(input), s->bytes}).unwrap_err().code,
              capability_errc::not_found) << "a destroyed key verifies nothing";
}

TEST(MemorySignatureVerifier, AnArmedFailureIsReturnedOnce)
{
    memory_signer keys = memory_signer::deterministic();
    const std::string id{keys.create({signature_algorithm::hs256, "jwt"})->key_id.view()};
    const auto s = keys.sign({id, bytes(input), payload_kind::message});
    memory_signature_verifier v{keys};
    v.fail_next(capability_errc::timeout, "kms_timeout {\"key\":\"never kept\"}");
    const verify_request r{signature_algorithm::hs256, id, 1, {}, bytes(input), s->bytes};
    const auto e = v.verify(r).unwrap_err();
    EXPECT_EQ(e.code, capability_errc::timeout);
    EXPECT_EQ(e.provider_code.view(), "kms_timeout");
    EXPECT_TRUE(v.verify(r).is_ok());
}

} // namespace
