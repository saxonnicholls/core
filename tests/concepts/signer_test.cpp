//
//  signer_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/signer.hpp>: the models and the non-models, the JOSE names.
//

#include <snicholls/core/concepts/signer.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>

#include <gtest/gtest.h>

#include <string_view>

using namespace snicholls;

namespace {

static_assert(signer<memory_signer>);
static_assert(signer<signer_interface>);
static_assert(signer<signer_adapter<memory_signer>>);
static_assert(signer<signer_adapter<memory_signer&>>);
static_assert(signer<signer_ref>);

struct no_public_keys {
    capability_result<signature> sign(const sign_request&);
};
struct public_key_of_not_const {
    capability_result<signature> sign(const sign_request&);
    capability_result<public_key> public_key_of(const key_ref&);
};
static_assert(!signer<no_public_keys>);
static_assert(!signer<public_key_of_not_const>);
static_assert(!signer<int>);

static_assert(jose_name(signature_algorithm::es256) == "ES256" && jose_name(signature_algorithm::eddsa) == "EdDSA");
static_assert(jose_name(signature_algorithm::es256k) == "ES256K" && jose_name(signature_algorithm::hs256) == "HS256");
static_assert(jose_name(signature_algorithm::rs256) == "RS256");
static_assert(jose_name(static_cast<signature_algorithm>(0)) == "unknown");
static_assert(!is_asymmetric(signature_algorithm::hs256) && is_asymmetric(signature_algorithm::eddsa));
static_assert(key_ref{"k", 0} == key_ref{"k", 0});

TEST(SignerConcept, SignaturesCompareByEveryMember)
{
    const signature a{signature_algorithm::hs256, "key-1", 1, {std::byte{1}}};
    signature b{signature_algorithm::hs256, "key-1", 1, {std::byte{1}}};   // built, not copied: GCC 13 warns falsely on a one-byte vector copy
    EXPECT_EQ(a, b);
    b.key_version = 2;
    EXPECT_NE(a, b);
    EXPECT_EQ((public_key{}), (public_key{}));
}

} // namespace
