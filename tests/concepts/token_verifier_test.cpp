//
//  token_verifier_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/token_verifier.hpp>: the models and the non-models, the claims' lookups.
//

#include <snicholls/core/concepts/token_verifier.hpp>
#include <snicholls/core/interfaces/token_verifier_reference.hpp>

#include <gtest/gtest.h>

#include <string_view>

using namespace snicholls;

namespace {

static_assert(token_verifier<memory_token_verifier>);
static_assert(token_verifier<token_verifier_interface>);
static_assert(token_verifier<token_verifier_adapter<memory_token_verifier>>);
static_assert(token_verifier<token_verifier_adapter<memory_token_verifier&>>);
static_assert(token_verifier<token_verifier_ref>);

struct verify_not_const {
    capability_result<token_claims> verify(std::string_view);
};
struct verify_returns_a_bool {
    bool verify(std::string_view) const;
};
struct no_verify {};
static_assert(!token_verifier<verify_not_const>);
static_assert(!token_verifier<verify_returns_a_bool>);
static_assert(!token_verifier<no_verify>);
static_assert(!token_verifier<int>);

TEST(TokenVerifierConcept, ClaimsFindTheirRolesAndScopes)
{
    token_claims c;
    c.roles = {"admin", "member"};
    c.scopes = {"payments.refund"};
    EXPECT_TRUE(c.has_role("admin"));
    EXPECT_FALSE(c.has_role("owner"));
    EXPECT_TRUE(c.has_scope("payments.refund"));
    EXPECT_FALSE(c.has_scope("payments"));
    EXPECT_FALSE(token_claims{}.has_role(""));
}

TEST(TokenVerifierConcept, ClaimsCompareByEveryMember)
{
    token_claims a;
    a.subject = uuid::from_u64(1, 2);
    token_claims b = a;
    EXPECT_EQ(a, b);
    b.method = sign_in_method::wallet;
    EXPECT_NE(a, b);
    b = a;
    b.token_owner = uuid::from_u64(3, 4);
    EXPECT_NE(a, b);
    EXPECT_TRUE(token_claims{}.subject.is_nil());
}

} // namespace
