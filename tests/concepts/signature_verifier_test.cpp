//
//  signature_verifier_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/signature_verifier.hpp>: the models and the non-models, the request borrows.
//

#include <snicholls/core/concepts/signature_verifier.hpp>
#include <snicholls/core/interfaces/signature_verifier_reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(signature_verifier<memory_signature_verifier>);
static_assert(signature_verifier<signature_verifier_interface>);
static_assert(signature_verifier<signature_verifier_adapter<memory_signature_verifier>>);
static_assert(signature_verifier<signature_verifier_adapter<memory_signature_verifier&>>);
static_assert(signature_verifier<signature_verifier_ref>);

struct verify_not_const {
    capability_result<void> verify(const verify_request&);
};
struct verify_returns_a_bool {
    bool verify(const verify_request&) const;
};
static_assert(!signature_verifier<verify_not_const>);
static_assert(!signature_verifier<verify_returns_a_bool>);
static_assert(!signature_verifier<int>);
static_assert(std::is_trivially_copyable_v<verify_request>, "a request is borrowed views: nothing to copy");

TEST(SignatureVerifierConcept, ARequestBorrowsItsBytes)
{
    const std::array<std::byte, 2> message{std::byte{1}, std::byte{2}};
    const verify_request r{signature_algorithm::hs256, "key-1", 1, {}, message, {}};
    EXPECT_EQ(r.message.data(), message.data());
    EXPECT_TRUE(r.public_key.empty());
    EXPECT_EQ(verify_request{}.algorithm, signature_algorithm::eddsa);
    EXPECT_EQ(verify_request{}.key_version, 0u);
}

} // namespace
