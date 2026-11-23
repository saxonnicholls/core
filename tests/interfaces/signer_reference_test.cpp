//
//  signer_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/signer_reference.hpp>: memory_signer is the reference key store, signing.
//

#include <snicholls/core/interfaces/signer_reference.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(std::is_same_v<memory_signer, memory_key_store>);

TEST(MemorySigner, SignsWithTheKeysItsStoreHolds)
{
    memory_signer s = memory_signer::deterministic();
    const auto k = s.create({signature_algorithm::hs256, "jwt"});
    ASSERT_TRUE(k.is_ok());
    const std::string text = "a.b";
    const auto sig = s.sign({k->key_id.view(), std::as_bytes(std::span<const char>(text.data(), text.size())), payload_kind::message});
    ASSERT_TRUE(sig.is_ok());
    EXPECT_EQ(sig->algorithm, signature_algorithm::hs256);
    EXPECT_EQ(jose_name(sig->algorithm), "HS256");
    EXPECT_TRUE(s.verify(*sig, std::as_bytes(std::span<const char>(text.data(), text.size()))));
}

} // namespace
