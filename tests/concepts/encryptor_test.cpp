//
//  encryptor_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/encryptor.hpp>: the models and the non-models, key ids and the envelope's header.
//

#include <snicholls/core/concepts/encryptor.hpp>
#include <snicholls/core/interfaces/encryptor_reference.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

static_assert(encryptor<insecure_test_encryptor>);
static_assert(encryptor<encryptor_interface>);
static_assert(encryptor<encryptor_adapter<insecure_test_encryptor>>);
static_assert(encryptor<encryptor_adapter<insecure_test_encryptor&>>);
static_assert(encryptor<encryptor_ref>);

using bytes = std::span<const std::byte>;
struct throws_instead {
    std::vector<std::byte> encrypt(std::string_view, bytes, bytes);
    std::vector<std::byte> decrypt(bytes, bytes) const;
    std::string_view algorithm() const;
};
struct no_aad {
    capability_result<std::vector<std::byte>> encrypt(std::string_view, bytes);
    capability_result<std::vector<std::byte>> decrypt(bytes) const;
    std::string_view algorithm() const;
};
struct decrypt_not_const {
    capability_result<std::vector<std::byte>> encrypt(std::string_view, bytes, bytes);
    capability_result<std::vector<std::byte>> decrypt(bytes, bytes);
    std::string_view algorithm() const;
};
static_assert(!encryptor<throws_instead>);
static_assert(!encryptor<no_aad>);
static_assert(!encryptor<decrypt_not_const>);
static_assert(!encryptor<int>);

static_assert(is_key_id("k1") && is_key_id("kms:prod.2026-11") && !is_key_id("kms:prod/2026-11"));
static_assert(!is_key_id("") && !is_key_id("has space") && !is_key_id("quote\"") && !is_key_id(std::string_view("a\0b", 3)));
static_assert(is_key_id(std::string_view("0123456789012345678901234567890123456789012345678901234567890123")));
static_assert(!is_key_id(std::string_view("01234567890123456789012345678901234567890123456789012345678901234")));

TEST(EncryptorConcept, TheHeaderCarriesTheKeyId)
{
    const std::vector<std::byte> h = sealed_header("kms:prod.2026-11");
    ASSERT_EQ(h.size(), 2u + 16u);
    EXPECT_EQ(h[0], sealed_format_v1);
    EXPECT_EQ(h[1], std::byte{16});
    EXPECT_EQ(*sealed_key_id(h), "kms:prod.2026-11");
}

TEST(EncryptorConcept, AMalformedHeaderIsInvalid)
{
    EXPECT_EQ(sealed_key_id({}).unwrap_err().code, capability_errc::invalid_argument);
    std::vector<std::byte> h = sealed_header("k1");
    h.pop_back();   // the id is cut short
    EXPECT_EQ(sealed_key_id(h).unwrap_err().code, capability_errc::invalid_argument);
    std::vector<std::byte> zero{sealed_format_v1, std::byte{0}};
    EXPECT_TRUE(sealed_key_id(zero).is_err());
    std::vector<std::byte> other_format = sealed_header("k1");
    other_format[0] = std::byte{2};
    EXPECT_TRUE(sealed_key_id(other_format).is_err());
    std::vector<std::byte> bad = sealed_header("k1");
    bad[2] = std::byte{' '};
    EXPECT_EQ(sealed_key_id(bad).unwrap_err().message, std::string_view{"the envelope's key id is not one"});
}

} // namespace
