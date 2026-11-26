//
//  encryptor_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/encryptor_reference.hpp>: the test double's envelope layout, determinism, key
//  management and its name, which says it is not encryption.
//

#include <snicholls/core/interfaces/encryptor_reference.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

std::vector<std::byte> bytes_of(std::string_view s)
{
    std::vector<std::byte> out;
    for (const char c : s)
        out.push_back(static_cast<std::byte>(c));
    return out;
}

TEST(InsecureTestEncryptor, SaysItIsNotEncryption)
{
    const insecure_test_encryptor e;
    EXPECT_NE(e.algorithm().find("not encryption"), std::string_view::npos);
    EXPECT_NE(e.algorithm().find("TEST-ONLY"), std::string_view::npos);
}

TEST(InsecureTestEncryptor, TheEnvelopeIsHeaderNonceCiphertextTag)
{
    auto e = test::make_test_encryptor();
    const auto plain = bytes_of("hello");
    const auto sealed = *e.encrypt("k1", plain, {});
    EXPECT_EQ(sealed.size(), 2 + 2 + insecure_test_encryptor::nonce_size + plain.size() + insecure_test_encryptor::tag_size);
    EXPECT_EQ(*sealed_key_id(sealed), "k1");
    EXPECT_EQ(sealed[4 + 7], std::byte{1});   // the first nonce counts from 1
}

TEST(InsecureTestEncryptor, IsDeterministicFromAFreshCopy)
{
    const auto a = test::make_test_encryptor();
    auto x = a;
    auto y = a;
    const auto plain = bytes_of("same");
    EXPECT_EQ(*x.encrypt("k1", plain, {}), *y.encrypt("k1", plain, {}));
    EXPECT_NE(*x.encrypt("k1", plain, {}), *x.encrypt("k1", plain, {}));
}

TEST(InsecureTestEncryptor, AddAndRemoveKeys)
{
    insecure_test_encryptor e;
    const std::array<std::byte, 32> k{};
    const std::array<std::byte, 16> short_key{};
    EXPECT_EQ(e.add_key("bad id", k).unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(e.add_key("k", short_key).unwrap_err().message, std::string_view{"a key is 32 bytes"});
    EXPECT_TRUE(e.add_key("k", k).is_ok());
    const auto sealed = *e.encrypt("k", bytes_of("x"), {});
    EXPECT_TRUE(e.remove_key("k"));
    EXPECT_FALSE(e.remove_key("k"));
    EXPECT_EQ(e.decrypt(sealed, {}).unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(e.encrypt("k", bytes_of("x"), {}).unwrap_err().code, capability_errc::not_found);
}

TEST(InsecureTestEncryptor, ReplacingAKeyMeansOldEnvelopesNoLongerOpen)
{
    auto e = test::make_test_encryptor();
    const auto sealed = *e.encrypt("k1", bytes_of("x"), {});
    std::array<std::byte, 32> other{};
    other.fill(std::byte{9});
    ASSERT_TRUE(e.add_key("k1", other).is_ok());
    EXPECT_EQ(e.decrypt(sealed, {}).unwrap_err().code, capability_errc::signature_invalid);
}

TEST(InsecureTestEncryptor, ATooShortEnvelopeIsInvalid)
{
    const auto e = test::make_test_encryptor();
    const auto header = sealed_header("k1");
    EXPECT_EQ(e.decrypt(header, {}).unwrap_err().message, std::string_view{"the envelope is too short"});
}

} // namespace
