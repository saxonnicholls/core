//
//  key_store_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/key_store_reference.hpp>: the HS256 tag is core's HMAC-SHA256, versions, the clock and the injected failure.
//

#include <snicholls/core/interfaces/key_store_reference.hpp>
#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <span>
#include <string>
#include <type_traits>

using namespace snicholls;

namespace {

const key_spec hs256{signature_algorithm::hs256, "jwt-test"};

std::span<const std::byte> bytes_of(const std::string& s) { return std::as_bytes(std::span<const char>(s.data(), s.size())); }

TEST(MemoryKeyStore, AnHs256TagIsCoresHmacSha256UnderTheKeyMaterial)
{
    memory_key_store s = memory_key_store::deterministic();   // key material: bytes 0, 1, ..., 31
    const auto k = s.create(hs256);
    ASSERT_TRUE(k.is_ok());
    EXPECT_EQ(k->key_id, std::string_view{"key-1"});
    const std::string text = "header.payload";
    const auto sig = s.sign({k->key_id.view(), bytes_of(text), payload_kind::message});
    ASSERT_TRUE(sig.is_ok());
    std::string key(32, '\0');
    for (std::size_t i = 0; i < key.size(); ++i)
        key[i] = static_cast<char>(i);
    const auto expected = hmac<sha256>::mac(key, text);
    ASSERT_EQ(sig->bytes.size(), 32u);
    EXPECT_TRUE(std::equal(sig->bytes.begin(), sig->bytes.end(), expected.bytes.begin()));
}

TEST(MemoryKeyStore, ARetiredVersionStillVerifiesAndADestroyedOneDoesNot)
{
    memory_key_store s = memory_key_store::deterministic();
    const auto k = s.create(hs256);
    const std::array<std::byte, 3> p{std::byte{1}, std::byte{2}, std::byte{3}};
    const auto old_sig = s.sign({k->key_id.view(), p, payload_kind::message});
    ASSERT_TRUE(s.rotate(k->key_id.view()).is_ok());
    const auto new_sig = s.sign({k->key_id.view(), p, payload_kind::message});
    EXPECT_EQ(old_sig->key_version, 1u);
    EXPECT_EQ(new_sig->key_version, 2u);
    EXPECT_NE(old_sig->bytes, new_sig->bytes);
    EXPECT_TRUE(s.verify(*old_sig, p));
    EXPECT_TRUE(s.verify(*new_sig, p));
    signature forged = *new_sig;
    forged.bytes[0] ^= std::byte{1};
    EXPECT_FALSE(s.verify(forged, p));
    ASSERT_TRUE(s.destroy(k->key_id.view()).is_ok());
    EXPECT_FALSE(s.verify(*old_sig, p));
}

// Any utc_clock, at any precision: this one reads seconds and never moves.
struct fixed_clock {
    [[nodiscard]] constexpr utc_s now() const noexcept { return utc_s::from_unix_seconds(42); }
};
static_assert(utc_clock<fixed_clock> && key_store<basic_memory_key_store<fixed_clock>>);
static_assert(std::is_copy_constructible_v<memory_key_store>, "the clock is owned: nothing captures the store");

TEST(MemoryKeyStore, CreatedAtComesFromTheInjectedClock)
{
    memory_key_store s = memory_key_store::deterministic();
    EXPECT_EQ(s.create(hs256)->created_at.unix_seconds(), 1'800'000'000);
    basic_memory_key_store<fixed_clock> t{[](std::span<std::byte> out) { std::fill(out.begin(), out.end(), std::byte{7}); }, {}};
    EXPECT_EQ(t.create(hs256)->created_at.unix_seconds(), 42);
    s.clock().advance(std::chrono::milliseconds{1'999});   // floored to the second it records
    EXPECT_EQ(s.create(hs256)->created_at.unix_seconds(), 1'800'000'001);
}

TEST(MemoryKeyStore, PublicKeysAreUnsupportedDigestsRefusedAndUnknownVersionsNotFound)
{
    memory_key_store s = memory_key_store::deterministic();
    const auto k = s.create(hs256);
    EXPECT_EQ(s.public_key_of({k->key_id.view(), 1}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(s.public_key_of({k->key_id.view(), 2}).unwrap_err().code, capability_errc::not_found);
    const std::array<std::byte, 32> digest{};
    EXPECT_EQ(s.sign({k->key_id.view(), digest, payload_kind::digest}).unwrap_err().code, capability_errc::unsupported);
}

TEST(MemoryKeyStore, FailNextFailsTheNextMutationOnly)
{
    memory_key_store s = memory_key_store::deterministic();
    s.fail_next(capability_errc::timeout, "kms_timeout secret=abc");
    const auto r = s.create(hs256);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().code, capability_errc::timeout);
    EXPECT_EQ(r.unwrap_err().provider_code, std::string_view{"kms_timeout"});
    EXPECT_TRUE(s.create(hs256).is_ok());
}

} // namespace
