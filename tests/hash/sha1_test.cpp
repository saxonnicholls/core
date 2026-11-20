//
//  sha1_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/sha1.hpp>: the FIPS 180-4 examples, NIST's byte-oriented ShortMsg and Monte
//  Carlo vectors, Python's hashlib on random lengths, every split, and digests computed at compile time.
//

#include <snicholls/core/hash/sha1.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

namespace {

using namespace snicholls;
using snicholls::test::hex;

static_assert(sha1::hash("").hex().view() == "da39a3ee5e6b4b0d3255bfef95601890afd80709");
static_assert(sha1::hash("abc").hex().view() == "a9993e364706816aba3e25717850c26c9cd0d89d");
static_assert(sha1::hash("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").hex().view() ==
              "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
static_assert(sha1::hash("The quick brown fox jumps over the lazy dog").hex().view() == "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12");
static_assert(sha1::digest_size == 20 && sha1::block_size == 64 && hash_function<sha1>);
static_assert(std::is_trivially_copyable_v<sha1> && sizeof(sha1) == 5 * 4 + 64 + sizeof(std::size_t) + 8 + 4);
static_assert(noexcept(sha1::hash("")));

TEST(Sha1Vectors, OneMillionAs)
{
    EXPECT_EQ(sha1::hash(std::string(1000000, 'a')).hex().view(), "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
}

TEST(Sha1Vectors, NistShortMessagesAndMonteCarlo)
{
    for (const auto& m : test::read_nist_short("SHA1ShortMsg.rsp"))
        ASSERT_EQ(hex(sha1::hash(m.message).bytes), m.digest) << "length " << m.message.size();
    const auto seed = test::unhex(test::read_rsp_field("SHA1Monte.rsp", "Seed"));
    EXPECT_EQ(test::sha2_monte<sha1>(seed), test::read_monte_expected("SHA1Monte.rsp", "MD"));
}

TEST(Sha1Oracle, PythonHashlibAtBlockBoundariesAndRandomLengths)
{
    for (const auto& c : test::read_oracle("oracle_sha.txt", "sha1"))
        ASSERT_EQ(hex(sha1::hash(test::oracle_message(c.length)).bytes), c.hex) << "line " << c.line;
}

TEST(Sha1Incremental, EverySplitGivesTheOneShotDigest)
{
    for (const std::size_t n : {0u, 1u, 55u, 56u, 57u, 63u, 64u, 65u, 119u, 120u, 128u, 200u}) {
        const auto m = test::oracle_message(n);
        const auto whole = sha1::hash(m);
        for (std::size_t a = 0; a <= n; ++a)
            for (std::size_t b = a; b <= n; b += 5)
                ASSERT_EQ(test::hash_in_parts<sha1>(m, a, b), whole) << "n=" << n << " a=" << a << " b=" << b;
        ASSERT_EQ(test::hash_bytewise<sha1>(m), whole);
    }
}

} // namespace
