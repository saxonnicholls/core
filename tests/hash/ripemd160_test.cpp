//
//  ripemd160_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/ripemd160.hpp>: the designers' published test vectors (Bosselaers' RIPEMD-160
//  page), Bitcoin's hash160 of a known public key, Python's hashlib on random lengths, every split, and digests
//  computed at compile time.
//

#include <snicholls/core/hash/ripemd160.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <type_traits>

namespace {

using namespace snicholls;
using snicholls::test::hex;

static_assert(ripemd160::hash("").hex().view() == "9c1185a5c5e9fc54612808977ee8f548b2258d31");
static_assert(ripemd160::hash("a").hex().view() == "0bdc9d2d256b3ee9daae347be6f4dc835a467ffe");
static_assert(ripemd160::hash("abc").hex().view() == "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc");
static_assert(ripemd160::hash("message digest").hex().view() == "5d0689ef49d2fae572b881b123a85ffa21595f36");
static_assert(ripemd160::hash("abcdefghijklmnopqrstuvwxyz").hex().view() == "f71c27109c692c1b56bbdceb5b9d2865b3708dbc");
static_assert(ripemd160::hash("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").hex().view() ==
              "12a053384a9c0c88e405a06c27dcf49ada62eb2b");
static_assert(ripemd160::hash("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789").hex().view() ==
              "b0e20b6e3116640286ed3a87a5713079b21f5189");
static_assert(ripemd160::hash("12345678901234567890123456789012345678901234567890123456789012345678901234567890").hex().view() ==
              "9b752e45573d4b39f4dbd3323cab82bf63326bfb");
static_assert(ripemd160::digest_size == 20 && ripemd160::block_size == 64 && hash_function<ripemd160>);
static_assert(std::is_trivially_copyable_v<ripemd160> && noexcept(ripemd160::hash("")));

TEST(Ripemd160Vectors, OneMillionAs)
{
    EXPECT_EQ(ripemd160::hash(std::string(1000000, 'a')).hex().view(), "52783243c1697bdbe16d37f97f68f08325dc1528");
}

TEST(Ripemd160Vectors, BitcoinHash160OfTheGeneratorPoint)
{
    // The compressed public key of private key 1 (the secp256k1 generator), whose P2PKH address is
    // 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH: hash160 = 751e76e8199196d454941c45d1b3a323f1433bd6
    const auto pubkey = test::unhex("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    EXPECT_EQ(ripemd160::hash(sha256::hash(pubkey)).hex().view(), "751e76e8199196d454941c45d1b3a323f1433bd6");
}

TEST(Ripemd160Oracle, PythonHashlibAtBlockBoundariesAndRandomLengths)
{
    for (const auto& c : test::read_oracle("oracle_sha.txt", "ripemd160"))
        ASSERT_EQ(hex(ripemd160::hash(test::oracle_message(c.length)).bytes), c.hex) << "line " << c.line;
}

TEST(Ripemd160Incremental, EverySplitGivesTheOneShotDigest)
{
    for (const std::size_t n : {0u, 1u, 55u, 56u, 57u, 63u, 64u, 65u, 128u, 200u}) {
        const auto m = test::oracle_message(n);
        const auto whole = ripemd160::hash(m);
        for (std::size_t a = 0; a <= n; ++a)
            for (std::size_t b = a; b <= n; b += 5)
                ASSERT_EQ(test::hash_in_parts<ripemd160>(m, a, b), whole) << "n=" << n << " a=" << a << " b=" << b;
        ASSERT_EQ(test::hash_bytewise<ripemd160>(m), whole);
    }
}

} // namespace
