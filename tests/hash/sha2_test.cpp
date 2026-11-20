//
//  sha2_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/sha2.hpp>: the FIPS 180-4 examples (NIST's "Examples with intermediate values"),
//  NIST's byte-oriented ShortMsg and Monte Carlo vectors (tests/fixtures/hash/nist/), Python's hashlib on random
//  lengths (tests/fixtures/hash/oracle_sha.txt), every split of an input, and digests computed at compile time.
//

#include <snicholls/core/hash/sha2.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls;
using snicholls::test::hex;

constexpr std::string_view two_blocks_256 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
constexpr std::string_view two_blocks_512 = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";

// FIPS 180-4 examples, computed by the compiler
static_assert(sha224::hash("abc").hex().view() == "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7");
static_assert(sha256::hash("abc").hex().view() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
static_assert(sha384::hash("abc").hex().view() ==
              "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7");
static_assert(sha512::hash("abc").hex().view() ==
              "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd"
              "454d4423643ce80e2a9ac94fa54ca49f");
static_assert(sha512_224::hash("abc").hex().view() == "4634270f707b6a54daae7530460842e20e37ed265ceee9a43e8924aa");
static_assert(sha512_256::hash("abc").hex().view() == "53048e2681941ef99b2e29b76b4c7dabe4c2d0c634fc6d46e0e2f13107e7af23");
static_assert(sha256::hash("").hex().view() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
static_assert(sha256::hash(two_blocks_256).hex().view() == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
static_assert(sha224::hash(two_blocks_256).hex().view() == "75388b16512776cc5dba5da1fd890150b0c6455cb4f58b1952522525");
static_assert(sha384::hash(two_blocks_512).hex().view() ==
              "09330c33f71147e83d192fc782cd1b4753111b173b3b05d22fa08086e3b0f712fcc7c71a557e2db966c3e9fa91746039");
static_assert(sha512::hash(two_blocks_512).hex().view() ==
              "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433a"
              "c7d329eeb6dd26545e96e55b874be909");
static_assert(sha512_224::hash(two_blocks_512).hex().view() == "23fec5bb94d60b23308192640b0c453335d664734fe40e7268674af9");
static_assert(sha512_256::hash(two_blocks_512).hex().view() == "3928e184fb8690f840da3988121d31be65cb9d3ef83ee6146feac861e19b563a");

// Sizes, block sizes and the interface
static_assert(sha224::digest_size == 28 && sha256::digest_size == 32 && sha384::digest_size == 48);
static_assert(sha512::digest_size == 64 && sha512_224::digest_size == 28 && sha512_256::digest_size == 32);
static_assert(sha256::block_size == 64 && sha512::block_size == 128 && sha384::block_size == 128);
static_assert(hash_function<sha224> && hash_function<sha256> && hash_function<sha384> && hash_function<sha512>);
static_assert(hash_function<sha512_224> && hash_function<sha512_256>);
static_assert(std::is_same_v<decltype(sha256::hash("")), digest<32>>);
static_assert(noexcept(sha256::hash("")) && noexcept(sha512{}.update(std::string_view{})) && noexcept(sha512{}.finish()));

// Layout contract: a hasher is its state, one block of buffer and two counters, trivially copyable
static_assert(std::is_trivially_copyable_v<sha256> && std::is_trivially_copyable_v<sha512>);
static_assert(sizeof(sha256) == 8 * 4 + 64 + sizeof(std::size_t) + 8);
static_assert(sizeof(sha512) == 8 * 8 + 128 + sizeof(std::size_t) + 8);

TEST(Sha2Vectors, FipsExamplesAtRunTime)
{
    EXPECT_EQ(sha256::hash(std::string("abc")).hex().view(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha512_256::hash(std::string(two_blocks_512)).hex().view(),
              "3928e184fb8690f840da3988121d31be65cb9d3ef83ee6146feac861e19b563a");
}

TEST(Sha2Vectors, OneMillionAs)
{
    const std::string a(1000000, 'a');
    EXPECT_EQ(sha224::hash(a).hex().view(), "20794655980c91d8bbb4c1ea97618a4bf03f42581948b2ee4ee7ad67");
    EXPECT_EQ(sha256::hash(a).hex().view(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    EXPECT_EQ(sha384::hash(a).hex().view(),
              "9d0e1809716474cb086e834e310a4a1ced149e9c00f248527972cec5704c2a5b07b8b3dc38ecc4ebae97ddd87f3d8985");
    EXPECT_EQ(sha512::hash(a).hex().view(),
              "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31b"
              "eb009c5c2c49aa2e4eadb217ad8cc09b");
}

template <typename H>
void expect_nist(const std::string& prefix)
{
    std::size_t n = 0;
    for (const auto& m : test::read_nist_short(prefix + "ShortMsg.rsp")) {
        ASSERT_EQ(hex(H::hash(m.message).bytes), m.digest) << prefix << " ShortMsg length " << m.message.size();
        ++n;
    }
    EXPECT_GE(n, 65u);
    const auto seed = test::unhex(test::read_rsp_field(prefix + "Monte.rsp", "Seed"));
    EXPECT_EQ(test::sha2_monte<H>(seed), test::read_monte_expected(prefix + "Monte.rsp", "MD")) << prefix << " Monte";
}

TEST(Sha2Vectors, NistShortMessagesAndMonteCarlo)
{
    expect_nist<sha224>("SHA224");
    expect_nist<sha256>("SHA256");
    expect_nist<sha384>("SHA384");
    expect_nist<sha512>("SHA512");
    expect_nist<sha512_224>("SHA512_224");
    expect_nist<sha512_256>("SHA512_256");
}

template <typename H>
void expect_oracle(const char* name)
{
    for (const auto& c : test::read_oracle("oracle_sha.txt", name))
        ASSERT_EQ(hex(H::hash(test::oracle_message(c.length)).bytes), c.hex) << name << " line " << c.line;
}

TEST(Sha2Oracle, PythonHashlibAtBlockBoundariesAndRandomLengths)
{
    expect_oracle<sha224>("sha224");
    expect_oracle<sha256>("sha256");
    expect_oracle<sha384>("sha384");
    expect_oracle<sha512>("sha512");
    expect_oracle<sha512_224>("sha512_224");
    expect_oracle<sha512_256>("sha512_256");
}

template <typename H>
void expect_every_split()
{
    for (const std::size_t n : {0u, 1u, 55u, 56u, 63u, 64u, 65u, 111u, 112u, 127u, 128u, 129u, 255u, 257u}) {
        const auto m = test::oracle_message(n);
        const auto whole = H::hash(m);
        for (std::size_t a = 0; a <= n; ++a)
            for (std::size_t b = a; b <= n; b += (n > 64 ? 7 : 1))
                ASSERT_EQ(test::hash_in_parts<H>(m, a, b), whole) << "n=" << n << " a=" << a << " b=" << b;
        ASSERT_EQ(test::hash_bytewise<H>(m), whole) << "n=" << n;
    }
}

TEST(Sha2Incremental, EverySplitGivesTheOneShotDigest)
{
    expect_every_split<sha224>();
    expect_every_split<sha256>();
    expect_every_split<sha384>();
    expect_every_split<sha512>();
    expect_every_split<sha512_224>();
    expect_every_split<sha512_256>();
}

TEST(Sha2Input, EveryCarrierOfTheSameBytesGivesTheSameDigest)
{
    const auto expected = sha256::hash("abc");
    const std::string s = "abc";
    const char* p = "abc";
    const std::vector<std::uint8_t> v{'a', 'b', 'c'};
    const std::array<std::byte, 3> a{std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    EXPECT_EQ(sha256::hash(s), expected);
    EXPECT_EQ(sha256::hash(std::string_view{s}), expected);
    EXPECT_EQ(sha256::hash(p), expected);
    EXPECT_EQ(sha256::hash(v), expected);
    EXPECT_EQ(sha256::hash(a), expected);
    EXPECT_EQ(sha256::hash(std::span<const std::byte>(a)), expected);
    EXPECT_EQ(sha256::hash(u8"abc"), expected);
    EXPECT_EQ(sha256::hash(static_cast<const char*>(nullptr)), sha256::hash(""));   // a null C string is empty

    // A digest is an input: Bitcoin's double SHA-256
    const auto once = sha256::hash("hello");
    EXPECT_EQ(sha256::hash(once).hex().view(), "9595c9df90075148eb06860365df33584b75bff782a510c6cd4883a419833d50");
}

TEST(Sha2Incremental, UpdateChainsAndACopyForksTheComputation)
{
    sha256 h;
    h.update("ab");
    sha256 fork = h;
    EXPECT_EQ(h.update("c").finish(), sha256::hash("abc"));
    EXPECT_EQ(fork.update("d").finish(), sha256::hash("abd"));
}

} // namespace
