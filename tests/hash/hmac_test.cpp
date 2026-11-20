//
//  hmac_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/hmac.hpp>: RFC 4231's test cases 1 to 7 for HMAC-SHA-224/256/384/512 and
//  RFC 2202's for HMAC-SHA-1 (parsed from the RFC text), Python's hmac over SHA-1, SHA-256, SHA-512 and SHA3-256
//  with keys shorter than, equal to and longer than the block, every split, verify, compile-time tags.
//

#include <snicholls/core/hash/blake2.hpp>
#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/sha1.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/hash/sha3.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls;
using snicholls::test::hex;
using snicholls::test::unhex;

static_assert(hmac<sha256>::mac("Jefe", "what do ya want for nothing?").hex().view() ==
              "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
static_assert(hmac<sha256>::verify("Jefe", "what do ya want for nothing?", hmac<sha256>::mac("Jefe", "what do ya want for nothing?")));
static_assert(!hmac<sha256>::verify("Jefe", "what do ya want for nothing!", hmac<sha256>::mac("Jefe", "what do ya want for nothing?")));
static_assert(hmac<sha512>::digest_size == 64 && hmac<sha3_256>::block_size == 136 && hmac<blake2b<32>>::block_size == 128);
static_assert(std::is_same_v<hmac<sha1>::digest_type, digest<20>> && std::is_trivially_copyable_v<hmac<sha256>>);
static_assert(sizeof(hmac<sha256>) == 2 * sizeof(sha256));
static_assert(noexcept(hmac<sha512>::mac("k", "m")) && noexcept(hmac<sha512>::verify("k", "m", "t")));

struct rfc4231_case
{
    const char *key, *data, *sha224, *sha256, *sha384, *sha512;
};

// RFC 4231 section 4, test cases 1 to 7 (case 5's tags are truncated to 128 bits)
constexpr rfc4231_case rfc4231[] = {
    {"0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b", "4869205468657265", "896fb1128abbdf196832107cd49df33f47b4b1169912ba4f53684b22",
     "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
     "afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59cfaea9ea9076ede7f4af152e8b2fa9cb6",
     "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854"},
    {"4a656665", "7768617420646f2079612077616e7420666f72206e6f7468696e673f", "a30e01098bc6dbbf45690f3a7e9e6d0f8bbea2a39e6148008fd05e44",
     "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
     "af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e8e2240ca5e69e2c78b3239ecfab21649",
     "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea2505549758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737"},
    {"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
     "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
     "7fb3cb3588c6c1f6ffa9694d7d6ad2649365b0c1f65d69d1ec8333ea", "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
     "88062608d3e6ad8a0aa2ace014c8a86f0aa635d947ac9febe83ef4e55966144b2a5ab39dc13814b94e3ab6e101a34f27",
     "fa73b0089d56a284efb0f0756c890be9b1b5dbdd8ee81a3655f83e33b2279d39bf3e848279a722c806b485a47e67c807b946a337bee8942674278859e13292fb"},
    {"0102030405060708090a0b0c0d0e0f10111213141516171819",
     "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
     "6c11506874013cac6a2abc1bb382627cec6a90d86efc012de7afec5a", "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b",
     "3e8a69b7783c25851933ab6290af6ca77a9981480850009cc5577c6e1f573b4e6801dd23c4a7d679ccf8a386c674cffb",
     "b0ba465637458c6990e5a8c5f61d4af7e576d97ff94b872de76f8050361ee3dba91ca5c11aa25eb4d679275cc5788063a5f19741120c4f2de2adebeb10a298dd"},
    {"0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c", "546573742057697468205472756e636174696f6e", "0e2aea68a90c8d37c988bcdb9fca6fa8",
     "a3b6167473100ee06e0c796c2955552b", "3abf34c3503b2a23a46efc619baef897", "415fad6271580a531d4179bc891d87a6"},
    {"aa x 131",
     "54657374205573696e67204c6172676572205468616e20426c6f636b2d53697a65204b6579202d2048617368204b6579204669727374",
     "95e9a0db962095adaebe9b2d6f0dbce2d499f112f2d2b7273fa6870e", "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
     "4ece084485813e9088d2c63a041bc5b44f9ef1012a2b588f3cd11f05033ac4c60c2ef6ab4030fe8296248df163f44952",
     "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f3526b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598"},
    {"aa x 131",
     "5468697320697320612074657374207573696e672061206c6172676572207468616e20626c6f636b2d73697a65206b657920616e642061206c61726765"
     "72207468616e20626c6f636b2d73697a6520646174612e20546865206b6579206e6565647320746f20626520686173686564206265666f7265206265"
     "696e6720757365642062792074686520484d414320616c676f726974686d2e",
     "3a854166ac5d9f023f54d517d0b39dbd946770db9c2b95c9f6f565d1", "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2",
     "6617178e941f020d351e2f254e8fd32c602420feb0b8fb9adccebb82461e99c5a678cc31e799176d3860e6110c46523e",
     "e37b6a775dc87dbaa4dfa9f96e5e3ffddebd71f8867289865df5a32d20cdc944b6022cac3c4982b10d5eeb55c3e4de15134676fb6de0446065c97440fa8c6a58"},
};

template <typename H>
std::string tag(const std::string& key, const std::string& data, std::size_t hex_digits)
{
    const std::string k = key == "aa x 131" ? std::string(262, 'a') : key;   // cases 6 and 7: 131 bytes of 0xaa
    return hex(hmac<H>::mac(unhex(k), unhex(data)).bytes).substr(0, hex_digits);
}

TEST(HmacVectors, Rfc4231Sha2)
{
    for (const auto& c : rfc4231) {
        EXPECT_EQ(tag<sha224>(c.key, c.data, std::string(c.sha224).size()), c.sha224) << c.key;
        EXPECT_EQ(tag<sha256>(c.key, c.data, std::string(c.sha256).size()), c.sha256) << c.key;
        EXPECT_EQ(tag<sha384>(c.key, c.data, std::string(c.sha384).size()), c.sha384) << c.key;
        EXPECT_EQ(tag<sha512>(c.key, c.data, std::string(c.sha512).size()), c.sha512) << c.key;
    }
}

TEST(HmacVectors, Rfc2202Sha1)
{
    const std::string aa80(160, 'a');
    EXPECT_EQ(tag<sha1>("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b", "4869205468657265", 40), "b617318655057264e28bc0b6fb378c8ef146be00");
    EXPECT_EQ(hex(hmac<sha1>::mac("Jefe", "what do ya want for nothing?").bytes), "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79");
    EXPECT_EQ(tag<sha1>("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", std::string(100, 'd'), 40), "125d7342b9ac11cd91a39af48aa17b4f63f175d3");
    std::string cd;
    for (int i = 0; i < 50; ++i) cd += "cd";
    EXPECT_EQ(tag<sha1>("0102030405060708090a0b0c0d0e0f10111213141516171819", cd, 40), "4c9007f4026250c6bc8414f9bf50c86c2d7235da");
    EXPECT_EQ(hex(hmac<sha1>::mac(unhex("0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c"), "Test With Truncation").bytes),
              "4c1a03424b55e07fe7f27be1d58bb9324a9a5a04");
    EXPECT_EQ(hex(hmac<sha1>::mac(unhex(aa80), "Test Using Larger Than Block-Size Key - Hash Key First").bytes),
              "aa4ae5e15272d00e95705637ce8a3b55ed402112");
    EXPECT_EQ(hex(hmac<sha1>::mac(unhex(aa80), "Test Using Larger Than Block-Size Key and Larger Than One Block-Size Data").bytes),
              "e8e99d0f45237d786d6bbaa7965c7808bbff1a91");
}

template <typename H>
void expect_oracle(const char* name)
{
    for (const auto& c : test::read_oracle("oracle_mac_kdf.txt", name))
        ASSERT_EQ(hex(hmac<H>::mac(test::oracle_key(c.params.at(0)), test::oracle_message(c.length)).bytes), c.hex)
            << name << " line " << c.line;
}

TEST(HmacOracle, PythonHmacWithKeysAroundTheBlockSize)
{
    expect_oracle<sha1>("hmac_sha1");
    expect_oracle<sha256>("hmac_sha256");
    expect_oracle<sha512>("hmac_sha512");
    expect_oracle<sha3_256>("hmac_sha3_256");
}

TEST(HmacIncremental, EverySplitAndACopiedKeyGiveTheOneShotTag)
{
    const auto key = test::oracle_key(70);
    const hmac<sha256> keyed{key};
    for (const std::size_t n : {0u, 1u, 63u, 64u, 65u, 200u}) {
        const auto m = test::oracle_message(n);
        const auto whole = hmac<sha256>::mac(key, m);
        for (std::size_t a = 0; a <= n; a += 3) {
            hmac<sha256> h = keyed;   // the key's two compressions, reused
            h.update(std::span<const std::byte>(m).first(a));
            h.update(std::span<const std::byte>(m).subspan(a));
            ASSERT_EQ(h.finish(), whole) << "n=" << n << " a=" << a;
        }
    }
}

TEST(HmacVerify, ATruncatedOrAlteredTagIsRejected)
{
    const auto t = hmac<sha256>::mac("key", "message");
    EXPECT_TRUE(hmac<sha256>::verify("key", "message", t));
    EXPECT_FALSE(hmac<sha256>::verify("key", "message", t.first<16>()));
    auto altered = t;
    altered[31] ^= std::byte{1};
    EXPECT_FALSE(hmac<sha256>::verify("key", "message", altered));
    EXPECT_FALSE(hmac<sha256>::verify("other key", "message", t));
}

} // namespace
