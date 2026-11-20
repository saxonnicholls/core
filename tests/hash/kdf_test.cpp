//
//  kdf_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/kdf.hpp>: RFC 5869's test cases 1 to 7 (HKDF-SHA256 and HKDF-SHA1), RFC 6070's
//  PBKDF2-HMAC-SHA1 vectors, RFC 7914's PBKDF2-HMAC-SHA256 vectors, BIP-39's first test seed (PBKDF2-HMAC-SHA512,
//  2048 iterations), Python's hashlib.pbkdf2_hmac and an HKDF written from Python's hmac on random inputs, every
//  error, compile-time derivations.
//

#include <snicholls/core/hash/kdf.hpp>
#include <snicholls/core/hash/sha1.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace snicholls;
using snicholls::test::hex;
using snicholls::test::unhex;

// Compile time: RFC 6070's second vector and RFC 5869's first
static_assert(pbkdf2_hmac<sha1, 20>("password", "salt", 2).unwrap_unchecked().hex().view() == "ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957");
static_assert(pbkdf2_hmac<sha256, 4>("p", "s", 0).unwrap_err_unchecked() == kdf_error::zero_iterations);
constexpr std::array<unsigned char, 22> rfc5869_ikm{0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b,
                                                    0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b};
static_assert(hkdf<sha256, 42>("", rfc5869_ikm, "").hex().view() ==
              "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8");
static_assert(noexcept(hkdf_extract<sha256>("s", "k")) && noexcept(pbkdf2_hmac<sha256, 32>("p", "s", 1)));

struct rfc5869_case
{
    bool sha1;
    const char *ikm, *salt, *info;
    std::size_t length;
    const char *prk, *okm;
};

// RFC 5869 Appendix A (case 7's salt is "not provided", which HKDF treats as HashLen zeros, the same as empty)
constexpr rfc5869_case rfc5869[] = {
    {false, "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b", "000102030405060708090a0b0c", "f0f1f2f3f4f5f6f7f8f9", 42,
     "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5",
     "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"},
    {false,
     "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f",
     "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
     "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
     82, "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244",
     "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71cc30c58179ec3e87c14c01d5c1f3434f1d87"},
    {false, "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b", "", "", 42, "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04",
     "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"},
    {true, "0b0b0b0b0b0b0b0b0b0b0b", "000102030405060708090a0b0c", "f0f1f2f3f4f5f6f7f8f9", 42, "9b6c18c432a7bf8f0e71c8eb88f4b30baa2ba243",
     "085a01ea1b10f36933068b56efa5ad81a4f14b822f5b091568a9cdd4f155fda2c22e422478d305f3f896"},
    {true,
     "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f",
     "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
     "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
     82, "8adae09a2a307059478d309b26c4115a224cfaf6",
     "0bd770a74d1160f7c9f12cd5912a06ebff6adcae899d92191fe4305673ba2ffe8fa3f1a4e5ad79f3f334b3b202b2173c486ea37ce3d397ed034c7f9dfeb15c5e927336d0441f4c4300e2cff0d0900b52d3b4"},
    {true, "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b", "", "", 42, "da8c8a73c7fa77288ec6f5e7c297786aa0d32d01",
     "0ac1af7002b3d761d1e55298da9d0506b9ae52057220a306e07b6b87e8df21d0ea00033de03984d34918"},
    {true, "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c", "", "", 42, "2adccada18779e7c2077ad2eb19d3f3e731385dd",
     "2c91117204d745f3500d636a62f64f0ab3bae548aa53d423b0d1f27ebba6f5e5673a081d70cce7acfc48"},
};

template <typename H>
void expect_rfc5869(const rfc5869_case& c)
{
    const auto prk = hkdf_extract<H>(unhex(c.salt), unhex(c.ikm));
    EXPECT_EQ(hex(prk.bytes), c.prk);
    std::vector<std::byte> okm(c.length);
    ASSERT_TRUE(hkdf_expand<H>(prk, unhex(c.info), okm));
    EXPECT_EQ(hex(okm), c.okm);
    std::vector<std::byte> again(c.length);
    ASSERT_TRUE(hkdf<H>(unhex(c.salt), unhex(c.ikm), unhex(c.info), again));
    EXPECT_EQ(again, okm);
}

TEST(KdfVectors, Rfc5869Hkdf)
{
    for (const auto& c : rfc5869) {
        if (c.sha1)
            expect_rfc5869<sha1>(c);
        else
            expect_rfc5869<sha256>(c);
    }
}

std::string pbkdf2_sha1(std::string_view password, std::string_view salt, std::uint32_t iterations, std::size_t length)
{
    std::vector<std::byte> out(length);
    EXPECT_TRUE(pbkdf2_hmac<sha1>(password, salt, iterations, out));
    return hex(out);
}

TEST(KdfVectors, Rfc6070Pbkdf2Sha1)
{
    EXPECT_EQ(pbkdf2_sha1("password", "salt", 1, 20), "0c60c80f961f0e71f3a9b524af6012062fe037a6");
    EXPECT_EQ(pbkdf2_sha1("password", "salt", 2, 20), "ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957");
    EXPECT_EQ(pbkdf2_sha1("password", "salt", 4096, 20), "4b007901b765489abead49d926f721d065a429c1");
    EXPECT_EQ(pbkdf2_sha1("passwordPASSWORDpassword", "saltSALTsaltSALTsaltSALTsaltSALTsalt", 4096, 25),
              "3d2eec4fe41c849b80c8d83662c0e44a8b291a964cf2f07038");
    EXPECT_EQ(pbkdf2_sha1(std::string_view("pass\0word", 9), std::string_view("sa\0lt", 5), 4096, 16), "56fa6aa75548099dcc37d7f03425e0c3");
#ifdef NDEBUG
    EXPECT_EQ(pbkdf2_sha1("password", "salt", 16777216, 20), "eefe3d61cd4da4e4e9945b3d6ba2158c2634e984");   // 33.5 million HMACs
#endif
}

TEST(KdfVectors, Rfc7914Pbkdf2Sha256)
{
    std::vector<std::byte> out(64);
    ASSERT_TRUE(pbkdf2_hmac<sha256>("passwd", "salt", 1, out));
    EXPECT_EQ(hex(out), "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783");
    ASSERT_TRUE(pbkdf2_hmac<sha256>("Password", "NaCl", 80000, out));
    EXPECT_EQ(hex(out), "4ddcd8f60b98be21830cee5ef22701f9641a4418d04c0414aeff08876b34ab56a1d425a1225833549adb841b51c9b3176a272bdebba1d078478f62b397f33c8d");
}

TEST(KdfVectors, Bip39SeedIsPbkdf2Sha512Over2048Iterations)
{
    // BIP-39's first English test vector: entropy 00...00, passphrase "TREZOR"
    const std::string mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    const auto seed = pbkdf2_hmac<sha512, 64>(mnemonic, "mnemonicTREZOR", 2048);
    ASSERT_TRUE(seed);
    EXPECT_EQ(seed.unwrap_unchecked().hex().view(),
              "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e53495531f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04");
}

TEST(KdfOracle, PythonPbkdf2AndHkdfOnRandomInputs)
{
    const auto salt = test::oracle_key(16);
    const auto run = [&]<typename H>(const char* name) {
        for (const auto& c : test::read_oracle("oracle_mac_kdf.txt", name)) {
            std::vector<std::byte> out(100);
            ASSERT_TRUE(pbkdf2_hmac<H>(test::oracle_message(c.length), salt, static_cast<std::uint32_t>(c.params.at(0)), out));
            ASSERT_EQ(hex(out), c.hex) << name << " line " << c.line;
        }
    };
    run.template operator()<sha1>("pbkdf2_sha1");
    run.template operator()<sha256>("pbkdf2_sha256");
    run.template operator()<sha512>("pbkdf2_sha512");
    for (const auto& c : test::read_oracle("oracle_mac_kdf.txt", "hkdf_sha256")) {
        std::vector<std::byte> out(c.params.at(0));
        ASSERT_TRUE(hkdf<sha256>(salt, test::oracle_message(c.length), "snicholls-core hkdf", out));
        ASSERT_EQ(hex(out), c.hex) << "line " << c.line;
    }
}

TEST(KdfErrors, BadArgumentsAreErrorsAndWriteNothing)
{
    std::vector<std::byte> out(32, std::byte{0x5a});
    EXPECT_EQ(pbkdf2_hmac<sha256>("p", "s", 0, out).unwrap_err_unchecked(), kdf_error::zero_iterations);
    EXPECT_EQ(pbkdf2_hmac<sha256>("p", "s", 1, std::span<std::byte>{}).unwrap_err_unchecked(), kdf_error::empty_output);
    EXPECT_EQ(hkdf_expand<sha256>(sha256::hash("k"), "", std::span<std::byte>{}).unwrap_err_unchecked(), kdf_error::empty_output);
    std::vector<std::byte> too_long(255 * 32 + 1);
    EXPECT_EQ(hkdf_expand<sha256>(sha256::hash("k"), "", too_long).unwrap_err_unchecked(), kdf_error::output_too_long);
    EXPECT_EQ(hkdf<sha256>("s", "k", "i", too_long).unwrap_err_unchecked(), kdf_error::output_too_long);
    for (const std::byte b : out)
        EXPECT_EQ(b, std::byte{0x5a});
    std::vector<std::byte> longest(255 * 32);
    EXPECT_TRUE(hkdf_expand<sha256>(sha256::hash("k"), "", longest));
}

} // namespace
