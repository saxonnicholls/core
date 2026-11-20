//
//  sha3_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/sha3.hpp>: FIPS 202's examples, NIST's byte-oriented SHA-3 ShortMsg and Monte
//  Carlo vectors and SHAKE Monte Carlo vectors, Ethereum's well-known Keccak-256 values, Python's hashlib and
//  PyCryptodome on random lengths and output lengths, every split, every squeeze split, compile-time digests.
//

#include <snicholls/core/hash/sha3.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls;
using snicholls::test::hex;

static_assert(sha3_224::hash("").hex().view() == "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7");
static_assert(sha3_256::hash("").hex().view() == "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
static_assert(sha3_256::hash("abc").hex().view() == "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");
static_assert(sha3_384::hash("abc").hex().view() ==
              "ec01498288516fc926459f58e2c6ad8df9b473cb0fc08c2596da7cf0e49be4b298d88cea927ac7f539f1edf228376d25");
static_assert(sha3_512::hash("abc").hex().view() ==
              "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0");
static_assert(shake128::hash<32>("").hex().view() == "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26");
static_assert(shake256::hash<32>("").hex().view() == "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f");

// Ethereum: the empty hash, an ERC-20 selector and the Transfer event topic, as compile-time constants
static_assert(keccak256::hash("").hex().view() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
static_assert(keccak256::hash("transfer(address,uint256)").first<4>().hex().view() == "a9059cbb");
static_assert(keccak256::hash("Transfer(address,address,uint256)").hex().view() ==
              "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef");

static_assert(sha3_256::block_size == 136 && sha3_512::block_size == 72 && shake128::block_size == 168);
static_assert(hash_function<sha3_224> && hash_function<sha3_256> && hash_function<sha3_384> && hash_function<sha3_512>);
static_assert(hash_function<keccak256> && !hash_function<shake128>);
static_assert(std::is_trivially_copyable_v<keccak256> && std::is_trivially_copyable_v<shake256>);
static_assert(noexcept(keccak256::hash("")) && noexcept(shake128::hash<16>("")));

TEST(Sha3Vectors, KeccakF1600OfTheZeroState)
{
    // The first lanes of Keccak-f[1600] applied to the all-zero state (the Keccak team's KeccakF-1600 intermediate values)
    std::array<std::uint64_t, 25> a{};
    keccak_f1600(a);
    EXPECT_EQ(a[0], 0xF1258F7940E1DDE7ull);
    EXPECT_EQ(a[1], 0x84D5CCF933C0478Aull);
    EXPECT_EQ(a[24], 0xEAF1FF7B5CECA249ull);
}

TEST(Sha3Vectors, OneMillionAs)
{
    const std::string a(1000000, 'a');
    EXPECT_EQ(sha3_256::hash(a).hex().view(), "5c8875ae474a3634ba4fd55ec85bffd661f32aca75c6d699d0cdcb6c115891c1");
}

template <typename H>
void expect_nist(const std::string& prefix)
{
    for (const auto& m : test::read_nist_short(prefix + "ShortMsg.rsp"))
        ASSERT_EQ(hex(H::hash(m.message).bytes), m.digest) << prefix << " length " << m.message.size();
    const auto seed = test::unhex(test::read_rsp_field(prefix + "Monte.rsp", "Seed"));
    EXPECT_EQ(test::sha3_monte<H>(seed), test::read_monte_expected(prefix + "Monte.rsp", "MD")) << prefix;
}

TEST(Sha3Vectors, NistShortMessagesAndMonteCarlo)
{
    expect_nist<sha3_224>("SHA3_224");
    expect_nist<sha3_256>("SHA3_256");
    expect_nist<sha3_384>("SHA3_384");
    expect_nist<sha3_512>("SHA3_512");
}

// SHAKEVS 6.3: the output length of each step is chosen by the last 16 bits of the previous output
template <typename X>
void expect_shake_monte(const std::string& file)
{
    const std::size_t min_bytes = std::stoull(test::read_rsp_field(file, "Minimum Output Length (bits)")) / 8;
    const std::size_t max_bytes = std::stoull(test::read_rsp_field(file, "Maximum Output Length (bits)")) / 8;
    std::vector<std::byte> output = test::unhex(test::read_rsp_field(file, "Msg"));
    std::size_t length = max_bytes;
    std::vector<std::string> got;
    for (int j = 0; j < 100; ++j) {
        for (int i = 0; i < 1000; ++i) {
            std::vector<std::byte> msg(16);
            for (std::size_t k = 0; k < 16 && k < output.size(); ++k)
                msg[k] = output[k];
            X x;
            x.update(msg);
            output.assign(length, std::byte{});
            x.squeeze(output);
            const unsigned right = (static_cast<unsigned>(output[length - 2]) << 8) | static_cast<unsigned>(output[length - 1]);
            length = min_bytes + right % (max_bytes - min_bytes + 1);
        }
        got.push_back(hex(output));
    }
    EXPECT_EQ(got, test::read_monte_expected(file, "Output")) << file;
}

TEST(Sha3Vectors, NistShakeMonteCarlo)
{
    expect_shake_monte<shake128>("SHAKE128Monte.rsp");
    expect_shake_monte<shake256>("SHAKE256Monte.rsp");
}

template <typename H>
void expect_oracle(const char* file, const char* name)
{
    for (const auto& c : test::read_oracle(file, name))
        ASSERT_EQ(hex(H::hash(test::oracle_message(c.length)).bytes), c.hex) << name << " line " << c.line;
}

template <typename X>
void expect_xof_oracle(const char* name)
{
    for (const auto& c : test::read_oracle("oracle_shake.txt", name)) {
        X x;
        x.update(test::oracle_message(c.length));
        std::vector<std::byte> out(c.params.at(0));
        x.squeeze(out);
        ASSERT_EQ(hex(out), c.hex) << name << " line " << c.line;
    }
}

TEST(Sha3Oracle, PythonHashlibAndPyCryptodomeAtBlockBoundariesAndRandomLengths)
{
    expect_oracle<sha3_224>("oracle_sha3.txt", "sha3_224");
    expect_oracle<sha3_256>("oracle_sha3.txt", "sha3_256");
    expect_oracle<sha3_384>("oracle_sha3.txt", "sha3_384");
    expect_oracle<sha3_512>("oracle_sha3.txt", "sha3_512");
    expect_oracle<keccak256>("oracle_sha3.txt", "keccak256");
    expect_xof_oracle<shake128>("shake128");
    expect_xof_oracle<shake256>("shake256");
}

template <typename H>
void expect_every_split()
{
    for (const std::size_t n : {0u, 1u, 71u, 72u, 73u, 103u, 104u, 135u, 136u, 137u, 143u, 144u, 167u, 168u, 169u, 300u}) {
        const auto m = test::oracle_message(n);
        const auto whole = H::hash(m);
        for (std::size_t a = 0; a <= n; a += 3)
            for (std::size_t b = a; b <= n; b += 11)
                ASSERT_EQ(test::hash_in_parts<H>(m, a, b), whole) << "n=" << n << " a=" << a << " b=" << b;
        ASSERT_EQ(test::hash_bytewise<H>(m), whole);
    }
}

TEST(Sha3Incremental, EverySplitGivesTheOneShotDigest)
{
    expect_every_split<sha3_224>();
    expect_every_split<sha3_256>();
    expect_every_split<sha3_384>();
    expect_every_split<sha3_512>();
    expect_every_split<keccak256>();
}

TEST(Sha3Shake, EverySqueezeSplitGivesTheSameStream)
{
    shake128 whole_x;
    whole_x.update("stream");
    std::vector<std::byte> whole(500);
    whole_x.squeeze(whole);
    for (std::size_t a = 0; a <= 500; a += 7) {
        shake128 x;
        x.update("str");
        x.update("eam");
        std::vector<std::byte> parts(500);
        x.squeeze(std::span<std::byte>(parts).first(a));
        x.squeeze(std::span<std::byte>(parts).subspan(a));
        ASSERT_EQ(parts, whole) << "a=" << a;
    }
    // finish<N> is the first N bytes, and a shorter output is a prefix of a longer one
    EXPECT_EQ(hex(shake128::hash<32>("stream").bytes), hex(std::span<const std::byte>(whole).first(32)));
}

TEST(Sha3Keccak, EthereumAddressIsTheLastTwentyBytesOfThePublicKeyHash)
{
    // The uncompressed public key of private key 1, without its 0x04 prefix; its address is
    // 0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf
    const auto pub = test::unhex(
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");
    const auto h = keccak256::hash(pub);
    EXPECT_EQ(hex(std::span<const std::byte>(h.bytes).subspan(12)), "7e5f4552091a69125d5dfcb7b8c2659029395bdf");
}

} // namespace
