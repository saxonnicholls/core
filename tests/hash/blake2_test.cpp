//
//  blake2_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/blake2.hpp>: RFC 7693's Appendix A and B examples and its Appendix E self-test
//  (every output length class, keyed and unkeyed, folded into one hash), Python's hashlib keyed and unkeyed with
//  every output length on random inputs, every split, the key-length error, compile-time digests.
//

#include <snicholls/core/hash/blake2.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace snicholls;
using snicholls::test::hex;

// RFC 7693 Appendices A and B
static_assert(blake2b_512::hash("abc").hex().view() ==
              "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923");
static_assert(blake2s_256::hash("abc").hex().view() == "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982");
static_assert(blake2b_256::hash("").hex().view() == "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8");
static_assert(blake2s_256::keyed_hash("", "abc") == blake2s_256::hash("abc"));   // an empty key is the unkeyed hash

static_assert(blake2b_512::block_size == 128 && blake2s_256::block_size == 64);
static_assert(blake2b<20>::digest_size == 20 && blake2b_512::max_key_size == 64 && blake2s_256::max_key_size == 32);
static_assert(hash_function<blake2b<1>> && hash_function<blake2b_512> && hash_function<blake2s<16>>);
static_assert(std::is_trivially_copyable_v<blake2b_512> && std::is_trivially_copyable_v<blake2s_256>);
static_assert(noexcept(blake2b_512::hash("")) && !noexcept(blake2b_512{std::string_view{}}));

// RFC 7693 Appendix E: the deterministic sequence of the self-test
std::vector<std::byte> selftest_seq(std::size_t len, std::uint32_t seed)
{
    std::vector<std::byte> out(len);
    std::uint32_t a = 0xDEAD4BADu * seed, b = 1;
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint32_t t = a + b;
        a = b;
        b = t;
        out[i] = static_cast<std::byte>(t >> 24);
    }
    return out;
}

template <template <std::size_t> class B, std::size_t... Out>
std::string self_test(std::initializer_list<std::size_t> in_lengths, std::index_sequence<Out...>)
{
    B<32> grand;
    const auto one = [&]<std::size_t N>() {
        for (const std::size_t in_len : in_lengths) {
            const auto in = selftest_seq(in_len, static_cast<std::uint32_t>(in_len));
            grand.update(B<N>::hash(in));
            grand.update(B<N>::keyed_hash(selftest_seq(N, N), in));
        }
    };
    (one.template operator()<Out>(), ...);
    return std::string(grand.finish().hex().view());
}

TEST(Blake2Vectors, Rfc7693SelfTest)
{
    EXPECT_EQ((self_test<blake2b>({0, 3, 128, 129, 255, 1024}, std::index_sequence<20, 32, 48, 64>{})),
              "c23a7800d98123bd10f506c61e29da5603d763b8bbad2e737f5e765a7bccd475");
    EXPECT_EQ((self_test<blake2s>({0, 3, 64, 65, 255, 1024}, std::index_sequence<16, 20, 28, 32>{})),
              "6a411f08ce25adcdfb02aba641451cec53c598b24f4fc787fbdc88797f4c1dfe");
}

TEST(Blake2Vectors, ReferenceKeyedVectors)
{
    // blake2b-kat.txt and blake2s-kat.txt of the reference: key 00..3f (00..1f), input 00..ff for lengths 0 and 255
    std::vector<std::uint8_t> key(64), in(255);
    for (std::size_t i = 0; i < key.size(); ++i) key[i] = static_cast<std::uint8_t>(i);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<std::uint8_t>(i);
    EXPECT_EQ(blake2b_512::keyed_hash(key, std::span<const std::uint8_t>(in).first(0)).hex().view(),
              "10ebb67700b1868efb4417987acf4690ae9d972fb7a590c2f02871799aaa4786b5e996e8f0f4eb981fc214b005f42d2ff4233499391653df7aefcbc13fc51568");
    EXPECT_EQ(blake2b_512::keyed_hash(key, in).hex().view(),
              "142709d62e28fcccd0af97fad0f8465b971e82201dc51070faa0372aa43e92484be1c1e73ba10906d5d1853db6a4106e0a7bf9800d373d6dee2d46d62ef2a461");
    const std::vector<std::uint8_t> key32(key.begin(), key.begin() + 32);
    EXPECT_EQ(blake2s_256::keyed_hash(key32, std::span<const std::uint8_t>(in).first(0)).hex().view(),
              "48a8997da407876b3d79c0d92325ad3b89cbb754d86ab71aee047ad345fd2c49");
    EXPECT_EQ(blake2s_256::keyed_hash(key32, in).hex().view(), "3fb735061abc519dfe979e54c1ee5bfad0a9d858b3315bad34bde999efd724dd");
}

template <typename B>
void expect_plain(const char* name)
{
    for (const auto& c : test::read_oracle("oracle_blake2.txt", name))
        ASSERT_EQ(hex(B::hash(test::oracle_message(c.length)).bytes), c.hex) << name << " line " << c.line;
}

// The output length is a template argument, so each oracle line's length picks its instantiation
template <template <std::size_t> class B, std::size_t... N>
std::string digest_with_length(std::size_t length, const std::vector<std::byte>& key, const std::vector<std::byte>& in,
                               std::index_sequence<N...>)
{
    std::string out;
    ((length == N + 1 ? (out = hex(B<N + 1>::keyed_hash(key, in).bytes), 0) : 0), ...);
    return out;
}

TEST(Blake2Oracle, PythonHashlibKeyedAndUnkeyedWithEveryOutputLength)
{
    expect_plain<blake2b_512>("blake2b");
    expect_plain<blake2s_256>("blake2s");
    const std::vector<std::byte> no_key;
    for (const auto& c : test::read_oracle("oracle_blake2.txt", "blake2b_out"))
        ASSERT_EQ(digest_with_length<blake2b>(c.params.at(0), no_key, test::oracle_message(c.length), std::make_index_sequence<64>{}), c.hex)
            << "line " << c.line;
    for (const auto& c : test::read_oracle("oracle_blake2.txt", "blake2s_out"))
        ASSERT_EQ(digest_with_length<blake2s>(c.params.at(0), no_key, test::oracle_message(c.length), std::make_index_sequence<32>{}), c.hex)
            << "line " << c.line;
    for (const auto& c : test::read_oracle("oracle_blake2.txt", "blake2b_key"))
        ASSERT_EQ(hex(blake2b_512::keyed_hash(test::oracle_key(c.params.at(0)), test::oracle_message(c.length)).bytes), c.hex)
            << "line " << c.line;
    for (const auto& c : test::read_oracle("oracle_blake2.txt", "blake2s_key"))
        ASSERT_EQ(hex(blake2s_256::keyed_hash(test::oracle_key(c.params.at(0)), test::oracle_message(c.length)).bytes), c.hex)
            << "line " << c.line;
}

template <typename B>
void expect_every_split(const std::vector<std::byte>& key)
{
    for (const std::size_t n : {0u, 1u, 63u, 64u, 65u, 127u, 128u, 129u, 256u, 300u}) {
        const auto m = test::oracle_message(n);
        const auto whole = B::keyed_hash(key, m);
        for (std::size_t a = 0; a <= n; a += 3)
            for (std::size_t b = a; b <= n; b += 7) {
                B h{key};
                h.update(std::span<const std::byte>(m).first(a));
                h.update(std::span<const std::byte>(m).subspan(a, b - a));
                h.update(std::span<const std::byte>(m).subspan(b));
                ASSERT_EQ(h.finish(), whole) << "n=" << n << " a=" << a << " b=" << b;
            }
    }
}

TEST(Blake2Incremental, EverySplitGivesTheOneShotDigestKeyedOrNot)
{
    expect_every_split<blake2b_512>({});
    expect_every_split<blake2b_512>(test::oracle_key(64));
    expect_every_split<blake2s_256>({});
    expect_every_split<blake2s_256>(test::oracle_key(17));
}

TEST(Blake2Errors, AKeyLongerThanTheVariantAllowsThrowsLengthError)
{
    EXPECT_THROW(blake2b_512{std::string(65, 'k')}, std::length_error);
    EXPECT_THROW(blake2s_256{std::string(33, 'k')}, std::length_error);
    EXPECT_NO_THROW(blake2b_512{std::string(64, 'k')});
    EXPECT_NO_THROW(blake2s_256{std::string(32, 'k')});
}

} // namespace
