//
//  blake3_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/blake3.hpp>: the official test_vectors.json (BLAKE3 1.5.4,
//  tests/fixtures/hash/blake3_test_vectors.json, unmodified): hash, keyed hash and derive key, 131 extended output
//  bytes each, at every published input length from 0 to 102,400; the official `blake3` Python package on random
//  lengths; every split; seeking into the output; the key error; compile-time digests.
//

#include <snicholls/core/hash/blake3.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls;
using snicholls::test::hex;

static_assert(blake3::hash("").hex().view() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");
static_assert(blake3::hash("abc").hex().view() == "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85");
static_assert(blake3::keyed_hash("whats the Elvish word for friend", "").hex().view() ==
              "92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26");
static_assert(blake3::derive_key("BLAKE3 2019-12-27 16:29:52 test vectors context", "").hex().view() ==
              "2cc39783c223154fea8dfb7c1b1660f2ac2dcbd1c1de8277b0b0dd39b7e50d7d");
static_assert(blake3::hash("").first<8>() == blake3{}.finish<64>().first<8>());
static_assert(hash_function<blake3> && blake3::digest_size == 32 && blake3::block_size == 64);
static_assert(std::is_trivially_copyable_v<blake3>);
static_assert(noexcept(blake3::hash("")) && noexcept(blake3::derive_key("c", "m")) && !noexcept(blake3{std::string_view{}}));

struct official_case
{
    std::size_t input_len = 0;
    std::string hash, keyed_hash, derive_key;
};

std::string json_string(const std::string& text, const std::string& field, std::size_t& at)
{
    const std::size_t k = text.find("\"" + field + "\"", at);
    const std::size_t open = text.find('"', text.find(':', k) + 1);
    const std::size_t close = text.find('"', open + 1);
    at = close + 1;
    return text.substr(open + 1, close - open - 1);
}

std::vector<official_case> read_official()
{
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/hash/blake3_test_vectors.json", std::ios::binary);
    if (!in) throw std::runtime_error("blake3_test_vectors.json not found");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::vector<official_case> out;
    for (std::size_t at = text.find("\"cases\""); (at = text.find("\"input_len\"", at)) != std::string::npos;) {
        official_case c;
        c.input_len = std::stoull(text.substr(text.find(':', at) + 1));
        c.hash = json_string(text, "hash", at);
        c.keyed_hash = json_string(text, "keyed_hash", at);
        c.derive_key = json_string(text, "derive_key", at);
        out.push_back(std::move(c));
    }
    return out;
}

std::vector<std::byte> official_input(std::size_t n)
{
    std::vector<std::byte> in(n);
    for (std::size_t i = 0; i < n; ++i)
        in[i] = static_cast<std::byte>(i % 251);
    return in;
}

TEST(Blake3Vectors, OfficialTestVectorsInEveryModeWithExtendedOutput)
{
    const auto cases = read_official();
    ASSERT_EQ(cases.size(), 35u);
    for (const auto& c : cases) {
        const auto in = official_input(c.input_len);
        std::vector<std::byte> out(131);

        blake3 h;
        h.update(in);
        h.finish_into(out);
        ASSERT_EQ(hex(out), c.hash) << "hash, length " << c.input_len;
        ASSERT_EQ(hex(h.finish().bytes), c.hash.substr(0, 64));

        blake3 k{"whats the Elvish word for friend"};
        k.update(in);
        k.finish_into(out);
        ASSERT_EQ(hex(out), c.keyed_hash) << "keyed_hash, length " << c.input_len;

        auto d = blake3::key_derivation("BLAKE3 2019-12-27 16:29:52 test vectors context");
        d.update(in);
        d.finish_into(out);
        ASSERT_EQ(hex(out), c.derive_key) << "derive_key, length " << c.input_len;

        // Seeking: any window of the stream is the same bytes
        for (const std::size_t seek : {1u, 63u, 64u, 65u, 100u}) {
            std::vector<std::byte> tail(131 - seek);
            h.finish_into(tail, seek);
            ASSERT_EQ(hex(tail), c.hash.substr(2 * seek)) << "seek " << seek << ", length " << c.input_len;
        }
    }
}

TEST(Blake3Oracle, OfficialPythonPackageOnRandomLengths)
{
    for (const auto& c : test::read_oracle("oracle_blake3.txt", "blake3"))
        ASSERT_EQ(hex(blake3::hash(test::oracle_message(c.length)).bytes), c.hex) << "line " << c.line;
    for (const auto& c : test::read_oracle("oracle_blake3.txt", "blake3_xof")) {
        blake3 h;
        h.update(test::oracle_message(c.length));
        std::vector<std::byte> out(c.params.at(0));
        h.finish_into(out);
        ASSERT_EQ(hex(out), c.hex) << "line " << c.line;
    }
    const auto key = test::oracle_key(32);
    for (const auto& c : test::read_oracle("oracle_blake3.txt", "blake3_keyed"))
        ASSERT_EQ(hex(blake3::keyed_hash(key, test::oracle_message(c.length)).bytes), c.hex) << "line " << c.line;
    for (const auto& c : test::read_oracle("oracle_blake3.txt", "blake3_derive")) {
        auto d = blake3::key_derivation("snicholls-core 2026-11-20 test vectors derive_key");
        d.update(test::oracle_message(c.length));
        ASSERT_EQ(hex(d.finish<48>().bytes), c.hex) << "line " << c.line;
    }
}

TEST(Blake3Incremental, EverySplitGivesTheOneShotDigest)
{
    for (const std::size_t n : {0u, 1u, 63u, 64u, 65u, 1023u, 1024u, 1025u, 2048u, 2049u, 3073u, 5000u}) {
        const auto m = test::oracle_message(n);
        const auto whole = blake3::hash(m);
        const std::size_t step = n > 1100 ? 97 : 13;
        for (std::size_t a = 0; a <= n; a += step)
            for (std::size_t b = a; b <= n; b += step + 4)
                ASSERT_EQ(test::hash_in_parts<blake3>(m, a, b), whole) << "n=" << n << " a=" << a << " b=" << b;
        if (n < 2100) ASSERT_EQ(test::hash_bytewise<blake3>(m), whole);
    }
}

TEST(Blake3Incremental, FinishIsConstSoAHasherCanContinue)
{
    blake3 h;
    h.update("ab");
    EXPECT_EQ(h.finish(), blake3::hash("ab"));
    h.update("c");
    EXPECT_EQ(h.finish(), blake3::hash("abc"));
}

TEST(Blake3Errors, AKeyThatIsNotThirtyTwoBytesThrowsLengthError)
{
    EXPECT_THROW(blake3{std::string(31, 'k')}, std::length_error);
    EXPECT_THROW((void)blake3::keyed_hash(std::string(33, 'k'), "m"), std::length_error);
    EXPECT_NO_THROW(blake3{std::string(32, 'k')});
}

} // namespace
