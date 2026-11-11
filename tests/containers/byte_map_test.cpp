//
//  byte_map_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/byte_map.hpp>: keys of every length from 0 to 255 bytes, hits and
//  misses that share a head or a length, the batched probe and find, the reverse table, every error with the map
//  unchanged after it, the BPE fixture's vocabulary against bpe_encode_piece, a large random set against
//  std::unordered_map, and building and looking up at compile time.
//

#include <snicholls/core/containers/byte_map.hpp>
#include <snicholls/core/ml/bpe.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace std::string_view_literals;
namespace containers = snicholls::containers;
using containers::byte_map;
using containers::byte_map_error;
using containers::byte_map_no_value;
using containers::byte_map_view;

// The layout and the compile-time build
static_assert(sizeof(containers::byte_map_slot) == 16);
static_assert(byte_map_no_value == snicholls::ml::bpe_no_rank);
static_assert(byte_map_view::required_slots(0) == 16 && byte_map_view::required_slots(8) == 16 &&
              byte_map_view::required_slots(9) == 32 && byte_map_view::required_slots(100'256) == 262'144);

constexpr std::array<std::pair<std::string_view, std::uint32_t>, 6> tiny{
    {{"a", 0}, {"b", 1}, {"ab", 2}, {"abab", 3}, {"ba", 4}, {"a longer key than eight bytes", 5}}};
static_assert(byte_map_view::required_arena(tiny) == 1 + 1 + 2 + 4 + 2 + 29);

constexpr byte_map<16, 64, 8> compiled{tiny};
static_assert(compiled("ab"sv) == 2 && compiled("abab"sv) == 3 && compiled("a longer key than eight bytes"sv) == 5);
static_assert(compiled("aba"sv) == byte_map_no_value && compiled(""sv) == byte_map_no_value &&
              compiled("a longer key than eight bytez"sv) == byte_map_no_value);
static_assert(compiled.key_of(4) == "ba"sv && compiled.key_of(6).empty() && compiled.key_of(99).empty());
static_assert(compiled.size() == 6 && compiled.capacity() == 16 && compiled.arena_used() == 39);
static_assert(compiled.find("ba"sv, compiled.probe("ba"sv)) == 4);
static_assert(snicholls::ml::bpe_vocabulary<const byte_map<16, 64, 8>>);
static_assert(snicholls::ml::bpe_vocabulary<const byte_map_view>);

// A view over arrays at compile time: the same answers, and the errors as values
constexpr bool view_at_compile_time()
{
    std::array<containers::byte_map_slot, 16> slots{};
    std::array<char, 16> arena{};
    byte_map_view m(slots, arena);
    if (m.insert("abc", 7).is_err() || m.insert("abc", 8).err() != byte_map_error::duplicate)
        return false;
    return m("abc"sv) == 7 && m("ab"sv) == byte_map_no_value && m.size() == 1;
}
static_assert(view_at_compile_time());

// The encoder of a piece, with the map as its vocabulary, at compile time
constexpr bool encodes_at_compile_time()
{
    std::array<std::uint32_t, 8> out{};
    std::array<snicholls::ml::bpe_part, snicholls::ml::bpe_scratch_size(6)> scratch{};
    const auto r = snicholls::ml::bpe_encode_piece("ababba", compiled, out, scratch);
    return r.unwrap() == 2 && out[0] == 3 && out[1] == 4;   // a b a b b a -> [ab]... -> [abab][b][a] -> [abab][ba]
}
static_assert(encodes_at_compile_time());

// Storage for a run-time view, sized for n keys of up to `bytes` bytes in all
struct storage {
    std::vector<containers::byte_map_slot> slots;
    std::vector<char> arena;
    std::vector<containers::byte_map_key_ref> by_value;
    byte_map_view view;
    storage(std::size_t n, std::size_t bytes, std::size_t values = 0)
        : slots(byte_map_view::required_slots(n)), arena(bytes), by_value(values), view(slots, arena, by_value)
    {
    }
};

TEST(ContainersByteMap, EveryKeyLengthFrom0To255IsFoundAndItsNeighboursAreNot)
{
    // Keys "", "x", "xx", ... 255 x's, plus the same lengths of y's: equal lengths, heads that differ, and heads
    // that agree (past eight bytes) with tails that differ in the last byte
    storage s(600, 2 * 256 * 256);
    std::vector<std::string> keys;
    for (std::size_t n = 0; n <= 255; ++n) {
        keys.emplace_back(n, 'x');
        if (n > 0) {
            std::string y(n, 'x');
            y.back() = 'y';
            keys.push_back(y);
        }
    }
    for (std::uint32_t i = 0; i < keys.size(); ++i)
        ASSERT_TRUE(s.view.insert(keys[i], i).is_ok()) << "key of " << keys[i].size() << " bytes";
    EXPECT_EQ(s.view.size(), keys.size());
    for (std::uint32_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(s.view(keys[i]), i);
        const auto pr = s.view.probe(keys[i]);
        EXPECT_EQ(s.view.find(keys[i], pr), i);
    }
    for (std::size_t n = 1; n <= 255; ++n) {   // the same length, a byte changed in the middle and at the end
        std::string z(n, 'x');
        z[n / 2] = 'z';
        EXPECT_FALSE(s.view.contains(z)) << n;
        z.back() = 'z';
        EXPECT_FALSE(s.view.contains(z)) << n;
    }
    EXPECT_FALSE(s.view.contains(std::string(256, 'x')));   // longer than any key
}

TEST(ContainersByteMap, KeysWithZeroBytesAndHighBytesAreTheirOwn)
{
    storage s(8, 64);
    const std::string zero1("\0", 1), zero2("\0\0", 2), hi("\xff\xfe\x80", 3);
    ASSERT_TRUE(s.view.insert(zero1, 1).is_ok());
    ASSERT_TRUE(s.view.insert(zero2, 2).is_ok());
    ASSERT_TRUE(s.view.insert(hi, 3).is_ok());
    EXPECT_EQ(s.view(zero1), 1u);
    EXPECT_EQ(s.view(zero2), 2u);
    EXPECT_EQ(s.view(hi), 3u);
    EXPECT_EQ(s.view(""sv), byte_map_no_value);   // the empty key's head is zero too, but its length is not 1
    EXPECT_EQ(s.view(std::string("\0\0\0", 3)), byte_map_no_value);
}

TEST(ContainersByteMap, TheEmptyKeyIsAKeyLikeAnyOther)
{
    storage s(4, 8);
    ASSERT_TRUE(s.view.insert("", 9).is_ok());
    EXPECT_EQ(s.view(""sv), 9u);
    EXPECT_EQ(s.view.insert("", 10).err(), byte_map_error::duplicate);
}

TEST(ContainersByteMap, TheReverseTableGivesEachValuesKey)
{
    storage s(4, 38, 8);   // 4 + 5 + 29 bytes, exactly
    ASSERT_TRUE(s.view.build(std::array<std::pair<std::string_view, std::uint32_t>, 3>{{{"zero", 0}, {"seven", 7}, {"a long key, longer than eight", 3}}}).is_ok());
    EXPECT_EQ(s.view.key_of(0), "zero");
    EXPECT_EQ(s.view.key_of(7), "seven");
    EXPECT_EQ(s.view.key_of(3), "a long key, longer than eight");
    EXPECT_TRUE(s.view.key_of(1).empty());
    EXPECT_TRUE(s.view.key_of(8).empty());   // past the table
    storage none(4, 32);
    ASSERT_TRUE(none.view.insert("k", 5).is_ok());
    EXPECT_TRUE(none.view.key_of(5).empty());   // no reverse table
}

TEST(ContainersByteMap, EveryErrorLeavesTheMapAsItWas)
{
    storage s(2, 8, 4);   // 16 slots, so 8 keys at most; 8 arena bytes; values 0 .. 3
    ASSERT_TRUE(s.view.insert("abc", 0).is_ok());
    const auto same = [&] { return s.view.size() == 1 && s.view.arena_used() == 3 && s.view("abc"sv) == 0 && s.view.key_of(0) == "abc"; };

    EXPECT_EQ(s.view.insert("abc", 1).err(), byte_map_error::duplicate);   // the key
    EXPECT_TRUE(same());
    EXPECT_EQ(s.view.insert("xyz", 0).err(), byte_map_error::duplicate);   // the value
    EXPECT_TRUE(same());
    EXPECT_FALSE(s.view.contains("xyz"));
    EXPECT_EQ(s.view.insert(std::string(256, 'k'), 1).err(), byte_map_error::key_too_long);
    EXPECT_TRUE(same());
    EXPECT_EQ(s.view.insert("toolong", 1).err(), byte_map_error::arena_full);   // 3 + 7 > 8
    EXPECT_TRUE(same());
    EXPECT_FALSE(s.view.contains("toolong"));
    EXPECT_EQ(s.view.insert("k", 4).err(), byte_map_error::bad_value);   // past the reverse table
    EXPECT_EQ(s.view.insert("k", byte_map_no_value).err(), byte_map_error::bad_value);
    EXPECT_TRUE(same());

    storage full(8, 64);   // 16 slots: the ninth key would make it more than half full
    for (std::uint32_t i = 0; i < 8; ++i)
        ASSERT_TRUE(full.view.insert(std::string(1, static_cast<char>('a' + i)), i).is_ok());
    EXPECT_EQ(full.view.insert("i", 8).err(), byte_map_error::too_full);
    EXPECT_EQ(full.view.size(), 8u);
    EXPECT_FALSE(full.view.contains("i"));
}

TEST(ContainersByteMap, BuildStopsAtTheFirstFailureAndKeepsWhatCameBefore)
{
    storage s(8, 64);
    const std::array<std::pair<std::string_view, std::uint32_t>, 4> entries{{{"a", 1}, {"b", 2}, {"a", 3}, {"c", 4}}};
    EXPECT_EQ(s.view.build(entries).err(), byte_map_error::duplicate);
    EXPECT_EQ(s.view.size(), 2u);
    EXPECT_EQ(s.view("b"sv), 2u);
    EXPECT_FALSE(s.view.contains("c"));
}

TEST(ContainersByteMap, AViewWithNoStorageMissesAndRefuses)
{
    byte_map_view m;
    EXPECT_EQ(m("a"sv), byte_map_no_value);
    EXPECT_EQ(m.find("a", m.probe("a")), byte_map_no_value);
    EXPECT_EQ(m.insert("a", 1).err(), byte_map_error::too_full);
    EXPECT_EQ(m.capacity(), 0u);
}

TEST(ContainersByteMap, OnlyThePowerOfTwoPartOfTheSlotsIsUsed)
{
    std::vector<containers::byte_map_slot> slots(100);
    std::vector<char> arena(16);
    byte_map_view m(slots, arena);
    EXPECT_EQ(m.capacity(), 64u);
}

TEST(ContainersByteMap, ConstructingAViewClearsItsStorage)
{
    std::vector<containers::byte_map_slot> slots(16);
    std::vector<char> arena(16);
    std::vector<containers::byte_map_key_ref> by_value(4);
    {
        byte_map_view m(slots, arena, by_value);
        ASSERT_TRUE(m.insert("abc", 2).is_ok());
    }
    byte_map_view again(slots, arena, by_value);
    EXPECT_FALSE(again.contains("abc"));
    EXPECT_TRUE(again.key_of(2).empty());
}

TEST(ContainersByteMap, ACopyOfAnOwningMapIsAMapOfItsOwn)
{
    byte_map<16, 32, 4> a;
    ASSERT_TRUE(a.insert("one", 1).is_ok());
    auto b = a;
    ASSERT_TRUE(b.insert("two", 2).is_ok());
    EXPECT_EQ(a("one"sv), 1u);
    EXPECT_FALSE(a.contains("two"));
    EXPECT_EQ(b("one"sv), 1u);
    EXPECT_EQ(b("two"sv), 2u);
    EXPECT_EQ(b.key_of(2), "two");
}

std::string unbase64(std::string_view in)
{
    if (in == "-")
        return {};
    static constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::uint32_t acc = 0;
    int bits = 0;
    for (const char c : in) {
        if (c == '=')
            break;
        acc = (acc << 6) | static_cast<std::uint32_t>(alphabet.find(c));
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((acc >> bits) & 0xff));
        }
    }
    return out;
}

TEST(ContainersByteMap, TheBpeFixturesVocabularyEncodesEveryPieceAsTiktokenDoes)
{
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/bpe_fixture.txt");
    ASSERT_TRUE(in.good()) << "missing tests/fixtures/ml/bpe_fixture.txt";
    std::vector<std::pair<std::string, std::uint32_t>> vocab;
    std::vector<std::pair<std::string, std::vector<std::uint32_t>>> pieces;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ls(line);
        std::string kind, b;
        ls >> kind >> b;
        std::vector<std::uint32_t> nums;
        for (std::uint32_t v; ls >> v;)
            nums.push_back(v);
        if (kind == "vocab")
            vocab.emplace_back(unbase64(b), nums.at(0));
        else if (kind == "piece")
            pieces.emplace_back(unbase64(b), std::move(nums));
    }
    ASSERT_EQ(vocab.size(), 512u);
    storage s(vocab.size(), byte_map_view::required_arena(vocab), vocab.size());
    ASSERT_TRUE(s.view.build(vocab).is_ok());
    for (const auto& [bytes, r] : vocab) {
        ASSERT_EQ(s.view(bytes), r);
        ASSERT_EQ(s.view.key_of(r), bytes);
    }
    ASSERT_GT(pieces.size(), 300u);
    for (const auto& [piece, want] : pieces) {
        std::vector<std::uint32_t> out(piece.size());
        std::vector<snicholls::ml::bpe_part> scratch(snicholls::ml::bpe_scratch_size(piece.size()));
        const auto r = snicholls::ml::bpe_encode_piece(piece, s.view, out, scratch);
        ASSERT_TRUE(r.is_ok());
        out.resize(r.unwrap());
        EXPECT_EQ(out, want) << "piece '" << piece << "'";
    }
}

TEST(ContainersByteMap, ARandomHundredThousandKeysAgreeWithUnorderedMap)
{
    // Lengths skewed short, as a vocabulary is; queries half hits, half random strings, in batches of 32
    std::mt19937_64 rng(20261111);
    std::unordered_map<std::string, std::uint32_t> want;
    std::vector<std::string> keys;
    while (keys.size() < 100'000) {
        std::string k(1 + rng() % (rng() % 4 == 0 ? 40 : 10), ' ');
        for (char& c : k)
            c = static_cast<char>(rng() % 256);
        if (want.emplace(k, static_cast<std::uint32_t>(keys.size())).second)
            keys.push_back(k);
    }
    storage s(keys.size(), byte_map_view::required_arena(keys), keys.size());
    for (std::uint32_t i = 0; i < keys.size(); ++i)
        ASSERT_TRUE(s.view.insert(keys[i], i).is_ok());
    EXPECT_EQ(s.view.capacity(), 262'144u);
    std::vector<std::string> queries;
    for (std::size_t i = 0; i < 200'000; ++i) {
        if (i % 2 == 0) {
            queries.push_back(keys[rng() % keys.size()]);
        } else {
            std::string q(1 + rng() % 12, ' ');
            for (char& c : q)
                c = static_cast<char>(rng() % 256);
            queries.push_back(q);
        }
    }
    std::array<containers::byte_map_probe, 32> probes{};
    for (std::size_t b = 0; b < queries.size(); b += 32) {
        for (std::size_t i = 0; i < 32; ++i)
            probes[i] = s.view.probe(queries[b + i]);
        for (std::size_t i = 0; i < 32; ++i) {
            const auto it = want.find(queries[b + i]);
            ASSERT_EQ(s.view.find(queries[b + i], probes[i]), it == want.end() ? byte_map_no_value : it->second);
        }
    }
}

} // namespace
