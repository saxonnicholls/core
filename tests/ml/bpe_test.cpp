//
//  bpe_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/bpe.hpp>: every piece of the committed fixture encodes to tiktoken's tokens
//  (and so rustbpe's), every text to the concatenation, and decodes back to its bytes; ties go to the
//  leftmost pair; the errors; and encoding at compile time with a vocabulary the compiler holds.
//

#include <snicholls/core/ml/bpe.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;

// A five-token vocabulary over "ab": a=0, b=1, ab=2, aba=3 is absent, abab=4. Merges by rank: ab first.
constexpr std::array<std::pair<std::string_view, std::uint32_t>, 4> tiny{{{"a", 0}, {"b", 1}, {"ab", 2}, {"abab", 3}}};
constexpr auto tiny_rank = [](std::string_view s) {
    for (const auto& [k, v] : tiny)
        if (k == s)
            return v;
    return ml::bpe_no_rank;
};

constexpr std::size_t encode_count(std::string_view s)
{
    std::array<std::uint32_t, 16> out{};
    std::array<ml::bpe_part, 17> scratch{};
    return ml::bpe_encode_piece(s, tiny_rank, out, scratch).unwrap();
}
static_assert(encode_count("") == 0 && encode_count("a") == 1 && encode_count("abab") == 1 && encode_count("ababab") == 2);
static_assert(encode_count("aab") == 2 && encode_count("ba") == 2);
static_assert(ml::bpe_scratch_size(10) == 11);

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

struct fixture {
    std::unordered_map<std::string, std::uint32_t> ranks;
    std::vector<std::string> bytes_of;
    std::vector<std::pair<std::string, std::vector<std::uint32_t>>> pieces, texts;
};

fixture load()
{
    fixture f;
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/bpe_fixture.txt");
    EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/bpe_fixture.txt";
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ls(line);
        std::string kind, b;
        ls >> kind >> b;
        std::vector<std::uint32_t> nums;
        for (std::uint32_t v; ls >> v;)
            nums.push_back(v);
        if (kind == "vocab") {
            f.ranks.emplace(unbase64(b), nums.at(0));
            if (f.bytes_of.size() <= nums[0])
                f.bytes_of.resize(nums[0] + 1);
            f.bytes_of[nums[0]] = unbase64(b);
        } else if (kind == "piece") {
            f.pieces.emplace_back(unbase64(b), std::move(nums));
        } else if (kind == "text") {
            f.texts.emplace_back(unbase64(b), std::move(nums));
        }
    }
    return f;
}

const fixture& data()
{
    static const fixture f = load();
    return f;
}

std::vector<std::uint32_t> encode(std::string_view piece)
{
    const auto& ranks = data().ranks;
    const auto rank = [&](std::string_view s) {
        const auto it = ranks.find(std::string(s));
        return it == ranks.end() ? ml::bpe_no_rank : it->second;
    };
    std::vector<std::uint32_t> out(piece.size());
    std::vector<ml::bpe_part> scratch(ml::bpe_scratch_size(piece.size()));
    const auto r = ml::bpe_encode_piece(piece, rank, out, scratch);
    EXPECT_TRUE(r.is_ok());
    out.resize(r.is_ok() ? r.unwrap() : 0);
    return out;
}

TEST(MlBpe, EveryFixturePieceEncodesToTiktokensTokens)
{
    ASSERT_EQ(data().ranks.size(), 512u);
    ASSERT_GT(data().pieces.size(), 300u);
    for (const auto& [piece, want] : data().pieces)
        EXPECT_EQ(encode(piece), want) << "piece '" << piece << "'";
}

TEST(MlBpe, EveryFixtureTextIsItsPiecesConcatenatedAndDecodesBack)
{
    // The pieces of each text are listed before it; walk them in order and compare to the text's tokens
    std::size_t p = 0;
    for (const auto& [text, want] : data().texts) {
        std::vector<std::uint32_t> got;
        std::string joined;
        while (joined.size() < text.size() && p < data().pieces.size()) {
            joined += data().pieces[p].first;
            const auto t = encode(data().pieces[p].first);
            got.insert(got.end(), t.begin(), t.end());
            ++p;
        }
        ASSERT_EQ(joined, text);
        EXPECT_EQ(got, want) << "text '" << text << "'";
        std::vector<char> back(text.size() + 1);
        const auto bytes = [&](std::uint32_t id) { return id < data().bytes_of.size() ? std::string_view(data().bytes_of[id]) : std::string_view(); };
        const auto r = ml::bpe_decode(want, bytes, back);
        ASSERT_TRUE(r.is_ok());
        EXPECT_EQ(std::string(back.data(), r.unwrap()), text);
    }
    EXPECT_EQ(p, data().pieces.size());
}

TEST(MlBpe, TiesMergeTheLeftmostPair)
{
    // "aaa" with aa ranked: the leftmost pair merges first, giving [aa][a], never [a][aa]
    const auto rank = [](std::string_view s) -> std::uint32_t {
        if (s == "a") return 0;
        if (s == "aa") return 1;
        return ml::bpe_no_rank;
    };
    std::array<std::uint32_t, 3> out{};
    std::array<ml::bpe_part, 4> scratch{};
    ASSERT_EQ(ml::bpe_encode_piece("aaa", rank, out, scratch).unwrap(), 2u);
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 0u);
}

TEST(MlBpe, ErrorsAreReportedNotThrown)
{
    std::array<std::uint32_t, 2> small{};
    std::array<ml::bpe_part, 8> scratch{};
    EXPECT_EQ(ml::bpe_encode_piece("abba", tiny_rank, small, scratch).err(), ml::bpe_error::output_too_small);
    EXPECT_EQ(small[0], 0u);   // untouched
    std::array<std::uint32_t, 8> out{};
    std::array<ml::bpe_part, 2> tight{};
    EXPECT_EQ(ml::bpe_encode_piece("abba", tiny_rank, out, tight).err(), ml::bpe_error::scratch_too_small);
    EXPECT_EQ(ml::bpe_encode_piece("abca", tiny_rank, out, scratch).err(), ml::bpe_error::unknown_bytes);   // c is no token
    EXPECT_EQ(ml::bpe_encode_piece("abab", tiny_rank, small, tight).unwrap(), 1u);   // a whole-piece token needs no scratch
    std::array<char, 2> two{};
    const std::array<std::uint32_t, 2> ids{2, 2};
    const auto bytes = [](std::uint32_t id) { return id == 2 ? std::string_view("ab") : std::string_view(); };
    EXPECT_EQ(ml::bpe_decode(ids, bytes, two).err(), ml::bpe_error::output_too_small);
    const std::array<std::uint32_t, 1> unknown{9};
    EXPECT_EQ(ml::bpe_decode(unknown, bytes, two).err(), ml::bpe_error::unknown_bytes);
}

} // namespace
