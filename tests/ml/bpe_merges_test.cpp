//
//  bpe_merges_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/bpe_merges.hpp>: merge_encode against a direct implementation of the merge rule
//  (rescan for the best pair, leftmost on ties) on random pieces and random merge lists, priorities that tie, the
//  pair that decides (not the bytes), ignore_merges, both units, both fallbacks, the errors; the merge index; GPT-2's
//  byte map, all 256 entries, against the definition. Token-for-token agreement with llama.cpp is in
//  tests/ml/tokenizer_test.cpp.
//

#include <snicholls/core/ml/bpe_merges.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;

// GPT-2's map: printable bytes are themselves, the rest from U+0100 in byte order
static_assert(ml::byte_to_unicode['a'] == U'a' && ml::byte_to_unicode[' '] == 0x120 && ml::byte_to_unicode['\n'] == 0x10A);
static_assert(ml::byte_to_unicode[0] == 0x100 && ml::byte_to_unicode[0xAD] == 0x143 && ml::byte_to_unicode[0xFF] == 0xFF);
static_assert(ml::unicode_to_byte[0x120] == ' ' && ml::unicode_to_byte[0x80] == -1 && ml::unicode_to_byte[0x143] == 0xAD);
static_assert(ml::merge_never<std::uint32_t> == 0xFFFFFFFFu && ml::merge_never<float> > 1e38f);
static_assert(ml::merge_scratch_size(10) == 41);
static_assert(ml::split_merge("\xC4\xA0 t").first == "\xC4\xA0" && ml::split_merge("\xC4\xA0 t").second == "t");
static_assert(ml::split_merge("  x").first == " " && ml::split_merge("ab").first.empty());

// A vocabulary over raw bytes at compile time: merges "a b" then "ab c"; "bc" is a token but not a merge
constexpr std::array<std::pair<std::string_view, std::string_view>, 2> merges{{{"a", "b"}, {"ab", "c"}}};
constexpr std::array<std::string_view, 6> pieces{"a", "b", "c", "ab", "abc", "bc"};
constexpr std::size_t encode_code(std::string_view s, bool ignore = false)
{
    std::array<std::uint32_t, 2> morder{};
    ml::build_merge_index(merges, morder);
    const ml::merge_table table{merges, morder};
    std::array<std::uint32_t, 6> order{};
    ml::build_token_index(pieces, order);
    const ml::token_trie trie{pieces, order};
    std::array<std::uint32_t, 16> out{};
    std::array<ml::merge_entry<std::uint32_t>, ml::merge_scratch_size(16)> scratch{};
    const auto none = [](unsigned char) { return ml::no_token; };
    const std::size_t n = ml::merge_encode(s, table, trie, none, out, scratch, {ml::merge_units::bytes, ignore}).unwrap();
    std::size_t code = n;
    for (std::size_t i = 0; i < n; ++i)
        code = code * 10 + out[i];
    return code;
}
static_assert(encode_code("abc") == 1 * 10 + 4);           // a b c -> ab c -> abc
static_assert(encode_code("bc") == (2 * 10 + 1) * 10 + 2);   // "bc" is a token, but no merge makes it
static_assert(encode_code("bc", true) == 1 * 10 + 5);      // ignore_merges: the whole piece is a token
static_assert(encode_code("cab") == (2 * 10 + 2) * 10 + 3);
static_assert(encode_code("") == 0);

// The rule written directly: rescan for the lowest-ranked adjacent pair, leftmost on ties
std::vector<std::string> reference(const std::vector<std::string>& units, const std::map<std::pair<std::string, std::string>, int>& rank)
{
    std::vector<std::string> parts = units;
    for (;;) {
        int best = -1;
        std::size_t at = 0;
        for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
            const auto it = rank.find({parts[i], parts[i + 1]});
            if (it != rank.end() && (best < 0 || it->second < best)) {
                best = it->second;
                at = i;
            }
        }
        if (best < 0)
            return parts;
        parts[at] += parts[at + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(at) + 1);
    }
}

TEST(MlBpeMerges, AgreesWithTheRuleOnRandomPiecesAndMergeLists)
{
    snicholls::stats::wyrand rng(11);
    for (int round = 0; round < 300; ++round) {
        // Random merges over {a, b, c}: pairs of strings built from earlier merges, with repeated ranks allowed
        std::vector<std::string> tokens{"a", "b", "c"};
        std::vector<std::pair<std::string, std::string>> list;
        std::map<std::pair<std::string, std::string>, int> rank;
        for (int m = 0, count = static_cast<int>(rng() % 20); m < count; ++m) {
            const std::string l = tokens[rng() % tokens.size()], r = tokens[rng() % tokens.size()];
            if (rank.count({l, r}))
                continue;
            rank[{l, r}] = static_cast<int>(list.size());
            list.emplace_back(l, r);
            tokens.push_back(l + r);
        }
        std::vector<std::pair<std::string_view, std::string_view>> views(list.begin(), list.end());
        std::vector<std::uint32_t> morder(views.size());
        ml::build_merge_index(views, morder);
        const ml::merge_table table{views, morder};
        std::vector<std::string_view> tv(tokens.begin(), tokens.end());
        std::vector<std::uint32_t> order(tv.size());
        ml::build_token_index(tv, order);
        const ml::token_trie trie{tv, order};
        for (int q = 0; q < 20; ++q) {
            std::string piece;
            for (std::size_t k = 0, len = rng() % 24; k < len; ++k)
                piece.push_back("abc"[rng() % 3]);
            std::vector<std::string> units;
            for (const char c : piece)
                units.emplace_back(1, c);
            const auto want = reference(units, rank);
            std::vector<std::uint32_t> out(piece.size());
            std::vector<ml::merge_entry<std::uint32_t>> scratch(ml::merge_scratch_size(piece.size()));
            const auto n = ml::merge_encode(piece, table, trie, [](unsigned char) { return ml::no_token; }, out, scratch).unwrap();
            ASSERT_EQ(n, want.size()) << piece;
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_EQ(tv[out[i]], want[i]) << piece;
        }
    }
}

TEST(MlBpeMerges, EqualPrioritiesMergeTheLeftmostPairFirst)
{
    // Score-style priorities that tie: "aa" from either side; the leftmost goes first, so "aaa" is "aa" "a"
    const std::array<std::string_view, 2> pieces{"a", "aa"};
    std::array<std::uint32_t, 2> order{};
    ml::build_token_index(pieces, order);
    const ml::token_trie trie{pieces, order};
    const auto priority = [&](std::string_view l, std::string_view r) {
        return trie.find(std::string_view(l.data(), l.size() + r.size())) == ml::no_token ? ml::merge_never<float> : -1.0f;
    };
    std::array<std::uint32_t, 8> out{};
    std::array<ml::merge_entry<float>, ml::merge_scratch_size(8)> scratch{};
    const auto n = ml::merge_encode("aaa", priority, trie, [](unsigned char) { return ml::no_token; }, out, scratch).unwrap();
    ASSERT_EQ(n, 2u);
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 0u);
}

TEST(MlBpeMerges, BytesFallBackWhenAPartIsNotAToken)
{
    // Code points as units: "é" is not a token, so its two bytes are
    const std::array<std::string_view, 1> pieces{"x"};
    std::array<std::uint32_t, 1> order{};
    ml::build_token_index(pieces, order);
    const ml::token_trie trie{pieces, order};
    const auto never = [](std::string_view, std::string_view) { return ml::merge_never<std::uint32_t>; };
    const auto bytes = [](unsigned char b) { return 1000u + b; };
    std::array<std::uint32_t, 8> out{};
    std::array<ml::merge_entry<std::uint32_t>, ml::merge_scratch_size(8)> scratch{};
    auto n = ml::merge_encode("x\xC3\xA9", never, trie, bytes, out, scratch).unwrap();
    EXPECT_EQ(std::vector<std::uint32_t>(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n)), (std::vector<std::uint32_t>{0, 1000 + 0xC3, 1000 + 0xA9}));
    // byte_level: the part is in GPT-2's alphabet, and falls back per byte it spells ("Ġ" spells 0x20)
    n = ml::merge_encode("\xC4\xA0", never, trie, bytes, out, scratch, {ml::merge_units::code_points, false, true}).unwrap();
    ASSERT_EQ(n, 1u);
    EXPECT_EQ(out[0], 1000u + 0x20);
    // A byte without a token is an error, and so are short spans
    const auto none = [](unsigned char) { return ml::no_token; };
    EXPECT_EQ(ml::merge_encode("y", never, trie, none, out, scratch).unwrap_err(), ml::bpe_error::unknown_bytes);
    std::array<std::uint32_t, 1> one{};
    EXPECT_EQ(ml::merge_encode("xx", never, trie, none, one, scratch).unwrap_err(), ml::bpe_error::output_too_small);
    std::array<ml::merge_entry<std::uint32_t>, 4> small{};
    EXPECT_EQ(ml::merge_encode("xx", never, trie, none, out, small).unwrap_err(), ml::bpe_error::scratch_too_small);
    EXPECT_EQ(ml::merge_encode("x", never, trie, none, std::span<std::uint32_t>{}, scratch, {ml::merge_units::bytes, true}).unwrap_err(),
              ml::bpe_error::output_too_small);
}

TEST(MlBpeMerges, TheMergeIndexFindsTheFirstRankOfAPair)
{
    const std::array<std::pair<std::string_view, std::string_view>, 4> list{{{"b", "c"}, {"a", "b"}, {"b", "c"}, {"ab", "c"}}};
    std::array<std::uint32_t, 4> order{};
    ml::build_merge_index(list, order);
    const ml::merge_table table{list, order};
    EXPECT_EQ(table.rank("b", "c"), 0u);
    EXPECT_EQ(table.rank("a", "b"), 1u);
    EXPECT_EQ(table("ab", "c"), 3u);
    EXPECT_EQ(table.rank("a", "c"), ml::merge_never<std::uint32_t>);
}

TEST(MlBpeMerges, TheByteLevelAlphabetRoundTripsEveryByte)
{
    std::string all;
    for (int b = 0; b < 256; ++b)
        all.push_back(static_cast<char>(b));
    std::string mapped(512, '\0');
    mapped.resize(ml::byte_level_encode(all, std::span<char>(mapped)).unwrap());
    // The 94 printable ASCII bytes are one byte each; the other 162 map past U+007F, two bytes each
    EXPECT_EQ(mapped.size(), 94u + 2u * 162u);
    std::string back(256, '\0');
    back.resize(ml::byte_level_decode(mapped, std::span<char>(back)).unwrap());
    EXPECT_EQ(back, all);
    // Distinct bytes map to distinct code points
    std::vector<bool> seen(324, false);
    for (int b = 0; b < 256; ++b) {
        EXPECT_FALSE(seen[ml::byte_to_unicode[b]]);
        seen[ml::byte_to_unicode[b]] = true;
    }
    std::array<char, 1> one{};
    EXPECT_EQ(ml::byte_level_encode("\n", one).unwrap_err(), ml::bpe_error::output_too_small);
    EXPECT_EQ(ml::byte_level_decode("\xE2\x96\x81", back).unwrap_err(), ml::bpe_error::unknown_bytes);
    EXPECT_EQ(ml::byte_level_decode("ab", one).unwrap_err(), ml::bpe_error::output_too_small);
}

} // namespace
