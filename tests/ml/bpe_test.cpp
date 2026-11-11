//
//  bpe_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/bpe.hpp>: every piece of the committed fixture encodes to tiktoken's tokens
//  (and so rustbpe's), every text to the concatenation, and decodes back to its bytes; ties go to the
//  leftmost pair; the errors; and encoding at compile time with a vocabulary the compiler holds.
//  bpe_encode_long_piece: the same tokens as bpe_encode_piece on the fixture, on random pieces over a
//  three-letter alphabet with vocabularies dense in ties, and on long pieces; its ties and errors.
//

#include <snicholls/core/ml/bpe.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
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

// bpe_encode_long_piece agrees with bpe_encode_piece at compile time, on the tiny vocabulary plus "ba"
constexpr auto tiny_ba_rank = [](std::string_view s) { return s == "ba" ? std::uint32_t{4} : tiny_rank(s); };
constexpr bool agree(std::string_view s)
{
    std::array<std::uint32_t, 16> a{}, b{};
    std::array<ml::bpe_part, ml::bpe_long_scratch_size(16)> scratch{};
    const auto ra = ml::bpe_encode_piece(s, tiny_ba_rank, a, scratch);
    const auto rb = ml::bpe_encode_long_piece(s, tiny_ba_rank, b, scratch);
    if (ra.unwrap() != rb.unwrap())
        return false;
    for (std::size_t i = 0; i < ra.unwrap(); ++i)
        if (a[i] != b[i])
            return false;
    return true;
}
static_assert(agree("") && agree("a") && agree("ab") && agree("abab") && agree("ababab") && agree("bababa") && agree("aabbab"));
static_assert(ml::bpe_long_scratch_size(10) == 51);

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

auto lookup(const std::unordered_map<std::string, std::uint32_t>& ranks)
{
    return [&ranks](std::string_view s) {
        const auto it = ranks.find(std::string(s));
        return it == ranks.end() ? ml::bpe_no_rank : it->second;
    };
}

template <typename Rank>
std::vector<std::uint32_t> encode_short(std::string_view piece, Rank&& rank)
{
    std::vector<std::uint32_t> out(piece.size());
    std::vector<ml::bpe_part> scratch(ml::bpe_scratch_size(piece.size()));
    const auto r = ml::bpe_encode_piece(piece, rank, out, scratch);
    EXPECT_TRUE(r.is_ok());
    out.resize(r.is_ok() ? r.unwrap() : 0);
    return out;
}

template <typename Rank>
std::vector<std::uint32_t> encode_long(std::string_view piece, Rank&& rank)
{
    std::vector<std::uint32_t> out(piece.size());
    std::vector<ml::bpe_part> scratch(ml::bpe_long_scratch_size(piece.size()));
    const auto r = ml::bpe_encode_long_piece(piece, rank, out, scratch);
    EXPECT_TRUE(r.is_ok());
    out.resize(r.is_ok() ? r.unwrap() : 0);
    return out;
}

std::vector<std::uint32_t> encode(std::string_view piece) { return encode_short(piece, lookup(data().ranks)); }

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

TEST(MlBpeLong, EveryFixturePieceEncodesToTiktokensTokens)
{
    ASSERT_GT(data().pieces.size(), 300u);
    for (const auto& [piece, want] : data().pieces)
        EXPECT_EQ(encode_long(piece, lookup(data().ranks)), want) << "piece '" << piece << "'";
}

TEST(MlBpeLong, RandomPiecesDenseInTiesGiveTheRescansTokens)
{
    // Every string of 1 to 4 letters over {a, b, c}, ranked at random (single bytes first), two thirds of the
    // longer ones kept: equal pairs at different places are everywhere, which is where leftmost-first matters
    std::vector<std::string> all;
    for (std::size_t len = 1, count = 3; len <= 4; ++len, count *= 3)
        for (std::size_t k = 0; k < count; ++k) {
            std::string s;
            for (std::size_t i = 0, x = k; i < len; ++i, x /= 3)
                s.push_back("abc"[x % 3]);
            all.push_back(s);
        }
    snicholls::stats::wyrand rng(20261111);
    std::vector<std::uint32_t> order(all.size());
    for (std::uint32_t i = 0; i < order.size(); ++i)
        order[i] = i;
    for (int round = 0; round < 50; ++round) {
        snicholls::stats::shuffle(std::span(order).subspan(3), rng);
        std::unordered_map<std::string, std::uint32_t> ranks;
        for (std::size_t i = 0; i < all.size(); ++i)
            if (all[i].size() == 1 || snicholls::stats::uniform_index(rng, 3) != 0)
                ranks.emplace(all[i], order[i]);
        for (int t = 0; t < 400; ++t) {
            std::string s(1 + snicholls::stats::uniform_index(rng, 64), 'a');
            for (char& c : s)
                c = "abc"[snicholls::stats::uniform_index(rng, 3)];
            ASSERT_EQ(encode_long(s, lookup(ranks)), encode_short(s, lookup(ranks))) << "round " << round << ", piece '" << s << "'";
        }
    }
}

TEST(MlBpeLong, LongPiecesGiveTheRescansTokens)
{
    // Shapes a split pattern keeps whole: a run of spaces, of letters, of punctuation; long enough to cross the
    // threshold, short enough for the O(n^2) reference to stay quick
    snicholls::stats::wyrand rng(7);
    for (const std::size_t n : {ml::bpe_long_piece_threshold, std::size_t{4096}}) {
        std::string letters(n, 'a'), punct(n, '!');
        for (char& c : letters)
            c = static_cast<char>('a' + snicholls::stats::uniform_index(rng, 26));
        for (char& c : punct)
            c = "!#$%&*+-/<=>?@^_|~"[snicholls::stats::uniform_index(rng, 18)];
        for (const std::string& s : {std::string(n, ' '), letters, punct})
            EXPECT_EQ(encode_long(s, lookup(data().ranks)), encode_short(s, lookup(data().ranks)))
                << n << " bytes starting '" << s.substr(0, 8) << "'";
    }
}

TEST(MlBpeLong, TiesMergeTheLeftmostPair)
{
    const auto rank = [](std::string_view s) -> std::uint32_t {
        if (s == "a") return 0;
        if (s == "aa") return 1;
        return ml::bpe_no_rank;
    };
    std::array<std::uint32_t, 5> out{};
    std::array<ml::bpe_part, ml::bpe_long_scratch_size(5)> scratch{};
    ASSERT_EQ(ml::bpe_encode_long_piece("aaa", rank, out, scratch).unwrap(), 2u);   // [aa][a], never [a][aa]
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 0u);
    ASSERT_EQ(ml::bpe_encode_long_piece("aaaaa", rank, out, scratch).unwrap(), 3u);   // [aa][aa][a]
    EXPECT_EQ(out[0], 1u);
    EXPECT_EQ(out[1], 1u);
    EXPECT_EQ(out[2], 0u);
}

TEST(MlBpeLong, ErrorsAreReportedNotThrown)
{
    std::array<std::uint32_t, 2> small{};
    std::array<ml::bpe_part, ml::bpe_long_scratch_size(8)> scratch{};
    EXPECT_EQ(ml::bpe_encode_long_piece("abba", tiny_rank, small, scratch).err(), ml::bpe_error::output_too_small);
    EXPECT_EQ(small[0], 0u);   // untouched
    std::array<std::uint32_t, 8> out{};
    std::array<ml::bpe_part, ml::bpe_scratch_size(4)> tight{};   // enough for bpe_encode_piece, not for this
    EXPECT_EQ(ml::bpe_encode_long_piece("abba", tiny_rank, out, tight).err(), ml::bpe_error::scratch_too_small);
    EXPECT_EQ(ml::bpe_encode_long_piece("abca", tiny_rank, out, scratch).err(), ml::bpe_error::unknown_bytes);
    std::array<ml::bpe_part, 0> none{};
    EXPECT_EQ(ml::bpe_encode_long_piece("abab", tiny_rank, small, none).unwrap(), 1u);   // a whole-piece token needs no scratch
    EXPECT_EQ(ml::bpe_encode_long_piece("", tiny_rank, std::span<std::uint32_t>{}, none).unwrap(), 0u);
}

} // namespace
