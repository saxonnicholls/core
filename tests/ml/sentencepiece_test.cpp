//
//  sentencepiece_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/sentencepiece.hpp>: escaping and the dummy prefix, merging by score with ties to the
//  left, byte fallback (<0xNN>, then the single byte), decoding (spaces, bytes, user-defined and control tokens, the
//  leading space), the errors, and encoding at compile time. Token-for-token agreement with llama.cpp on Llama's and
//  Phi-3's vocabularies is in tests/ml/tokenizer_test.cpp.
//

#include <snicholls/core/ml/sentencepiece.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

// A small SentencePiece vocabulary: "▁" is \xE2\x96\x81
constexpr std::array<std::string_view, 9> pieces{"<unk>", "\xE2\x96\x81", "h", "i", "\xE2\x96\x81h", "hi", "\xE2\x96\x81hi", "<0x21>", "x"};
constexpr std::array<float, 9> scores{0, -1, -2, -2, -3, -1, -0.5f, 0, -9};
constexpr std::array<ml::token_kind, 9> kinds{ml::token_kind::unknown, ml::token_kind::normal, ml::token_kind::normal,
                                              ml::token_kind::normal, ml::token_kind::normal, ml::token_kind::normal,
                                              ml::token_kind::normal, ml::token_kind::byte, ml::token_kind::user_defined};

struct vocab
{
    std::array<std::uint32_t, 9> order{};
    constexpr vocab() { ml::build_token_index(pieces, order); }
    constexpr ml::token_trie trie() const { return {pieces, order}; }
};

constexpr std::size_t encode_code(std::string_view text)
{
    const vocab v;
    const auto trie = v.trie();
    const auto score = [](std::uint32_t id) { return scores[id]; };
    const ml::byte_fallback bytes = ml::make_byte_fallback(trie);
    std::array<char, 64> escaped{};
    const std::size_t n = ml::sentencepiece_escape(text, escaped, true).unwrap();
    std::array<std::uint32_t, 64> out{};
    std::array<ml::merge_entry<float>, ml::merge_scratch_size(64)> scratch{};
    const std::size_t k = ml::sentencepiece_bpe_encode(std::string_view(escaped.data(), n), trie, score, bytes, out, scratch).unwrap();
    std::size_t code = k;
    for (std::size_t i = 0; i < k; ++i)
        code = code * 10 + out[i];
    return code;
}
// "hi" -> "▁hi": "▁h" (-3) and "hi" (-1) compete, "hi" wins, then "▁" + "hi" = "▁hi"
static_assert(encode_code("hi") == 1 * 10 + 6);
// "!": no piece, falls back to <0x21>; "x" has no <0x78>, so the single-byte piece "x" is its fallback
static_assert(encode_code("!") == (2 * 10 + 1) * 10 + 7);
static_assert(encode_code("") == 0);
static_assert(ml::sentencepiece_escaped_size(4) == 15);

TEST(MlSentencePiece, EscapingWritesSpacesAndTheDummyPrefix)
{
    std::array<char, 32> out{};
    auto n = ml::sentencepiece_escape(" a b", out, true).unwrap();
    EXPECT_EQ(std::string_view(out.data(), n), "\xE2\x96\x81\xE2\x96\x81" "a\xE2\x96\x81" "b");
    n = ml::sentencepiece_escape("ab", out, false).unwrap();
    EXPECT_EQ(std::string_view(out.data(), n), "ab");
    EXPECT_EQ(ml::sentencepiece_escape("", out, true).unwrap(), 0u);
    std::array<char, 3> small{};
    EXPECT_EQ(ml::sentencepiece_escape("a", small, true).unwrap_err(), ml::bpe_error::output_too_small);
    EXPECT_EQ(ml::sentencepiece_escape("abcd", small, false).unwrap_err(), ml::bpe_error::output_too_small);
    EXPECT_EQ(ml::sentencepiece_escape("a b", std::span<char>(small).first(2), false).unwrap_err(), ml::bpe_error::output_too_small);
}

TEST(MlSentencePiece, ByteFallbackFindsHexTokensThenSingleBytes)
{
    const vocab v;
    const auto f = ml::make_byte_fallback(v.trie());
    EXPECT_EQ(f('!'), 7u);
    EXPECT_EQ(f('x'), 8u);
    EXPECT_EQ(f('h'), 2u);
    EXPECT_EQ(f('?'), ml::no_token);
}

TEST(MlSentencePiece, DecodingRendersPiecesBytesAndSkipsControlTokens)
{
    const auto piece = [](std::uint32_t id) { return id < pieces.size() ? pieces[id] : std::string_view{}; };
    const auto kind = [](std::uint32_t id) { return kinds[id]; };
    std::array<char, 32> out{};
    const std::array<std::uint32_t, 5> tokens{6, 7, 0, 1, 8};
    auto n = ml::sentencepiece_decode(tokens, piece, kind, out).unwrap();
    EXPECT_EQ(std::string_view(out.data(), n), "hi! x");
    n = ml::sentencepiece_decode(tokens, piece, kind, out, {false, true}).unwrap();
    EXPECT_EQ(std::string_view(out.data(), n), " hi!<unk> x");
    const std::array<std::uint32_t, 1> bad{99};
    EXPECT_EQ(ml::sentencepiece_decode(bad, piece, kind, out).unwrap_err(), ml::bpe_error::unknown_bytes);
    std::array<char, 2> small{};
    EXPECT_EQ(ml::sentencepiece_decode(tokens, piece, kind, small).unwrap_err(), ml::bpe_error::output_too_small);
    const std::array<std::uint32_t, 2> bytes{7, 7};
    EXPECT_EQ(ml::sentencepiece_decode(bytes, piece, kind, std::span<char>(small).first(1)).unwrap_err(), ml::bpe_error::output_too_small);
}

TEST(MlSentencePiece, EncodeThenDecodeIsTheText)
{
    const vocab v;
    const auto trie = v.trie();
    const auto score = [](std::uint32_t id) { return scores[id]; };
    const auto bytes = ml::make_byte_fallback(trie);
    for (const std::string_view text : {"hi", "hi hi", " hi!", "hhii", "!!"}) {
        std::vector<char> e(ml::sentencepiece_escaped_size(text.size()));
        const auto n = ml::sentencepiece_escape(text, e, true).unwrap();
        std::vector<std::uint32_t> out(n);
        std::vector<ml::merge_entry<float>> scratch(ml::merge_scratch_size(n));
        const auto k = ml::sentencepiece_bpe_encode(std::string_view(e.data(), n), trie, score, bytes, out, scratch).unwrap();
        std::vector<char> back(64);
        const auto m = ml::sentencepiece_decode(std::span<const std::uint32_t>(out.data(), k), [](std::uint32_t id) { return pieces[id]; },
                                                [](std::uint32_t id) { return kinds[id]; }, back)
                           .unwrap();
        EXPECT_EQ(std::string_view(back.data(), m), text);
    }
}

} // namespace
