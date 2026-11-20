//
//  wordpiece_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/wordpiece.hpp>: Hugging Face's convention ("##" continuations) on a small vocabulary,
//  BERT's basic tokenisation (lowercase, accents, punctuation, CJK, control characters, U+0130), the word limit,
//  words that cannot be cut, GGUF's convention, decoding both ways, the errors, and encoding at compile time.
//  Token-for-token agreement with llama.cpp on BGE's vocabulary is in tests/ml/tokenizer_test.cpp.
//

#include <snicholls/core/ml/wordpiece.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr std::array<std::string_view, 12> pieces{"[UNK]", "un", "##aff", "##able", "!", "cafe", "i", "##\xCC\x87",
                                                   "\xE4\xB8\xAD", "\xE6\x96\x87", "[CLS]", "##s"};

struct vocab
{
    std::array<std::uint32_t, 12> order{};
    constexpr vocab() { ml::build_token_index(pieces, order); }
    constexpr ml::token_trie trie() const { return {pieces, order}; }
};

std::vector<std::uint32_t> encode(std::string_view text, ml::wordpiece_options o = {})
{
    const vocab v;
    std::vector<std::uint32_t> out(3 * text.size() + 3);
    std::vector<char> scratch(ml::wordpiece_scratch_size(text.size()));
    out.resize(ml::wordpiece_encode(text, v.trie(), 0, o, out, scratch).unwrap());
    return out;
}

constexpr std::size_t compile_time_count(std::string_view text)
{
    const vocab v;
    std::array<std::uint32_t, 32> out{};
    std::array<char, 64> scratch{};
    return ml::wordpiece_encode(text, v.trie(), 0, {}, out, scratch).unwrap();
}
static_assert(compile_time_count("unaffable!") == 4 && compile_time_count("") == 0 && compile_time_count("  \t ") == 0);
static_assert(ml::is_bert_cjk(0x4E2D) && ml::is_bert_cjk(0x2B920) && !ml::is_bert_cjk(0x2B820) && !ml::is_bert_cjk(0x3042));
static_assert(ml::is_bert_punctuation('$') && ml::is_bert_punctuation(0x3001) && !ml::is_bert_punctuation('a'));
static_assert(ml::wordpiece_scratch_size(5) == 19);

TEST(MlWordPiece, GreedyLongestMatchWithContinuations)
{
    EXPECT_EQ(encode("unaffable"), (std::vector<std::uint32_t>{1, 2, 3}));
    EXPECT_EQ(encode("UnAffable!"), (std::vector<std::uint32_t>{1, 2, 3, 4}));
    EXPECT_EQ(encode("unaffables"), (std::vector<std::uint32_t>{1, 2, 3, 11}));
    EXPECT_EQ(encode("unx cafe"), (std::vector<std::uint32_t>{0, 5}));   // "unx" cannot be cut: [UNK]
    EXPECT_EQ(encode("affable"), (std::vector<std::uint32_t>{0}));       // a continuation cannot start a word
}

TEST(MlWordPiece, BasicTokenisationNormalisesAndSplits)
{
    EXPECT_EQ(encode("Caf\xC3\xA9"), (std::vector<std::uint32_t>{5}));                // accents stripped
    EXPECT_EQ(encode("CAFE\xCC\x81"), (std::vector<std::uint32_t>{5}));               // a combining accent too
    EXPECT_EQ(encode("\xE4\xB8\xAD\xE6\x96\x87"), (std::vector<std::uint32_t>{8, 9}));   // each ideograph a word
    EXPECT_EQ(encode("un\x01" "aff\xE2\x80\x8B" "able"), (std::vector<std::uint32_t>{1, 2, 3}));   // \p{C} dropped
    EXPECT_EQ(encode("cafe!cafe"), (std::vector<std::uint32_t>{5, 4, 5}));
    // U+0130 lowercases to "i" + U+0307 (the full mapping); without strip_accents the mark stays
    ml::wordpiece_options keep;
    keep.strip_accents = false;
    EXPECT_EQ(encode("\xC4\xB0", keep), (std::vector<std::uint32_t>{6, 7}));
    EXPECT_EQ(encode("\xC4\xB0"), (std::vector<std::uint32_t>{6}));
    ml::wordpiece_options cased;
    cased.lowercase = false;
    EXPECT_EQ(encode("Cafe", cased), (std::vector<std::uint32_t>{0}));
}

TEST(MlWordPiece, LongWordsAreUnknown)
{
    ml::wordpiece_options o;
    o.max_word_chars = 5;
    EXPECT_EQ(encode("unaffable cafe", o), (std::vector<std::uint32_t>{0, 5}));
    o.max_word_chars = 0;
    EXPECT_EQ(encode("unaffable", o), (std::vector<std::uint32_t>{1, 2, 3}));
}

TEST(MlWordPiece, GgufsConventionMarksWordStarts)
{
    const std::array<std::string_view, 4> g{"[UNK]", "\xE2\x96\x81" "un", "aff", "able"};
    std::array<std::uint32_t, 4> order{};
    ml::build_token_index(g, order);
    ml::wordpiece_options o;
    o.word_prefix = "\xE2\x96\x81";
    o.continuation_prefix = "";
    std::array<std::uint32_t, 8> out{};
    std::array<char, 64> scratch{};
    const auto n = ml::wordpiece_encode("unaffable", {g, order}, 0, o, out, scratch).unwrap();
    EXPECT_EQ(std::vector<std::uint32_t>(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(n)), (std::vector<std::uint32_t>{1, 2, 3}));
    std::array<char, 32> text{};
    const auto piece = [&](std::uint32_t id) { return g[id]; };
    const auto m = ml::wordpiece_decode(std::span<const std::uint32_t>(out.data(), n), piece, [](std::uint32_t) { return false; }, o, text).unwrap();
    EXPECT_EQ(std::string_view(text.data(), m), "unaffable");
}

TEST(MlWordPiece, DecodingJoinsContinuationsAndSkipsSpecialTokens)
{
    const auto piece = [](std::uint32_t id) { return id < pieces.size() ? pieces[id] : std::string_view{}; };
    const auto skip = [](std::uint32_t id) { return id == 10; };
    std::array<char, 64> text{};
    const std::array<std::uint32_t, 6> tokens{10, 1, 2, 3, 4, 5};
    const auto n = ml::wordpiece_decode(tokens, piece, skip, {}, text).unwrap();
    EXPECT_EQ(std::string_view(text.data(), n), "unaffable ! cafe");
    const std::array<std::uint32_t, 1> bad{77};
    EXPECT_EQ(ml::wordpiece_decode(bad, piece, skip, {}, text).unwrap_err(), ml::wordpiece_error::unknown_id);
    std::array<char, 3> small{};
    EXPECT_EQ(ml::wordpiece_decode(tokens, piece, skip, {}, small).unwrap_err(), ml::wordpiece_error::output_too_small);
}

TEST(MlWordPiece, ShortSpansAreErrors)
{
    const vocab v;
    std::array<std::uint32_t, 1> one{};
    std::array<char, 64> scratch{};
    EXPECT_EQ(ml::wordpiece_encode("unaffable", v.trie(), 0, {}, one, scratch).unwrap_err(), ml::wordpiece_error::output_too_small);
    EXPECT_EQ(ml::wordpiece_encode("cafe cafe", v.trie(), 0, {}, one, scratch).unwrap_err(), ml::wordpiece_error::output_too_small);
    std::array<std::uint32_t, 8> out{};
    std::array<char, 2> small{};
    EXPECT_EQ(ml::wordpiece_encode("cafe", v.trie(), 0, {}, out, small).unwrap_err(), ml::wordpiece_error::scratch_too_small);
}

} // namespace
