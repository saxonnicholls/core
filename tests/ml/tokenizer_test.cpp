//
//  tokenizer_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/tokenizer.hpp>: token for token llama.cpp on its own vocabulary test vectors (the
//  .inp strings of models/ggml-vocab-*.gguf, whose .out files hold Hugging Face's tokens and agree with llama.cpp)
//  and edge cases, for sixteen vocabularies across the four model types, on the committed subsets in
//  tests/fixtures/ml/tokenizers/ (scripts/generate_tokenizer_fixtures.cpp); detokenisation back to the text; the
//  documented differences from llama.cpp; the errors.
//

#include <snicholls/core/ml/tokenizer.hpp>

#include "../support/tokenizer_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::test::hex;
using snicholls::test::owned_tokenizer;

std::string fixture(const std::string& name)
{
    return std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/tokenizers/" + name + ".txt";
}

// A vocabulary's cases: core's tokens (llama.cpp's unless the fixture records a difference); returns how many
// cases agree with llama.cpp
std::size_t check_vocabulary(const std::string& name, ml::tokenizer_model model)
{
    owned_tokenizer t;
    t.load(fixture(name));
    EXPECT_EQ(t.cfg.model, model) << name;
    EXPECT_GE(t.cases.size(), 50u) << name;
    std::size_t agree = 0;
    for (const auto& c : t.cases) {
        EXPECT_EQ(t.tokenize(c.text), c.core) << name << " on " << hex(c.text);
        agree += !c.differs;
    }
    return agree;
}

TEST(MlTokenizer, SentencePieceBpeMatchesLlamaCppTokenForToken)
{
    EXPECT_EQ(check_vocabulary("llama-spm", ml::tokenizer_model::sentencepiece_bpe), 59u);
    EXPECT_EQ(check_vocabulary("phi-3", ml::tokenizer_model::sentencepiece_bpe), 58u);
}

TEST(MlTokenizer, ByteLevelBpeMatchesLlamaCppTokenForTokenForEveryPreTokenizer)
{
    for (const char* name : {"gpt-2", "llama-bpe", "qwen2", "deepseek-llm", "deepseek-coder", "falcon", "starcoder", "refact", "command-r", "mpt"})
        EXPECT_EQ(check_vocabulary(name, ml::tokenizer_model::bpe), 60u) << name;
    EXPECT_EQ(check_vocabulary("qwen35", ml::tokenizer_model::bpe), 64u);
}

TEST(MlTokenizer, SentencePieceStyleBpeMatchesLlamaCppTokenForToken)
{
    EXPECT_EQ(check_vocabulary("gemma-4", ml::tokenizer_model::bpe), 60u);
}

TEST(MlTokenizer, WordPieceMatchesLlamaCppTokenForToken)
{
    EXPECT_EQ(check_vocabulary("bert-bge", ml::tokenizer_model::wordpiece), 60u);
}

TEST(MlTokenizer, UnigramMatchesLlamaCppTokenForTokenWithTheCharsmap)
{
    EXPECT_EQ(check_vocabulary("nomic-bert-moe", ml::tokenizer_model::unigram), 61u);
}

// Where core differs from llama.cpp, and why: each difference the fixtures record is one of these
TEST(MlTokenizer, TheDifferencesFromLlamaCppAreTheDocumentedOnes)
{
    const std::string invalid = "a\x80" "b \xE2\x82 c\xC0\xAF d\xED\xA0\x80 \xF4\x90\x80\x80 z";
    for (const char* name : {"llama-spm", "phi-3", "gpt-2", "llama-bpe", "qwen2", "qwen35", "deepseek-llm", "deepseek-coder", "falcon",
                             "starcoder", "refact", "command-r", "mpt", "gemma-4", "bert-bge", "nomic-bert-moe"}) {
        owned_tokenizer t;
        t.load(fixture(name));
        for (const auto& c : t.cases) {
            if (!c.differs)
                continue;
            const bool sentencepiece = t.cfg.model == ml::tokenizer_model::sentencepiece_bpe;
            const bool explained =
                (sentencepiece && c.text.find('\x80') != std::string::npos) ||   // llama.cpp cuts invalid UTF-8 by lead byte
                c.text == invalid ||   // llama.cpp decodes the encoded surrogate ED A0 80 as a code point; core as U+FFFD
                (std::string(name) == "phi-3" && c.text.find("</s>") != std::string::npos) ||   // llama.cpp re-marks "</s>" as control
                (std::string(name) == "bert-bge" && c.text.find("\xED\x95\x9C") != std::string::npos);   // Hangul: NFD's jamo, as Hugging Face
            EXPECT_TRUE(explained) << name << " differs on " << hex(c.text);
        }
    }
}

// Decoding gives the text back exactly (BPE: with invalid UTF-8 repaired to U+FFFD; SentencePiece keeps the bytes),
// except after an added token, as llama.cpp: a SentencePiece fragment after one gets its own dummy prefix, which
// decodes as a space, and a user-defined token renders as written.
TEST(MlTokenizer, ByteLevelAndSentencePieceBpeDecodeBackToTheText)
{
    std::size_t checked = 0;
    for (const char* name : {"gpt-2", "llama-bpe", "qwen2", "qwen35", "deepseek-llm", "deepseek-coder", "falcon", "starcoder", "refact",
                             "command-r", "mpt", "gemma-4", "llama-spm", "phi-3"}) {
        owned_tokenizer t;
        t.load(fixture(name));
        for (const auto& c : t.cases) {
            bool added = false;
            for (const auto id : c.core)
                added = added || t.view.kind(id) == ml::token_kind::user_defined;
            if (added)
                continue;
            // A BPE model sees invalid UTF-8 as U+FFFD, so it decodes to the repaired text
            std::string want = c.text;
            if (t.cfg.model == ml::tokenizer_model::bpe) {
                want.assign(3 * c.text.size(), '\0');
                want.resize(ml::replace_invalid_utf8(c.text, std::span<char>(want)));
            }
            EXPECT_EQ(t.detokenize(c.core), want) << name << " on " << hex(c.text);
            ++checked;
        }
    }
    EXPECT_GT(checked, 800u);
}

// Unigram and WordPiece normalise, so decoding gives the normalised text: compare with what was fed in
TEST(MlTokenizer, UnigramAndWordPieceDecodeToTheNormalisedText)
{
    owned_tokenizer u;
    u.load(fixture("nomic-bert-moe"));
    EXPECT_EQ(u.detokenize(u.tokenize("Hello  world   again    ")), "Hello world again");
    EXPECT_EQ(u.detokenize(u.tokenize("\xEF\xAC\x81 \xE2\x91\xA0 \xEF\xBC\xA1\xEF\xBC\xA2\xEF\xBC\xA3")), "fi 1 ABC");   // NFKC by the charsmap
    owned_tokenizer w;
    w.load(fixture("bert-bge"));
    EXPECT_EQ(w.detokenize(w.tokenize("Hello World!")), "hello world !");
    EXPECT_EQ(w.detokenize(w.tokenize("caf\xC3\xA9 cafe\xCC\x81")), "cafe cafe");
}

TEST(MlTokenizer, ErrorsAreValues)
{
    owned_tokenizer t;
    t.load(fixture("gpt-2"));
    std::array<std::uint32_t, 1> one{};
    std::vector<char> text(256);
    std::vector<ml::merge_entry<std::uint32_t>> ranked(ml::merge_scratch_size(256));
    EXPECT_EQ(ml::tokenize(t.view, "Hello world", one, {text, ranked, {}, {}}).unwrap_err(), ml::tokenizer_error::output_too_small);
    std::array<std::uint32_t, 16> out{};
    EXPECT_EQ(ml::tokenize(t.view, "Hello world", out, {}).unwrap_err(), ml::tokenizer_error::scratch_too_small);
    std::array<char, 2> small{};
    const std::array<std::uint32_t, 1> hello{t.tokenize("Hello")[0]};
    EXPECT_EQ(ml::detokenize(t.view, hello, small).unwrap_err(), ml::tokenizer_error::output_too_small);
    const std::array<std::uint32_t, 1> bad{0xFFFFFFF0u};
    EXPECT_EQ(ml::detokenize(t.view, bad, text).unwrap_err(), ml::tokenizer_error::unknown_bytes);
    EXPECT_EQ(ml::tokenize(t.view, "", out, {}).unwrap(), 0u);
}

// A whole BPE tokenizer the compiler holds: GPT-2's alphabet, a merge list, Llama 3's split
constexpr std::array<std::string_view, 7> tiny_pieces{"a", "b", "\xC4\xA0", "ab", "\xC4\xA0" "a", "\xC4\xA0" "ab", "c"};
constexpr std::array<std::pair<std::string_view, std::string_view>, 3> tiny_merges{
    {{"\xC4\xA0", "a"}, {"\xC4\xA0" "a", "b"}, {"a", "b"}}};
constexpr std::size_t compile_time_tokens(std::string_view text)
{
    std::array<std::uint32_t, 7> order{};
    ml::build_token_index(tiny_pieces, order);
    std::array<std::uint32_t, 3> merge_order{};
    ml::build_merge_index(tiny_merges, merge_order);
    ml::tokenizer_vocab v;
    v.model = ml::tokenizer_model::bpe;
    v.pieces = tiny_pieces;
    v.vocab = {tiny_pieces, order};
    v.merges = {tiny_merges, merge_order};
    v.pre = ml::pretokenizer::llama3;
    v.bytes = ml::make_byte_level_fallback(v.vocab);
    std::array<char, 64> t{};
    std::array<ml::merge_entry<std::uint32_t>, ml::merge_scratch_size(64)> r{};
    std::array<std::uint32_t, 32> out{};
    const std::size_t n = ml::tokenize(v, text, out, {t, r, {}, {}}).unwrap();
    std::size_t code = n;
    for (std::size_t i = 0; i < n; ++i)
        code = code * 8 + out[i];
    return code;
}
// "ab ab c" -> "ab", " ab", " c" = [3], [5], [2, 6]: " ab" merges "Ġa" first, which outranks "ab"; with "ab" first it
// would be [2, 3]
static_assert(compile_time_tokens("ab ab c") == ((((4 * 8 + 3) * 8 + 5) * 8 + 2) * 8 + 6));

} // namespace
