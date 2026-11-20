//
//  pretokenize_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/pretokenize.hpp>: every family's split against Python's regex module running the
//  original expressions on the committed fixture (llama.cpp's test strings and edge cases); the patterns'
//  alternatives one by one at compile time; invalid UTF-8; the composition of splits; the GGUF names; and the
//  added-token split.
//

#include <snicholls/core/ml/pretokenize.hpp>

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

std::vector<std::string> split(ml::pretokenizer kind, std::string_view text)
{
    std::vector<std::string> pieces;
    ml::pretokenize(kind, text, [&](std::string_view p) { pieces.emplace_back(p); });
    return pieces;
}

// The match length at 0, at compile time
template <typename P>
constexpr std::size_t match(const P& p, std::string_view s)
{
    return p(s, 0);
}

// GPT-2: contractions are case-sensitive, a space joins the run after it, whitespace leaves its last unit for the
// next piece
static_assert(match(ml::gpt2_pattern{}, "'s x") == 2 && match(ml::gpt2_pattern{}, "'S x") == 1);
static_assert(match(ml::gpt2_pattern{}, " hello world") == 6 && match(ml::gpt2_pattern{}, " 123x") == 4);
static_assert(match(ml::gpt2_pattern{}, " !!x") == 3 && match(ml::gpt2_pattern{}, "   x") == 2);
static_assert(match(ml::gpt2_pattern{}, "   ") == 3 && match(ml::gpt2_pattern{}, " x") == 2);
// Llama 3: case-insensitive contractions, a non-letter lead joins letters, three digits, newlines end a
// whitespace run
static_assert(match(ml::llama3_pattern{}, "'LL") == 3 && match(ml::llama3_pattern{}, "'Re") == 3);
static_assert(match(ml::llama3_pattern{}, "!hello") == 6 && match(ml::llama3_pattern{}, "\nhello") == 0 + 1);
static_assert(match(ml::llama3_pattern{}, "12345") == 3 && match(ml::qwen2_pattern{}, "12345") == 1);
static_assert(match(ml::llama3_pattern{}, "  \n  x") == 3 && match(ml::llama3_pattern{}, " ...\n\nx") == 6);
// 'ſ' folds to s
static_assert(match(ml::llama3_pattern{}, "'\xC5\xBF") == 3 && match(ml::gpt2_pattern{}, "'\xC5\xBF") == 1);
// GPT-4o: an upper run backtracks to the last letter after which a lower one follows ("HTTPServer" is one piece)
static_assert(match(ml::gpt4o_pattern{}, "HTTPServer") == 10 && match(ml::gpt4o_pattern{}, "Server") == 6);
static_assert(match(ml::gpt4o_pattern{}, "iPhone") == 1 && match(ml::gpt4o_pattern{}, "ABC def") == 3);
static_assert(match(ml::gpt4o_pattern{}, "don't") == 5 && match(ml::gpt4o_pattern{}, "a//\n/") == 1);
static_assert(match(ml::gpt4o_pattern{}, " //\n/x") == 5);
// DeepSeek V3: ASCII punctuation leads ASCII letters
static_assert(match(ml::deepseek_v3_pattern{}, ".com") == 4 && match(ml::deepseek_v3_pattern{}, "!!") == 2);
static_assert(match(ml::newline_pattern{}, "ab\n\ncd") == 2 && match(ml::newline_pattern{}, "\n\ncd") == 2);
static_assert(match(ml::trailing_space_pattern{}, "  ") == 2 && match(ml::trailing_space_pattern{}, "  x") == 0);
static_assert(match(ml::three_ascii_digits_pattern{}, "1234") == 3 && match(ml::three_ascii_digits_pattern{}, "12") == 0);

// A whole split at compile time
constexpr std::size_t count_pieces(std::string_view s)
{
    std::size_t n = 0;
    ml::pretokenize(ml::pretokenizer::llama3, s, [&](std::string_view) { ++n; });
    return n;
}
static_assert(count_pieces("") == 0 && count_pieces("Hello world") == 2 && count_pieces("I'm 12345 ok!\n") == 7);

static_assert(ml::pretokenizer_for("llama-bpe").unwrap() == ml::pretokenizer::llama3);
static_assert(ml::pretokenizer_for("deepseek-v3").unwrap() == ml::pretokenizer::deepseek_v3);
static_assert(ml::pretokenizer_for("no such").is_err());

TEST(MlPretokenize, EveryFamilyMatchesPythonRegexOnTheFixture)
{
    const auto lines = snicholls::test::fixture_lines("pretokenize_fixture.txt");
    std::vector<std::string> texts;
    std::size_t at = 0;
    for (; at < lines.size() && lines[at][0] == "text"; ++at)
        texts.push_back(snicholls::test::unhex(lines[at][1]));
    ASSERT_GT(texts.size(), 80u);
    std::size_t families = 0;
    while (at < lines.size()) {
        ASSERT_EQ(lines[at][0], "family");
        const std::string name = lines[at][1];
        const auto kind = [&] {
            constexpr std::pair<std::string_view, ml::pretokenizer> names[] = {
                {"gpt2", ml::pretokenizer::gpt2},          {"llama3", ml::pretokenizer::llama3},
                {"qwen2", ml::pretokenizer::qwen2},        {"qwen35", ml::pretokenizer::qwen35},
                {"gpt4o", ml::pretokenizer::gpt4o},        {"deepseek_v3", ml::pretokenizer::deepseek_v3},
                {"deepseek_llm", ml::pretokenizer::deepseek_llm}, {"deepseek_coder", ml::pretokenizer::deepseek_coder},
                {"falcon", ml::pretokenizer::falcon},      {"starcoder", ml::pretokenizer::starcoder},
                {"newlines", ml::pretokenizer::newlines}};
            for (const auto& [n, k] : names)
                if (n == name)
                    return k;
            throw std::runtime_error("unknown family " + name);
        }();
        ++at;
        ++families;
        for (const std::string& text : texts) {
            ASSERT_EQ(lines[at][0], "pieces");
            std::vector<std::string> want;
            std::size_t off = 0;
            for (std::size_t i = 1; i < lines[at].size(); ++i) {
                const std::size_t n = std::stoul(lines[at][i]);
                want.push_back(text.substr(off, n));
                off += n;
            }
            EXPECT_EQ(split(kind, text), want) << name << " on \"" << text << "\" (" << snicholls::test::hex(text) << ")";
            ++at;
        }
    }
    EXPECT_EQ(families, 11u);
}

TEST(MlPretokenize, InvalidBytesAreOneUnitEachAndThePiecesCoverTheText)
{
    // A lone continuation byte, a truncated sequence, an overlong form and an encoded surrogate
    const std::string text = "a\x80" "b \xE2\x82 c\xC0\xAF d\xED\xA0\x80";
    for (const auto kind : {ml::pretokenizer::gpt2, ml::pretokenizer::llama3, ml::pretokenizer::gpt4o, ml::pretokenizer::deepseek_v3}) {
        std::string joined;
        for (const auto& p : split(kind, text)) {
            EXPECT_FALSE(p.empty());
            joined += p;
        }
        EXPECT_EQ(joined, text);
    }
    // Under GPT-2 an invalid byte is a symbol: it does not join the letters around it
    EXPECT_EQ(split(ml::pretokenizer::gpt2, "a\x80" "b"), (std::vector<std::string>{"a", "\x80", "b"}));
    // Under Llama 3 it is a lead, like punctuation, and joins the letters after it
    EXPECT_EQ(split(ml::pretokenizer::llama3, "a\x80" "b"), (std::vector<std::string>{"a", "\x80" "b"}));
}

TEST(MlPretokenize, SplitSequenceAppliesEachPatternToThePiecesOfTheLast)
{
    std::vector<std::string> pieces;
    ml::split_sequence("ab 1234567 cd", [&](std::string_view p) { pieces.emplace_back(p); }, ml::digits_pattern<3>{},
                       ml::gpt2_pattern{});
    EXPECT_EQ(pieces, (std::vector<std::string>{"ab", " ", "123", "456", "7", " cd"}));
    pieces.clear();
    ml::split_isolated("x1y22z", ml::digits_pattern<>{}, [&](std::string_view p) { pieces.emplace_back(p); });
    EXPECT_EQ(pieces, (std::vector<std::string>{"x", "1", "y", "22", "z"}));
}

TEST(MlPretokenize, GgufNamesMapToTheirFamilies)
{
    EXPECT_EQ(ml::pretokenizer_for("gpt-2").unwrap(), ml::pretokenizer::gpt2);
    EXPECT_EQ(ml::pretokenizer_for("qwen2").unwrap(), ml::pretokenizer::qwen2);
    EXPECT_EQ(ml::pretokenizer_for("qwen35").unwrap(), ml::pretokenizer::qwen35);
    EXPECT_EQ(ml::pretokenizer_for("gpt-4o").unwrap(), ml::pretokenizer::gpt4o);
    EXPECT_EQ(ml::pretokenizer_for("command-r").unwrap(), ml::pretokenizer::starcoder);
    EXPECT_EQ(ml::pretokenizer_for("gemma4").unwrap(), ml::pretokenizer::newlines);
    EXPECT_EQ(ml::pretokenizer_for("").unwrap_err(), ml::pretokenize_error::unknown_name);
}

TEST(MlPretokenize, AddedTokensAreCutOutLeftmostLongest)
{
    const std::array<std::string_view, 4> pieces{"<s>", "<|end|>", "<|end|>x", "  "};
    std::array<std::uint32_t, 4> order{};
    const std::size_t n = ml::build_token_index(pieces, order);
    const ml::token_trie added{pieces, std::span<const std::uint32_t>(order).first(n)};
    std::vector<std::string> out;
    ml::split_added_tokens(
        "a<s>b<|end|>xy   z<s><s>", added, [&](std::string_view t) { out.emplace_back("text:" + std::string(t)); },
        [&](std::uint32_t id) { out.emplace_back("id:" + std::to_string(id)); });
    EXPECT_EQ(out, (std::vector<std::string>{"text:a", "id:0", "text:b", "id:2", "text:y", "id:3", "text: z", "id:0", "id:0"}));
    out.clear();
    ml::split_added_tokens("plain", ml::token_trie{}, [&](std::string_view t) { out.emplace_back(t); }, [](std::uint32_t) {});
    EXPECT_EQ(out, (std::vector<std::string>{"plain"}));
}

} // namespace
