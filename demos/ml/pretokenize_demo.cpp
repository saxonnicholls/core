//
//  pretokenize_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/pretokenize.hpp>: one text split by GPT-2's, Llama 3's, GPT-4o's and DeepSeek V3's
//  pre-tokenizers, a split counted at compile time, a family chosen by its GGUF name, a sequence of splits built by
//  hand, and added tokens cut out of raw text. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/pretokenize.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

std::string show(ml::pretokenizer kind, std::string_view text)
{
    std::string s;
    ml::pretokenize(kind, text, [&](std::string_view piece) { s += "[" + std::string(piece) + "]"; });
    return s;
}

//! [pretokenize]
// Llama 3's split at compile time: contractions, a lead joined to letters, digits three at a time
constexpr std::size_t pieces(std::string_view text)
{
    std::size_t n = 0;
    ml::pretokenize(ml::pretokenizer::llama3, text, [&](std::string_view) { ++n; });
    return n;
}
static_assert(pieces("I'm 12345 ok!\n") == 7);   // I, 'm, " ", 123, 45, " ok", "!\n"
//! [pretokenize]

} // namespace

int main()
{
    std::printf("snicholls-core pretokenize demo\n");
    bool ok = true;
    const std::string_view text = "HTTPServer's 2026 price:  $1234!";
    const auto g = show(ml::pretokenizer::gpt2, text), l = show(ml::pretokenizer::llama3, text);
    const auto o = show(ml::pretokenizer::gpt4o, text), d = show(ml::pretokenizer::deepseek_v3, text);
    std::printf("  gpt2        %s\n  llama3      %s\n  gpt4o       %s\n  deepseek_v3 %s\n", g.c_str(), l.c_str(), o.c_str(), d.c_str());
    ok &= g == "[HTTPServer]['s][ 2026][ price][:][ ][ $][1234][!]";
    ok &= l == "[HTTPServer]['s][ ][202][6][ price][:][ ][ $][123][4][!]";
    ok &= o == "[HTTPServer's][ ][202][6][ price][:][ ][ $][123][4][!]";
    ok &= d == "[HTTPServer]['s][ ][202][6][ price][:][ ][ $][123][4][!]";

    // The family of a GGUF file's tokenizer.ggml.pre
    const auto kind = ml::pretokenizer_for("qwen2");
    std::printf("  \"qwen2\" -> %s\n", kind.is_ok() && kind.unwrap() == ml::pretokenizer::qwen2 ? "qwen2" : "?");
    ok &= kind.is_ok() && ml::pretokenizer_for("unknown").is_err();

    // A sequence built by hand: digits alone, then GPT-2's pattern over every piece
    std::vector<std::string> seq;
    ml::split_sequence("ab12 cd", [&](std::string_view p) { seq.emplace_back(p); }, ml::digits_pattern<1>{}, ml::gpt2_pattern{});
    ok &= seq == std::vector<std::string>{"ab", "1", "2", " cd"};

    // Added tokens first: "<|end|>" is cut out of the raw text and never split
    const std::array<std::string_view, 1> added{"<|end|>"};
    std::array<std::uint32_t, 1> order{};
    ml::build_token_index(added, order);
    std::string cut;
    ml::split_added_tokens("hi<|end|>yo", {added, order}, [&](std::string_view t) { cut += "[" + std::string(t) + "]"; },
                           [&](std::uint32_t id) { cut += "#" + std::to_string(id); });
    std::printf("  added tokens: %s\n", cut.c_str());
    ok &= cut == "[hi]#0[yo]";

    std::printf(ok ? "pretokenize demo: ok\n" : "pretokenize demo: FAILED\n");
    return ok ? 0 : 1;
}
