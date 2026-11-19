//
//  tokenizer_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/tokenizer.hpp>: a byte-level BPE tokenizer filled in as a GGUF loader would fill
//  it (pieces in GPT-2's alphabet, merges split from "left right" strings, Llama 3's pre-tokenizer by name, an
//  added token), text tokenized and detokenized through the seam, at compile time and at run time; and a too-small
//  output. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/tokenizer.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;

//! [tokenizer]
// What a loader reads from a file: pieces by id ("Ġ" is a space), merges as strings, the pre-tokenizer's name
constexpr std::array<std::string_view, 8> pieces{"h", "i", "\xC4\xA0", "hi", "\xC4\xA0hi", "!", "<|end|>", "\xC4\xA0h"};
constexpr std::array<std::string_view, 3> merge_lines{"h i", "\xC4\xA0 hi", "\xC4\xA0 h"};
constexpr std::array<ml::token_kind, 8> kinds{ml::token_kind::normal, ml::token_kind::normal, ml::token_kind::normal,
                                              ml::token_kind::normal, ml::token_kind::normal, ml::token_kind::normal,
                                              ml::token_kind::user_defined, ml::token_kind::normal};

struct loaded
{
    std::array<std::pair<std::string_view, std::string_view>, 3> merges{};
    std::array<std::uint32_t, 8> order{}, added_order{};
    std::array<std::uint32_t, 3> merge_order{};
    std::size_t added = 0;

    constexpr loaded()
    {
        for (std::size_t i = 0; i < merges.size(); ++i)
            merges[i] = ml::split_merge(merge_lines[i]);
        ml::build_merge_index(merges, merge_order);
        ml::build_token_index(pieces, order);
        added = ml::build_token_index(pieces, added_order, [](std::uint32_t id) { return kinds[id] == ml::token_kind::user_defined; });
    }

    constexpr ml::tokenizer_vocab vocab() const
    {
        ml::tokenizer_vocab v;
        v.model = ml::tokenizer_model::bpe;
        v.pieces = pieces;
        v.kinds = kinds;
        v.vocab = {pieces, order};
        v.added = {pieces, std::span<const std::uint32_t>(added_order).first(added)};
        v.merges = {merges, merge_order};
        v.pre = ml::pretokenizer_for("llama-bpe").unwrap();
        v.bytes = ml::make_byte_level_fallback(v.vocab);
        return v;
    }
};

constexpr std::size_t count(std::string_view text)
{
    const loaded l;
    const ml::tokenizer_vocab v = l.vocab();
    std::array<char, 256> t{};
    std::array<ml::merge_entry<std::uint32_t>, 1024> r{};
    std::array<std::uint32_t, 64> out{};
    return ml::tokenize(v, text, out, {t, r, {}, {}}).unwrap();
}
static_assert(count("hi hi!<|end|>") == 4);   // "hi", "Ġhi", "!", "<|end|>"
//! [tokenizer]

} // namespace

int main()
{
    std::printf("snicholls-core tokenizer demo\n");
    bool ok = true;
    const loaded l;
    const ml::tokenizer_vocab v = l.vocab();
    const std::string_view text = "hi hi!<|end|> h";
    const ml::tokenizer_scratch_need need(v, text.size());
    std::vector<char> t(need.text);
    std::vector<ml::merge_entry<std::uint32_t>> r(need.ranked);
    std::vector<std::uint32_t> out(3 * text.size() + 3);
    const auto n = ml::tokenize(v, text, out, {t, r, {}, {}}).unwrap();
    std::printf("  \"hi hi!<|end|> h\" ->");
    for (std::size_t i = 0; i < n; ++i)
        std::printf(" %u", out[i]);
    std::printf("\n");
    ok &= n == 5 && out[0] == 3 && out[1] == 4 && out[2] == 5 && out[3] == 6 && out[4] == 7;

    std::vector<char> back(64);
    const auto m = ml::detokenize(v, std::span<const std::uint32_t>(out.data(), n), back).unwrap();
    std::printf("  detokenized: \"%.*s\"\n", static_cast<int>(m), back.data());
    ok &= std::string_view(back.data(), m) == text;

    std::array<std::uint32_t, 2> two{};
    const auto small = ml::tokenize(v, text, two, {t, r, {}, {}});
    std::printf("  two slots: %s\n", small.is_err() ? "output_too_small" : "?");
    ok &= small.is_err() && small.unwrap_err() == ml::tokenizer_error::output_too_small;

    std::printf(ok ? "tokenizer demo: ok\n" : "tokenizer demo: FAILED\n");
    return ok ? 0 : 1;
}
