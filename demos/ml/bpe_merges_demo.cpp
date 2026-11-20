//
//  bpe_merges_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/bpe_merges.hpp>: a byte-level vocabulary and merge list in GPT-2's alphabet held by
//  the compiler, a piece mapped into the alphabet and merged at compile time, the pair that decides, byte fallback,
//  decoding back, and the errors. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/bpe_merges.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <utility>

namespace {

namespace ml = snicholls::ml;

//! [bpe_merges]
// A vocabulary as a byte-level tokenizer file writes it: " " is "Ġ" (U+0120) in GPT-2's alphabet
constexpr std::array<std::string_view, 7> pieces{"l", "o", "w", "\xC4\xA0", "lo", "low", "\xC4\xA0low"};
constexpr std::array<std::pair<std::string_view, std::string_view>, 3> merges{
    {{"l", "o"}, {"lo", "w"}, {"\xC4\xA0", "low"}}};   // in priority order: "l o" first

constexpr auto encode(std::string_view raw)
{
    std::array<std::uint32_t, 7> order{};
    ml::build_token_index(pieces, order);
    const ml::token_trie vocab{pieces, order};
    std::array<std::uint32_t, 3> merge_order{};
    ml::build_merge_index(merges, merge_order);
    const ml::merge_table table{merges, merge_order};
    // Map the piece into the alphabet, then merge its code points
    std::array<char, 32> mapped{};
    const std::size_t m = ml::byte_level_encode(raw, mapped).unwrap();
    std::array<std::uint32_t, 16> out{};
    std::array<ml::merge_entry<std::uint32_t>, ml::merge_scratch_size(32)> scratch{};
    const auto bytes = [&](unsigned char b) {   // each byte's own token, by its spelling in the alphabet
        char s[4]{};
        return vocab.find(std::string_view(s, ml::encode_utf8(ml::byte_to_unicode[b], s)));
    };
    const std::size_t n = ml::merge_encode(std::string_view(mapped.data(), m), table, vocab, bytes, out, scratch,
                                           {ml::merge_units::code_points, false, true}).unwrap();
    return std::pair{out, n};
}
static_assert(encode(" low").second == 1 && encode(" low").first[0] == 6);   // l o -> lo, lo w -> low, Ġ low
static_assert(encode("owl").second == 3);                                    // no merge applies: o w l
//! [bpe_merges]

} // namespace

int main()
{
    std::printf("snicholls-core bpe_merges demo\n");
    bool ok = true;
    const auto [tokens, n] = encode(" lowlow");
    std::printf("  \" lowlow\" ->");
    for (std::size_t i = 0; i < n; ++i)
        std::printf(" %u", tokens[i]);
    std::printf("\n");
    ok &= n == 2 && tokens[0] == 6 && tokens[1] == 5;

    // Decoding: the pieces, concatenated, back through the alphabet
    std::array<char, 32> text{};
    std::size_t at = 0;
    for (std::size_t i = 0; i < n; ++i)
        at += ml::byte_level_decode(pieces[tokens[i]], std::span<char>(text).subspan(at)).unwrap();
    std::printf("  decoded: \"%.*s\"\n", static_cast<int>(at), text.data());
    ok &= std::string_view(text.data(), at) == " lowlow";

    // GPT-2's byte map: printable bytes are themselves, the others moved past U+00FF
    std::printf("  byte 0x20 is U+%04X, 0x0A is U+%04X, 0x41 is U+%04X\n", static_cast<unsigned>(ml::byte_to_unicode[0x20]),
                static_cast<unsigned>(ml::byte_to_unicode[0x0A]), static_cast<unsigned>(ml::byte_to_unicode[0x41]));
    ok &= ml::byte_to_unicode[0x20] == 0x120 && ml::unicode_to_byte[0x120] == 0x20;

    // A byte the vocabulary lacks is an error
    std::array<std::uint32_t, 4> out{};
    std::array<ml::merge_entry<std::uint32_t>, ml::merge_scratch_size(4)> scratch{};
    const auto never = [](std::string_view, std::string_view) { return ml::merge_never<std::uint32_t>; };
    const auto bad = ml::merge_encode("z", never, [](std::string_view) { return ml::no_token; },
                                      [](unsigned char) { return ml::no_token; }, out, scratch);
    std::printf("  \"z\": %s\n", bad.is_err() ? "unknown_bytes" : "?");
    ok &= bad.is_err() && bad.unwrap_err() == ml::bpe_error::unknown_bytes;

    std::printf(ok ? "bpe_merges demo: ok\n" : "bpe_merges demo: FAILED\n");
    return ok ? 0 : 1;
}
