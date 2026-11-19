//
//  sentencepiece_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/sentencepiece.hpp>: a SentencePiece BPE vocabulary with scores held by the compiler,
//  text escaped with the dummy prefix and merged by score at compile time, byte fallback for a character no piece
//  covers, and decoding back. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/sentencepiece.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <utility>

namespace {

namespace ml = snicholls::ml;

//! [sentencepiece]
// Pieces, scores and kinds by id, as a SentencePiece model lists them ("▁" is a space)
constexpr std::array<std::string_view, 8> pieces{"<unk>", "\xE2\x96\x81", "H", "i", "Hi", "\xE2\x96\x81Hi", "<0x21>", "<0x3F>"};
constexpr std::array<float, 8> scores{0, -2, -3, -3, -1, -0.5f, 0, 0};
constexpr std::array<ml::token_kind, 8> kinds{ml::token_kind::unknown, ml::token_kind::normal, ml::token_kind::normal,
                                              ml::token_kind::normal, ml::token_kind::normal, ml::token_kind::normal,
                                              ml::token_kind::byte, ml::token_kind::byte};

constexpr auto encode(std::string_view text)
{
    std::array<std::uint32_t, 8> order{};
    ml::build_token_index(pieces, order);
    const ml::token_trie vocab{pieces, order};
    const ml::byte_fallback bytes = ml::make_byte_fallback(vocab);   // "<0x21>" is the fallback for '!'
    std::array<char, 48> escaped{};
    const std::size_t e = ml::sentencepiece_escape(text, escaped, true).unwrap();   // "Hi!" -> "▁Hi!"
    std::array<std::uint32_t, 48> out{};
    std::array<ml::merge_entry<float>, ml::merge_scratch_size(48)> scratch{};
    const auto score = [](std::uint32_t id) { return scores[id]; };
    const std::size_t n = ml::sentencepiece_bpe_encode(std::string_view(escaped.data(), e), vocab, score, bytes, out, scratch).unwrap();
    return std::pair{out, n};
}
// "▁Hi!": "Hi" (-1) beats "▁" + "H"; then "▁" + "Hi" is "▁Hi" (-0.5); "!" has no piece, so its byte token
static_assert(encode("Hi!").second == 2 && encode("Hi!").first[0] == 5 && encode("Hi!").first[1] == 6);
//! [sentencepiece]

} // namespace

int main()
{
    std::printf("snicholls-core sentencepiece demo\n");
    bool ok = true;
    const auto [tokens, n] = encode("Hi Hi?");
    std::printf("  \"Hi Hi?\" ->");
    for (std::size_t i = 0; i < n; ++i)
        std::printf(" %u", tokens[i]);
    std::printf("\n");
    ok &= n == 3 && tokens[0] == 5 && tokens[1] == 5 && tokens[2] == 7;

    std::array<char, 32> text{};
    const auto m = ml::sentencepiece_decode(std::span<const std::uint32_t>(tokens.data(), n), [](std::uint32_t id) { return pieces[id]; },
                                            [](std::uint32_t id) { return kinds[id]; }, text);
    std::printf("  decoded: \"%.*s\"\n", static_cast<int>(m.unwrap()), text.data());
    ok &= std::string_view(text.data(), m.unwrap()) == "Hi Hi?";

    std::printf(ok ? "sentencepiece demo: ok\n" : "sentencepiece demo: FAILED\n");
    return ok ? 0 : 1;
}
