//
//  unigram_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/unigram.hpp>: a Unigram vocabulary with log-probabilities held by the compiler,
//  text normalised (dummy prefix, merged spaces) and segmented by Viterbi at compile time, an unknown run, and
//  decoding back. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/unigram.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <utility>

namespace {

namespace ml = snicholls::ml;

//! [unigram]
constexpr std::array<std::string_view, 6> pieces{"<unk>", "\xE2\x96\x81", "\xE2\x96\x81" "un", "do", "\xE2\x96\x81" "undo", "able"};
constexpr std::array<float, 6> scores{0, -3, -4, -4, -6, -3};

constexpr auto encode(std::string_view text)
{
    std::array<std::uint32_t, 6> order{};
    const std::size_t v = ml::build_token_index(pieces, order, [](std::uint32_t id) { return id != 0; });   // not <unk>
    const ml::token_trie vocab{pieces, std::span<const std::uint32_t>(order).first(v)};
    // No charsmap here; a model's comes from its file (tokenizer.ggml.precompiled_charsmap)
    std::array<char, 64> norm{};
    const std::size_t n = ml::unigram_normalize(text, {}, {}, {}, norm).unwrap();   // "  undoable " -> "▁undoable"
    std::array<std::uint32_t, 64> out{};
    std::array<ml::unigram_lattice_entry, 65> lattice{};
    const ml::unigram_unknown unknown{0, ml::unigram_unknown_score(-6)};
    const std::size_t k = ml::unigram_encode(std::string_view(norm.data(), n), vocab, [](std::uint32_t id) { return scores[id]; },
                                             [](std::uint32_t) { return ml::token_kind::normal; }, unknown, out, lattice)
                              .unwrap();
    return std::pair{out, k};
}
// "▁undo" + "able" scores -9, beating "▁un" + "do" + "able" at -11
static_assert(encode("  undoable ").second == 2 && encode("  undoable ").first[0] == 4 && encode("  undoable ").first[1] == 5);
//! [unigram]

} // namespace

int main()
{
    std::printf("snicholls-core unigram demo\n");
    bool ok = true;
    const auto [tokens, n] = encode("undo xyz able");
    std::printf("  \"undo xyz able\" ->");
    for (std::size_t i = 0; i < n; ++i)
        std::printf(" %u", tokens[i]);
    std::printf("   (xyz is one unknown)\n");
    ok &= n == 5 && tokens[0] == 4 && tokens[1] == 1 && tokens[2] == 0 && tokens[3] == 1 && tokens[4] == 5;

    std::array<char, 32> text{};
    const auto m = ml::sentencepiece_decode(std::span<const std::uint32_t>(tokens.data(), n), [](std::uint32_t id) { return pieces[id]; },
                                            [](std::uint32_t id) { return id == 0 ? ml::token_kind::unknown : ml::token_kind::normal; },
                                            text);
    std::printf("  decoded: \"%.*s\"\n", static_cast<int>(m.unwrap()), text.data());
    ok &= std::string_view(text.data(), m.unwrap()) == "undo  able";

    std::printf(ok ? "unigram demo: ok\n" : "unigram demo: FAILED\n");
    return ok ? 0 : 1;
}
