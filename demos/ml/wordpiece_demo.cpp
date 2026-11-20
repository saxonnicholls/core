//
//  wordpiece_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/wordpiece.hpp>: a BERT vocabulary held by the compiler, text lowercased,
//  accent-stripped, split at punctuation and cut into "##" continuations at compile time, a word that cannot be
//  cut, and decoding back. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/wordpiece.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <utility>

namespace {

namespace ml = snicholls::ml;

//! [wordpiece]
constexpr std::array<std::string_view, 7> pieces{"[UNK]", "[CLS]", "un", "##aff", "##able", "cafe", "!"};

constexpr auto encode(std::string_view text)
{
    std::array<std::uint32_t, 7> order{};
    ml::build_token_index(pieces, order);
    const ml::token_trie vocab{pieces, order};
    std::array<std::uint32_t, 32> out{};
    std::array<char, 64> word{};
    const std::size_t n = ml::wordpiece_encode(text, vocab, 0, {}, out, word).unwrap();
    return std::pair{out, n};
}
// "UnAffable Café!" -> un ##aff ##able, cafe, !
static_assert(encode("UnAffable Caf\xC3\xA9!").second == 5 && encode("UnAffable Caf\xC3\xA9!").first[3] == 5);
//! [wordpiece]

} // namespace

int main()
{
    std::printf("snicholls-core wordpiece demo\n");
    bool ok = true;
    const auto [tokens, n] = encode("unaffable unx!");
    std::printf("  \"unaffable unx!\" ->");
    for (std::size_t i = 0; i < n; ++i)
        std::printf(" %u", tokens[i]);
    std::printf("   (unx cannot be cut: [UNK])\n");
    ok &= n == 5 && tokens[0] == 2 && tokens[3] == 0 && tokens[4] == 6;

    std::array<char, 64> text{};
    const auto m = ml::wordpiece_decode(std::span<const std::uint32_t>(tokens.data(), n), [](std::uint32_t id) { return pieces[id]; },
                                        [](std::uint32_t id) { return id == 1; }, {}, text);
    std::printf("  decoded: \"%.*s\"\n", static_cast<int>(m.unwrap()), text.data());
    ok &= std::string_view(text.data(), m.unwrap()) == "unaffable [UNK] !";

    std::printf(ok ? "wordpiece demo: ok\n" : "wordpiece demo: FAILED\n");
    return ok ? 0 : 1;
}
