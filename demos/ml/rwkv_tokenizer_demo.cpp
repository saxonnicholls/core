//
//  rwkv_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/rwkv_tokenizer.hpp>: an RWKV vocabulary's escaped pieces read back, indexed, and text
//  tokenized by greedy longest match at compile time and at run time, with an unknown byte. Exits non-zero if a
//  result is not what it documents.
//

#include <snicholls/core/ml/rwkv_tokenizer.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

//! [rwkv]
// Greedy longest match: "the" beats "th" beats "t"
constexpr ml::static_token_trie<5> vocab{std::array<std::string_view, 5>{"t", "th", "the", " ", "\n"}};
constexpr std::size_t count(std::string_view text)
{
    std::array<std::uint32_t, 32> out{};
    return ml::rwkv_encode(text, vocab.view(), 99, out).unwrap();
}
static_assert(count("the th t\n") == 6);   // "the", " ", "th", " ", "t", "\n"
//! [rwkv]

} // namespace

int main()
{
    std::printf("snicholls-core rwkv demo\n");
    bool ok = true;
    // A vocabulary file's pieces are escaped; read them back once, when loading
    const std::vector<std::string_view> escaped{"\\n", "\\x41B", "C\\\\", "\\t"};
    std::vector<std::string> storage;
    for (const auto e : escaped) {
        std::string s(e.size(), '\0');
        s.resize(ml::rwkv_unescape(e, std::span<char>(s)).unwrap());
        storage.push_back(s);
    }
    ok &= storage[0] == "\n" && storage[1] == "AB" && storage[2] == "C\\" && storage[3] == "\t";
    const std::vector<std::string_view> pieces(storage.begin(), storage.end());
    std::vector<std::uint32_t> order(pieces.size());
    ml::build_token_index(pieces, order);
    std::array<std::uint32_t, 16> out{};
    const auto n = ml::rwkv_encode("AB\nAZ", {pieces, order}, 99, out).unwrap();
    std::printf("  \"AB\\nAZ\" ->");
    for (std::size_t i = 0; i < n; ++i)
        std::printf(" %u", out[i]);
    std::printf("   (A and Z alone are unknown)\n");
    ok &= n == 4 && out[0] == 1 && out[1] == 0 && out[2] == 99 && out[3] == 99;

    std::printf(ok ? "rwkv demo: ok\n" : "rwkv demo: FAILED\n");
    return ok ? 0 : 1;
}
