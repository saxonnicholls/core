//
//  token_trie_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/token_trie.hpp>: a vocabulary indexed by the compiler, exact lookups, the longest
//  token at the start of a text, and a byte-by-byte walk listing every token that is a prefix; a run-time index
//  with a filter. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/token_trie.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

//! [token_trie]
// A vocabulary the compiler sorts: ids are positions in the array
constexpr ml::static_token_trie<6> vocab{std::array<std::string_view, 6>{"<unk>", "un", "unaff", "able", "a", "unaffable"}};
static_assert(vocab.view().find("able") == 3 && vocab.view().find("abl") == ml::no_token);

// Every token that is a prefix of "unaffable!", by walking the trie one byte at a time
constexpr std::size_t prefixes()
{
    const ml::token_trie t = vocab.view();
    std::size_t found = 0;
    auto c = t.root();
    for (const char b : std::string_view("unaffable!"))
        if (!t.advance(c, b))
            break;
        else if (t.at(c) != ml::no_token)
            ++found;   // "un", "unaff", "unaffable"
    return found;
}
static_assert(prefixes() == 3);
//! [token_trie]

} // namespace

int main()
{
    std::printf("snicholls-core token_trie demo\n");
    bool ok = true;
    std::size_t n = 0;
    const std::uint32_t id = vocab.view().longest_prefix("unaffables", n);
    std::printf("  longest prefix of \"unaffables\": id %u, %zu bytes\n", id, n);
    ok &= id == 5 && n == 9;

    // At run time over pieces a tokenizer file holds, keeping only some ids (Unigram keeps no control tokens)
    const std::vector<std::string_view> pieces{"<s>", "hello", "</s>", "hell", "o"};
    std::vector<std::uint32_t> order(pieces.size());
    order.resize(ml::build_token_index(pieces, order, [&](std::uint32_t i) { return pieces[i][0] != '<'; }));
    const ml::token_trie trie{pieces, order};
    std::printf("  %zu of %zu pieces indexed; \"hello\" is %u, \"<s>\" is %s\n", order.size(), pieces.size(), trie.find("hello"),
                trie.find("<s>") == ml::no_token ? "not indexed" : "?");
    ok &= order.size() == 3 && trie.find("hello") == 1 && trie.find("<s>") == ml::no_token;

    std::printf(ok ? "token_trie demo: ok\n" : "token_trie demo: FAILED\n");
    return ok ? 0 : 1;
}
