//
//  token_trie_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/token_trie.hpp>: find, longest_prefix and the cursor walk against a linear scan on
//  random vocabularies over a small alphabet (dense in shared prefixes) with bytes above 0x7F; duplicates (the
//  largest id wins); the empty piece; a filtered index; and a trie the compiler builds.
//

#include <snicholls/core/ml/token_trie.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr ml::static_token_trie<6> fixed{std::array<std::string_view, 6>{"he", "hello", "help", "h", "world", "\xE2\x96\x81" "a"}};
static_assert(fixed.view().find("hello") == 1 && fixed.view().find("hel") == ml::no_token && fixed.view().find("") == ml::no_token);
static_assert(fixed.view().find("\xE2\x96\x81" "a") == 5 && fixed.view()("world") == 4);
constexpr std::uint64_t longest(std::string_view s)
{
    std::size_t n = 0;
    return std::uint64_t{fixed.view().longest_prefix(s, n)} * 100 + n;
}
static_assert(longest("helpers") == 2 * 100 + 4 && longest("hex") == 0 * 100 + 2 && longest("x") == std::uint64_t{ml::no_token} * 100);
static_assert(noexcept(fixed.view().find(std::string_view{})));

TEST(MlTokenTrie, AgreesWithALinearScanOnRandomVocabularies)
{
    snicholls::stats::wyrand rng(7);
    const std::string alphabet = "ab\xC3\xA9";
    for (int round = 0; round < 50; ++round) {
        std::vector<std::string> storage;
        const std::size_t v = 1 + rng() % 60;
        for (std::size_t i = 0; i < v; ++i) {
            std::string p;
            for (std::size_t k = 0, len = rng() % 6; k < len; ++k)
                p.push_back(alphabet[rng() % alphabet.size()]);
            storage.push_back(p);
        }
        const std::vector<std::string_view> pieces(storage.begin(), storage.end());
        std::vector<std::uint32_t> order(pieces.size());
        ASSERT_EQ(ml::build_token_index(pieces, order), pieces.size());
        const ml::token_trie trie{pieces, order};
        const auto scan = [&](std::string_view s) {
            std::uint32_t best = ml::no_token;
            for (std::uint32_t id = 0; id < pieces.size(); ++id)
                if (pieces[id] == s)
                    best = id;   // the largest id
            return best;
        };
        for (int q = 0; q < 200; ++q) {
            std::string text;
            for (std::size_t k = 0, len = rng() % 8; k < len; ++k)
                text.push_back(alphabet[rng() % alphabet.size()]);
            EXPECT_EQ(trie.find(text), scan(text));
            // The cursor walk finds exactly the prefixes that are pieces
            auto c = trie.root();
            std::size_t best_len = 0;
            std::uint32_t best = scan("");
            for (std::size_t i = 0; i < text.size() && trie.advance(c, text[i]); ++i) {
                EXPECT_EQ(trie.at(c), scan(text.substr(0, i + 1)));
                if (trie.at(c) != ml::no_token) {
                    best = trie.at(c);
                    best_len = i + 1;
                }
            }
            std::size_t n = 99;
            const std::uint32_t lp = trie.longest_prefix(text, n);
            if (best_len > 0) {
                EXPECT_EQ(lp, best);
                EXPECT_EQ(n, best_len);
            } else {
                EXPECT_EQ(n, 0u);
            }
        }
    }
}

TEST(MlTokenTrie, AFilterKeepsOnlyTheChosenIds)
{
    const std::array<std::string_view, 4> pieces{"<s>", "a", "</s>", "b"};
    std::array<std::uint32_t, 4> order{};
    const std::size_t n = ml::build_token_index(pieces, order, [](std::uint32_t id) { return id % 2 == 1; });
    ASSERT_EQ(n, 2u);
    const ml::token_trie trie{pieces, std::span<const std::uint32_t>(order).first(n)};
    EXPECT_EQ(trie.find("a"), 1u);
    EXPECT_EQ(trie.find("<s>"), ml::no_token);
    const ml::token_trie empty{};
    EXPECT_EQ(empty.find("a"), ml::no_token);
    auto c = empty.root();
    EXPECT_TRUE(c.empty());
    EXPECT_FALSE(empty.advance(c, 'a'));
}

TEST(MlTokenTrie, TokenKindsAreGgufsNumbers)
{
    EXPECT_EQ(static_cast<int>(ml::token_kind::normal), 1);
    EXPECT_EQ(static_cast<int>(ml::token_kind::control), 3);
    EXPECT_EQ(static_cast<int>(ml::token_kind::user_defined), 4);
    EXPECT_EQ(static_cast<int>(ml::token_kind::byte), 6);
}

} // namespace
