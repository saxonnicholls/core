//
//  unigram_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/unigram.hpp>: the precompiled charsmap (a double array built by
//  tests/support/tokenizer_fixture.hpp, longest match, malformed blobs), the normaliser's whitespace rules (dummy
//  prefix, merged and kept spaces, suffix, user-defined pieces, invalid bytes), and Viterbi against an exhaustive
//  search over every segmentation on random texts; unknown runs, user-defined scores, the errors. Token-for-token
//  agreement with llama.cpp on nomic-embed's XLM-R vocabulary is in tests/ml/tokenizer_test.cpp.
//

#include <snicholls/core/ml/unigram.hpp>
#include <snicholls/core/stats/random.hpp>

#include "../support/tokenizer_fixture.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

std::string normalize(std::string_view text, const ml::precompiled_charsmap& map, ml::unigram_normalizer_options o,
                      const ml::token_trie& user = {})
{
    std::string out(ml::unigram_normalized_size(text.size(), map), '\0');
    out.resize(ml::unigram_normalize(text, map, user, o, std::span<char>(out)).unwrap());
    return out;
}

const std::string sp = "\xE2\x96\x81";

TEST(MlUnigram, TheCharsmapFindsTheLongestKey)
{
    const std::string blob = snicholls::test::build_charsmap({{"a", "A"}, {"ab", "X"}, {"abc", ""}, {"\xEF\xAC\x81", "fi"}, {"q", "qq"}});
    const ml::precompiled_charsmap map{blob};
    EXPECT_TRUE(map.valid());
    const auto m = [&](std::string_view s) {
        const auto r = map.match(s, 0);
        return std::string(r.replacement) + "/" + std::to_string(r.consumed);
    };
    EXPECT_EQ(m("a"), "A/1");
    EXPECT_EQ(m("abd"), "X/2");
    EXPECT_EQ(m("abcd"), "/3");
    EXPECT_EQ(m("\xEF\xAC\x81!"), "fi/3");
    EXPECT_EQ(m("b"), "/0");
    EXPECT_EQ(m(""), "/0");
    EXPECT_EQ(map.max_replacement(), 2u);
    // A blob too short for its header, or whose array runs past its end, matches nothing and is not valid
    EXPECT_FALSE((ml::precompiled_charsmap{std::string_view("\x08\x00", 2)}.valid()));
    EXPECT_FALSE((ml::precompiled_charsmap{std::string_view("\xFF\x00\x00\x00", 4)}.valid()));
    EXPECT_EQ((ml::precompiled_charsmap{std::string_view("\xFF\x00\x00\x00", 4)}.match("a", 0).consumed), 0u);
    EXPECT_TRUE(ml::precompiled_charsmap{}.valid());
    EXPECT_EQ(ml::precompiled_charsmap{}.match("a", 0).consumed, 0u);
}

TEST(MlUnigram, TheNormaliserFollowsSentencePiecesWhitespaceRules)
{
    const std::string blob = snicholls::test::build_charsmap({{"\t", " "}, {"\xEF\xAC\x81", "fi"}});
    const ml::precompiled_charsmap map{blob};
    ml::unigram_normalizer_options o;   // dummy prefix, merged spaces, escaped
    EXPECT_EQ(normalize("  a \t b  ", map, o), sp + "a" + sp + "b");
    EXPECT_EQ(normalize("\xEF\xAC\x81x", map, o), sp + "fix");
    EXPECT_EQ(normalize("", map, o), "");
    EXPECT_EQ(normalize("   ", map, o), "");
    o.remove_extra_whitespaces = false;
    EXPECT_EQ(normalize(" a  b", map, o), sp + sp + "a" + sp + sp + "b");
    o.add_dummy_prefix = false;
    EXPECT_EQ(normalize("a b", map, o), "a" + sp + "b");
    o.escape_whitespaces = false;
    EXPECT_EQ(normalize("a b", map, o), "a b");
    o = {};
    o.treat_whitespace_as_suffix = true;
    EXPECT_EQ(normalize("a b", map, o), sp + "a" + sp + "b" + sp);
    o = {};
    // Invalid bytes become U+FFFD; user-defined pieces pass through before the charsmap sees them
    EXPECT_EQ(normalize("a\x80", map, o), sp + "a\xEF\xBF\xBD");
    const std::array<std::string_view, 1> user{"\xEF\xAC\x81"};
    std::array<std::uint32_t, 1> order{};
    ml::build_token_index(user, order);
    EXPECT_EQ(normalize("\xEF\xAC\x81", map, o, {user, order}), sp + "\xEF\xAC\x81");
    std::array<char, 2> small{};
    EXPECT_EQ(ml::unigram_normalize("abc", map, {}, o, small).unwrap_err(), ml::unigram_error::output_too_small);
    EXPECT_EQ(ml::unigram_normalize("a", ml::precompiled_charsmap{std::string_view("\xFF\x00\x00\x00", 4)}, {}, o, small).unwrap_err(),
              ml::unigram_error::bad_charsmap);
}

TEST(MlUnigram, ViterbiFindsTheBestSegmentationOnRandomTexts)
{
    snicholls::stats::wyrand rng(5);
    for (int round = 0; round < 100; ++round) {
        std::vector<std::string> storage{"<unk>"};
        std::vector<float> scores{0};
        for (std::size_t i = 0, n = 2 + rng() % 12; i < n; ++i) {
            std::string p;
            for (std::size_t k = 0, len = 1 + rng() % 3; k < len; ++k)
                p.push_back("abc"[rng() % 3]);
            storage.push_back(p);
            scores.push_back(-static_cast<float>(1 + rng() % 1000000) / 100000.0f);
        }
        const std::vector<std::string_view> pieces(storage.begin(), storage.end());
        std::vector<std::uint32_t> order(pieces.size());
        order.resize(ml::build_token_index(pieces, order, [](std::uint32_t id) { return id != 0; }));
        const ml::token_trie trie{pieces, order};
        const double unk = -20;
        for (int q = 0; q < 10; ++q) {
            std::string text;
            for (std::size_t k = 0, len = rng() % 10; k < len; ++k)
                text.push_back("abcd"[rng() % 4]);
            // The definition: every segmentation into pieces, or into single unknown bytes where no one-byte piece
            // exists, the best kept (relaxed in the same order, so ties fall the same way); consecutive unknowns are one
            std::vector<double> best(text.size() + 1, -1e300);
            std::vector<std::pair<std::size_t, std::uint32_t>> from(text.size() + 1, {0, 0});
            best[0] = 0;
            for (std::size_t i = 0; i < text.size(); ++i) {
                for (std::size_t len = 1; i + len <= text.size(); ++len) {
                    const std::uint32_t id = trie.find(std::string_view(text).substr(i, len));
                    if (id != ml::no_token && best[i] + scores[id] > best[i + len]) {
                        best[i + len] = best[i] + scores[id];
                        from[i + len] = {i, id};
                    }
                }
                if (trie.find(std::string_view(text).substr(i, 1)) == ml::no_token && best[i] + unk > best[i + 1]) {
                    best[i + 1] = best[i] + unk;
                    from[i + 1] = {i, 0};
                }
            }
            std::vector<std::uint32_t> want;
            for (std::size_t at = text.size(); at > 0; at = from[at].first)
                if (!(from[at].second == 0 && !want.empty() && want.back() == 0))
                    want.push_back(from[at].second);
            std::reverse(want.begin(), want.end());
            std::vector<std::uint32_t> out(text.size());
            std::vector<ml::unigram_lattice_entry> lattice(text.size() + 1);
            const auto n = ml::unigram_encode(text, trie, [&](std::uint32_t id) { return scores[id]; },
                                              [](std::uint32_t) { return ml::token_kind::normal; }, {0, unk}, out, lattice)
                               .unwrap();
            out.resize(n);
            EXPECT_EQ(out, want) << text;
        }
    }
}

TEST(MlUnigram, UnknownRunsAreOneTokenAndUserDefinedPiecesScoreZero)
{
    const std::array<std::string_view, 4> pieces{"<unk>", "a", "ab", "b"};
    const std::array<float, 4> scores{0, -1, -5, -1};
    std::array<std::uint32_t, 4> order{};
    const std::size_t n = ml::build_token_index(pieces, order, [](std::uint32_t id) { return id != 0; });
    const ml::token_trie trie{pieces, std::span<const std::uint32_t>(order).first(n)};
    std::array<std::uint32_t, 16> out{};
    std::array<ml::unigram_lattice_entry, 17> lattice{};
    const auto score = [&](std::uint32_t id) { return scores[id]; };
    const auto normal = [](std::uint32_t) { return ml::token_kind::normal; };
    auto k = ml::unigram_encode("axyb", trie, score, normal, {0, -10}, out, lattice).unwrap();
    EXPECT_EQ(std::vector<std::uint32_t>(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(k)), (std::vector<std::uint32_t>{1, 0, 3}));
    k = ml::unigram_encode("ab", trie, score, normal, {0, -10}, out, lattice).unwrap();
    EXPECT_EQ(std::vector<std::uint32_t>(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(k)), (std::vector<std::uint32_t>{1, 3}));
    const auto ab_user = [](std::uint32_t id) { return id == 2 ? ml::token_kind::user_defined : ml::token_kind::normal; };
    k = ml::unigram_encode("ab", trie, score, ab_user, {0, -10}, out, lattice).unwrap();
    EXPECT_EQ(std::vector<std::uint32_t>(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(k)), (std::vector<std::uint32_t>{2}));
    EXPECT_EQ(ml::unigram_encode("", trie, score, normal, {0, -10}, out, lattice).unwrap(), 0u);
    EXPECT_EQ(ml::unigram_encode("ab", trie, score, normal, {0, -10}, out, std::span(lattice).first(2)).unwrap_err(),
              ml::unigram_error::lattice_too_small);
    EXPECT_EQ(ml::unigram_encode("ab", trie, score, normal, {0, -10}, std::span(out).first(1), lattice).unwrap_err(),
              ml::unigram_error::output_too_small);
    EXPECT_DOUBLE_EQ(ml::unigram_unknown_score(-20.5f), -30.5);
}

// The whole of it at compile time: a charsmap-free normaliser, a fixed vocabulary
constexpr std::size_t compile_time_count()
{
    constexpr std::array<std::string_view, 4> pieces{"<unk>", "\xE2\x96\x81" "a", "b", "\xE2\x96\x81"};
    std::array<std::uint32_t, 4> order{};
    const std::size_t n = ml::build_token_index(pieces, order, [](std::uint32_t id) { return id != 0; });
    const ml::token_trie trie{pieces, std::span<const std::uint32_t>(order).first(n)};
    std::array<char, 32> norm{};
    const std::size_t len = ml::unigram_normalize("a b", {}, {}, {}, norm).unwrap();
    std::array<std::uint32_t, 32> out{};
    std::array<ml::unigram_lattice_entry, 33> lattice{};
    return ml::unigram_encode(std::string_view(norm.data(), len), trie, [](std::uint32_t) { return -1.0f; },
                              [](std::uint32_t) { return ml::token_kind::normal; }, {0, -10}, out, lattice)
        .unwrap();
}
static_assert(compile_time_count() == 3);   // "▁a", "▁", "b"

} // namespace
