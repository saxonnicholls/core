//
//  tokenizer_fixture.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The committed tokenizer fixtures (tests/fixtures/ml/): hex helpers, a line reader, a tokenizer over owned
//  vectors (what scripts/generate_tokenizer_fixtures.cpp writes and tests/ml/tokenizer_test.cpp reads back), and a
//  builder of SentencePiece's precompiled charsmap (a XOR-compressed double array) from (key, replacement) pairs.
//  Test support only; the library never reads files.
//

#ifndef snicholls_core_tests_support_tokenizer_fixture_hpp
#define snicholls_core_tests_support_tokenizer_fixture_hpp

#include <snicholls/core/ml/tokenizer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::test
{

inline std::string unhex(std::string_view in)
{
    if (in == "-")
        return {};
    if (in.size() % 2 != 0)
        throw std::runtime_error("tokenizer fixture: odd hex string");
    const auto digit = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        throw std::runtime_error("tokenizer fixture: bad hex digit");
    };
    std::string out;
    for (std::size_t i = 0; i < in.size(); i += 2)
        out.push_back(static_cast<char>(digit(in[i]) * 16 + digit(in[i + 1])));
    return out;
}

inline std::string hex(std::string_view in)
{
    static constexpr char digits[] = "0123456789abcdef";
    if (in.empty())
        return "-";
    std::string out;
    for (const char c : in) {
        out.push_back(digits[static_cast<unsigned char>(c) >> 4]);
        out.push_back(digits[static_cast<unsigned char>(c) & 15]);
    }
    return out;
}

// A file's lines, split into words, skipping comments
inline std::vector<std::vector<std::string>> read_lines(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("tokenizer fixture: cannot open " + path);
    std::vector<std::vector<std::string>> lines;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream words(line);
        std::vector<std::string> w;
        for (std::string word; words >> word;)
            w.push_back(word);
        lines.push_back(std::move(w));
    }
    return lines;
}

#ifdef SNICHOLLS_CORE_FIXTURE_DIR
inline std::vector<std::vector<std::string>> fixture_lines(const std::string& name)
{
    return read_lines(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/" + name);
}
#endif

// SentencePiece's precompiled charsmap from (key, replacement) pairs: a trie of the keys laid out as a XOR double
// array, every node's children at p ^ c for a block position p of its own and its BASE stored as index ^ p, and a
// key's end as a child labelled 0 holding the replacement's offset (Darts-clone's format, which
// ml::precompiled_charsmap reads).
inline std::string build_charsmap(const std::map<std::string, std::string>& entries)
{
    if (entries.empty())
        return {};
    struct node
    {
        std::map<unsigned char, std::size_t> children;
        long value = -1;
    };
    std::vector<node> trie(1);
    std::string strings;
    for (const auto& [key, replacement] : entries) {
        std::size_t n = 0;
        for (const char c : key) {
            const auto b = static_cast<unsigned char>(c);
            if (b == 0)
                throw std::runtime_error("build_charsmap: a key holds NUL");
            auto it = trie[n].children.find(b);
            if (it == trie[n].children.end()) {
                trie.push_back({});
                it = trie[n].children.emplace(b, trie.size() - 1).first;
            }
            n = it->second;
        }
        trie[n].value = static_cast<long>(strings.size());
        strings += replacement;
        strings.push_back('\0');
    }
    std::vector<std::uint32_t> units(1, 0);
    std::vector<bool> used(1, true);   // index 0 is the root
    std::set<std::uint32_t> blocks;
    const auto ensure = [&](std::size_t i) {
        if (i >= units.size()) {
            units.resize(i + 1, 0);
            used.resize(i + 1, false);
        }
    };
    // Breadth first: (trie node, its index in the array)
    std::vector<std::pair<std::size_t, std::uint32_t>> queue{{0, 0}};
    for (std::size_t q = 0; q < queue.size(); ++q) {
        const auto [t, x] = queue[q];
        std::vector<unsigned> labels;
        if (trie[t].value >= 0)
            labels.push_back(0);
        for (const auto& [c, child] : trie[t].children)
            labels.push_back(c);
        std::uint32_t p = 1;
        for (;; ++p) {
            if (blocks.count(p))
                continue;
            bool free = true;
            for (const unsigned c : labels) {
                const std::uint32_t i = p ^ c;
                if (i == 0 || (i < used.size() && used[i])) {
                    free = false;
                    break;
                }
            }
            if (free)
                break;
        }
        blocks.insert(p);
        const std::uint32_t base = x ^ p;
        if (base >= (1u << 21))
            throw std::runtime_error("build_charsmap: the array outgrew the unshifted BASE");
        units[x] |= base << 10;
        if (trie[t].value >= 0) {
            ensure(p);
            used[p] = true;
            units[p] = 0x80000000u | static_cast<std::uint32_t>(trie[t].value);
            units[x] |= 1u << 8;
        }
        for (const auto& [c, child] : trie[t].children) {
            const std::uint32_t i = p ^ c;
            ensure(i);
            used[i] = true;
            units[i] |= c;
            queue.emplace_back(child, i);
        }
    }
    std::string blob;
    const auto put32 = [&](std::uint32_t v) {
        for (int k = 0; k < 4; ++k)
            blob.push_back(static_cast<char>((v >> (8 * k)) & 0xFF));
    };
    put32(static_cast<std::uint32_t>(4 * units.size()));
    for (const auto u : units)
        put32(u);
    return blob + strings;
}

// Everything a tokenizer fixture records about a tokenizer
struct tokenizer_config
{
    ml::tokenizer_model model = ml::tokenizer_model::bpe;
    std::string pre = "-";
    bool byte_level = true, ignore_merges = false, escape_whitespaces = false, add_space_prefix = true;
    ml::unigram_normalizer_options normalizer;
    bool lowercase = true, strip_accents = true;
    std::string word_prefix, continuation_prefix = "##";
    std::size_t max_word_chars = 100;
    std::uint32_t unknown = ml::no_token;
    double unknown_score = 0;
};

inline const char* model_name(ml::tokenizer_model m)
{
    switch (m) {
        case ml::tokenizer_model::sentencepiece_bpe: return "sentencepiece_bpe";
        case ml::tokenizer_model::bpe: return "bpe";
        case ml::tokenizer_model::unigram: return "unigram";
        case ml::tokenizer_model::wordpiece: return "wordpiece";
        case ml::tokenizer_model::rwkv: return "rwkv";
    }
    return "?";
}

inline ml::tokenizer_model model_from(std::string_view name)
{
    for (const auto m : {ml::tokenizer_model::sentencepiece_bpe, ml::tokenizer_model::bpe, ml::tokenizer_model::unigram,
                         ml::tokenizer_model::wordpiece, ml::tokenizer_model::rwkv})
        if (name == model_name(m))
            return m;
    throw std::runtime_error("tokenizer fixture: unknown model");
}

// One test text, llama.cpp's tokens, and core's where they differ (empty when they agree)
struct tokenizer_case
{
    std::string text;
    std::vector<std::uint32_t> llama;
    std::vector<std::uint32_t> core;
    bool differs = false;
};

// A tokenizer over owned vectors; index() rebuilds its view
struct owned_tokenizer
{
    tokenizer_config cfg;
    std::vector<std::string> storage;           // the pieces' bytes, when read from a fixture
    std::vector<std::string_view> pieces;
    std::vector<float> scores;
    std::vector<ml::token_kind> kinds;
    std::vector<bool> present;
    std::vector<std::uint32_t> order, added_order, merge_order;
    std::vector<std::pair<std::string_view, std::string_view>> merges;
    std::vector<std::string> merge_storage;
    std::string charsmap;
    std::map<std::string, std::string> charsmap_entries;
    std::vector<tokenizer_case> cases;
    ml::tokenizer_vocab view;

    owned_tokenizer() = default;
    owned_tokenizer(const owned_tokenizer&) = delete;
    owned_tokenizer& operator=(const owned_tokenizer&) = delete;

    void index()
    {
        view = {};
        view.model = cfg.model;
        view.pieces = pieces;
        view.scores = scores;
        view.kinds = kinds;
        const auto in_vocab = [&](std::uint32_t id) {
            if (!present[id])
                return false;
            if (cfg.model != ml::tokenizer_model::unigram)
                return true;
            const auto k = kinds[id];
            return k == ml::token_kind::normal || k == ml::token_kind::user_defined || k == ml::token_kind::unused;
        };
        order.assign(pieces.size(), 0);
        order.resize(ml::build_token_index(pieces, order, in_vocab));
        added_order.assign(pieces.size(), 0);
        added_order.resize(ml::build_token_index(pieces, added_order,
                                                 [&](std::uint32_t id) { return present[id] && kinds[id] == ml::token_kind::user_defined; }));
        merge_order.assign(merges.size(), 0);
        ml::build_merge_index(merges, merge_order);
        view.vocab = {pieces, order};
        view.added = {pieces, added_order};
        view.merges = {merges, merge_order};
        if (cfg.pre != "-")
            view.pre = ml::pretokenizer_for(cfg.pre).unwrap();
        view.byte_level = cfg.byte_level;
        view.ignore_merges = cfg.ignore_merges;
        view.escape_whitespaces = cfg.escape_whitespaces;
        view.add_space_prefix = cfg.add_space_prefix;
        view.charsmap = {charsmap};
        view.normalizer = cfg.normalizer;
        view.wordpiece.lowercase = cfg.lowercase;
        view.wordpiece.strip_accents = cfg.strip_accents;
        view.wordpiece.word_prefix = cfg.word_prefix;
        view.wordpiece.continuation_prefix = cfg.continuation_prefix;
        view.wordpiece.max_word_chars = cfg.max_word_chars;
        view.unknown = cfg.unknown;
        view.unknown_score = cfg.unknown_score;
        const auto find = [&](std::string_view s) { return view.vocab.find(s); };
        view.bytes = cfg.model == ml::tokenizer_model::bpe && cfg.byte_level ? ml::make_byte_level_fallback(find) : ml::make_byte_fallback(find);
    }

    std::vector<std::uint32_t> tokenize(std::string_view text) const
    {
        const ml::tokenizer_scratch_need need(view, text.size());
        std::vector<char> t(need.text);
        std::vector<ml::merge_entry<std::uint32_t>> r(need.ranked);
        std::vector<ml::merge_entry<float>> s(need.scored);
        std::vector<ml::unigram_lattice_entry> l(need.lattice);
        std::vector<std::uint32_t> out(3 * text.size() + 3);
        const auto n = ml::tokenize(view, text, out, {t, r, s, l});
        if (n.is_err())
            throw std::runtime_error("tokenize failed");
        out.resize(n.unwrap());
        return out;
    }

    std::string detokenize(const std::vector<std::uint32_t>& tokens) const
    {
        std::string out(16 * tokens.size() + 16, '\0');
        const auto n = ml::detokenize(view, tokens, std::span<char>(out));
        if (n.is_err())
            throw std::runtime_error("detokenize failed");
        out.resize(n.unwrap());
        return out;
    }

    // Reads a fixture written by scripts/generate_tokenizer_fixtures.cpp
    void load(const std::string& path)
    {
        const auto lines = read_lines(path);
        std::map<std::uint32_t, std::size_t> slot;
        std::size_t size = 0;
        std::vector<std::pair<std::string, std::string>> merge_text;
        for (const auto& w : lines) {
            const std::string& k = w[0];
            if (k == "model") {
                cfg.model = model_from(w[1]);
            } else if (k == "pre") {
                cfg.pre = w[1];
            } else if (k == "flags") {
                const auto b = [&](std::size_t i) { return w.at(i) == "1"; };
                cfg.byte_level = b(1);
                cfg.ignore_merges = b(2);
                cfg.escape_whitespaces = b(3);
                cfg.add_space_prefix = b(4);
                cfg.normalizer = {b(5), b(6), b(7), b(8)};
            } else if (k == "wordpiece") {
                cfg.lowercase = w.at(1) == "1";
                cfg.strip_accents = w.at(2) == "1";
                cfg.word_prefix = unhex(w.at(3));
                cfg.continuation_prefix = unhex(w.at(4));
                cfg.max_word_chars = std::stoul(w.at(5));
            } else if (k == "unknown") {
                const long id = std::stol(w.at(1));
                cfg.unknown = id < 0 ? ml::no_token : static_cast<std::uint32_t>(id);
                cfg.unknown_score = std::strtod(w.at(2).c_str(), nullptr);
            } else if (k == "size") {
                size = std::stoul(w.at(1));
                storage.assign(size, {});
                scores.assign(size, 0.0f);
                kinds.assign(size, ml::token_kind::undefined);
                present.assign(size, false);
            } else if (k == "token") {
                const auto id = static_cast<std::uint32_t>(std::stoul(w.at(1)));
                kinds.at(id) = static_cast<ml::token_kind>(std::stoi(w.at(2)));
                scores.at(id) = std::strtof(w.at(3).c_str(), nullptr);
                storage.at(id) = unhex(w.at(4));
                present.at(id) = true;
            } else if (k == "merge") {
                merge_text.emplace_back(unhex(w.at(1)), unhex(w.at(2)));
            } else if (k == "charsmap") {
                charsmap_entries[unhex(w.at(1))] = unhex(w.at(2));
            } else if (k == "case") {
                cases.push_back({unhex(w.at(1)), {}, {}, false});
            } else if (k == "llama" || k == "core") {
                auto& v = k == "llama" ? cases.back().llama : cases.back().core;
                for (std::size_t i = 1; i < w.size(); ++i)
                    v.push_back(static_cast<std::uint32_t>(std::stoul(w[i])));
                if (k == "core")
                    cases.back().differs = true;
            } else {
                throw std::runtime_error("tokenizer fixture: unknown line " + k);
            }
        }
        pieces.assign(storage.begin(), storage.end());
        merge_storage.clear();
        merge_storage.reserve(2 * merge_text.size());
        for (const auto& [l, r] : merge_text) {
            merge_storage.push_back(l);
            merge_storage.push_back(r);
        }
        merges.clear();
        for (std::size_t i = 0; i < merge_text.size(); ++i)
            merges.emplace_back(merge_storage[2 * i], merge_storage[2 * i + 1]);
        if (cfg.model == ml::tokenizer_model::unigram)
            charsmap = build_charsmap(charsmap_entries);
        for (auto& c : cases)
            if (!c.differs)
                c.core = c.llama;
        index();
    }
};

} // namespace snicholls::test

#endif // snicholls_core_tests_support_tokenizer_fixture_hpp
