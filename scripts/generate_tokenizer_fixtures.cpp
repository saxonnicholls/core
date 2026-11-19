//
//  generate_tokenizer_fixtures.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Writes tests/fixtures/ml/tokenizers/<vocab>.txt from llama.cpp itself: for each test text, the tokens
//  llama_tokenize gives (add_special and parse_special off, as llama.cpp's test-tokenizer-0 runs) on the vocabulary
//  in models/ggml-vocab-<vocab>.gguf, and the part of that vocabulary the texts can reach. The texts are llama.cpp's
//  own test strings (the .inp file next to each vocabulary, whose .out file holds the same tokens from Hugging Face)
//  and the edge cases below.
//
//  The part of the vocabulary written is every token whose piece occurs inside a text as the model sees it (the
//  pre-tokenised pieces in GPT-2's alphabet, the escaped text, the normalised text, the words), every merge whose two
//  sides concatenated does, the byte tokens, and for Unigram the charsmap entries the texts reach. Every lookup the
//  model can make on a text is of such a substring, so on the subset it gets the answers it gets on the whole
//  vocabulary, and the tokens are the same. The program checks that: core's tokens on the whole vocabulary and on
//  the subset must agree with each other, and are reported against llama.cpp's.
//
//  Not part of the build: core has no dependencies. Built once against a llama.cpp checkout:
//
//    cmake -S <llama.cpp> -B build -DLLAMA_BUILD_TESTS=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF
//    cmake --build build --target llama
//    c++ -std=c++20 -O2 -I<core>/include -I<core>/tests -I<llama.cpp>/include -I<llama.cpp>/ggml/include \
//        <core>/scripts/generate_tokenizer_fixtures.cpp -Lbuild/bin -lllama -Wl,-rpath,build/bin -o gen
//    ./gen <llama.cpp> <core>/tests/fixtures/ml/tokenizers
//

#include <snicholls/core/ml/tokenizer.hpp>

#include "support/tokenizer_fixture.hpp"

#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ml = snicholls::ml;
using snicholls::test::hex;

namespace {

// The edge cases every vocabulary is also given: empty text, whitespace runs, emoji, CJK, combining marks, NFKC
// forms, case, contractions, digits, code, invalid UTF-8 (with and without an encoded surrogate, which llama.cpp
// accepts as a code point), and text that spells special tokens.
const std::vector<std::string> edge_cases = {
    "",
    "   ",
    "\t\t\n\n  \n \r\n",
    "Hello  world   again    ",
    "emoji \xF0\x9F\x99\x82\xF0\x9F\x9A\x80\xF0\x9F\x94\xA5 \xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA7 "
    "\xE2\x9D\xA4\xEF\xB8\x8F \xF0\x9F\x87\xA6\xF0\x9F\x87\xBA",
    "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xAE\xE3\x83\x86\xE3\x82\xAD\xE3\x82\xB9\xE3\x83\x88 \xE4\xB8\xAD\xE6\x96\x87"
    "\xE5\x88\x86\xE8\xAF\x8D \xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4",
    "e\xCC\x81 n\xCC\x83 a\xCC\x88\xCC\x84 \xC7\x95 caf\xC3\xA9 cafe\xCC\x81",
    "\xEF\xAC\x81 \xEF\xAC\x80 \xE2\x91\xA0 \xEF\xBC\xA1\xEF\xBC\xA2\xEF\xBC\xA3 \xEF\xBD\xB6\xEF\xBE\x80 \xE2\x84\xAB \xC2\xB5",
    "\xC4\xB0stanbul \xCE\xA3\xCE\x8A\xCE\xA3\xCE\xA5\xCE\xA6\xCE\x9F\xCE\xA3 Stra\xC3\x9F" "e",
    "camelCase HTTPServer it's IT'S we'VE don't",
    "1234567890 \xD9\xA1\xD9\xA2\xD9\xA3 \xC2\xBD",
    "def f(x):\n\treturn x**2\n\n",
    "a\x80" "b \xE2\x82 c\xC0\xAF d\xED\xA0\x80 \xF4\x90\x80\x80 z",
    "a\x80" "b \xE2\x82 c\xC0\xAF \xF4\x90\x80\x80 z\xFF",
    "<s> </s> [CLS] <|endoftext|> <unk>",
};

// A GGUF file's metadata, read in place
struct gguf
{
    std::vector<char> data;
    std::map<std::string, std::string_view> strings;
    std::map<std::string, std::uint64_t> integers;
    std::map<std::string, std::vector<std::string_view>> string_arrays;
    std::map<std::string, std::vector<float>> float_arrays;
    std::map<std::string, std::vector<std::int32_t>> int_arrays;
    std::map<std::string, std::string_view> byte_arrays;

    explicit gguf(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        data.assign(std::istreambuf_iterator<char>(in), {});
        std::size_t at = 0;
        const auto take = [&](std::size_t n) {
            const char* p = data.data() + at;
            at += n;
            if (at > data.size())
                throw std::runtime_error("gguf: truncated");
            return p;
        };
        const auto u32 = [&] { std::uint32_t v; std::memcpy(&v, take(4), 4); return v; };
        const auto u64 = [&] { std::uint64_t v; std::memcpy(&v, take(8), 8); return v; };
        const auto str = [&] { const std::uint64_t n = u64(); return std::string_view(take(n), n); };
        if (std::string_view(take(4), 4) != "GGUF")
            throw std::runtime_error("gguf: not a GGUF file");
        u32();
        u64();
        const std::uint64_t kvs = u64();
        const auto scalar_size = [](std::uint32_t t) -> std::size_t {
            switch (t) {
                case 0: case 1: case 7: return 1;
                case 2: case 3: return 2;
                case 4: case 5: case 6: return 4;
                case 10: case 11: case 12: return 8;
                default: return 0;
            }
        };
        for (std::uint64_t k = 0; k < kvs; ++k) {
            const std::string key(str());
            const std::uint32_t type = u32();
            if (type == 8) {
                strings[key] = str();
            } else if (type == 9) {
                const std::uint32_t item = u32();
                const std::uint64_t n = u64();
                if (item == 8) {
                    auto& v = string_arrays[key];
                    for (std::uint64_t i = 0; i < n; ++i)
                        v.push_back(str());
                } else if (item == 6) {
                    auto& v = float_arrays[key];
                    v.resize(n);
                    std::memcpy(v.data(), take(4 * n), 4 * n);
                } else if (item == 5 || item == 4) {
                    auto& v = int_arrays[key];
                    v.resize(n);
                    std::memcpy(v.data(), take(4 * n), 4 * n);
                } else if (item == 0 || item == 1) {
                    byte_arrays[key] = std::string_view(take(n), n);
                } else {
                    take(scalar_size(item) * n);
                }
            } else {
                const std::size_t n = scalar_size(type);
                std::uint64_t v = 0;
                std::memcpy(&v, take(n), n);
                integers[key] = v;
            }
        }
    }

    bool flag(const std::string& key, bool fallback) const
    {
        const auto it = integers.find("tokenizer.ggml." + key);
        return it == integers.end() ? fallback : it->second != 0;
    }
};

using snicholls::test::model_name;
using owned = snicholls::test::owned_tokenizer;

std::vector<std::uint32_t> llama_tokens(const llama_vocab* vocab, const std::string& text)
{
    std::vector<llama_token> out(4 * text.size() + 16);
    const int n = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), out.data(), static_cast<int>(out.size()), false, false);
    if (n < 0)
        throw std::runtime_error("llama_tokenize: output too small");
    return {out.begin(), out.begin() + n};
}

std::string ids(const std::vector<std::uint32_t>& v)
{
    std::string s;
    for (const auto id : v)
        s += " " + std::to_string(id);
    return s;
}

// The strings the model can look up for a text are substrings of these
std::vector<std::string> reachable(const owned& o, const std::string& text)
{
    std::vector<std::string> spans;
    const auto& v = o.view;
    std::vector<std::string> fragments;
    ml::split_added_tokens(text, v.added, [&](std::string_view f) { fragments.emplace_back(f); }, [](std::uint32_t) {});
    for (const auto& f : fragments) {
        switch (v.model) {
            case ml::tokenizer_model::sentencepiece_bpe: {
                std::string e(ml::sentencepiece_escaped_size(f.size()), '\0');
                e.resize(ml::sentencepiece_escape(f, std::span<char>(e), v.add_space_prefix).unwrap());
                spans.push_back(e);
                break;
            }
            case ml::tokenizer_model::bpe: {
                std::string src = f;
                if (!ml::is_valid_utf8(src)) {
                    src.assign(3 * f.size(), '\0');
                    src.resize(ml::replace_invalid_utf8(f, std::span<char>(src)));
                }
                if (v.escape_whitespaces) {
                    const std::string repaired = src;
                    src.assign(ml::sentencepiece_escaped_size(repaired.size()), '\0');
                    src.resize(ml::sentencepiece_escape(repaired, std::span<char>(src), false).unwrap());
                }
                ml::pretokenize(v.pre, src, [&](std::string_view p) {
                    if (!v.byte_level) {
                        spans.emplace_back(p);
                        return;
                    }
                    std::string m(2 * p.size(), '\0');
                    m.resize(ml::byte_level_encode(p, std::span<char>(m)).unwrap());
                    spans.push_back(m);
                });
                break;
            }
            case ml::tokenizer_model::unigram: {
                std::string nrm(ml::unigram_normalized_size(f.size(), v.charsmap), '\0');
                nrm.resize(ml::unigram_normalize(f, v.charsmap, v.added, v.normalizer, std::span<char>(nrm)).unwrap());
                spans.push_back(nrm);
                break;
            }
            case ml::tokenizer_model::wordpiece:
            case ml::tokenizer_model::rwkv: {
                // BERT's basic tokenisation, with "▁" before every word: the pieces looked up are substrings
                const auto& o = v.wordpiece;
                std::string norm(ml::escaped_space);
                const auto put = [&](char32_t c) {
                    char buf[4];
                    norm.append(buf, ml::encode_utf8(c, buf));
                };
                for (std::size_t i = 0; i < f.size();) {
                    const auto d = ml::decode_utf8(f, i);
                    i += d.length;
                    const char32_t cp = d.code_point;
                    if (ml::is_whitespace(cp)) {
                        norm += ml::escaped_space;
                        continue;
                    }
                    if (cp == 0 || cp == ml::replacement_character || ml::is_other(ml::category(cp)))
                        continue;
                    std::array<char32_t, ml::max_decomposition> parts{};
                    const std::size_t n = o.strip_accents ? ml::decompose(cp, parts) : (parts[0] = cp, std::size_t{1});
                    for (std::size_t k = 0; k < n; ++k) {
                        const char32_t q = parts[k];
                        if (o.strip_accents && ml::category(q) == ml::unicode_category::nonspacing_mark)
                            continue;
                        const bool alone = ml::is_bert_cjk(cp) || ml::is_bert_punctuation(q);
                        if (alone)
                            norm += ml::escaped_space;
                        if (o.lowercase && q == U'\u0130') {
                            put(U'i');
                            put(U'\u0307');
                        } else {
                            put(o.lowercase ? ml::to_lower(q) : q);
                        }
                        if (alone)
                            norm += ml::escaped_space;
                    }
                }
                spans.push_back(norm);
                break;
            }
        }
    }
    return spans;
}

void write(const std::string& llama_root, const std::string& name, const std::string& out_dir)
{
    const std::string path = llama_root + "/models/ggml-vocab-" + name + ".gguf";
    const gguf g(path);
    owned full;
    const std::string model(g.strings.at("tokenizer.ggml.model"));
    const auto pre = g.strings.count("tokenizer.ggml.pre") ? std::string(g.strings.at("tokenizer.ggml.pre")) : std::string("-");
    snicholls::test::tokenizer_config& c = full.cfg;
    if (model == "llama") {
        c.model = ml::tokenizer_model::sentencepiece_bpe;
        c.add_space_prefix = g.flag("add_space_prefix", true);
    } else if (model == "gpt2") {
        c.model = ml::tokenizer_model::bpe;
        c.pre = pre;
        c.ignore_merges = pre == "llama-bpe";
    } else if (model == "gemma4") {
        c.model = ml::tokenizer_model::bpe;
        c.pre = "gemma4";
        c.byte_level = false;
        c.escape_whitespaces = true;
    } else if (model == "bert") {
        c.model = ml::tokenizer_model::wordpiece;
        c.word_prefix = std::string(ml::escaped_space);
        c.continuation_prefix = "";
        c.max_word_chars = 0;
        c.unknown = static_cast<std::uint32_t>(g.integers.at("tokenizer.ggml.unknown_token_id"));
    } else if (model == "t5") {
        c.model = ml::tokenizer_model::unigram;
        c.normalizer.add_dummy_prefix = g.flag("add_space_prefix", true);
        c.normalizer.remove_extra_whitespaces = g.flag("remove_extra_whitespaces", true);
        c.unknown = static_cast<std::uint32_t>(g.integers.at("tokenizer.ggml.unknown_token_id"));
        full.charsmap = std::string(g.byte_arrays.at("tokenizer.ggml.precompiled_charsmap"));
    } else {
        throw std::runtime_error("unsupported model " + model);
    }
    const auto& tokens = g.string_arrays.at("tokenizer.ggml.tokens");
    full.pieces = tokens;
    full.present.assign(tokens.size(), true);
    full.scores = g.float_arrays.count("tokenizer.ggml.scores") ? g.float_arrays.at("tokenizer.ggml.scores") : std::vector<float>{};
    for (const auto t : g.int_arrays.at("tokenizer.ggml.token_type"))
        full.kinds.push_back(static_cast<ml::token_kind>(t));
    if (g.string_arrays.count("tokenizer.ggml.merges"))
        for (const auto m : g.string_arrays.at("tokenizer.ggml.merges"))
            full.merges.push_back(ml::split_merge(m));
    if (c.model == ml::tokenizer_model::unigram) {
        float lowest = 1e30f;
        for (std::size_t i = 0; i < tokens.size(); ++i)
            if (full.kinds[i] == ml::token_kind::normal)
                lowest = std::min(lowest, full.scores[i]);
        c.unknown_score = ml::unigram_unknown_score(lowest);
    }
    full.index();

    // The texts: llama.cpp's, then the edge cases
    std::vector<std::string> texts;
    {
        // nomic-bert-moe ships without its own .inp: the same strings as the others
        std::ifstream probe(path + ".inp");
        std::ifstream in(probe ? path + ".inp" : llama_root + "/models/ggml-vocab-llama-spm.gguf.inp", std::ios::binary);
        const std::string all((std::istreambuf_iterator<char>(in)), {});
        const std::string sep = "\n__ggml_vocab_test__\n";
        for (std::size_t at = 0; at < all.size();) {
            const std::size_t end = all.find(sep, at);
            texts.push_back(all.substr(at, end - at));
            at = end == std::string::npos ? all.size() : end + sep.size();
        }
    }
    texts.insert(texts.end(), edge_cases.begin(), edge_cases.end());

    // The reachable part of the vocabulary
    std::unordered_set<std::string> subs;
    for (const auto& t : texts)
        for (const auto& s : reachable(full, t))
            for (std::size_t i = 0; i < s.size(); ++i)
                for (std::size_t j = i + 1; j <= s.size() && j - i <= 96; ++j)
                    subs.insert(s.substr(i, j - i));
    owned sub;
    sub.cfg = c;
    sub.charsmap = full.charsmap;   // replaced below for Unigram
    sub.pieces.assign(tokens.size(), {});
    sub.scores.assign(full.scores.size(), 0.0f);
    sub.kinds.assign(tokens.size(), ml::token_kind::undefined);
    sub.present.assign(tokens.size(), false);
    const auto keep = [&](std::size_t id) {
        sub.present[id] = true;
        sub.pieces[id] = tokens[id];
        if (!full.scores.empty())
            sub.scores[id] = full.scores[id];
        sub.kinds[id] = full.kinds[id];
    };
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        const auto k = full.kinds[id];
        if (subs.count(std::string(tokens[id])) || k == ml::token_kind::byte || id == c.unknown ||
            (k == ml::token_kind::user_defined && std::any_of(texts.begin(), texts.end(), [&](const std::string& t) { return t.find(tokens[id]) != std::string::npos; })))
            keep(id);
    }
    for (std::size_t b = 0; b < 256; ++b) {   // the byte fallback's targets
        if (const auto id = full.view.bytes.ids[b]; id != ml::no_token)
            keep(id);
    }
    for (const auto& [l, r] : full.merges)
        if (subs.count(std::string(l) + std::string(r)))
            sub.merges.emplace_back(l, r);
    // Unigram: the charsmap entries the texts reach, as (key, replacement)
    std::map<std::string, std::string> charsmap_entries;
    if (c.model == ml::tokenizer_model::unigram) {
        for (const auto& t : texts)
            for (std::size_t i = 0; i < t.size();) {
                const auto m = full.view.charsmap.match(t, i);
                if (m.consumed) {
                    charsmap_entries[t.substr(i, m.consumed)] = std::string(m.replacement);
                    i += m.consumed;
                } else {
                    i += 1;
                }
            }
    }
    if (c.model == ml::tokenizer_model::unigram)
        sub.charsmap = snicholls::test::build_charsmap(charsmap_entries);   // the tests rebuild it the same way
    sub.index();

    std::ostringstream o;
    o << "# llama.cpp's tokens for ggml-vocab-" << name << ".gguf, through scripts/generate_tokenizer_fixtures.cpp.\n"
      << "# Never edit by hand. token <id> <kind> <score> <piece hex>; merge <left hex> <right hex> in rank order;\n"
      << "# charsmap <key hex> <replacement hex>; case <text hex>, llama <ids>, and core <ids> where core differs.\n";
    o << "model " << model_name(c.model) << "\npre " << c.pre << "\n";
    o << "flags " << c.byte_level << ' ' << c.ignore_merges << ' ' << c.escape_whitespaces << ' ' << c.add_space_prefix << ' '
      << c.normalizer.add_dummy_prefix << ' ' << c.normalizer.remove_extra_whitespaces << ' ' << c.normalizer.escape_whitespaces << ' '
      << c.normalizer.treat_whitespace_as_suffix << "\n";
    o << "wordpiece " << c.lowercase << ' ' << c.strip_accents << ' ' << hex(c.word_prefix) << ' ' << hex(c.continuation_prefix) << ' '
      << c.max_word_chars << "\n";
    char score[64];
    std::snprintf(score, sizeof score, "%.17g", c.unknown_score);
    o << "unknown " << static_cast<std::int64_t>(c.unknown == ml::no_token ? -1 : static_cast<std::int64_t>(c.unknown)) << ' ' << score << "\n";
    o << "size " << tokens.size() << "\n";
    std::size_t kept = 0;
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        if (!sub.present[id])
            continue;
        ++kept;
        std::snprintf(score, sizeof score, "%.9g", full.scores.empty() ? 0.0 : static_cast<double>(full.scores[id]));
        o << "token " << id << ' ' << static_cast<int>(full.kinds[id]) << ' ' << score << ' ' << hex(tokens[id]) << "\n";
    }
    for (const auto& [l, r] : sub.merges)
        o << "merge " << hex(l) << ' ' << hex(r) << "\n";
    for (const auto& [k, r] : charsmap_entries)
        o << "charsmap " << hex(k) << ' ' << hex(r) << "\n";

    // llama.cpp's tokens, and core's on the whole vocabulary and on the subset
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* lm = llama_model_load_from_file(path.c_str(), mp);
    if (!lm)
        throw std::runtime_error("llama.cpp cannot load " + path);
    const llama_vocab* lv = llama_model_get_vocab(lm);
    std::size_t agree = 0, differ = 0;
    for (const auto& t : texts) {
        const auto want = llama_tokens(lv, t);
        const auto core_full = full.tokenize(t);
        const auto core_sub = sub.tokenize(t);
        if (core_full != core_sub)
            throw std::runtime_error(name + ": the subset changes core's tokens for " + hex(t));
        o << "case " << hex(t) << "\nllama" << ids(want) << "\n";
        if (core_full != want) {
            o << "core" << ids(core_full) << "\n";
            ++differ;
            std::fprintf(stderr, "  %s: core differs on %s\n    llama%s\n    core %s\n", name.c_str(), hex(t).c_str(), ids(want).c_str(),
                         ids(core_full).c_str());
        } else {
            ++agree;
        }
    }
    llama_model_free(lm);
    std::ofstream(out_dir + "/" + name + ".txt", std::ios::binary) << o.str();
    std::printf("%-16s %-18s %6zu of %6zu tokens, %6zu merges, %3zu charsmap entries: %zu texts agree, %zu differ\n", name.c_str(),
                model_name(c.model), kept, tokens.size(), sub.merges.size(), charsmap_entries.size(), agree, differ);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <llama.cpp> <out dir> [vocab ...]\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    std::vector<std::string> names = {"llama-spm", "phi-3", "gpt-2", "llama-bpe", "qwen2", "qwen35", "deepseek-llm", "deepseek-coder",
                                      "falcon", "starcoder", "refact", "command-r", "mpt", "gemma-4", "bert-bge", "nomic-bert-moe"};
    if (argc > 3)
        names.assign(argv + 3, argv + argc);
    try {
        for (const auto& n : names)
            write(argv[1], n, argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
