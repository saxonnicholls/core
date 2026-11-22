//
//  unigram_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::unigram_encode against llama.cpp's Viterbi written out (llm_tokenizer_ugm_session: the lattice
//  in a std::vector, the vocabulary in naive_trie, a std::map of children per node), on the same normalised text
//  (the committed nomic-embed fixture's texts through unigram_normalize, repeated) and the fixture's vocabulary and
//  scores. main() refuses to benchmark if any text's tokens differ.
//

#include <snicholls/core/ml/unigram.hpp>

#include "../../tests/support/tokenizer_fixture.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

struct node_trie
{
    std::map<char, node_trie> children;
    bool has_value = false;
    std::uint32_t value = 0;
};

struct data_t
{
    snicholls::test::owned_tokenizer t;
    std::vector<std::string> texts;   // normalised
    node_trie trie;
    std::size_t bytes = 0;
};

const data_t& data()
{
    static const data_t& d = *[] {
        auto* made = new data_t;
        made->t.load(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/tokenizers/nomic-bert-moe.txt");
        for (int rep = 0; rep < 8; ++rep)
            for (const auto& c : made->t.cases) {
                std::string n(ml::unigram_normalized_size(c.text.size(), made->t.view.charsmap), '\0');
                n.resize(ml::unigram_normalize(c.text, made->t.view.charsmap, {}, made->t.view.normalizer, std::span<char>(n)).unwrap());
                made->bytes += n.size();
                made->texts.push_back(n);
            }
        for (const std::uint32_t id : made->t.order) {
            node_trie* n = &made->trie;
            for (const char c : made->t.pieces[id])
                n = &n->children[c];
            n->has_value = true;
            n->value = id;
        }
        return made;
    }();
    return d;
}

std::uint64_t core_encode(const data_t& d)
{
    std::vector<std::uint32_t> out(512);
    std::vector<ml::unigram_lattice_entry> lattice(513);
    std::uint64_t h = 0;
    const auto score = [&](std::uint32_t id) { return d.t.view.score(id); };
    const auto kind = [&](std::uint32_t id) { return d.t.view.kind(id); };
    for (const auto& text : d.texts) {
        const std::size_t n = ml::unigram_encode(text, d.t.view.vocab, score, kind, {d.t.view.unknown, d.t.view.unknown_score}, out, lattice).unwrap();
        for (std::size_t i = 0; i < n; ++i)
            h = h * 31 + out[i];
    }
    return h;
}

// llama.cpp's llm_tokenizer_ugm_session::tokenize after normalisation, transliterated
std::uint64_t hand_encode(const data_t& d)
{
    struct best
    {
        std::uint32_t token;
        std::size_t from;
        double score;
    };
    const std::uint32_t unk = d.t.view.unknown;
    std::uint64_t h = 0;
    std::vector<std::uint32_t> out;
    for (const auto& text : d.texts) {
        const std::size_t n = text.size();
        if (n == 0)
            continue;
        std::vector<best> results(n + 1, {unk, 0, -1.7976931348623157e308});
        results[0] = {unk, 0, 0};
        for (std::size_t at = 0; at < n;) {
            const std::size_t units = ml::decode_utf8(text, at).length;
            bool single = false;
            const best current = results[at];
            const node_trie* node = &d.trie;
            for (std::size_t p = at; p < n; ++p) {
                const auto it = node->children.find(text[p]);
                if (it == node->children.end())
                    break;
                node = &it->second;
                if (node->has_value) {
                    if (p + 1 - at == units)
                        single = true;
                    const std::uint32_t id = node->value;
                    const double s = current.score + (d.t.view.kind(id) == ml::token_kind::user_defined ? 0.0 : d.t.view.score(id));
                    if (s > results[p + 1].score)
                        results[p + 1] = {id, at, s};
                }
            }
            if (!single && current.score + d.t.view.unknown_score > results[at + units].score)
                results[at + units] = {unk, at, current.score + d.t.view.unknown_score};
            at += units;
        }
        out.clear();
        bool prev_unknown = false;
        for (std::size_t at = n;;) {
            const best& b = results[at];
            const bool u = b.token == unk;
            if (!(prev_unknown && u))
                out.push_back(b.token);
            if (b.from == 0)
                break;
            prev_unknown = u;
            at = b.from;
        }
        std::reverse(out.begin(), out.end());
        for (const auto id : out)
            h = h * 31 + id;
    }
    return h;
}

void BM_Unigram_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(core_encode(data()));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * data().bytes));
}
BENCHMARK(BM_Unigram_Core);

void BM_Unigram_LlamaCppLoop(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(hand_encode(data()));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * data().bytes));
}
BENCHMARK(BM_Unigram_LlamaCppLoop);

} // namespace

int main(int argc, char** argv)
{
    if (core_encode(data()) != hand_encode(data())) {
        std::fprintf(stderr, "unigram benchmark: fairness check FAILED - the tokens differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
