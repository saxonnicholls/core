//
//  bpe_merges_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::merge_encode against llama.cpp's merge loop written out (llm_tokenizer_bpe_session: a vector of
//  symbols per piece, a std::priority_queue of bigrams each holding its text as a std::string, the merge ranks in
//  a std::map of string pairs, the vocabulary in a std::unordered_map), on the same pieces: the texts of the
//  committed Llama 3 fixture, split by Llama 3's pattern and written in GPT-2's alphabet, with the fixture's
//  vocabulary and merges. main() refuses to benchmark if any piece's tokens differ.
//

#include <snicholls/core/ml/bpe_merges.hpp>
#include <snicholls/core/ml/pretokenize.hpp>

#include "../../tests/support/tokenizer_fixture.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;

struct data_t
{
    snicholls::test::owned_tokenizer t;
    std::vector<std::string> pieces;   // in GPT-2's alphabet
    std::map<std::pair<std::string, std::string>, int> ranks;
    std::unordered_map<std::string, std::uint32_t> ids;
};

const data_t& data()
{
    static const data_t& d = *[] {
        auto* d = new data_t;
        d->t.load(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/tokenizers/llama-bpe.txt");
        for (int rep = 0; rep < 8; ++rep)
            for (const auto& c : d->t.cases) {
                std::string text(3 * c.text.size(), '\0');
                text.resize(ml::replace_invalid_utf8(c.text, std::span<char>(text)));
                ml::pretokenize(ml::pretokenizer::llama3, text, [&](std::string_view p) {
                    std::string m(2 * p.size(), '\0');
                    m.resize(ml::byte_level_encode(p, std::span<char>(m)).unwrap());
                    d->pieces.push_back(m);
                });
            }
        for (std::size_t i = 0; i < d->t.merges.size(); ++i)
            d->ranks.emplace(std::pair{std::string(d->t.merges[i].first), std::string(d->t.merges[i].second)}, static_cast<int>(i));
        for (std::uint32_t id = 0; id < d->t.pieces.size(); ++id)
            if (d->t.present[id])
                d->ids[std::string(d->t.pieces[id])] = id;
        return d;
    }();
    return d;
}

std::uint64_t core_encode(const data_t& d)
{
    std::vector<std::uint32_t> out(256);
    std::vector<ml::merge_entry<std::uint32_t>> scratch(ml::merge_scratch_size(256));
    std::uint64_t h = 0;
    for (const auto& p : d.pieces) {
        const std::size_t n = ml::merge_encode(p, d.t.view.merges, d.t.view.vocab, d.t.view.bytes, out, scratch).unwrap();
        for (std::size_t i = 0; i < n; ++i)
            h = h * 31 + out[i];
    }
    return h;
}

// llama.cpp's llm_tokenizer_bpe_session::tokenize for one word, transliterated
struct symbol
{
    int prev, next;
    const char* text;
    std::size_t n;
};
struct bigram
{
    int left, right;
    std::string text;
    int rank;
    std::size_t size;
};
struct later
{
    bool operator()(const bigram& l, const bigram& r) const { return l.rank > r.rank || (l.rank == r.rank && l.left > r.left); }
};

std::uint64_t hand_encode(const data_t& d)
{
    std::uint64_t h = 0;
    std::vector<symbol> symbols;
    for (const auto& word : d.pieces) {
        symbols.clear();
        std::priority_queue<bigram, std::vector<bigram>, later> queue;
        int index = 0;
        for (std::size_t offset = 0; offset < word.size();) {
            const std::size_t len = std::min(word.size() - offset, static_cast<std::size_t>(ml::decode_utf8(word, offset).length));
            symbols.push_back({index - 1, offset + len == word.size() ? -1 : index + 1, word.data() + offset, len});
            offset += len;
            ++index;
        }
        const auto add = [&](int left, int right) {
            if (left == -1 || right == -1)
                return;
            std::string l(symbols[left].text, symbols[left].n), r(symbols[right].text, symbols[right].n);
            const auto it = d.ranks.find({l, r});
            if (it == d.ranks.end())
                return;
            queue.push({left, right, l + r, it->second, l.size() + r.size()});
        };
        for (int i = 1; i < static_cast<int>(symbols.size()); ++i)
            add(i - 1, i);
        while (!queue.empty()) {
            const bigram b = queue.top();
            queue.pop();
            symbol& left = symbols[b.left];
            symbol& right = symbols[b.right];
            if (left.n == 0 || right.n == 0 || std::string(left.text, left.n) + std::string(right.text, right.n) != b.text)
                continue;
            left.n += right.n;
            right.n = 0;
            left.next = right.next;
            if (right.next >= 0)
                symbols[right.next].prev = b.left;
            add(left.prev, b.left);
            add(b.left, left.next);
        }
        for (const auto& s : symbols)
            if (s.n > 0)
                h = h * 31 + d.ids.at(std::string(s.text, s.n));
    }
    return h;
}

void BM_BpeMerges_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(core_encode(data()));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * data().pieces.size()));
}
BENCHMARK(BM_BpeMerges_Core);

void BM_BpeMerges_LlamaCppLoop(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(hand_encode(data()));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * data().pieces.size()));
}
BENCHMARK(BM_BpeMerges_LlamaCppLoop);

} // namespace

int main(int argc, char** argv)
{
    if (core_encode(data()) != hand_encode(data())) {
        std::fprintf(stderr, "bpe_merges benchmark: fairness check FAILED - the tokens differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
