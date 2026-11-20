//
//  token_trie_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::token_trie against the node trie llama.cpp builds for Unigram and RWKV (naive_trie: a std::map
//  of children per node), on the same vocabulary (every piece of the committed tokenizer fixtures, about 13,000)
//  and the same text: the longest token at every byte of the corpus (the walk Unigram, WordPiece and RWKV make).
//  Exact lookups are set against std::unordered_map. main() refuses to benchmark if any answer differs.
//

#include <snicholls/core/ml/token_trie.hpp>

#include "../../tests/support/tokenizer_fixture.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

namespace ml = snicholls::ml;

struct node_trie   // llama.cpp's naive_trie
{
    std::map<char, node_trie> children;
    bool has_value = false;
    std::uint32_t value = 0;

    void insert(std::string_view key, std::uint32_t v)
    {
        node_trie* n = this;
        for (const char c : key)
            n = &n->children[c];
        n->has_value = true;
        n->value = v;
    }
};

struct data_t
{
    std::vector<std::string> storage;
    std::vector<std::string_view> pieces;
    std::vector<std::uint32_t> order;
    node_trie nodes;
    std::unordered_map<std::string_view, std::uint32_t> hash;
    std::string text;
    std::vector<std::string> queries;
};

const data_t& data()
{
    static const data_t d = [] {
        data_t d;
        std::set<std::string> unique;
        std::string all;
        for (const char* name : {"llama-spm", "gpt-2", "llama-bpe", "qwen2", "deepseek-llm", "falcon", "gemma-4", "bert-bge", "nomic-bert-moe"}) {
            snicholls::test::owned_tokenizer t;
            t.load(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/tokenizers/" + name + ".txt");
            for (std::size_t i = 0; i < t.pieces.size(); ++i)
                if (t.present[i] && !t.pieces[i].empty())
                    unique.insert(std::string(t.pieces[i]));
            for (const auto& c : t.cases)
                all += c.text;
        }
        d.storage.assign(unique.begin(), unique.end());
        d.pieces.assign(d.storage.begin(), d.storage.end());
        d.order.resize(d.pieces.size());
        ml::build_token_index(d.pieces, d.order);
        for (std::uint32_t id = 0; id < d.pieces.size(); ++id) {
            d.nodes.insert(d.pieces[id], id);
            d.hash[d.pieces[id]] = id;
        }
        while (d.text.size() < (1u << 15))
            d.text += all;
        for (std::size_t i = 0; i + 8 < d.text.size(); i += 3)
            d.queries.push_back(d.text.substr(i, 1 + i % 7));
        return d;
    }();
    return d;
}

std::uint64_t core_walks(const data_t& d)
{
    const ml::token_trie trie{d.pieces, d.order};
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < d.text.size(); ++i) {
        std::size_t n = 0;
        sum += trie.longest_prefix(std::string_view(d.text).substr(i), n) + n;
    }
    return sum;
}

std::uint64_t node_walks(const data_t& d)
{
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < d.text.size(); ++i) {
        const node_trie* n = &d.nodes;
        std::uint32_t best = ml::no_token;
        std::size_t len = 0;
        for (std::size_t j = i; j < d.text.size(); ++j) {
            const auto it = n->children.find(d.text[j]);
            if (it == n->children.end())
                break;
            n = &it->second;
            if (n->has_value) {
                best = n->value;
                len = j + 1 - i;
            }
        }
        sum += best + len;
    }
    return sum;
}

std::uint64_t core_finds(const data_t& d)
{
    const ml::token_trie trie{d.pieces, d.order};
    std::uint64_t sum = 0;
    for (const auto& q : d.queries)
        sum += trie.find(q);
    return sum;
}

std::uint64_t hash_finds(const data_t& d)
{
    std::uint64_t sum = 0;
    for (const auto& q : d.queries) {
        const auto it = d.hash.find(q);
        sum += it == d.hash.end() ? ml::no_token : it->second;
    }
    return sum;
}

void BM_TokenTrieWalk_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(core_walks(data()));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * data().text.size()));
}
BENCHMARK(BM_TokenTrieWalk_Core);

void BM_TokenTrieWalk_NodeTrie(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(node_walks(data()));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * data().text.size()));
}
BENCHMARK(BM_TokenTrieWalk_NodeTrie);

void BM_TokenTrieFind_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(core_finds(data()));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * data().queries.size()));
}
BENCHMARK(BM_TokenTrieFind_Core);

void BM_TokenTrieFind_UnorderedMap(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(hash_finds(data()));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * data().queries.size()));
}
BENCHMARK(BM_TokenTrieFind_UnorderedMap);

} // namespace

int main(int argc, char** argv)
{
    if (core_walks(data()) != node_walks(data()) || core_finds(data()) != hash_finds(data())) {
        std::fprintf(stderr, "token_trie benchmark: fairness check FAILED - the answers differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
