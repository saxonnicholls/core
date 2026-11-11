//
//  bpe_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::bpe_encode_piece against tiktoken's byte_pair_merge transliterated by hand (a std::vector of
//  parts per piece, erase to merge), with the same vocabulary lookup, over the pieces of the committed fixture
//  (tests/fixtures/ml/bpe_fixture.txt). main() runs a fairness check first and refuses to benchmark if any
//  piece's tokens differ (CLAUDE.md "Zero overhead"). The comparison with tiktoken and rustbpe themselves is
//  under Performance in ml/bpe.hpp. A second group sets bpe_encode_long_piece against bpe_encode_piece on long
//  pieces of random letters (the same pieces for both, checked to give the same tokens), for the crossover that
//  bpe_long_piece_threshold records.
//

#include <snicholls/core/ml/bpe.hpp>
#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;

std::string unbase64(std::string_view in)
{
    if (in == "-")
        return {};
    static constexpr std::string_view a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    std::uint32_t acc = 0;
    int bits = 0;
    for (const char c : in) {
        if (c == '=')
            break;
        acc = (acc << 6) | static_cast<std::uint32_t>(a.find(c));
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((acc >> bits) & 0xff));
        }
    }
    return out;
}

struct data_t {
    std::unordered_map<std::string_view, std::uint32_t> ranks;
    std::vector<std::string> storage;
    std::vector<std::string> pieces;
};

data_t load()
{
    data_t d;
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/bpe_fixture.txt");
    std::string line;
    std::vector<std::pair<std::string, std::uint32_t>> kv;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kind, b;
        ls >> kind >> b;
        if (kind == "vocab") {
            std::uint32_t r = 0;
            ls >> r;
            kv.emplace_back(unbase64(b), r);
        } else if (kind == "piece") {
            d.pieces.push_back(unbase64(b));
        }
    }
    d.storage.reserve(kv.size());
    for (auto& [k, r] : kv) {
        d.storage.push_back(std::move(k));
        d.ranks.emplace(d.storage.back(), r);
    }
    return d;
}

const data_t& data()
{
    static const data_t d = load();
    return d;
}

std::uint32_t rank(std::string_view s)
{
    const auto& m = data().ranks;
    const auto it = m.find(s);
    return it == m.end() ? ml::bpe_no_rank : it->second;
}

// By hand: tiktoken's byte_pair_merge and byte_pair_encode, line for line, with a vector per piece
void encode_by_hand(std::string_view piece, std::vector<std::uint32_t>& out)
{
    if (piece.empty())
        return;
    if (const std::uint32_t whole = rank(piece); whole != ml::bpe_no_rank) {
        out.push_back(whole);
        return;
    }
    std::vector<std::pair<std::size_t, std::uint32_t>> parts;
    parts.reserve(piece.size() + 1);
    std::pair<std::uint32_t, std::size_t> min_rank{ml::bpe_no_rank, 0};
    for (std::size_t i = 0; i + 1 < piece.size(); ++i) {
        const std::uint32_t r = rank(piece.substr(i, 2));
        if (r < min_rank.first)
            min_rank = {r, i};
        parts.emplace_back(i, r);
    }
    parts.emplace_back(piece.size() - 1, ml::bpe_no_rank);
    parts.emplace_back(piece.size(), ml::bpe_no_rank);
    const auto get_rank = [&](std::size_t i) {
        return i + 3 < parts.size() ? rank(piece.substr(parts[i].first, parts[i + 3].first - parts[i].first)) : ml::bpe_no_rank;
    };
    while (min_rank.first != ml::bpe_no_rank) {
        const std::size_t i = min_rank.second;
        if (i > 0)
            parts[i - 1].second = get_rank(i - 1);
        parts[i].second = get_rank(i);
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(i) + 1);
        min_rank = {ml::bpe_no_rank, 0};
        for (std::size_t k = 0; k + 1 < parts.size(); ++k)
            if (parts[k].second < min_rank.first)
                min_rank = {parts[k].second, k};
    }
    for (std::size_t k = 0; k + 1 < parts.size(); ++k)
        out.push_back(rank(piece.substr(parts[k].first, parts[k + 1].first - parts[k].first)));
}

void encode_core(std::string_view piece, std::vector<std::uint32_t>& out, std::vector<std::uint32_t>& buf, std::vector<ml::bpe_part>& scratch)
{
    if (buf.size() < piece.size()) {
        buf.resize(piece.size());
        scratch.resize(ml::bpe_scratch_size(piece.size()));
    }
    const auto r = ml::bpe_encode_piece(piece, rank, std::span<std::uint32_t>(buf.data(), piece.size()), scratch);
    out.insert(out.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(r.unwrap()));
}

std::size_t total_bytes()
{
    std::size_t n = 0;
    for (const auto& p : data().pieces)
        n += p.size();
    return n;
}

void BM_BpeEncode_Core(benchmark::State& state)
{
    std::vector<std::uint32_t> out, buf;
    std::vector<ml::bpe_part> scratch;
    out.reserve(1 << 16);
    for (auto _ : state) {
        out.clear();
        for (const auto& p : data().pieces)
            encode_core(p, out, buf, scratch);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(total_bytes()));
}
BENCHMARK(BM_BpeEncode_Core);

void BM_BpeEncode_HandWritten(benchmark::State& state)
{
    std::vector<std::uint32_t> out;
    out.reserve(1 << 16);
    for (auto _ : state) {
        out.clear();
        for (const auto& p : data().pieces)
            encode_by_hand(p, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(total_bytes()));
}
BENCHMARK(BM_BpeEncode_HandWritten);

// A piece of n random letters: what a split pattern keeps whole as one word-like run
std::string long_piece(std::size_t n)
{
    snicholls::stats::wyrand rng(n);
    std::string s(n, 'a');
    for (char& c : s)
        c = static_cast<char>('a' + snicholls::stats::uniform_index(rng, 26));
    return s;
}

template <bool Long>
std::size_t encode_long_or_short(std::string_view piece, std::vector<std::uint32_t>& out, std::vector<ml::bpe_part>& scratch)
{
    return Long ? ml::bpe_encode_long_piece(piece, rank, out, scratch).unwrap() : ml::bpe_encode_piece(piece, rank, out, scratch).unwrap();
}

template <bool Long>
void BM_BpeLongPiece(benchmark::State& state)
{
    const std::string piece = long_piece(static_cast<std::size_t>(state.range(0)));
    std::vector<std::uint32_t> out(piece.size());
    std::vector<ml::bpe_part> scratch(ml::bpe_long_scratch_size(piece.size()));
    for (auto _ : state)
        benchmark::DoNotOptimize(encode_long_or_short<Long>(piece, out, scratch));
    state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(piece.size()));
}
BENCHMARK(BM_BpeLongPiece<false>)->Name("BM_BpeLongPiece_Rescan")->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);
BENCHMARK(BM_BpeLongPiece<true>)->Name("BM_BpeLongPiece_Heap")->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

bool fair()
{
    for (const std::size_t n : {std::size_t{64}, std::size_t{256}, std::size_t{1024}, std::size_t{4096}}) {
        const std::string piece = long_piece(n);
        std::vector<std::uint32_t> a(n), b(n);
        std::vector<ml::bpe_part> scratch(ml::bpe_long_scratch_size(n));
        const std::size_t na = encode_long_or_short<false>(piece, a, scratch);
        const std::size_t nb = encode_long_or_short<true>(piece, b, scratch);
        if (na != nb || !std::equal(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(na), b.begin()))
            return false;
    }
    if (data().pieces.empty())
        return false;
    std::vector<std::uint32_t> a, b, buf;
    std::vector<ml::bpe_part> scratch;
    for (const auto& p : data().pieces) {
        a.clear();
        b.clear();
        encode_core(p, a, buf, scratch);
        encode_by_hand(p, b);
        if (a != b)
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "bpe benchmark: fairness check FAILED - the pairs disagree (or the fixture is missing); not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
