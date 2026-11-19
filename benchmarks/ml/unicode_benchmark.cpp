//
//  unicode_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::decode_utf8 and category over a mixed-script corpus (the texts of the committed tokenizer
//  fixtures, repeated), against the hand-written equivalent llama.cpp uses: the same decoding written out and a
//  flat table of every code point's category (1.1 MB, filled once from core's own answers, so the two agree by
//  construction). Both sum the category indices; main() refuses to benchmark if the sums differ.
//

#include <snicholls/core/ml/unicode.hpp>

#include "../../tests/support/tokenizer_fixture.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ml = snicholls::ml;

const std::string& corpus()
{
    static const std::string text = [] {
        snicholls::test::owned_tokenizer t;
        t.load(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/tokenizers/llama-bpe.txt");
        std::string all;
        for (const auto& c : t.cases)
            all += c.text;
        std::string out;
        while (out.size() < (1u << 16))
            out += all;
        return out;
    }();
    return text;
}

const std::vector<std::uint8_t>& flat_table()
{
    static const std::vector<std::uint8_t> table = [] {
        std::vector<std::uint8_t> t(0x110000);
        for (char32_t cp = 0; cp < 0x110000; ++cp)
            t[cp] = static_cast<std::uint8_t>(ml::category(cp));
        return t;
    }();
    return table;
}

std::uint64_t core_sum(std::string_view text)
{
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < text.size();) {
        const auto d = ml::decode_utf8(text, i);
        sum += static_cast<std::uint64_t>(ml::category(d.code_point));
        i += d.length;
    }
    return sum;
}

// The same decoding, written out, and one load from the flat table
std::uint64_t hand_sum(std::string_view text, const std::uint8_t* table)
{
    std::uint64_t sum = 0;
    const auto* p = reinterpret_cast<const unsigned char*>(text.data());
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned b0 = p[i];
        char32_t cp = 0xFFFD;
        std::size_t len = 1;
        const auto c = [&](std::size_t k, unsigned lo = 0x80, unsigned hi = 0xBF) { return i + k < n && p[i + k] >= lo && p[i + k] <= hi; };
        if (b0 < 0x80) {
            cp = b0;
        } else if (b0 >= 0xC2 && b0 <= 0xDF && c(1)) {
            cp = ((b0 & 0x1F) << 6) | (p[i + 1] & 0x3F);
            len = 2;
        } else if (b0 >= 0xE0 && b0 <= 0xEF && c(1, b0 == 0xE0 ? 0xA0 : 0x80, b0 == 0xED ? 0x9F : 0xBF) && c(2)) {
            cp = ((b0 & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F);
            len = 3;
        } else if (b0 >= 0xF0 && b0 <= 0xF4 && c(1, b0 == 0xF0 ? 0x90 : 0x80, b0 == 0xF4 ? 0x8F : 0xBF) && c(2) && c(3)) {
            cp = ((b0 & 0x07) << 18) | ((p[i + 1] & 0x3F) << 12) | ((p[i + 2] & 0x3F) << 6) | (p[i + 3] & 0x3F);
            len = 4;
        }
        sum += table[cp];
        i += len;
    }
    return sum;
}

void BM_UnicodeCategory_Core(benchmark::State& state)
{
    const std::string& text = corpus();
    for (auto _ : state)
        benchmark::DoNotOptimize(core_sum(text));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * text.size()));
}
BENCHMARK(BM_UnicodeCategory_Core);

void BM_UnicodeCategory_FlatTable(benchmark::State& state)
{
    const std::string& text = corpus();
    const std::uint8_t* table = flat_table().data();
    for (auto _ : state)
        benchmark::DoNotOptimize(hand_sum(text, table));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * text.size()));
}
BENCHMARK(BM_UnicodeCategory_FlatTable);

} // namespace

int main(int argc, char** argv)
{
    if (core_sum(corpus()) != hand_sum(corpus(), flat_table().data())) {
        std::fprintf(stderr, "unicode benchmark: fairness check FAILED - the sums differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
