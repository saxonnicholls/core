//
//  pretokenize_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml's Llama 3 split against llama.cpp's general technique for a split pattern: every code point
//  collapsed to one byte for its category (letters to 0xD2, numbers to 0xD1, whitespace to 0x0B, ...) and the
//  expression run by std::regex over the collapsed text, \p{L} and \p{N} rewritten as classes of those bytes. Both
//  count the pieces and their code points over the same corpus (the texts of the committed tokenizer fixtures,
//  repeated, with invalid UTF-8 repaired); main() refuses to benchmark if they disagree.
//

#include <snicholls/core/ml/pretokenize.hpp>

#include "../../tests/support/tokenizer_fixture.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <regex>
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
            all += c.text + "\n";
        std::string repaired(3 * all.size(), '\0');
        repaired.resize(ml::replace_invalid_utf8(all, std::span<char>(repaired)));
        std::string out;
        while (out.size() < (1u << 15))
            out += repaired;
        return out;
    }();
    return text;
}

// Pieces and their total code points, hashed together
std::uint64_t core_split(std::string_view text)
{
    std::uint64_t h = 0;
    ml::pretokenize(ml::pretokenizer::llama3, text, [&](std::string_view piece) {
        std::uint64_t cps = 0;
        for (std::size_t i = 0; i < piece.size(); i += ml::decode_utf8(piece, i).length)
            ++cps;
        h = h * 31 + cps;
    });
    return h;
}

std::uint64_t regex_split(std::string_view text)
{
    static const std::regex expr(
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\xD2A-Za-z\\xD10-9]?[\\xD2A-Za-z]+|[\\xD10-9]{1,3}"
        "| ?[^\\s\\xD2A-Za-z\\xD10-9]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        std::regex::optimize | std::regex::nosubs);
    std::string collapsed;
    collapsed.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        const auto d = ml::decode_utf8(text, i);
        i += d.length;
        const char32_t cp = d.code_point;
        const ml::unicode_category c = ml::category(cp);
        char b = '\xD0';
        if (cp < 128)
            b = static_cast<char>(cp);
        else if (ml::is_whitespace(cp))
            b = '\x0B';
        else if (ml::is_letter(c))
            b = '\xD2';
        else if (ml::is_number(c))
            b = '\xD1';
        else if (ml::is_punctuation(c))
            b = '\xD3';
        else if (ml::is_mark(c))
            b = '\xD4';
        else if (ml::is_symbol(c))
            b = '\xD5';
        collapsed.push_back(b);
    }
    std::uint64_t h = 0;
    std::ptrdiff_t at = 0;
    for (auto it = std::sregex_iterator(collapsed.begin(), collapsed.end(), expr); it != std::sregex_iterator(); ++it) {
        if (it->position() > at)
            h = h * 31 + static_cast<std::uint64_t>(it->position() - at);
        h = h * 31 + static_cast<std::uint64_t>(it->length());
        at = it->position() + it->length();
    }
    if (at < static_cast<std::ptrdiff_t>(collapsed.size()))
        h = h * 31 + static_cast<std::uint64_t>(static_cast<std::ptrdiff_t>(collapsed.size()) - at);
    return h;
}

void BM_PretokenizeLlama3_Core(benchmark::State& state)
{
    const std::string& text = corpus();
    for (auto _ : state)
        benchmark::DoNotOptimize(core_split(text));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * text.size()));
}
BENCHMARK(BM_PretokenizeLlama3_Core);

void BM_PretokenizeLlama3_CollapsedStdRegex(benchmark::State& state)
{
    const std::string& text = corpus();
    for (auto _ : state)
        benchmark::DoNotOptimize(regex_split(text));
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations() * text.size()));
}
BENCHMARK(BM_PretokenizeLlama3_CollapsedStdRegex);

} // namespace

int main(int argc, char** argv)
{
    if (core_split(corpus()) != regex_split(corpus())) {
        std::fprintf(stderr, "pretokenize benchmark: fairness check FAILED - the splits differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
