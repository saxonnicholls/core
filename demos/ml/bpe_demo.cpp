//
//  bpe_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/bpe.hpp>: a small byte-level vocabulary held by the compiler, pieces encoded
//  at compile time and at run time, decoded back, and an error. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/ml/bpe.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <utility>

namespace {

namespace ml = snicholls::ml;

//! [bpe]
// A byte-level vocabulary: the bytes of "lowest", then merges in rank order (as a trainer would emit them).
// A token's rank is its id. A real vocabulary is a hash map; this one is small enough to search.
constexpr std::array<std::pair<std::string_view, std::uint32_t>, 11> vocabulary{{
    {"l", 0}, {"o", 1}, {"w", 2}, {"e", 3}, {"s", 4}, {"t", 5}, {" ", 6},   // the bytes
    {"lo", 7}, {"low", 8}, {"es", 9}, {"est", 10},                         // the merges, lowest rank first
}};

constexpr std::uint32_t rank_of(std::string_view bytes)
{
    for (const auto& [b, r] : vocabulary)
        if (b == bytes)
            return r;
    return ml::bpe_no_rank;
}

constexpr std::string_view bytes_of(std::uint32_t id)
{
    for (const auto& [b, r] : vocabulary)
        if (r == id)
            return b;
    return {};
}

// Encoding a piece at compile time: "lowest" merges lo, then low, then es, then est.
constexpr auto lowest = [] {
    std::array<std::uint32_t, 6> out{};
    std::array<ml::bpe_part, ml::bpe_scratch_size(6)> scratch{};
    const std::size_t n = ml::bpe_encode_piece("lowest", rank_of, out, scratch).unwrap();
    return std::pair{out, n};
}();
static_assert(lowest.second == 2 && lowest.first[0] == 8 && lowest.first[1] == 10);
//! [bpe]

} // namespace

int main()
{
    std::printf("snicholls-core bpe demo\n");
    bool ok = true;

    std::printf("  \"lowest\" -> [%u %u]\n", lowest.first[0], lowest.first[1]);

    // At run time: a text the tokenizer split into the pieces "slow" and " lowest" (the split is the tokenizer's).
    std::array<std::uint32_t, 16> tokens{};
    std::array<ml::bpe_part, 17> scratch{};
    std::size_t n = 0;
    for (const std::string_view piece : {std::string_view("slow"), std::string_view(" lowest")}) {
        const auto r = ml::bpe_encode_piece(piece, rank_of, std::span(tokens).subspan(n), scratch);
        ok &= r.is_ok();
        n += r.unwrap();
    }
    std::printf("  \"slow\" \" lowest\" ->");
    for (std::size_t i = 0; i < n; ++i)
        std::printf(" %u", tokens[i]);
    std::printf("\n");
    ok &= n == 5 && tokens[0] == 4 && tokens[1] == 8 && tokens[2] == 6 && tokens[3] == 8 && tokens[4] == 10;

    std::array<char, 32> text{};
    const auto back = ml::bpe_decode(std::span<const std::uint32_t>(tokens.data(), n), bytes_of, text);
    std::printf("  decoded: \"%.*s\"\n", static_cast<int>(back.unwrap()), text.data());
    ok &= std::string_view(text.data(), back.unwrap()) == "slow lowest";

    // A byte the vocabulary lacks is an error, not a guess.
    const auto missing = ml::bpe_encode_piece("lox", rank_of, tokens, scratch);
    std::printf("  \"lox\": %s\n", missing.is_err() && missing.err() == ml::bpe_error::unknown_bytes ? "unknown_bytes" : "?");
    ok &= missing.is_err() && missing.err() == ml::bpe_error::unknown_bytes;

    std::printf(ok ? "bpe demo: ok\n" : "bpe demo: FAILED\n");
    return ok ? 0 : 1;
}
