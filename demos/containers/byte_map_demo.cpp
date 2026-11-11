//
//  byte_map_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/byte_map.hpp>: a small BPE vocabulary built by the compiler and used to
//  encode a piece; a larger one over the caller's storage, looked up in a batch with probe and find; decoding
//  through the reverse table; and an insert refused. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/containers/byte_map.hpp>
#include <snicholls/core/ml/bpe.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace containers = snicholls::containers;
using namespace std::string_view_literals;

int main()
{
    std::printf("snicholls-core byte_map demo\n");
    bool ok = true;

    //! [byte_map]
    // A vocabulary built by the compiler: 16 slots, 32 bytes of keys, a reverse table for values 0 .. 7
    constexpr containers::byte_map<16, 32, 8> vocab{std::array<std::pair<std::string_view, std::uint32_t>, 6>{
        {{"l", 0}, {"o", 1}, {"w", 2}, {"lo", 3}, {"low", 4}, {"ow", 5}}}};

    // It is an ml::bpe_vocabulary as it stands
    std::array<std::uint32_t, 8> tokens{};
    std::array<snicholls::ml::bpe_part, snicholls::ml::bpe_scratch_size(5)> scratch{};
    const auto n = snicholls::ml::bpe_encode_piece("lowow", vocab, tokens, scratch).unwrap();   // [low][ow]

    // Decoding reads the reverse table
    std::string text;
    for (std::size_t i = 0; i < n; ++i)
        text += vocab.key_of(tokens[i]);
    //! [byte_map]
    std::printf("  \"lowow\" -> %zu tokens: %u %u -> \"%s\"\n", n, tokens[0], tokens[1], text.c_str());
    ok = ok && n == 2 && tokens[0] == 4 && tokens[1] == 5 && text == "lowow";

    //! [byte_map_view]
    // A larger table over storage the caller owns, sized by the helpers
    const std::vector<std::pair<std::string, std::uint32_t>> words{
        {"the", 0}, {"quick", 1}, {"brown", 2}, {"fox", 3}, {"jumps", 4}, {"over", 5}, {"a lazy dog's back", 6}};
    std::vector<containers::byte_map_slot> slots(containers::byte_map_view::required_slots(words.size()));
    std::vector<char> arena(containers::byte_map_view::required_arena(words));
    containers::byte_map_view map(slots, arena);
    if (map.build(words).is_err())
        return 1;

    // Probe a batch first, then find each: the slots' cache misses overlap
    constexpr std::array<std::string_view, 4> queries{"fox"sv, "cat"sv, "a lazy dog's back"sv, "the"sv};
    std::array<containers::byte_map_probe, queries.size()> probes{};
    for (std::size_t i = 0; i < queries.size(); ++i)
        probes[i] = map.probe(queries[i]);
    std::array<std::uint32_t, queries.size()> found{};
    for (std::size_t i = 0; i < queries.size(); ++i)
        found[i] = map.find(queries[i], probes[i]);   // byte_map_no_value for "cat"
    //! [byte_map_view]
    for (std::size_t i = 0; i < queries.size(); ++i) {
        if (found[i] == containers::byte_map_no_value)
            std::printf("  %-18.*s not found\n", static_cast<int>(queries[i].size()), queries[i].data());
        else
            std::printf("  %-18.*s %u\n", static_cast<int>(queries[i].size()), queries[i].data(), found[i]);
    }
    ok = ok && found[0] == 3 && found[1] == containers::byte_map_no_value && found[2] == 6 && found[3] == 0;
    std::printf("  %zu keys in %zu slots, %zu arena bytes\n", map.size(), map.capacity(), map.arena_used());
    ok = ok && map.size() == 7 && map.capacity() == 16 && map.arena_used() == arena.size();

    // An insert that fails says why and changes nothing
    const auto again = map.insert("fox", 9);
    std::printf("  insert \"fox\" again: %s\n", again.is_err() && again.err() == containers::byte_map_error::duplicate ? "duplicate" : "?");
    ok = ok && again.is_err() && again.err() == containers::byte_map_error::duplicate && map("fox"sv) == 3;

    std::printf(ok ? "byte_map demo: ok\n" : "byte_map demo: FAILED\n");
    return ok ? 0 : 1;
}
