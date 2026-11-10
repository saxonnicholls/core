//
//  perfect_map_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/perfect_map.hpp>: integer, enumeration and text keys,
//  misses, a thousand integer keys and 512 text keys placed by the compiler.
//

#include <snicholls/core/containers/perfect_map.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

namespace {

using namespace std::string_view_literals;
using snicholls::containers::perfect_map;

enum class venue : std::uint8_t { lse, nyse, cme, ice };

constexpr perfect_map<std::string_view, double, 5> tick_size{
    std::array<std::pair<std::string_view, double>, 5>{{{"ES", 0.25}, {"NQ", 0.25}, {"CL", 0.01}, {"GC", 0.1}, {"ZN", 0.015625}}}};
static_assert(*tick_size.find("CL"sv) == 0.01 && *tick_size.find("ZN"sv) == 0.015625 && !tick_size.contains("XX"sv));
constexpr perfect_map<venue, int, 3> latency{std::array<std::pair<venue, int>, 3>{{{venue::lse, 3}, {venue::cme, 1}, {venue::ice, 2}}}};
static_assert(*latency.find(venue::cme) == 1 && latency.find(venue::nyse) == nullptr);

constexpr auto thousand = [] {
    std::array<std::pair<std::uint32_t, std::uint32_t>, 1000> e{};
    for (std::uint32_t i = 0; i < 1000; ++i)
        e[i] = {i * 7919u + 13u, i};
    return e;
}();
constexpr perfect_map<std::uint32_t, std::uint32_t, 1000> big{thousand};

TEST(PerfectMap, AThousandKeysEachInItsOwnSlot)
{
    for (std::uint32_t i = 0; i < 1000; ++i)
        ASSERT_EQ(*big.find(i * 7919u + 13u), i);
    EXPECT_EQ(big.find(4u), nullptr);
}

// 512 four-letter codes: each text key is hashed once while building, not once per comparison, so this
// fits the compiler's budget.
constexpr auto code_text = [] {
    std::array<char, 512 * 4> t{};
    for (std::size_t i = 0; i < 512; ++i)
        for (std::size_t j = 0, x = i * 97 + 13; j < 4; ++j, x /= 26)
            t[i * 4 + 3 - j] = static_cast<char>('A' + x % 26);
    return t;
}();
constexpr auto codes = [] {
    std::array<std::pair<std::string_view, std::uint32_t>, 512> e{};
    for (std::uint32_t i = 0; i < 512; ++i)
        e[i] = {std::string_view(code_text.data() + i * 4, 4), i};
    return e;
}();
constexpr perfect_map<std::string_view, std::uint32_t, 512> by_code{codes};
static_assert(*by_code.find(codes[511].first) == 511 && !by_code.contains("ZZZZ"sv));

TEST(PerfectMap, FiveHundredTextKeysBuiltByTheCompiler)
{
    for (const auto& [code, i] : codes)
        ASSERT_EQ(*by_code.find(code), i);
}

} // namespace
