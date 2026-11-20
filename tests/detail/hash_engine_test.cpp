//
//  hash_engine_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/hash_engine.hpp>: word loads and stores in both byte orders from every
//  one-byte unit, rotations, the block feeder's whole-block and buffered paths, and the volatile helpers.
//

#include <snicholls/core/detail/hash_engine.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace snicholls::detail;

constexpr unsigned char bytes[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};
static_assert(load_le_word<std::uint32_t>(bytes) == 0x67452301u && load_be_word<std::uint32_t>(bytes) == 0x01234567u);
static_assert(load_le_word<std::uint64_t>(bytes) == 0xefcdab8967452301ull && load_be_word<std::uint64_t>(bytes) == 0x0123456789abcdefull);
constexpr char8_t text[4] = {u8'a', u8'b', u8'c', u8'd'};
static_assert(load_be_word<std::uint32_t>(text) == 0x61626364u);
constexpr std::byte raw[2] = {std::byte{0xff}, std::byte{0x01}};
static_assert(load_le_word<std::uint16_t>(raw) == 0x01ffu);
static_assert(rotr<std::uint32_t>(1u, 1) == 0x80000000u && rotl<std::uint64_t>(0x8000000000000000ull, 1) == 1u);
static_assert(rotl<std::uint32_t>(0x12345678u, 8) == 0x34567812u);

constexpr bool stores_round_trip()
{
    std::array<std::byte, 8> out{};
    store_be_word(out.data(), std::uint64_t{0x0102030405060708ull});
    const bool be = load_be_word<std::uint64_t>(out.data()) == 0x0102030405060708ull && out[0] == std::byte{1};
    store_le_word(out.data(), std::uint32_t{0xaabbccddu});
    return be && out[0] == std::byte{0xdd} && load_le_word<std::uint32_t>(out.data()) == 0xaabbccddu;
}
static_assert(stores_round_trip());

TEST(HashEngineFeeder, WholeBlocksComeFromTheInputAndTheRestIsBuffered)
{
    block_feeder<4> f;
    std::vector<std::string> blocks;
    const auto record = [&](const auto* p) { blocks.emplace_back(reinterpret_cast<const char*>(p), 4); };
    const std::string input = "abcdefghij";
    f.feed(input.data(), 3, record);   // "abc" buffered
    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(f.fill, 3u);
    f.feed(input.data() + 3, 7, record);   // completes "abcd", then "efgh" straight from the input, "ij" buffered
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(blocks[0], "abcd");
    EXPECT_EQ(blocks[1], "efgh");
    EXPECT_EQ(f.fill, 2u);
    EXPECT_EQ(f.buf[0], 'i');
    f.feed(input.data(), 0, record);   // nothing
    EXPECT_EQ(blocks.size(), 2u);
}

TEST(HashEngineVolatile, ZeroesAndTestsThroughVolatileAccess)
{
    std::array<unsigned char, 5> b{1, 2, 3, 4, 5};
    volatile_zero(b.data() + 1, 3);
    EXPECT_EQ(b, (std::array<unsigned char, 5>{1, 0, 0, 0, 5}));
    EXPECT_TRUE(volatile_is_zero(0));
    EXPECT_FALSE(volatile_is_zero(4));
}

} // namespace
