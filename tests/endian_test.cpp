//
//  endian_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/endian.hpp>. The byte layouts are asserted at
//  compile time, which is also the proof that the functions are constexpr; the
//  run-time tests cover unaligned access, every byte type, and round trips.
//

#include <snicholls/core/endian.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using snicholls::byteswap;
using snicholls::load_be;
using snicholls::load_le;
using snicholls::store_be;
using snicholls::store_le;
using snicholls::to_be_bytes;
using snicholls::to_le_bytes;

// Layout, at compile time
static_assert(to_le_bytes(std::uint32_t{0x01020304}) == std::array<unsigned char, 4>{0x04, 0x03, 0x02, 0x01});
static_assert(to_be_bytes(std::uint32_t{0x01020304}) == std::array<unsigned char, 4>{0x01, 0x02, 0x03, 0x04});
static_assert(to_le_bytes(std::uint16_t{0xabcd}) == std::array<unsigned char, 2>{0xcd, 0xab});
static_assert(to_le_bytes(std::int8_t{-1}) == std::array<unsigned char, 1>{0xff});
static_assert(to_le_bytes(std::int32_t{-2}) == std::array<unsigned char, 4>{0xfe, 0xff, 0xff, 0xff});

// IEEE bits: 1.0f is 0x3f800000, 1.0 is 0x3ff0000000000000
static_assert(to_be_bytes(1.0f) == std::array<unsigned char, 4>{0x3f, 0x80, 0x00, 0x00});
static_assert(to_le_bytes(1.0) == std::array<unsigned char, 8>{0, 0, 0, 0, 0, 0, 0xf0, 0x3f});

// Round trips, at compile time
constexpr std::uint64_t round_trip_le(std::uint64_t v)
{
    std::array<unsigned char, 8> b{};
    store_le(b.data(), v);
    return load_le<std::uint64_t>(b.data());
}
constexpr double round_trip_be(double v)
{
    std::array<std::byte, 8> b{};
    store_be(b.data(), v);
    return load_be<double>(b.data());
}
static_assert(round_trip_le(0x0102030405060708ull) == 0x0102030405060708ull);
static_assert(round_trip_be(-2.5) == -2.5);

// byteswap
static_assert(byteswap(std::uint32_t{0x01020304}) == 0x04030201u);
static_assert(byteswap(std::uint16_t{0x1234}) == 0x3412u);
static_assert(byteswap(std::uint8_t{0x7f}) == 0x7fu);
static_assert(byteswap(std::uint64_t{0x0102030405060708ull}) == 0x0807060504030201ull);
static_assert(byteswap(std::int32_t{-2}) == std::int32_t(0xfeffffff));
static_assert(byteswap(byteswap(std::int64_t{-123456789})) == -123456789);

// Everything is noexcept
static_assert(noexcept(load_le<std::uint32_t>(static_cast<const char*>(nullptr))));
static_assert(noexcept(store_be(static_cast<char*>(nullptr), 1)));

// Only types with a defined byte representation are accepted
static_assert(snicholls::byte_order_value<int>);
static_assert(snicholls::byte_order_value<double>);
static_assert(!snicholls::byte_order_value<bool>);
static_assert(!snicholls::byte_order_value<long double>);
static_assert(!snicholls::byte_order_value<int*>);
static_assert(snicholls::byte_type<std::byte>);
static_assert(!snicholls::byte_type<int>);

enum class colour : std::uint16_t { red = 0x0102 };
static_assert(to_le_bytes(colour::red) == std::array<unsigned char, 2>{0x02, 0x01});

TEST(EndianUnaligned, LoadsAndStoresAtEveryOffset)
{
    std::vector<unsigned char> buf(32, 0);
    for (std::size_t off = 0; off < 8; ++off) {
        store_le(buf.data() + off, std::uint64_t{0x1122334455667788ull});
        EXPECT_EQ(load_le<std::uint64_t>(buf.data() + off), 0x1122334455667788ull) << "offset " << off;
        store_be(buf.data() + off, std::uint32_t{0xdeadbeef});
        EXPECT_EQ(load_be<std::uint32_t>(buf.data() + off), 0xdeadbeefu) << "offset " << off;
        EXPECT_EQ(buf[off], 0xde);
    }
}

TEST(EndianByteTypes, CharSignedCharAndStdByteAgree)
{
    char c[4];
    signed char s[4];
    std::byte b[4];
    store_le(c, std::int32_t{-1234567});
    store_le(s, std::int32_t{-1234567});
    store_le(b, std::int32_t{-1234567});
    EXPECT_EQ(load_le<std::int32_t>(c), -1234567);
    EXPECT_EQ(load_le<std::int32_t>(s), -1234567);
    EXPECT_EQ(load_le<std::int32_t>(b), -1234567);
    EXPECT_EQ(std::memcmp(c, s, 4), 0);
    EXPECT_EQ(std::memcmp(c, b, 4), 0);
}

TEST(EndianRoundTrip, ExtremesSurviveBothOrders)
{
    const auto check = [](auto v) {
        using T = decltype(v);
        unsigned char b[sizeof(T)];
        store_le(b, v);
        EXPECT_EQ(load_le<T>(b), v);
        store_be(b, v);
        EXPECT_EQ(load_be<T>(b), v);
    };
    check(std::numeric_limits<std::int64_t>::min());
    check(std::numeric_limits<std::int64_t>::max());
    check(std::numeric_limits<std::uint64_t>::max());
    check(std::numeric_limits<std::int16_t>::min());
    check(std::numeric_limits<double>::lowest());
    check(std::numeric_limits<double>::denorm_min());
    check(std::numeric_limits<float>::infinity());
    check(-0.0);
}

TEST(EndianRoundTrip, NanKeepsItsBits)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    unsigned char b[8];
    store_le(b, nan);
    const double back = load_le<double>(b);
    std::uint64_t x = 0, y = 0;
    std::memcpy(&x, &nan, 8);
    std::memcpy(&y, &back, 8);
    EXPECT_EQ(x, y);
}

TEST(EndianHost, LittleEndianLoadMatchesMemcpyOnLittleEndianHosts)
{
    if constexpr (std::endian::native == std::endian::little) {
        const unsigned char b[4] = {1, 2, 3, 4};
        std::uint32_t via_memcpy = 0;
        std::memcpy(&via_memcpy, b, 4);
        EXPECT_EQ(load_le<std::uint32_t>(b), via_memcpy);
        EXPECT_EQ(load_be<std::uint32_t>(b), byteswap(via_memcpy));
    }
    else {
        GTEST_SKIP() << "big-endian host";
    }
}

} // namespace
