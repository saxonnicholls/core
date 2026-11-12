//
//  mapped_header_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/mapped_header.hpp>.
//

// snicholls::containers::mapped_header: the encoded layout byte for byte, the round trip (at compile time too), and
// every refusal: short input, magic, version, size, checksum (any single flipped bit), count over capacity, element
// shape and type tag.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <gtest/gtest.h>

#include <snicholls/core/containers/mapped_header.hpp>
#include <snicholls/core/hash/wyhash.hpp>

namespace {

using namespace snicholls::containers;

struct point
{
    double x, y, z;
};

constexpr mapped_header sample = [] {
    mapped_header h = mapped_header_for<point>(0x1234, 5, 8);
    h.user = 42;
    return h;
}();

constexpr std::array<std::byte, 64> encoded = [] {
    std::array<std::byte, 64> b{};
    (void)encode_header(sample, b);
    return b;
}();

// The round trip and the checks run at compile time.
static_assert(decode_header(encoded).is_ok());
static_assert(*decode_header(encoded) == sample);
static_assert(validate_header(encoded, mapped_header_for<point>(0x1234)).is_ok());
static_assert(validate_header(encoded, mapped_header_for<float>(0x1234)).err() == header_error::element_mismatch);
static_assert(mapped_header_for<point>(7).element_size == 24 && mapped_header_for<point>(7).element_align == 8);
static_assert(mapped_header_magic == 0x524448504d414e53ull);

std::uint64_t le64(const std::array<std::byte, 64>& b, std::size_t at)
{
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < 8; ++i) v |= std::uint64_t(b[at + i]) << (8 * i);
    return v;
}

TEST(MappedHeaderEncode, WritesTheDocumentedLayoutLittleEndian)
{
    EXPECT_EQ(char(encoded[0]), 'S');
    EXPECT_EQ(char(encoded[7]), 'R');
    EXPECT_EQ(le64(encoded, 0), mapped_header_magic);
    EXPECT_EQ(le64(encoded, 8), (std::uint64_t{64} << 32) | mapped_header_version);
    EXPECT_EQ(le64(encoded, 16), 0x1234u);
    EXPECT_EQ(le64(encoded, 24), (std::uint64_t{8} << 32) | 24u);
    EXPECT_EQ(le64(encoded, 32), 5u);
    EXPECT_EQ(le64(encoded, 40), 8u);
    EXPECT_EQ(le64(encoded, 48), 42u);
    EXPECT_EQ(le64(encoded, 56), snicholls::wyhash(std::span<const std::byte>(encoded).first(56), mapped_header_magic));
}

TEST(MappedHeaderEncode, RefusesAShortBufferAndCountOverCapacityWritingNothing)
{
    std::array<std::byte, 64> out{};
    EXPECT_EQ(encode_header(sample, std::span<std::byte>(out).first(63)).err(), header_error::too_short);
    EXPECT_EQ(encode_header(mapped_header_for<point>(0, 9, 8), out).err(), header_error::count_over_capacity);
    EXPECT_EQ(out, (std::array<std::byte, 64>{}));
    std::array<std::byte, 100> longer{};
    EXPECT_TRUE(encode_header(sample, longer).is_ok());   // writes the first 64 only
    EXPECT_EQ(longer[64], std::byte{0});
    EXPECT_EQ(*decode_header(longer), sample);
}

TEST(MappedHeaderDecode, RefusesShortInputAndForeignBytes)
{
    EXPECT_EQ(decode_header(std::span<const std::byte>(encoded).first(63)).err(), header_error::too_short);
    EXPECT_EQ(decode_header(std::array<std::byte, 64>{}).err(), header_error::bad_magic);
}

TEST(MappedHeaderDecode, RefusesAnotherVersionOrSizeBeforeTheChecksum)
{
    auto b = encoded;
    b[8] = std::byte{2};
    EXPECT_EQ(decode_header(b).err(), header_error::bad_version);
    b = encoded;
    b[12] = std::byte{128};
    EXPECT_EQ(decode_header(b).err(), header_error::bad_size);
}

TEST(MappedHeaderDecode, AnySingleFlippedBitAfterTheFixedFieldsIsCaught)
{
    for (std::size_t byte = 16; byte < 64; ++byte)
        for (int bit = 0; bit < 8; ++bit) {
            auto b = encoded;
            b[byte] ^= std::byte(1u << bit);
            EXPECT_EQ(decode_header(b).err(), header_error::bad_checksum) << "byte " << byte << " bit " << bit;
        }
}

TEST(MappedHeaderDecode, RefusesCountOverCapacityEvenWithAValidChecksum)
{
    // Write the fields by hand, as a buggy writer would, with a correct checksum.
    auto b = encoded;
    b[32] = std::byte{9};
    const auto sum = snicholls::wyhash(std::span<const std::byte>(b).first(56), mapped_header_magic);
    for (std::size_t i = 0; i < 8; ++i) b[56 + i] = std::byte((sum >> (8 * i)) & 0xff);
    EXPECT_EQ(decode_header(b).err(), header_error::count_over_capacity);
}

TEST(MappedHeaderValidate, ChecksTheReadersElementShapeAndTag)
{
    EXPECT_TRUE(validate_header(encoded, mapped_header_for<point>(0x1234, 999, 1)).is_ok());   // counts ignored
    EXPECT_EQ(validate_header(encoded, mapped_header_for<point>(0x1235)).err(), header_error::type_mismatch);
    mapped_header other_align = mapped_header_for<point>(0x1234);
    other_align.element_align = 16;
    EXPECT_EQ(validate_header(encoded, other_align).err(), header_error::element_mismatch);
    auto b = encoded;
    b[0] = std::byte{0};
    EXPECT_EQ(validate_header(b, sample).err(), header_error::bad_magic);   // structure first
}

} // namespace
