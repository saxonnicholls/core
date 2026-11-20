//
//  crc_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/crc.hpp>: the check values of the CRC catalogue ("123456789"), Python's
//  zlib.crc32 and crcmod on random lengths, every split, resume, and the tables built at compile time.
//

#include <snicholls/core/hash/crc.hpp>

#include "../support/hash_fixtures.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>

namespace {

using namespace snicholls;

static_assert(crc32::hash("123456789") == 0xCBF43926u);
static_assert(crc32c::hash("123456789") == 0xE3069283u);
static_assert(crc64_xz::hash("123456789") == 0x995DC9BBDF1939FAull);
static_assert(crc32::hash("") == 0 && crc32c::hash("") == 0 && crc64_xz::hash("") == 0);
static_assert(crc32::hash("The quick brown fox jumps over the lazy dog") == 0x414FA339u);
static_assert(crc32::resume(crc32::hash("1234")).update("56789").finish() == 0xCBF43926u);
static_assert(crc64_xz::resume(crc64_xz::hash("12345678")).update("9").finish() == 0x995DC9BBDF1939FAull);
static_assert(std::is_same_v<decltype(crc32::hash("")), std::uint32_t> && std::is_same_v<crc64_xz::value_type, std::uint64_t>);
static_assert(sizeof(crc32) == 4 && sizeof(crc64_xz) == 8 && std::is_trivially_copyable_v<crc32c>);
static_assert(noexcept(crc32c::hash("")));
static_assert(detail::crc_tables<std::uint32_t, 0xEDB88320u>[0][1] == 0x77073096u);   // zlib's table, entry 1

std::string crc_hex(std::uint32_t v)
{
    char buf[9];
    std::snprintf(buf, sizeof buf, "%08x", static_cast<unsigned>(v));
    return buf;
}

std::string crc_hex(std::uint64_t v)
{
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

template <typename C>
void expect_oracle(const char* name)
{
    for (const auto& c : test::read_oracle("oracle_crc.txt", name))
        ASSERT_EQ(crc_hex(C::hash(test::oracle_message(c.length))), c.hex) << name << " line " << c.line;
}

TEST(CrcOracle, ZlibAndCrcmodAtEveryAlignmentAndRandomLengths)
{
    expect_oracle<crc32>("crc32");
    expect_oracle<crc32c>("crc32c");
    expect_oracle<crc64_xz>("crc64_xz");
}

template <typename C>
void expect_every_split()
{
    for (const std::size_t n : {0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u, 64u, 100u}) {
        const auto m = test::oracle_message(n);
        const auto whole = C::hash(m);
        for (std::size_t a = 0; a <= n; ++a)
            for (std::size_t b = a; b <= n; ++b) {
                ASSERT_EQ(test::hash_in_parts<C>(m, a, b), whole) << "n=" << n << " a=" << a << " b=" << b;
                // resume from the finished value of the prefix is the same as continuing
                ASSERT_EQ(C::resume(C::hash(std::span<const std::byte>(m).first(a))).update(std::span<const std::byte>(m).subspan(a)).finish(),
                          whole);
            }
    }
}

TEST(CrcIncremental, EverySplitAndEveryResumeGivesTheOneShotValue)
{
    expect_every_split<crc32>();
    expect_every_split<crc32c>();
    expect_every_split<crc64_xz>();
}

TEST(CrcIncremental, FinishIsConstSoMoreCanBeAbsorbed)
{
    crc32 c;
    c.update("1234");
    EXPECT_EQ(c.finish(), crc32::hash("1234"));
    c.update("56789");
    EXPECT_EQ(c.finish(), 0xCBF43926u);
}

} // namespace
