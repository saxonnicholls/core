//
//  sha1_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/sha1.hpp>.
//

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <snicholls/core/detail/sha1.hpp>

namespace {

using snicholls::detail::sha1;
using snicholls::detail::sha1_digest;

std::string hex(const sha1::digest_type& d)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string s;
    for (const std::uint8_t b : d) {
        s += digits[b >> 4];
        s += digits[b & 0x0F];
    }
    return s;
}

constexpr bool digest_starts(const sha1::digest_type& d, std::uint8_t a, std::uint8_t b, std::uint8_t c)
{
    return d[0] == a && d[1] == b && d[2] == c;
}

// FIPS 180-4 / NIST CAVP examples.
TEST(Sha1Vectors, FipsExamples)
{
    EXPECT_EQ(hex(sha1_digest("")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    EXPECT_EQ(hex(sha1_digest("abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
    EXPECT_EQ(hex(sha1_digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
              "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    EXPECT_EQ(hex(sha1_digest("The quick brown fox jumps over the lazy dog")),
              "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12");
}

TEST(Sha1Vectors, OneMillionAs)
{
    sha1 h;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i)
        h.update(chunk);
    EXPECT_EQ(hex(h.finish()), "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
}

TEST(Sha1Padding, EveryLengthAroundTheBlockBoundaryMatchesOneShotAndSplitUpdates)
{
    // 55, 56, 63, 64, 65 bytes exercise the one-block / two-block padding cases. Splitting the input at every
    // point must not change the digest.
    for (std::size_t n : {0U, 1U, 55U, 56U, 57U, 63U, 64U, 65U, 119U, 120U, 128U}) {
        const std::string text(n, 'x');
        const auto whole = sha1_digest(text);
        for (std::size_t cut = 0; cut <= n; ++cut) {
            sha1 h;
            h.update(std::string_view(text).substr(0, cut));
            h.update(std::string_view(text).substr(cut));
            ASSERT_EQ(h.finish(), whole) << "n=" << n << " cut=" << cut;
        }
    }
    // Two known lengths at the boundary.
    EXPECT_EQ(hex(sha1_digest(std::string(64, 'a'))), "0098ba824b5c16427bd7a1122a5a442a25ec644d");
    EXPECT_EQ(hex(sha1_digest(std::string(56, 'a'))), "c2db330f6083854c99d4b5bfb6e8f29f201be699");
}

TEST(Sha1Inputs, BytesAndTextAgree)
{
    const std::array<std::uint8_t, 3> u8{'a', 'b', 'c'};
    const std::array<std::byte, 3> b{std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    sha1 x;
    x.update(std::span<const std::uint8_t>(u8));
    sha1 y;
    y.update(std::span<const std::byte>(b));
    const auto expected = sha1_digest("abc");
    EXPECT_EQ(x.finish(), expected);
    EXPECT_EQ(y.finish(), expected);
}

TEST(Sha1Constexpr, DigestAtCompileTime)
{
    static_assert(digest_starts(sha1_digest("abc"), 0xa9, 0x99, 0x3e));
    static_assert(noexcept(sha1_digest(std::string_view{})));
    SUCCEED();
}

} // namespace
