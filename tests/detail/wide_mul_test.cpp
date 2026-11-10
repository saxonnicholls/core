//
//  wide_mul_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/wide_mul.hpp>: the constant-expression path
//  (four 32-bit multiplies) against known products, and the run-time path
//  (__int128 where the compiler has it) against the constant-expression one.
//

#include <snicholls/core/detail/wide_mul.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using snicholls::detail::umul128;

constexpr std::uint64_t all = ~std::uint64_t{0};

// Known products, folded by the compiler
static_assert(umul128(0, all).low == 0 && umul128(0, all).high == 0);
static_assert(umul128(1, all).low == all && umul128(1, all).high == 0);
static_assert(umul128(std::uint64_t{1} << 32, std::uint64_t{1} << 32).low == 0 &&
              umul128(std::uint64_t{1} << 32, std::uint64_t{1} << 32).high == 1);          // 2^64
static_assert(umul128(all, all).low == 1 && umul128(all, all).high == all - 1);           // (2^64-1)^2
// 0xdeadbeefcafebabe * 0x0123456789abcdef, computed with Python's integers
static_assert(umul128(0xdeadbeefcafebabeull, 0x0123456789abcdefull).high == 0x00fd5bdeeeb2a01dull);
static_assert(umul128(0xdeadbeefcafebabeull, 0x0123456789abcdefull).low == 0x7eb689f4ea447d62ull);
static_assert(umul128(3, 5).low == 15 && umul128(3, 5).high == 0);
static_assert(noexcept(umul128(1, 2)));

TEST(WideMul, RunTimePathAgreesWithTheConstantExpressionPath)
{
    // Pairs chosen to exercise every carry: the compile-time values are the 32-bit path by construction.
    constexpr std::uint64_t a[] = {0, 1, all, all - 1, 0x8000000000000000ull, 0xffffffff00000000ull, 0x00000000ffffffffull,
                                   0xdeadbeefcafebabeull, 0x9e3779b97f4a7c15ull, 12345678901234567ull};
    for (const auto x : a)
        for (const auto y : a) {
            const auto run = umul128(x, y);   // run time: __int128 where available
            // Recompute through the 32-bit path by forcing it: the same formula written out.
            const std::uint64_t xh = x >> 32, xl = static_cast<std::uint32_t>(x), yh = y >> 32, yl = static_cast<std::uint32_t>(y);
            const std::uint64_t hh = xh * yh, hl = xh * yl, lh = xl * yh, ll = xl * yl;
            const std::uint64_t mid = (ll >> 32) + static_cast<std::uint32_t>(hl) + static_cast<std::uint32_t>(lh);
            EXPECT_EQ(run.low, (mid << 32) | static_cast<std::uint32_t>(ll));
            EXPECT_EQ(run.high, hh + (hl >> 32) + (lh >> 32) + (mid >> 32));
        }
}

TEST(WideMul, MatchesTheKnownSquareOfTwoToTheSixtyThree)
{
    const auto p = umul128(std::uint64_t{1} << 63, std::uint64_t{1} << 63);   // 2^126
    EXPECT_EQ(p.low, 0u);
    EXPECT_EQ(p.high, std::uint64_t{1} << 62);
}

} // namespace
