//
//  mul_div_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/mul_div.hpp>: the 512-bit product and Knuth division, against the same values
//  computed by uint256's own divmod where they fit, and by exact identities where they do not.
//

#include <snicholls/core/detail/mul_div.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using snicholls::uint256;
using snicholls::detail::mul_div;

static_assert(mul_div(6, 7, 4)->first == uint256(10) && mul_div(6, 7, 4)->second == uint256(2));
static_assert(!mul_div(1, 1, 0));

TEST(MulDiv, AgreesWithUint256WhereTheProductFits)
{
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (int i = 0; i < 2000; ++i) {
        const uint256 a = uint256::from_limbs(next(), next() >> (i % 64));
        const uint256 b = uint256::from_limbs(next(), i % 3 == 0 ? next() >> 1 : 0);
        uint256 d = uint256::from_limbs(next() | 1, i % 5 == 0 ? next() : 0);
        const auto p = uint256::checked_mul(a, b);
        if (!p)
            continue;
        const auto [q, r] = p->divmod(d);
        const auto got = mul_div(a, b, d);
        ASSERT_TRUE(got);
        EXPECT_EQ(got->first, q);
        EXPECT_EQ(got->second, r);
    }
}

TEST(MulDiv, HandlesProductsBeyond256Bits)
{
    const uint256 max = uint256::max();
    // (2^256 - 1)^2 / (2^256 - 1) = 2^256 - 1, remainder 0
    EXPECT_EQ(mul_div(max, max, max)->first, max);
    // (2^256 - 1) * 2^255 / 2^255 = 2^256 - 1
    const uint256 half = uint256(1) << 255;
    EXPECT_EQ(mul_div(max, half, half)->first, max);
    EXPECT_TRUE(mul_div(max, half, half)->second.is_zero());
    // (2^256 - 1) * 3 / 4: quotient 3 * 2^254 - 1, remainder 1
    const auto q = mul_div(max, 3, 4);
    EXPECT_EQ(q->first, uint256(3) * (uint256(1) << 254) - 1);
    EXPECT_EQ(q->second, uint256(1));
    EXPECT_FALSE(mul_div(max, 2, 1));   // needs 257 bits
}

} // namespace
