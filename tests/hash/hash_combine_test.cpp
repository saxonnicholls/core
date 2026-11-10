//
//  hash/hash_combine_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/hash_combine.hpp>. ReferenceValues is new: it pins the algorithm to values
//  computed independently of this code.
//

#include <snicholls/core/hash/hash_combine.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <unordered_set>

namespace {

TEST(HashCombine, IsConstexprAndNoexcept)
{
    static_assert(noexcept(snicholls::hash_combine(0, 0)));
    constexpr std::size_t v = snicholls::hash_combine(0, 1);
    static_assert(v != 0);
    EXPECT_EQ(v, snicholls::hash_combine(0, 1));   // deterministic
}

TEST(HashCombine, IsOrderSensitive)
{
    // A chain from a fixed seed is order-sensitive: every step mixes the running seed. (A single step
    // adds seed and h before mixing, so hash_combine(1, 2) == hash_combine(2, 1) - as in Boost.)
    static_assert(snicholls::hash_combine(snicholls::hash_combine(0, 1), 2) != snicholls::hash_combine(snicholls::hash_combine(0, 2), 1));
    static_assert(snicholls::hash_combine(snicholls::hash_combine(0, 5), 5) != snicholls::hash_combine(snicholls::hash_combine(0, 0), 0));
}

TEST(HashCombine, MixesSmallInputsAcrossTheWord)
{
    // Consecutive small hashes (std::hash<int> is the identity on libc++ / libstdc++) must not stay
    // consecutive: every output differs from the others in many bits, including the high ones.
    std::unordered_set<std::size_t> seen;
    for (std::size_t i = 0; i < 1000; ++i) {
        const std::size_t x = snicholls::hash_combine(0, i);
        const std::size_t y = snicholls::hash_combine(0, i + 1);
        EXPECT_GE(std::popcount(static_cast<std::size_t>(x ^ y)), 8) << i;
        seen.insert(x);
    }
    EXPECT_EQ(seen.size(), 1000U);
    if constexpr (sizeof(std::size_t) == 8) {   // braces: EXPECT_NE expands to an if/else (GCC -Wdangling-else)
        EXPECT_NE(snicholls::hash_combine(0, 1) >> 32, 0U);
    }
}

TEST(HashCombine, ReferenceValues)
{
    // Computed with an independent transcription of Boost.ContainerHash 1.81's 64-bit hash_mix
    // (Python, arbitrary precision masked to 64 bits), not with this code.
    // On a 32-bit size_t the result is narrowed, so the 64-bit references apply to 64-bit platforms.
    if constexpr (sizeof(std::size_t) == 8) {
        constexpr std::uint64_t zero_zero = snicholls::hash_combine(0, 0);
        constexpr std::uint64_t zero_one = snicholls::hash_combine(0, 1);
        constexpr std::uint64_t chain = snicholls::hash_combine(snicholls::hash_combine(0, 1), 2);
        constexpr std::uint64_t wide = snicholls::hash_combine(static_cast<std::size_t>(0x123456789abcdef0ULL),
                                                               static_cast<std::size_t>(0xfedcba9876543210ULL));
        EXPECT_EQ(zero_zero, 0xa55db391e20904c2ULL);
        EXPECT_EQ(zero_one, 0x1ed1b5abbd8399b7ULL);
        EXPECT_EQ(chain, 0x30b3fc98529bf99eULL);
        EXPECT_EQ(wide, 0xd3978720ba3a3bbaULL);
    }
    SUCCEED();
}

} // namespace
