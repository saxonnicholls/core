//
//  random_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/random.hpp>: reproducibility, the generator concept, bounds,
//  uniformity by chi-square, and the shuffle.
//

#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace {

using snicholls::stats::wyrand;
namespace stats = snicholls::stats;

consteval std::uint64_t third_draw(std::uint64_t seed)
{
    wyrand g(seed);
    static_cast<void>(g());
    static_cast<void>(g());
    return g();
}
static_assert(third_draw(1) == third_draw(1) && third_draw(1) != third_draw(2));   // constexpr and seeded
static_assert(std::uniform_random_bit_generator<wyrand> && stats::full_range_generator<wyrand>);
static_assert(stats::full_range_generator<std::mt19937> && stats::full_range_generator<std::mt19937_64>);
static_assert(!stats::full_range_generator<std::minstd_rand>);   // not a full range: its draws would be biased
static_assert(sizeof(wyrand) == 8 && std::is_trivially_copyable_v<wyrand>);
static_assert(wyrand::stream(7, 0) != wyrand::stream(7, 1) && wyrand::stream(7, 1) == wyrand::stream(7, 1));

// A 32-bit generator's two draws make one 64-bit value, high half first
struct counter32 {
    using result_type = std::uint32_t;
    std::uint32_t next = 1;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return 0xFFFFFFFFu; }
    constexpr result_type operator()() { return next++; }
};
static_assert([] {
    counter32 g;
    return stats::random_u64(g) == ((std::uint64_t{1} << 32) | 2u);
}());

TEST(StatsRandom, UniformIndexStaysInRangeAndIsUniform)
{
    wyrand g(42);
    std::array<int, 7> hits{};
    const int draws = 700000;
    for (int i = 0; i < draws; ++i)
        ++hits[stats::uniform_index(g, 7)];
    double chi2 = 0;
    for (const int h : hits)
        chi2 += (h - draws / 7.0) * (h - draws / 7.0) / (draws / 7.0);
    EXPECT_LT(chi2, 22.46);   // 6 degrees of freedom, p = 0.001
    EXPECT_EQ(stats::uniform_index(g, 1), 0u);
}

TEST(StatsRandom, UniformIndex32IsUniformAndBounded)
{
    wyrand g(3);
    std::array<int, 5> hits{};
    const int draws = 500000;
    const auto refill = [&g] { return static_cast<std::uint32_t>(g()); };
    for (int i = 0; i < draws; ++i)
        ++hits[stats::uniform_index32(static_cast<std::uint32_t>(g()), 5, refill)];
    double chi2 = 0;
    for (const int h : hits)
        chi2 += (h - draws / 5.0) * (h - draws / 5.0) / (draws / 5.0);
    EXPECT_LT(chi2, 18.47);   // 4 degrees of freedom, p = 0.001
    for (int i = 0; i < 1000; ++i)
        EXPECT_LT(stats::uniform_index32(static_cast<std::uint32_t>(g()), 0xFFFFFFFFu, refill), 0xFFFFFFFFu);
}

TEST(StatsRandom, UniformRealIsInTheUnitIntervalWithMeanAHalf)
{
    wyrand g(9);
    double sum = 0;
    for (int i = 0; i < 100000; ++i) {
        const double u = stats::uniform_real(g);
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
        sum += u;
    }
    EXPECT_NEAR(sum / 100000, 0.5, 0.005);
}

TEST(StatsRandom, ShuffleIsAPermutationAndReproducible)
{
    std::vector<int> a(100), b(100);
    std::iota(a.begin(), a.end(), 0);
    std::iota(b.begin(), b.end(), 0);
    wyrand g1(5), g2(5);
    stats::shuffle(std::span<int>(a), g1);
    stats::shuffle(std::span<int>(b), g2);
    EXPECT_EQ(a, b);
    std::vector<int> sorted = a;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(sorted[static_cast<std::size_t>(i)], i);
    EXPECT_NE(a[0] * 1000 + a[1], 1);   // moved
}

// ---- known sequences, computed independently -----------------------------------------------------------------------
// From wyhash 4.2's published wyrand (state += 0x2d358dccaa6c78a5; mix(state, state ^ 0x8bb84b93962eacc9)), Lemire's
// draw and splitmix64, evaluated with Python's exact integers, never by core. A change here breaks every seeded result
// that callers stored, so these values are the contract.

constexpr std::array<std::uint64_t, 6> draws(std::uint64_t seed)
{
    wyrand g(seed);
    std::array<std::uint64_t, 6> out{};
    for (auto& x : out)
        x = g();
    return out;
}
static_assert(draws(0) == std::array<std::uint64_t, 6>{0x9a45cd888d59f0d6, 0x1445b6a189663f5, 0x1842218b97e7a496, 0x4dda1bc7277a55f9,
                                                       0x120d43ca60abacb4, 0x4d8c1a51fabcc9a2});
static_assert(draws(42) == std::array<std::uint64_t, 6>{0xca71d87c76983989, 0x7e5ba61552085fc6, 0xcdf101e3bab88b9f, 0xa3825ad73267808,
                                                        0x8ac0adc15d671c29, 0x74b0aa52525d790d});
static_assert(draws(~std::uint64_t{0}) == std::array<std::uint64_t, 6>{0x10ba1525e333549e, 0x205fb8bd591710a9, 0x9fdaea1b39c149ae,
                                                                       0x2fed759e18314395, 0x9ae538551694cf7c, 0x904f0d8d1ef7d30b});

constexpr std::array<std::uint64_t, 7> indices()
{
    wyrand g(7);
    const std::array<std::uint64_t, 7> n{1, 2, 3, 10, 1000, (std::uint64_t{1} << 63) + 1, ~std::uint64_t{0}};
    std::array<std::uint64_t, 7> out{};
    for (std::size_t i = 0; i < n.size(); ++i)
        out[i] = stats::uniform_index(g, n[i]);
    return out;
}
static_assert(indices() == std::array<std::uint64_t, 7>{0, 0, 2, 4, 116, 1526808058992259716ull, 14602987178384512604ull});

constexpr std::array<std::uint64_t, 3> stream_draws(std::uint64_t k)
{
    wyrand g = wyrand::stream(42, k);
    return {g(), g(), g()};
}
static_assert(stream_draws(0) == std::array<std::uint64_t, 3>{0x8a4c76c83510bc62, 0xc900a68564c62c0c, 0xeb6f695632327c2c});
static_assert(stream_draws(1) == std::array<std::uint64_t, 3>{0x7fc8f0cedce13bd5, 0x987ddd6c707b6c7c, 0x4d27dbe80445634e});
static_assert(stream_draws(1000) == std::array<std::uint64_t, 3>{0xfe0cc61c4e65b0aa, 0xf5996debfa5c9516, 0xa6cf25176b382ece});

TEST(StatsRandomKnownSequences, DrawsIndicesRealsAndStreamsMatchTheReferenceAtRunTime)
{
    EXPECT_EQ(draws(42)[3], 0xa3825ad73267808u);
    EXPECT_EQ(indices()[6], 14602987178384512604ull);
    EXPECT_EQ(stream_draws(1000)[0], 0xfe0cc61c4e65b0aau);
    wyrand g(7);
    EXPECT_EQ(stats::uniform_real(g), 0.773381802821399);
    EXPECT_EQ(stats::uniform_real(g), 0.2224859770550386);
    EXPECT_EQ(stats::uniform_real(g), 0.7182904091670301);
}

} // namespace
