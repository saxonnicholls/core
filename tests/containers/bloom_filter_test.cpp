//
//  bloom_filter_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/containers/bloom_filter.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

using snicholls::containers::bloom_filter;
using snicholls::containers::bloom_parameters;
using snicholls::containers::bloom_sizing;

std::vector<std::string> keys(std::size_t n, std::string_view prefix)
{
    std::vector<std::string> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        out.push_back(std::string{prefix} + std::to_string(i * 2654435761u));
    return out;
}

TEST(BloomSizing, FollowsTheTextbookFormula)
{
    // 1,000,000 keys at 0.5%: m = -n ln p / ln2^2 = 11,027,754 bits -> rounded up to 64; k = round(m/n ln2) = 8
    const auto p = bloom_sizing(1'000'000, 0.005);
    EXPECT_EQ(p.bits % 64, 0u);
    EXPECT_NEAR(static_cast<double>(p.bits), 11'027'754.0, 128.0);
    EXPECT_EQ(p.hashes, 8u);

    // 1% at 10,000 keys: 95,851 bits, k = 7
    const auto q = bloom_sizing(10'000, 0.01);
    EXPECT_NEAR(static_cast<double>(q.bits), 95'851.0, 64.0);
    EXPECT_EQ(q.hashes, 7u);

    // Nonsense inputs are made sane
    EXPECT_GE(bloom_sizing(0, 0.01).bits, 64u);
    EXPECT_EQ(bloom_sizing(100, 0.0).hashes, bloom_sizing(100, 0.01).hashes);
    EXPECT_EQ(bloom_sizing(100, 5.0).bits, bloom_sizing(100, 0.01).bits);
}

TEST(BloomFilter, NeverForgetsAKeyItWasGiven)
{
    const auto members = keys(10'000, "addr-");
    bloom_filter f(members.size(), 0.01);
    for (const auto& k : members)
        f.add(k);
    EXPECT_EQ(f.count(), members.size());
    for (const auto& k : members)
        EXPECT_TRUE(f.maybe_contains(k)) << k;
}

TEST(BloomFilter, FalsePositiveRateIsNearItsTarget)
{
    const auto members = keys(10'000, "in-");
    const auto strangers = keys(10'000, "out-");
    bloom_filter f(members.size(), 0.01);
    for (const auto& k : members)
        f.add(k);
    std::size_t hits = 0;
    for (const auto& k : strangers)
        hits += f.maybe_contains(k) ? 1 : 0;
    const double rate = static_cast<double>(hits) / static_cast<double>(strangers.size());
    EXPECT_LT(rate, 0.02) << "rate " << rate;   // target 1%; twice that is the alarm
    EXPECT_NEAR(f.estimated_false_positive_rate(), 0.01, 0.005);
}

TEST(BloomFilter, EmptyFilterContainsNothing)
{
    const bloom_filter f(1000, 0.01);
    EXPECT_TRUE(f.empty());
    EXPECT_EQ(f.count(), 0u);
    EXPECT_FALSE(f.maybe_contains("anything"));
    EXPECT_EQ(f.estimated_false_positive_rate(), 0.0);
    EXPECT_EQ(f.memory_bytes(), f.bits() / 8);

    const bloom_filter tiny;   // the default: 64 bits, one hash
    EXPECT_EQ(tiny.bits(), 64u);
    EXPECT_EQ(tiny.hashes(), 1u);
    EXPECT_FALSE(tiny.maybe_contains("x"));
}

TEST(BloomFilter, ClearForgetsEverythingAndKeepsTheSize)
{
    bloom_filter f(100, 0.01);
    f.add("one");
    ASSERT_TRUE(f.maybe_contains("one"));
    const auto bits = f.bits();
    f.clear();
    EXPECT_FALSE(f.maybe_contains("one"));
    EXPECT_EQ(f.count(), 0u);
    EXPECT_EQ(f.bits(), bits);
    for (const auto w : f.words())
        EXPECT_EQ(w, 0u);
}

TEST(BloomFilter, MergeIsTheUnion)
{
    bloom_filter a(1000, 0.01), b(1000, 0.01);
    a.add("in a");
    b.add("in b");
    a.merge(b);
    EXPECT_TRUE(a.maybe_contains("in a"));
    EXPECT_TRUE(a.maybe_contains("in b"));
    EXPECT_FALSE(b.maybe_contains("in a"));
    EXPECT_EQ(a.count(), 2u);
}

TEST(BloomFilter, TheSameBytesAreTheSameKeyInEveryForm)
{
    bloom_filter f(100, 0.01);
    const std::string text{"0xabc"};
    f.add(std::string_view{text});
    EXPECT_TRUE(f.maybe_contains(std::span<const std::byte>{reinterpret_cast<const std::byte*>(text.data()), text.size()}));
    EXPECT_TRUE(f.maybe_contains(std::span<const unsigned char>{reinterpret_cast<const unsigned char*>(text.data()), text.size()}));
    EXPECT_FALSE(f.maybe_contains("0xabd"));
}

TEST(BloomFilter, ExplicitParametersAreRoundedAndClamped)
{
    const bloom_filter f(bloom_parameters{100, 0});
    EXPECT_EQ(f.bits(), 128u);   // up to a multiple of 64
    EXPECT_EQ(f.hashes(), 1u);
    const bloom_filter g(bloom_parameters{1, 99});
    EXPECT_EQ(g.bits(), 64u);
    EXPECT_EQ(g.hashes(), 32u);
}

TEST(BloomFilter, WordsExposeTheBitsSet)
{
    bloom_filter f(bloom_parameters{64, 3});
    f.add("k");
    std::size_t set = 0;
    for (const auto w : f.words())
        for (int i = 0; i < 64; ++i)
            set += (w >> i) & 1u;
    EXPECT_GE(set, 1u);
    EXPECT_LE(set, 3u);   // three positions, possibly coinciding
}

} // namespace
