//
//  hierarchical_bitset_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/hierarchical_bitset.hpp>.
//

// snicholls::containers::hierarchical_bitset: model-checked against std::vector<bool> for sizes that need one to four
// levels, word boundaries, empty and full sets, resize.

#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/containers/hierarchical_bitset.hpp>

namespace {

using snicholls::containers::hierarchical_bitset;
constexpr auto npos = hierarchical_bitset::npos;

std::size_t model_next(const std::vector<bool>& v, std::size_t i)
{
    for (; i < v.size(); ++i)
        if (v[i]) return i;
    return npos;
}

std::size_t model_prev(const std::vector<bool>& v, std::size_t i)
{
    if (v.empty()) return npos;
    if (i >= v.size()) i = v.size() - 1;
    for (std::size_t k = i + 1; k-- > 0;)
        if (v[k]) return k;
    return npos;
}

TEST(HierarchicalBitsetBasics, EmptyAndZeroSize)
{
    hierarchical_bitset zero;
    EXPECT_EQ(zero.size(), 0u);
    EXPECT_FALSE(zero.any());
    EXPECT_EQ(zero.find_next(0), npos);
    EXPECT_EQ(zero.find_prev(0), npos);
    hierarchical_bitset b(100);
    EXPECT_FALSE(b.any());
    EXPECT_EQ(b.find_next(0), npos);
    EXPECT_EQ(b.find_prev(99), npos);
    EXPECT_EQ(b.find_next(1000), npos);
}

TEST(HierarchicalBitsetBasics, WordBoundaries)
{
    hierarchical_bitset b(64 * 64 * 2);
    for (std::size_t i : {0u, 63u, 64u, 4095u, 4096u, 8191u}) {
        b.set(i);
        EXPECT_TRUE(b.test(i));
    }
    EXPECT_EQ(b.find_next(1), 63u);
    EXPECT_EQ(b.find_next(65), 4095u);
    EXPECT_EQ(b.find_next(4097), 8191u);
    EXPECT_EQ(b.find_prev(4094), 64u);
    EXPECT_EQ(b.find_prev(8190), 4096u);
    EXPECT_EQ(b.find_prev(1000000), 8191u);   // clamped
    b.reset(63);
    EXPECT_EQ(b.find_next(1), 64u);
    b.clear();
    EXPECT_FALSE(b.any());
}

TEST(HierarchicalBitsetResize, ResizeClears)
{
    hierarchical_bitset b(10);
    b.set(3);
    b.resize(500);
    EXPECT_EQ(b.size(), 500u);
    EXPECT_FALSE(b.any());
}

void model_check(std::size_t n, unsigned seed)
{
    std::mt19937_64 rng(seed);
    hierarchical_bitset b(n);
    std::vector<bool> v(n);
    for (int step = 0; step < 20000; ++step) {
        const std::size_t i = rng() % n;
        if (rng() % 2) {
            b.set(i);
            v[i] = true;
        } else {
            b.reset(i);
            v[i] = false;
        }
        const std::size_t q = rng() % (n + 3);
        ASSERT_EQ(b.find_next(q), model_next(v, q)) << "n=" << n << " q=" << q;
        ASSERT_EQ(b.find_prev(q), model_prev(v, q)) << "n=" << n << " q=" << q;
        ASSERT_EQ(b.any(), model_next(v, 0) != npos);
    }
}

TEST(HierarchicalBitsetModel, OneLevel) { model_check(50, 1); }
TEST(HierarchicalBitsetModel, TwoLevels) { model_check(3000, 2); }
TEST(HierarchicalBitsetModel, ThreeLevels) { model_check(300000, 3); }

TEST(HierarchicalBitsetModel, SparseFourLevels)
{
    // Few bits in a big set: the find functions must climb and descend every level.
    constexpr std::size_t n = (std::size_t{1} << 24) + 5;
    hierarchical_bitset b(n);
    b.set(5);
    b.set(n - 1);
    EXPECT_EQ(b.find_next(6), n - 1);
    EXPECT_EQ(b.find_prev(n - 2), 5u);
    b.reset(5);
    EXPECT_EQ(b.find_prev(n - 2), npos);
    EXPECT_EQ(b.find_next(0), n - 1);
}

} // namespace
