//
//  dense_id_map_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/dense_id_map.hpp>.
//

// snicholls::containers::dense_id_map: model-checked against std::unordered_map for sequential (exchange-style), random and
// mixed ids - including ids below the window, far ahead of it (fallback) and fallback ids the window later covers.

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/containers/dense_id_map.hpp>

namespace {

using snicholls::containers::dense_id_map;

int values[1 << 16];
int* ptr(std::uint64_t k) { return &values[k & 0xffff]; }

TEST(DenseIdMapBasics, InsertFindEraseDuplicatesAndClear)
{
    dense_id_map<int*> m;
    EXPECT_TRUE(m.empty());
    EXPECT_TRUE(m.insert(100, ptr(100)));
    EXPECT_FALSE(m.insert(100, ptr(1)));
    ASSERT_NE(m.find(100), nullptr);
    EXPECT_EQ(*m.find(100), ptr(100));
    EXPECT_EQ(m.find(101), nullptr);
    EXPECT_TRUE(m.erase(100));
    EXPECT_FALSE(m.erase(100));
    EXPECT_TRUE(m.insert(5, ptr(5)));       // below the window: fallback
    EXPECT_EQ(m.outliers(), 1u);
    EXPECT_TRUE(m.contains(5));
    m.clear();
    EXPECT_TRUE(m.empty());
    EXPECT_FALSE(m.contains(5));
    const dense_id_map<int*>& cm = m;
    EXPECT_EQ(cm.find(5), nullptr);
}

// Regression (pre-0.1.0 review): insert(empty_key) reached flat_id_map's fallback, which counted it without storing it.
TEST(DenseIdMapBasics, InsertingTheReservedKeyIsRefusedAndChangesNothing)
{
    static int x = 0;
    using map = dense_id_map<int*>;
    map m;
    EXPECT_FALSE(m.insert(map::empty_key, &x));   // into an empty map: would have become the window's base
    EXPECT_TRUE(m.empty());
    ASSERT_TRUE(m.insert(1, &x));
    EXPECT_FALSE(m.insert(map::empty_key, &x));   // into the fallback
    EXPECT_EQ(m.size(), 1u);
    EXPECT_FALSE(m.contains(map::empty_key));
    EXPECT_FALSE(m.erase(map::empty_key));
    EXPECT_EQ(m.outliers(), 0u);
}

TEST(DenseIdMapWindow, SequentialIdsStayInTheWindowAndItSlides)
{
    dense_id_map<int*> m(16);
    for (std::uint64_t i = 1; i <= 100000; ++i) {
        ASSERT_TRUE(m.insert(i, ptr(i)));
        if (i > 64) {   // braces: ASSERT_* expands to an if/else (GCC -Wdangling-else)
            ASSERT_TRUE(m.erase(i - 64));   // a live set of 64 moving up, like a busy book
        }
    }
    EXPECT_EQ(m.size(), 64u);
    EXPECT_EQ(m.outliers(), 0u);
    EXPECT_LE(m.window(), 1024u);                    // the window slid up instead of growing to 100k
    EXPECT_EQ(*m.find(100000), ptr(100000));
}

TEST(DenseIdMapWindow, FarAheadIdsGoToTheFallbackAndMoveInWhenCovered)
{
    dense_id_map<int*> m(16, 1 << 12);
    ASSERT_TRUE(m.insert(1, ptr(1)));
    ASSERT_TRUE(m.insert(1'000'000, ptr(2)));       // beyond max_window: fallback
    EXPECT_EQ(m.outliers(), 1u);
    ASSERT_TRUE(m.insert(3000, ptr(3)));            // grows the window (within max_window)
    dense_id_map<int*> n(16, 1 << 20);
    ASSERT_TRUE(n.insert(10, ptr(10)));
    ASSERT_TRUE(n.insert(5, ptr(5)));               // below the window: fallback
    for (std::uint64_t i = 11; i < 5000; ++i) n.insert(i, ptr(i));
    ASSERT_TRUE(n.erase(10));
    for (std::uint64_t i = 11; i < 5000; ++i) n.erase(i);
    EXPECT_TRUE(n.contains(5));                     // still found wherever it lives
    EXPECT_EQ(*n.find(5), ptr(5));
    EXPECT_EQ(*m.find(1'000'000), ptr(2));
}

void model_check(unsigned seed, int mode)
{
    std::mt19937_64 rng(seed);
    dense_id_map<int*> m(8, 1 << 14);
    std::unordered_map<std::uint64_t, int*> model;
    std::uint64_t next = 1000;
    for (int step = 0; step < 200000; ++step) {
        std::uint64_t key;
        const auto op = rng() % 10;
        if (mode == 0) key = op < 5 ? next++ : next - 1 - rng() % 200;                  // sequential
        else if (mode == 1) key = rng() % 100000;                                        // random
        else key = rng() % 7 == 0 ? rng() % 2'000'000 : (op < 5 ? next++ : next - 1 - rng() % 300);   // mixed
        if (op < 5) {
            ASSERT_EQ(m.insert(key, ptr(key)), model.emplace(key, ptr(key)).second) << step;
        } else if (op < 8) {
            ASSERT_EQ(m.erase(key), model.erase(key) == 1) << step;
        } else {
            const auto it = model.find(key);
            int** got = m.find(key);
            ASSERT_EQ(got != nullptr, it != model.end()) << step;
            if (got) {
                ASSERT_EQ(*got, it->second);
            }
        }
        ASSERT_EQ(m.size(), model.size()) << step;
    }
    for (auto& [k, v] : model) ASSERT_EQ(*m.find(k), v);
}

TEST(DenseIdMapModel, SequentialIds) { model_check(1, 0); }
TEST(DenseIdMapModel, RandomIds) { model_check(2, 1); }
TEST(DenseIdMapModel, MixedIdsWithOutliers) { model_check(3, 2); }

} // namespace

namespace {
TEST(DenseIdMapWindow, ReusedIdBelowTheLowestLiveOneSurvivesGrowth)
{
    snicholls::containers::dense_id_map<int*> m(64);
    for (std::uint64_t i = 1; i <= 10; ++i) m.insert(i, ptr(i));
    m.erase(1);
    m.erase(2);
    ASSERT_TRUE(m.insert(1, ptr(1)));                       // below the lazily advanced front
    for (std::uint64_t i = 11; i <= 5000; ++i) m.insert(i, ptr(i));   // forces growth
    ASSERT_NE(m.find(1), nullptr);
    EXPECT_EQ(*m.find(1), ptr(1));
}
} // namespace
