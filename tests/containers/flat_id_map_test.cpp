//
//  flat_id_map_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/flat_id_map.hpp>.
//

// snicholls::containers::flat_id_map: model-checked against std::unordered_map (random and sequential ids, heavy erase so
// backward-shift deletion is exercised), duplicate inserts, growth, reserve, clear, for_each.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <random>
#include <unordered_map>

#include <gtest/gtest.h>

#include <snicholls/core/containers/flat_id_map.hpp>

namespace {

using snicholls::containers::flat_id_map;

TEST(FlatIdMapBasics, InsertFindEraseAndDuplicates)
{
    flat_id_map<int> m;
    EXPECT_TRUE(m.empty());
    EXPECT_TRUE(m.insert(5, 50));
    EXPECT_FALSE(m.insert(5, 99));   // unchanged
    ASSERT_NE(m.find(5), nullptr);
    EXPECT_EQ(*m.find(5), 50);
    EXPECT_TRUE(m.contains(5));
    EXPECT_EQ(m.find(6), nullptr);
    EXPECT_TRUE(m.erase(5));
    EXPECT_FALSE(m.erase(5));
    EXPECT_EQ(m.size(), 0u);
    const flat_id_map<int>& cm = m;
    EXPECT_EQ(cm.find(5), nullptr);
}

// Regression: find(empty_key) matched the first empty slot and returned garbage, and erase(empty_key) corrupted the
// size - so a book crashed on a cancel for order id UINT64_MAX.
TEST(FlatIdMapBasics, TheReservedKeyIsNeverFound)
{
    using map = snicholls::containers::flat_id_map<int>;
    map m;
    EXPECT_EQ(m.find(map::empty_key), nullptr);
    EXPECT_FALSE(m.contains(map::empty_key));
    EXPECT_FALSE(m.erase(map::empty_key));
    m.insert(1, 10);
    m.insert(2, 20);
    EXPECT_EQ(m.find(map::empty_key), nullptr);
    EXPECT_FALSE(m.erase(map::empty_key));
    EXPECT_EQ(m.size(), 2u);
    const map& cm = m;
    EXPECT_EQ(cm.find(map::empty_key), nullptr);
    EXPECT_EQ(*m.find(2), 20);
}

// Regression (pre-0.1.0 review): insert(empty_key) counted an entry it could not store, so size() grew and nothing
// could ever find or erase it.
TEST(FlatIdMapBasics, InsertingTheReservedKeyIsRefusedAndChangesNothing)
{
    using map = flat_id_map<int>;
    map m;
    EXPECT_FALSE(m.insert(map::empty_key, 7));
    EXPECT_EQ(m.size(), 0u);
    EXPECT_TRUE(m.empty());
    EXPECT_FALSE(m.contains(map::empty_key));
    m.insert(1, 10);
    EXPECT_FALSE(m.insert(map::empty_key, 8));
    EXPECT_EQ(m.size(), 1u);
    int visited = 0;
    m.for_each([&](std::uint64_t, int) { ++visited; });
    EXPECT_EQ(visited, 1);
}

// Regression (pre-0.1.0 review): capacity_for(n) looped forever for n > max/4 (2 * n overflowed).
TEST(FlatIdMapReserve, AnImpossibleSizeThrowsLengthErrorInsteadOfLooping)
{
    flat_id_map<int> m;
    EXPECT_THROW(m.reserve(std::numeric_limits<std::size_t>::max() / 2), std::length_error);
    const std::size_t huge = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW({ const flat_id_map<int> big(huge); }, std::length_error);
    EXPECT_TRUE(m.insert(1, 1));   // unchanged and usable
}

TEST(FlatIdMapGrowth, LoadFactorStaysAtMostOneHalf)
{
    flat_id_map<std::uint64_t> m(1);
    for (std::uint64_t i = 0; i < 10000; ++i) {
        ASSERT_TRUE(m.insert(i, i * 3));
        ASSERT_LE(m.size() * 2, m.capacity());
    }
    for (std::uint64_t i = 0; i < 10000; ++i) ASSERT_EQ(*m.find(i), i * 3);
}

TEST(FlatIdMapReserve, ReserveAvoidsGrowth)
{
    flat_id_map<int> m;
    m.reserve(1000);
    const auto cap = m.capacity();
    for (int i = 0; i < 1000; ++i) m.insert(static_cast<std::uint64_t>(i), i);
    EXPECT_EQ(m.capacity(), cap);
    m.clear();
    EXPECT_EQ(m.size(), 0u);
    EXPECT_EQ(m.capacity(), cap);
    EXPECT_EQ(m.find(3), nullptr);
}

TEST(FlatIdMapForEach, VisitsEveryEntryOnce)
{
    flat_id_map<int> m;
    for (int i = 1; i <= 100; ++i) m.insert(static_cast<std::uint64_t>(i * 7), i);
    long sum = 0;
    int n = 0;
    m.for_each([&](std::uint64_t k, int v) {
        EXPECT_EQ(k, static_cast<std::uint64_t>(v * 7));
        sum += v;
        ++n;
    });
    EXPECT_EQ(n, 100);
    EXPECT_EQ(sum, 5050);
}

void model_check(bool sequential, unsigned seed)
{
    std::mt19937_64 rng(seed);
    flat_id_map<std::uint64_t> m(4);
    std::unordered_map<std::uint64_t, std::uint64_t> model;
    std::uint64_t next = 1000;
    for (int step = 0; step < 200000; ++step) {
        const auto op = rng() % 10;
        std::uint64_t key = sequential ? (op < 5 ? next++ : next - 1 - rng() % 64) : rng() % 5000;
        if (op < 5) {
            const std::uint64_t v = rng();
            ASSERT_EQ(m.insert(key, v), model.emplace(key, v).second);
        } else if (op < 8) {
            ASSERT_EQ(m.erase(key), model.erase(key) == 1);
        } else {
            auto it = model.find(key);
            const std::uint64_t* got = m.find(key);
            ASSERT_EQ(got != nullptr, it != model.end());
            if (got) {   // braces: ASSERT_* expands to an if/else (GCC -Wdangling-else)
                ASSERT_EQ(*got, it->second);
            }
        }
        ASSERT_EQ(m.size(), model.size());
    }
    for (auto& [k, v] : model) ASSERT_EQ(*m.find(k), v);
}

TEST(FlatIdMapModel, RandomKeysMatchUnorderedMap) { model_check(false, 7); }
TEST(FlatIdMapModel, SequentialExchangeStyleIdsMatchUnorderedMap) { model_check(true, 9); }

} // namespace
