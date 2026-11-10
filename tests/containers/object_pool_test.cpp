//
//  object_pool_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/object_pool.hpp>.
//

// snicholls::containers::object_pool: stable addresses, LIFO reuse, chunk growth, strong guarantee when T's constructor
// throws, release_all, reserve, moves.

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/containers/object_pool.hpp>

namespace {

using snicholls::containers::object_pool;

struct item
{
    std::int64_t a = 0;
    std::int64_t b = 0;
    item() = default;
    item(std::int64_t x, std::int64_t y) noexcept : a(x), b(y) {}
};

struct throwing
{
    explicit throwing(bool fail)
    {
        if (fail) throw std::runtime_error("boom");
    }
};

TEST(ObjectPoolCreate, ConstructsWithArgumentsAndCountsLiveObjects)
{
    object_pool<item, 4> pool;
    EXPECT_EQ(pool.capacity(), 0u);
    item* p = pool.create(3, 4);
    EXPECT_EQ(p->a, 3);
    EXPECT_EQ(p->b, 4);
    EXPECT_EQ(pool.live(), 1u);
    EXPECT_EQ(pool.capacity(), 4u);
    pool.destroy(p);
    EXPECT_EQ(pool.live(), 0u);
}

TEST(ObjectPoolReuse, ReleasedSlotsAreReusedLastInFirstOut)
{
    object_pool<item, 8> pool;
    item* a = pool.create();
    item* b = pool.create();
    pool.destroy(a);
    pool.destroy(b);
    EXPECT_EQ(pool.create(), b);
    EXPECT_EQ(pool.create(), a);
}

TEST(ObjectPoolGrowth, AddressesStayStableAcrossManyChunks)
{
    object_pool<item, 16> pool;
    std::vector<item*> ptrs;
    for (int i = 0; i < 1000; ++i) ptrs.push_back(pool.create(i, -i));
    EXPECT_EQ(pool.capacity(), 1008u);   // 63 chunks of 16
    std::set<item*> unique(ptrs.begin(), ptrs.end());
    EXPECT_EQ(unique.size(), 1000u);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(ptrs[i]->a, i);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptrs[i]) % alignof(item), 0u);
    }
}

TEST(ObjectPoolExceptions, AThrowingConstructorLeavesThePoolUnchanged)
{
    object_pool<throwing, 4> pool;
    EXPECT_THROW((void)pool.create(true), std::runtime_error);
    EXPECT_EQ(pool.live(), 0u);
    throwing* ok = pool.create(false);
    EXPECT_EQ(pool.live(), 1u);
    pool.destroy(ok);
}

TEST(ObjectPoolRelease, ReleaseAllKeepsChunksAndForgetsObjects)
{
    object_pool<item, 4> pool;
    for (int i = 0; i < 10; ++i) (void)pool.create();
    const auto cap = pool.capacity();
    pool.release_all();
    EXPECT_EQ(pool.live(), 0u);
    std::set<item*> seen;
    for (int i = 0; i < 12; ++i) seen.insert(pool.create());
    EXPECT_EQ(seen.size(), 12u);
    EXPECT_EQ(pool.capacity(), cap);   // 12 slots fit in the 3 kept chunks
}

TEST(ObjectPoolReserve, ReserveAllocatesUpFront)
{
    object_pool<item, 10> pool;
    pool.reserve(25);
    EXPECT_EQ(pool.capacity(), 30u);
    for (int i = 0; i < 30; ++i) (void)pool.create();
    EXPECT_EQ(pool.capacity(), 30u);
}

TEST(ObjectPoolMove, MovedPoolKeepsObjectsValid)
{
    object_pool<std::string, 4> pool;
    std::string* s = pool.create("hello");
    object_pool<std::string, 4> other(std::move(pool));
    EXPECT_EQ(*s, "hello");
    EXPECT_EQ(other.live(), 1u);
    object_pool<std::string, 4> third;
    third = std::move(other);
    third.destroy(s);
    EXPECT_EQ(third.live(), 0u);
}

} // namespace
