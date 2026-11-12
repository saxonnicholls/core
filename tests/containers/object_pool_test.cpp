//
//  object_pool_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/object_pool.hpp>.
//

// snicholls::containers::object_pool: stable addresses, LIFO reuse, chunk growth, strong guarantee when T's constructor
// throws, release_all, reserve, moves, and chunks from a caller's std::pmr::memory_resource (counted, and failing).

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <new>
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

// Counts what passes through to new_delete_resource; with fail_after set, the allocation after that many throws.
class counting_resource final : public std::pmr::memory_resource
{
public:
    std::size_t allocations = 0;
    std::size_t deallocations = 0;
    std::size_t bytes_live = 0;
    long fail_after = -1;   // -1: never fail

private:
    void* do_allocate(std::size_t bytes, std::size_t align) override
    {
        if (fail_after >= 0 && static_cast<long>(allocations) >= fail_after) throw std::bad_alloc();
        ++allocations;
        bytes_live += bytes;
        return std::pmr::new_delete_resource()->allocate(bytes, align);
    }
    void do_deallocate(void* p, std::size_t bytes, std::size_t align) override
    {
        ++deallocations;
        bytes_live -= bytes;
        std::pmr::new_delete_resource()->deallocate(p, bytes, align);
    }
    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override { return this == &o; }
};

TEST(ObjectPoolResource, DefaultsToNewDeleteResource)
{
    object_pool<item, 4> pool;
    EXPECT_EQ(pool.resource(), std::pmr::new_delete_resource());
    object_pool<item, 4> null_given(nullptr);
    EXPECT_EQ(null_given.resource(), std::pmr::new_delete_resource());
}

TEST(ObjectPoolResource, ChunksComeFromTheResourceAndAllGoBack)
{
    counting_resource r;
    {
        object_pool<item, 16> pool(&r);
        EXPECT_EQ(pool.resource(), &r);
        EXPECT_EQ(r.allocations, 0u);   // nothing until the first create
        std::vector<item*> ptrs;
        for (int i = 0; i < 100; ++i) ptrs.push_back(pool.create(i, i));
        EXPECT_EQ(r.allocations, 7u);   // ceil(100 / 16)
        EXPECT_EQ(r.bytes_live, pool.capacity() * sizeof(item));
        for (item* p : ptrs) pool.destroy(p);
        EXPECT_EQ(r.deallocations, 0u);   // destroy recycles; nothing goes back early
        pool.release_all();
        EXPECT_EQ(r.deallocations, 0u);   // release_all keeps the chunks too
    }
    EXPECT_EQ(r.deallocations, 7u);
    EXPECT_EQ(r.bytes_live, 0u);
}

TEST(ObjectPoolResource, ReserveTakesWholeChunksFromTheResource)
{
    counting_resource r;
    object_pool<item, 10> pool(&r);
    pool.reserve(25);
    EXPECT_EQ(r.allocations, 3u);
    EXPECT_EQ(r.bytes_live, 30 * sizeof(item));
}

TEST(ObjectPoolResource, ChunksLiveInsideAMonotonicBuffer)
{
    alignas(std::max_align_t) std::byte buffer[4096];
    std::pmr::monotonic_buffer_resource arena(buffer, sizeof buffer, std::pmr::null_memory_resource());
    object_pool<item, 8> pool(&arena);
    std::vector<item*> ptrs;
    for (int i = 0; i < 64; ++i) ptrs.push_back(pool.create(i, 2 * i));
    for (int i = 0; i < 64; ++i) {
        EXPECT_GE(reinterpret_cast<std::byte*>(ptrs[i]), buffer);
        EXPECT_LT(reinterpret_cast<std::byte*>(ptrs[i]), buffer + sizeof buffer);
        EXPECT_EQ(ptrs[i]->b, 2 * i);
    }
}

TEST(ObjectPoolResource, AFailingResourceLeavesThePoolUnchangedAndUsable)
{
    counting_resource r;
    object_pool<item, 4> pool(&r);
    item* first[4];
    for (auto& p : first) p = pool.create(1, 1);   // fills chunk 1 exactly
    r.fail_after = 1;                               // chunk 2 will fail
    EXPECT_THROW((void)pool.create(2, 2), std::bad_alloc);
    EXPECT_EQ(pool.live(), 4u);
    EXPECT_EQ(pool.capacity(), 4u);
    for (item* p : first) EXPECT_EQ(p->a, 1);   // the live objects are untouched
    EXPECT_THROW(pool.reserve(8), std::bad_alloc);
    EXPECT_EQ(pool.capacity(), 4u);
    pool.destroy(first[2]);
    EXPECT_EQ(pool.create(3, 3), first[2]);   // the free list still works while the resource fails
    r.fail_after = -1;
    item* more = pool.create(4, 4);           // and growth resumes once it recovers
    EXPECT_EQ(more->a, 4);
    EXPECT_EQ(pool.capacity(), 8u);
    EXPECT_EQ(r.allocations, 2u);
}

TEST(ObjectPoolResource, AFailingResourceOnTheFirstChunkLeavesAnEmptyPool)
{
    counting_resource r;
    r.fail_after = 0;
    object_pool<item, 4> pool(&r);
    EXPECT_THROW((void)pool.create(), std::bad_alloc);
    EXPECT_EQ(pool.capacity(), 0u);
    EXPECT_EQ(pool.live(), 0u);
    r.fail_after = -1;
    EXPECT_EQ(pool.create(5, 6)->b, 6);
}

TEST(ObjectPoolResource, MovingBetweenResourcesGivesEachChunkBackToItsOwn)
{
    counting_resource ra;
    counting_resource rb;
    {
        object_pool<item, 4> a(&ra);
        object_pool<item, 4> b(&rb);
        item* pa = a.create(1, 2);
        (void)b.create(3, 4);
        b = std::move(a);   // b's chunk goes back to rb now; a's chunk is b's from here on
        EXPECT_EQ(rb.deallocations, 1u);
        EXPECT_EQ(b.resource(), &ra);
        EXPECT_EQ(pa->b, 2);
        object_pool<item, 4> c(std::move(b));
        EXPECT_EQ(c.resource(), &ra);
        EXPECT_EQ(c.live(), 1u);
        c.destroy(pa);
        auto& same = c;
        c = std::move(same);   // self-move is harmless
        EXPECT_EQ(c.capacity(), 4u);
    }
    EXPECT_EQ(ra.bytes_live, 0u);
    EXPECT_EQ(rb.bytes_live, 0u);
    EXPECT_EQ(ra.deallocations, 1u);
}

} // namespace
