//
//  object_pool_allocation_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Allocation-failure tests for <snicholls/core/containers/object_pool.hpp>,
//  in snicholls_core_allocation_tests (tests/support/failing_new.cpp replaces
//  operator new there). A regression test for the pre-0.1.0 review: acquire()
//  pointed next_ at a new chunk before storing it, so a failed push_back left
//  next_ dangling into the freed chunk. Since chunks come from a memory
//  resource, the chunk list grows first and the chunk is allocated second.
//

#include <snicholls/core/containers/object_pool.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory_resource>
#include <new>

#include "../support/failing_new.hpp"

namespace {

TEST(ObjectPoolAllocationFailure, AFailedChunkListGrowthLeavesThePoolUnchanged)
{
    snicholls::containers::object_pool<long, 4> pool;
    {
        const snicholls_test::failing_new fail(1);   // 1: the chunk list's growth, before any chunk exists
        EXPECT_THROW((void)pool.create(1), std::bad_alloc);
    }
    EXPECT_EQ(pool.capacity(), 0u);
    EXPECT_EQ(pool.live(), 0u);
    // Before the fix this returned a slot of the freed chunk (ASan: heap-use-after-free) without allocating one.
    long* p = pool.create(2);
    EXPECT_EQ(*p, 2);
    EXPECT_EQ(pool.capacity(), 4u);
    pool.destroy(p);
}

// libstdc++'s new_delete_resource takes the aligned operator new, which failing_new does not replace (libc++ goes
// through the plain one), so the chunk comes from a resource that calls the plain operator new on every library.
class plain_new_resource final : public std::pmr::memory_resource
{
    void* do_allocate(std::size_t bytes, std::size_t) override { return ::operator new(bytes); }
    void do_deallocate(void* p, std::size_t, std::size_t) override { ::operator delete(p); }
    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override { return this == &o; }
};

TEST(ObjectPoolAllocationFailure, AFailedChunkAllocationLeavesThePoolUnchanged)
{
    plain_new_resource plain;
    snicholls::containers::object_pool<long, 4> pool(&plain);
    {
        const snicholls_test::failing_new fail(2);   // 1: the chunk list's growth, 2: the chunk itself
        EXPECT_THROW((void)pool.create(1), std::bad_alloc);
    }
    EXPECT_EQ(pool.capacity(), 0u);
    long* p = pool.create(3);
    EXPECT_EQ(*p, 3);
    pool.destroy(p);
}

} // namespace
