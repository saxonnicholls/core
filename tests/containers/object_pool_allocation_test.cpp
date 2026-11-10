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
//  next_ dangling into the freed chunk.
//

#include <snicholls/core/containers/object_pool.hpp>

#include <gtest/gtest.h>

#include <new>

#include "../support/failing_new.hpp"

namespace {

TEST(ObjectPoolAllocationFailure, AFailedChunkListGrowthLeavesThePoolUnchanged)
{
    snicholls::containers::object_pool<long, 4> pool;
    {
        const snicholls_test::failing_new fail(2);   // 1: the chunk, 2: the chunk list's growth in push_back
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

TEST(ObjectPoolAllocationFailure, AFailedChunkAllocationLeavesThePoolUnchanged)
{
    snicholls::containers::object_pool<long, 4> pool;
    {
        const snicholls_test::failing_new fail(1);   // the chunk itself
        EXPECT_THROW((void)pool.create(1), std::bad_alloc);
    }
    EXPECT_EQ(pool.capacity(), 0u);
    long* p = pool.create(3);
    EXPECT_EQ(*p, 3);
    pool.destroy(p);
}

} // namespace
