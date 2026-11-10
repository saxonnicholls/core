//
//  dense_id_map_allocation_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Allocation-failure tests for <snicholls/core/containers/dense_id_map.hpp>,
//  in snicholls_core_allocation_tests (tests/support/failing_new.cpp replaces
//  operator new there). A regression test for the pre-0.1.0 review: grow()
//  installed the new window and base before allocating the list of fallback
//  keys the window would cover, so a bad_alloc there broke the documented
//  strong guarantee (a fallback key inside the window became unreachable).
//

#include <snicholls/core/containers/dense_id_map.hpp>

#include <gtest/gtest.h>

#include <new>

#include "../support/failing_new.hpp"

namespace {

TEST(DenseIdMapAllocationFailure, AFailedGrowthLeavesTheMapUnchanged)
{
    static int x = 0;
    snicholls::containers::dense_id_map<int*> m(64, 1000);
    ASSERT_TRUE(m.insert(0, &x));
    ASSERT_TRUE(m.insert(1010, &x));   // beyond max_window: the hash fallback
    {
        // Inserting 600 grows the window to 1024 slots, which then covers 1010.
        // 1: the new window, 2: the list of covered fallback keys.
        const snicholls_test::failing_new fail(2);
        EXPECT_THROW((void)m.insert(600, &x), std::bad_alloc);
    }
    EXPECT_EQ(m.size(), 2u);
    EXPECT_TRUE(m.contains(0));
    EXPECT_TRUE(m.contains(1010));   // was false: the key stayed in the fallback, inside the new window
    EXPECT_FALSE(m.contains(600));
    // and the map still works
    EXPECT_TRUE(m.insert(600, &x));
    EXPECT_TRUE(m.contains(600) && m.contains(1010));
    EXPECT_TRUE(m.erase(1010));
    EXPECT_EQ(m.size(), 2u);
}

TEST(DenseIdMapAllocationFailure, AFailedWindowAllocationLeavesTheMapUnchanged)
{
    static int x = 0;
    snicholls::containers::dense_id_map<int*> m(64, 1000);
    ASSERT_TRUE(m.insert(0, &x));
    {
        const snicholls_test::failing_new fail(1);   // the new window
        EXPECT_THROW((void)m.insert(600, &x), std::bad_alloc);
    }
    EXPECT_EQ(m.size(), 1u);
    EXPECT_TRUE(m.contains(0));
    EXPECT_TRUE(m.insert(600, &x));
}

} // namespace
