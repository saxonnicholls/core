//
//  task_pool_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/task_pool_reference.hpp>: deferred, in order, stepwise, nested submissions and a
//  throwing task.
//

#include <snicholls/core/interfaces/task_pool_reference.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace snicholls;

namespace {

TEST(DeferredTaskPool, RunsNothingUntilAsked)
{
    deferred_task_pool pool;
    int ran = 0;
    pool.submit([&ran] { ++ran; });
    EXPECT_EQ(ran, 0);
    EXPECT_EQ(pool.pending(), 1u);
    pool.wait_idle();
    EXPECT_EQ(ran, 1);
    EXPECT_EQ(pool.pending(), 0u);
    EXPECT_EQ(pool.worker_count(), 1u);
}

TEST(DeferredTaskPool, RunsInSubmissionOrderAndStepwise)
{
    deferred_task_pool pool;
    std::vector<int> order;
    for (int i = 0; i < 3; ++i)
        pool.submit([&order, i] { order.push_back(i); });
    EXPECT_TRUE(pool.run_one());
    EXPECT_EQ(order, std::vector<int>{0});
    pool.wait_idle();
    EXPECT_EQ(order, (std::vector<int>{0, 1, 2}));
    EXPECT_FALSE(pool.run_one());
    EXPECT_EQ(pool.ran(), 3u);
}

TEST(DeferredTaskPool, TasksSubmittedByTasksRunInTheSameWait)
{
    deferred_task_pool pool;
    std::vector<int> order;
    pool.submit([&] {
        order.push_back(1);
        pool.submit([&] { order.push_back(3); });
    });
    pool.submit([&] { order.push_back(2); });
    pool.wait_idle();
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
}

TEST(DeferredTaskPool, AnEmptyTaskIsIgnored)
{
    deferred_task_pool pool;
    pool.submit(pool_task_function{});
    EXPECT_EQ(pool.pending(), 0u);
}

TEST(DeferredTaskPool, AThrowingTaskLeavesTheRestQueued)
{
    deferred_task_pool pool;
    int ran = 0;
    pool.submit([] { throw std::runtime_error("task failed"); });
    pool.submit([&ran] { ++ran; });
    EXPECT_THROW(pool.wait_idle(), std::runtime_error);
    EXPECT_EQ(pool.pending(), 1u);
    pool.wait_idle();
    EXPECT_EQ(ran, 1);
}

} // namespace
