//
//  task_pool_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/task_pool_conformance.hpp>: the kit passes the reference and catches broken
//  pools: one that drops tasks, one that runs tasks twice, one that forgets nested submissions, one with no workers.
//

#include <snicholls/core/interfaces/task_pool_conformance.hpp>
#include <snicholls/core/interfaces/task_pool_reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace snicholls;

namespace {

enum class defect { drops_every_tenth, runs_twice, forgets_nested, no_workers };

template <defect D>
struct broken_pool {
    deferred_task_pool inner;
    int submitted = 0;
    bool running = false;
    void submit(pool_task_function t)
    {
        ++submitted;
        if (D == defect::drops_every_tenth && submitted % 10 == 0)
            return;
        if (D == defect::forgets_nested && running)
            return;
        if (D == defect::runs_twice)
            inner.submit(t);
        inner.submit(std::move(t));
    }
    void wait_idle()
    {
        running = true;
        inner.wait_idle();
        running = false;
    }
    std::size_t worker_count() const noexcept { return D == defect::no_workers ? 0 : 1; }
};

template <defect D>
bool fails_with(const std::string& what)
{
    broken_pool<D> p;
    const auto f = check_task_pool(p);
    return std::find(f.begin(), f.end(), what) != f.end();
}

TEST(TaskPoolConformance, TheReferencePasses)
{
    deferred_task_pool p;
    EXPECT_TRUE(check_task_pool(p).empty());
}

TEST(TaskPoolConformance, BrokenPoolsAreCaught)
{
    EXPECT_TRUE(fails_with<defect::drops_every_tenth>("every task runs exactly once (200 tasks, then wait_idle)"));
    EXPECT_TRUE(fails_with<defect::runs_twice>("every task runs exactly once (200 tasks, then wait_idle)"));
    EXPECT_TRUE(fails_with<defect::forgets_nested>("tasks submitted by tasks run before wait_idle returns"));
    EXPECT_TRUE(fails_with<defect::no_workers>("worker_count is at least 1"));
    EXPECT_FALSE(fails_with<defect::forgets_nested>("every task runs exactly once (200 tasks, then wait_idle)"));
}

} // namespace
