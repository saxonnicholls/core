//
//  task_pool_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/task_pool.hpp>: the kit passes on the reference in every form (the model, the
//  interface over an owning and a borrowing adapter, the value wrapper), and the adapters forward every member.
//

#include <snicholls/core/interfaces/task_pool.hpp>
#include <snicholls/core/interfaces/task_pool_conformance.hpp>
#include <snicholls/core/interfaces/task_pool_reference.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(std::is_abstract_v<task_pool_interface>);

std::string describe(const std::vector<std::string>& failures)
{
    std::string out;
    for (const auto& f : failures)
        out += "\n  " + f;
    return out;
}

TEST(TaskPoolInterface, TheKitPassesInEveryForm)
{
    deferred_task_pool model;
    EXPECT_TRUE(check_task_pool(model).empty()) << describe(check_task_pool(model));
    task_pool_adapter<deferred_task_pool> owned;
    task_pool_interface& owned_virtual = owned;
    EXPECT_TRUE(check_task_pool(owned_virtual).empty());
    deferred_task_pool viewed;
    task_pool_adapter<deferred_task_pool&> view{viewed};
    task_pool_interface& view_virtual = view;
    EXPECT_TRUE(check_task_pool(view_virtual).empty());
    EXPECT_GT(viewed.ran(), 200u);   // the view ran the kit's tasks on the model
    any_task_pool value = any_task_pool::of(deferred_task_pool{});
    EXPECT_TRUE(check_task_pool(value).empty());
}

// Counts every call, to see the adapters forward.
struct counting_pool {
    mutable int calls = 0;
    void submit(pool_task_function t) { ++calls; t(); }
    void wait_idle() { ++calls; }
    std::size_t worker_count() const noexcept { ++calls; return 3; }
};

TEST(TaskPoolInterface, TheAdaptersForwardEveryMember)
{
    counting_pool impl;
    task_pool_adapter<counting_pool&> view{impl};
    task_pool_interface& p = view;
    int ran = 0;
    p.submit([&ran] { ++ran; });
    p.wait_idle();
    EXPECT_EQ(p.worker_count(), 3u);
    EXPECT_EQ(impl.calls, 3);
    EXPECT_EQ(ran, 1);
    EXPECT_EQ(&view.implementation(), &impl);
    any_task_pool value{std::make_unique<task_pool_adapter<counting_pool>>()};
    value.submit([] {});
    EXPECT_EQ(value.worker_count(), 3u);
    EXPECT_EQ(static_cast<task_pool_adapter<counting_pool>&>(value.get()).implementation().calls, 2);
}

} // namespace
