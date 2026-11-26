//
//  task_pool_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/task_pool.hpp>: the trait and the concept agree on the models and the non-models,
//  and pool_task / is_pool_task_v, at compile time.
//

#include <snicholls/core/concepts/task_pool.hpp>
#include <snicholls/core/interfaces/task_pool_reference.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

using namespace snicholls;

namespace {

static_assert(SNICHOLLS_HAS_CONCEPTS == 1, "the tests build as C++20");

template <typename P>
constexpr bool both = is_task_pool_like_v<P> && task_pool_like<P>;
template <typename P>
constexpr bool neither = !is_task_pool_like_v<P> && !task_pool_like<P>;

static_assert(both<deferred_task_pool>);
static_assert(both<task_pool_interface>);
static_assert(both<task_pool_adapter<deferred_task_pool>>);
static_assert(both<task_pool_adapter<deferred_task_pool&>>);
static_assert(both<any_task_pool>);
static_assert(is_task_pool_like_v<const deferred_task_pool>);   // cv is looked through

struct worker_count_may_throw {
    void submit(pool_task_function);
    void wait_idle();
    std::size_t worker_count() const;
};
struct worker_count_int {
    void submit(pool_task_function);
    void wait_idle();
    int worker_count() const noexcept;
};
struct submit_returns_bool {
    bool submit(pool_task_function);
    void wait_idle();
    std::size_t worker_count() const noexcept;
};
struct no_wait {
    void submit(pool_task_function);
    std::size_t worker_count() const noexcept;
};
static_assert(neither<worker_count_may_throw>);
static_assert(neither<worker_count_int>);
static_assert(neither<submit_returns_bool>);
static_assert(neither<no_wait>);
static_assert(neither<int>);

static_assert(is_pool_task_v<void (*)()> && pool_task<void (*)()>);
static_assert(is_pool_task_v<int (*)()>);   // a result is discarded
static_assert(!is_pool_task_v<void (*)(int)> && !pool_task<void (*)(int)>);
static_assert(!is_pool_task_v<int>);
auto move_only = [p = std::unique_ptr<int>()] { static_cast<void>(p); };
static_assert(is_pool_task_v<decltype(move_only)> && pool_task<decltype(move_only)>);

// A pool written once with the macro: concept-constrained here, typename under C++17.
template <SNICHOLLS_TASK_POOL Pool>
std::size_t workers(const Pool& p) noexcept
{
    return p.worker_count();
}

TEST(TaskPoolConcept, TheMacroConstrainsATemplate)
{
    const deferred_task_pool pool;
    EXPECT_EQ(workers(pool), 1u);
}

} // namespace
