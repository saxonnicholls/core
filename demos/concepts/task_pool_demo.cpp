//
//  task_pool_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/task_pool.hpp>: an algorithm written once over any pool, constrained by the concept under C++20 and by the trait under C++17. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/task_pool.hpp>
#include <snicholls/core/interfaces/task_pool_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <vector>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}
} // namespace

namespace {
// Generic code: a parallel for over any pool (here, run deterministically by the reference).
template <SNICHOLLS_TASK_POOL Pool>
void parallel_for(Pool& pool, std::size_t n, std::vector<int>& out)
{
    static_assert(is_task_pool_like_v<Pool>);
    for (std::size_t i = 0; i < n; ++i)
        pool.submit([&out, i] { out[i] = static_cast<int>(i * i); });
    pool.wait_idle();
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/task_pool.hpp demo\n");
    //! [concept]
    deferred_task_pool pool;
    std::vector<int> squares(5);
    parallel_for(pool, squares.size(), squares);
    //! [concept]
    check(squares == std::vector<int>{0, 1, 4, 9, 16}, "every index ran");
    check(is_task_pool_like_v<deferred_task_pool> && !is_task_pool_like_v<int>, "is_task_pool_like_v");
    check(is_pool_task_v<void (*)()> && !is_pool_task_v<int>, "is_pool_task_v");
    return failures == 0 ? 0 : 1;
}
