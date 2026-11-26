//
//  task_pool_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/task_pool_reference.hpp>: tasks queued, stepped through one at a time, then drained in order. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/task_pool_reference.hpp>

#include <cstdio>
#include <string>

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

int main()
{
    std::printf("snicholls-core interfaces/task_pool_reference.hpp demo\n");
    //! [reference]
    deferred_task_pool pool;
    std::string log;
    pool.submit([&] { log += 'a'; pool.submit([&] { log += 'c'; }); });
    pool.submit([&] { log += 'b'; });
    const bool stepped = pool.run_one();   // runs "a" only
    const std::string after_one = log;
    pool.wait_idle();                      // then "b", then "c" that "a" submitted
    //! [reference]
    check(stepped && after_one == "a", "run_one runs one task");
    check(log == "abc", "wait_idle drains in submission order");
    check(pool.pending() == 0 && pool.ran() == 3 && pool.worker_count() == 1, "pending, ran, worker_count");
    return failures == 0 ? 0 : 1;
}
