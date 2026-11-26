//
//  task_pool_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/task_pool_conformance.hpp>: the kit run on the reference and on a pool that loses tasks. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/task_pool_conformance.hpp>
#include <snicholls/core/interfaces/task_pool_reference.hpp>

#include <cstddef>
#include <cstdio>

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
// Loses every seventh task.
struct leaky_pool {
    deferred_task_pool inner;
    int n = 0;
    void submit(pool_task_function t)
    {
        if (++n % 7 != 0)
            inner.submit(std::move(t));
    }
    void wait_idle() { inner.wait_idle(); }
    std::size_t worker_count() const noexcept { return 1; }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/task_pool_conformance.hpp demo\n");
    //! [kit]
    deferred_task_pool good;
    leaky_pool bad;
    const auto good_failures = check_task_pool(good);   // std::vector<std::string>: empty is a pass
    const auto bad_failures = check_task_pool(bad);
    //! [kit]
    check(good_failures.empty(), "the reference passes");
    check(!bad_failures.empty(), "a pool that loses tasks is caught");
    for (const auto& f : bad_failures)
        std::printf("    %s\n", f.c_str());
    return failures == 0 ? 0 : 1;
}
