//
//  task_pool_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/task_pool.hpp>: a pool chosen at run time through the interface, the adapter (owning and viewing) and the value wrapper. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/task_pool.hpp>
#include <snicholls/core/interfaces/task_pool_reference.hpp>

#include <cstdio>
#include <memory>

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
    std::printf("snicholls-core interfaces/task_pool.hpp demo\n");
    //! [virtual]
    std::unique_ptr<task_pool_interface> chosen = std::make_unique<task_pool_adapter<deferred_task_pool>>();
    int ran = 0;
    chosen->submit([&ran] { ++ran; });
    chosen->wait_idle();
    any_task_pool value{std::move(chosen)};   // a value that models the concept
    value.submit([&ran] { ++ran; });
    value.wait_idle();
    //! [virtual]
    check(ran == 2, "both tasks ran");
    check(value.worker_count() == 1, "worker_count through the value");
    deferred_task_pool model;
    task_pool_adapter<deferred_task_pool&> view{model};
    view.submit([&ran] { ++ran; });
    check(model.pending() == 1 && &view.implementation() == &model, "the view submits to the model");
    view.wait_idle();
    check(ran == 3, "and runs it");
    return failures == 0 ? 0 : 1;
}
