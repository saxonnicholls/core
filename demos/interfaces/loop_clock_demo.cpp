//
//  loop_clock_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/loop_clock.hpp>: a loop clock chosen at run time, its waker forwarded through the adapter, and the value wrapper. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/loop_clock.hpp>
#include <snicholls/core/interfaces/loop_clock_reference.hpp>

#include <chrono>
#include <cstdio>
#include <memory>

using namespace snicholls;
using namespace std::chrono_literals;

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
    std::printf("snicholls-core interfaces/loop_clock.hpp demo\n");
    //! [virtual]
    auto adapted = std::make_shared<loop_clock_adapter<manual_loop_clock>>();
    std::shared_ptr<loop_clock_interface> clock = adapted;   // what a loop holds
    int woken = 0;
    clock->set_waker([&woken] { ++woken; });                 // the loop installs its waker
    adapted->implementation().advance(250ms);                // the test moves time; the loop is woken
    //! [virtual]
    check(woken == 1, "the waker reached the model through the adapter");
    check(clock->now() == loop_time_point{250ms}, "now through the interface");
    any_loop_clock value = any_loop_clock::of(manual_loop_clock{loop_time_point{1s}});
    check(value.now() == loop_time_point{1s} && value.block_for(loop_time_point{2s}) == 0, "the value wrapper");
    return failures == 0 ? 0 : 1;
}
