//
//  loop_clock_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/loop_clock_reference.hpp>: a fixed clock that moves on command, never backwards, waking the loop. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/loop_clock_reference.hpp>

#include <chrono>
#include <cstdio>

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
    std::printf("snicholls-core interfaces/loop_clock_reference.hpp demo\n");
    //! [reference]
    manual_loop_clock clock{loop_time_point{10s}};
    int woken = 0;
    clock.set_waker([&woken] { ++woken; });
    clock.advance(5s);
    clock.advance(-1s);                       // ignored: time does not go back
    clock.advance_to(loop_time_point{1s});    // earlier: stays, but still wakes
    //! [reference]
    check(clock.now() == loop_time_point{15s}, "advanced to 15 s and no further back");
    check(woken == 2, "advance and advance_to woke the loop");
    check(clock.block_for(loop_time_point{1h}) == 0, "never asks the poller to block");
    return failures == 0 ? 0 : 1;
}
