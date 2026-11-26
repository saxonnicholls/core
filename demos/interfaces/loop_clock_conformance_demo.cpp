//
//  loop_clock_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/loop_clock_conformance.hpp>: the kit run on the reference, with and without the checks of a manual clock. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/loop_clock_conformance.hpp>
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
    std::printf("snicholls-core interfaces/loop_clock_conformance.hpp demo\n");
    //! [kit]
    manual_loop_clock clock;
    const auto any_clock = check_loop_clock(clock);                                   // every clock
    const auto manual = check_loop_clock_moves(clock, [&] { clock.advance(1ms); });   // one that moves on command
    //! [kit]
    check(any_clock.empty(), "the reference passes the clock checks");
    check(manual.empty(), "and the checks of a clock that moves on command");
    return failures == 0 ? 0 : 1;
}
