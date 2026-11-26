//
//  loop_clock_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/loop_clock.hpp>: a timer check written once over any loop clock, driven by a manual one. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/loop_clock.hpp>
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

namespace {
// Generic code: how long the poller may sleep before a timer is due, in any clock's timeline.
template <SNICHOLLS_LOOP_CLOCK Clock>
long long poll_timeout(const Clock& clock, loop_time_point next_timer) noexcept
{
    return clock.now() < next_timer ? clock.block_for(next_timer) : 0;
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/loop_clock.hpp demo\n");
    //! [concept]
    manual_loop_clock clock;
    const loop_time_point timer = clock.now() + 1h;
    const long long before = poll_timeout(clock, timer);   // 0: a manual clock is never waited for
    clock.advance(1h);                                     // an hour, in no wall time
    const bool due = !(clock.now() < timer);
    //! [concept]
    check(before == 0, "a manual clock never blocks the poller");
    check(due, "the timer is due");
    check(is_loop_clock_like_v<manual_loop_clock> && !is_loop_clock_like_v<int>, "is_loop_clock_like_v");
    return failures == 0 ? 0 : 1;
}
