//
//  utc_clock_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/utc_clock.hpp>: a component that owns its clock, a manual clock moved by a test, and
//  a clock of the caller's own. Exits non-zero if a value is not the one documented.
//

#include <snicholls/core/time/utc_clock.hpp>

#include <chrono>
#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [utc_clock]
// A component that stamps what it records takes its clock as a type and owns it: still copyable, no allocation.
template <snicholls::utc_clock Clock = snicholls::manual_utc_clock>
class journal {
public:
    explicit journal(Clock clock) : clock_(clock) {}
    snicholls::utc_ms record() const { return clock_.now(); }
    Clock& clock() noexcept { return clock_; }   // a test moves a manual clock through this

private:
    Clock clock_;
};

// A clock of the caller's own: anything whose now() is a utc of any precision.
struct market_open {
    [[nodiscard]] constexpr snicholls::utc_s now() const noexcept { return snicholls::utc_s::from_unix_seconds(1'800'000'000); }
};
static_assert(snicholls::utc_clock<market_open>);
//! [utc_clock]
} // namespace

int main()
{
    using namespace std::chrono_literals;
    std::printf("snicholls-core time/utc_clock demo\n");
    journal j{snicholls::manual_utc_clock{snicholls::utc_ms::from_unix_seconds(1'800'000'000)}};
    const auto first = j.record();
    j.clock().advance(90s);
    j.clock().advance(-1h);   // ignored: a manual clock never runs backwards
    const auto copy = j;      // copyable: the clock is a value
    std::printf("  first %lld ms, then %lld ms\n", static_cast<long long>(first.unix_millis()),
                static_cast<long long>(j.record().unix_millis()));
    check(j.record() - first == 90s && copy.record() == j.record(), "time moves only when told, and copies with the owner");
    check(journal<market_open>{{}}.record().unix_seconds() == 1'800'000'000, "a clock of the caller's own");
    return failures == 0 ? 0 : 1;
}
