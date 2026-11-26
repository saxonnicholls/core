//
//  time_zone_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/time_zone_conformance.hpp>: the kit run on a toy zone over three years of transitions. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/time_zone_conformance.hpp>
#include <snicholls/core/interfaces/time_zone_reference.hpp>

#include <chrono>
#include <cstdio>

using namespace snicholls;
using namespace std::chrono_literals;   // not all of std::chrono: libstdc++'s std::chrono::time_zone and local_info
                                        // would collide with core's
using std::chrono::sys_seconds;
using std::chrono::sys_days;
using std::chrono::year;

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
struct probe {
    two_rule_zone z = new_york_like_zone();
    auto& zone() { return z; }
    static sys_seconds from() { return sys_days{year{2025} / 1 / 1}; }
    static sys_seconds to() { return sys_days{year{2028} / 1 / 1}; }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/time_zone_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_time_zone_conformance<probe>();
    //! [kit]
    check(report.ok(), "the toy zone passes");
    check(report.checks == 6, "six checks ran");
    return failures == 0 ? 0 : 1;
}
