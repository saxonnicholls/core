//
//  time_zone_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/time_zone.hpp>: a zone chosen at run time through the virtual interface, the adapter and the handle. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/time_zone.hpp>
#include <snicholls/core/interfaces/time_zone_reference.hpp>

#include <chrono>
#include <cstdio>
#include <memory>

using namespace snicholls;
using namespace std::chrono_literals;   // not all of std::chrono: libstdc++'s std::chrono::time_zone and local_info
                                        // would collide with core's
using std::chrono::sys_days;
using std::chrono::hours;
using std::chrono::year;
using std::chrono::January;

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
    std::printf("snicholls-core interfaces/time_zone.hpp demo\n");
    //! [virtual]
    std::unique_ptr<time_zone_interface> chosen = std::make_unique<time_zone_adapter<two_rule_zone>>(sydney_like_zone());
    time_zone_ref zone{*chosen};   // copyable, models the concept
    const utc_s t{sys_days{year{2026} / January / 15}};
    const auto local = to_local(zone, t);
    //! [virtual]
    check(offset_at(zone, t) == hours{11}, "AEDT in January");
    check(*from_local(*chosen, local) == t, "the interface round-trips");
    check(zone.info(t.sys_time()).abbrev == std::string_view{"AEDT"} && chosen->name() == "Sydney-like", "info and name");
    return failures == 0 ? 0 : 1;
}
