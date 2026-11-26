//
//  time_zone_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/time_zone_reference.hpp>: a fixed offset, and a toy two-rule daylight-saving zone in the southern hemisphere. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/time_zone_reference.hpp>

#include <chrono>
#include <cstdio>
#include <string_view>

using namespace snicholls;
using namespace std::chrono;

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
    std::printf("snicholls-core interfaces/time_zone_reference.hpp demo\n");
    //! [reference]
    constexpr fixed_offset_zone india{"India-like", minutes{330}, "IST"};
    constexpr two_rule_zone sydney = sydney_like_zone();
    constexpr zone_info summer = sydney.info(sys_days{year{2026} / January / 1});
    constexpr local_info fold = sydney.info(local_days{year{2026} / April / 5} + hours{2} + minutes{30});
    //! [reference]
    check(india.info(sys_seconds{}).offset == minutes{330}, "one offset forever");
    check(summer.offset == hours{11} && summer.is_dst() && summer.abbrev == std::string_view{"AEDT"}, "AEDT in summer");
    check(summer.end == sys_days{year{2026} / April / 4} + hours{16}, "it ends at 03:00 AEDT on 5 April");
    check(fold.kind == local_time_kind::ambiguous && fold.first.offset == hours{11}, "02:30 on 5 April happens twice");
    return failures == 0 ? 0 : 1;
}
