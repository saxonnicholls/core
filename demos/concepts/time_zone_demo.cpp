//
//  time_zone_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/time_zone.hpp>: a market cut-off written once against the concept, gaps and folds resolved only by a named policy. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/time_zone.hpp>
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

namespace {
// Generic code: today's 16:00 cut-off in the market's zone, as an instant.
template <time_zone Z>
result<utc_s, time_zone_errc> cut_off(const Z& market, year_month_day day)
{
    return from_local(market, local_seconds{local_days{day} + hours{16}});
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/time_zone.hpp demo\n");
    //! [concept]
    constexpr two_rule_zone new_york = new_york_like_zone();
    const auto close = cut_off(new_york, year{2026} / July / 1);                       // 16:00 EDT
    const local_seconds in_gap = local_days{year{2026} / March / 8} + hours{2} + minutes{30};
    const auto refused = from_local(new_york, in_gap);                                 // rejected by default
    const auto moved = from_local(new_york, in_gap, {fold_policy::reject, gap_policy::next_valid});
    //! [concept]
    check(close && to_text(*close).view() == "2026-07-01T20:00:00Z", "16:00 EDT is 20:00Z");
    check(to_text(*close, offset_at(new_york, *close)).view() == "2026-07-01T16:00:00-04:00", "the text with its offset");
    check(refused.is_err() && refused.unwrap_err() == time_zone_errc::nonexistent_local_time, "02:30 on the spring-forward day is refused");
    check(moved && to_text(*moved).view() == "2026-03-08T07:00:00Z", "next_valid gives 03:00 EDT");
    check(to_local(new_york, *close).time_since_epoch() == (local_days{year{2026} / July / 1} + hours{16}).time_since_epoch(), "to_local");
    return failures == 0 ? 0 : 1;
}
