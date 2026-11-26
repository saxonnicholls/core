//
//  uuid_generator_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/uuid_generator_reference.hpp>: deterministic identifiers from wyrand and a
//  manual clock, monotonic within a millisecond. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/uuid_generator_reference.hpp>

#include <chrono>
#include <cstdio>

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
    std::printf("snicholls-core interfaces/uuid_generator_reference.hpp demo\n");
    //! [reference]
    basic_uuid_generator ids{stats::wyrand{7}, manual_utc_clock{utc_ms::from_unix_millis(1'800'000'000'000)}};
    const uuid a = *ids.v7();
    const uuid b = *ids.v7();                       // same millisecond: the counter orders them
    ids.clock().advance(std::chrono::milliseconds{5});
    const uuid c = *ids.v7();
    const uuid r = *ids.v4();
    //! [reference]
    check(a < b && b < c, "strictly increasing");
    check(a.timestamp() == b.timestamp() && c.timestamp()->time_since_epoch().count() == 1'800'000'000'005, "the clock's milliseconds");
    check(r.version() == 4, "a version 4");
    auto again = make_reference_uuid_generator(7, utc_ms::from_unix_millis(1'800'000'000'000));
    check(*again.v7() == a, "the same seed and clock make the same identifier");
    std::printf("    %s\n    %s\n", to_string(a).c_str(), to_string(b).c_str());
    return failures == 0 ? 0 : 1;
}
