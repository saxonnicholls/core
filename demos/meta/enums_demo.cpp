//
//  enums_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/meta/enums.hpp>: an order side and status printed, parsed and
//  listed with no hand-written tables. Exits non-zero if a name is not the one documented.
//

#include <snicholls/core/meta/enums.hpp>

#include <cstdint>
#include <cstdio>
#include <string>

namespace oms {
enum class status : std::uint8_t { new_order, partially_filled, filled, cancelled, rejected };
} // namespace oms

int main()
{
    std::printf("snicholls-core enums demo\n");
    bool ok = true;

    //! [enums]
    // Names, values and parsing straight from the declaration, at compile time.
    static_assert(snicholls::enum_name(oms::status::filled) == "filled");
    static_assert(snicholls::enum_count_v<oms::status> == 5);

    // From the wire: a name becomes an enumerator, or nothing.
    const auto parsed = snicholls::enum_cast<oms::status>("partially_filled");
    const auto rubbish = snicholls::enum_cast<oms::status>("filed");

    // Every enumerator, in order of value.
    std::string all;
    for (const oms::status s : snicholls::enum_values_v<oms::status>)
        all += std::string(snicholls::enum_name(s)) + ' ';
    //! [enums]

    std::printf("  %s\n  parsed %s, rubbish %s\n", all.c_str(), parsed ? "yes" : "no", rubbish ? "yes" : "no");
    ok &= all == "new_order partially_filled filled cancelled rejected " && parsed == oms::status::partially_filled && !rubbish;

    std::printf(ok ? "enums demo: ok\n" : "enums demo: FAILED\n");
    return ok ? 0 : 1;
}
