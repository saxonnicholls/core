//
//  checked_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/checked.hpp>: a position in lots that refuses to wrap, and a quantity
//  from the wire that refuses to be anything but a whole number. Exits non-zero if a result is not
//  the one documented.
//

#include <snicholls/core/checked.hpp>

#include <cstdint>
#include <cstdio>

int main()
{
    std::printf("snicholls-core checked demo\n");
    //! [checked]
    // Adding a fill to a position: an overflow is refused, never wrapped.
    const std::int64_t position = 9'000'000'000'000'000'000;
    const auto after = snicholls::checked_add<std::int64_t>(position, 500'000'000'000'000'000);

    // A quantity from the wire: the whole string, a whole number, and it must fit in 32 bits.
    const auto lots = snicholls::parse_number<std::int32_t>("1500");
    const auto rubbish = snicholls::parse_number<std::int32_t>("15e2");

    // A price that must be an exact number of ticks before it becomes an integer.
    const auto ticks = snicholls::numeric_cast<std::int32_t>(1234.0);
    //! [checked]
    std::printf("  overflow refused %d, lots %d, rubbish refused %d, ticks %d\n", !after.has_value(), lots.unwrap(), rubbish.is_err(), *ticks);
    const bool ok = !after && lots.unwrap() == 1500 && rubbish.is_err() && *ticks == 1234;
    std::printf(ok ? "checked demo: ok\n" : "checked demo: FAILED\n");
    return ok ? 0 : 1;
}
