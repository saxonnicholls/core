//
//  tabulate_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/tabulate.hpp>: functions of an index memoised by the compiler.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/tabulate.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/tabulate demo\n");

    //! [tabulate]
    // A discount factor per day for a year at 5%, computed by the compiler. At run time: one load.
    constexpr auto discount = fn::tabulate<366>([](std::size_t day) {
        double d = 1.0;
        for (std::size_t i = 0; i < day; ++i) d /= 1.0 + 0.05 / 365;
        return d;
    });
    static_assert(discount(0) == 1.0);
    //! [tabulate]
    check(discount(365) > 0.951 && discount(365) < 0.952, "a year at 5% daily discounts to about 0.9512");

    //! [binomial]
    // A row of Pascal's triangle, for a binomial tree of 30 steps.
    constexpr auto choose_30 = fn::tabulate<31>([](std::size_t k) {
        std::uint64_t r = 1;
        for (std::size_t i = 1; i <= k; ++i) r = r * (30 - k + i) / i;
        return r;
    });
    static_assert(choose_30(15) == 155'117'520);
    //! [binomial]
    check(choose_30(0) == 1 && choose_30(30) == 1, "the ends of the row are 1");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
