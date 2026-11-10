//
//  values_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/meta/values.hpp>: a hand-written table's rules stated where the table is. Exits non-zero if a rule is not the one documented.
//

#include <snicholls/core/meta/values.hpp>

#include <cstdio>

int main()
{
    std::printf("snicholls-core values demo\n");

    //! [values]
    // Tenor buckets in days and the codes that name them: in order, and no code twice.
    static_assert(snicholls::strictly_increasing_v<7, 30, 91, 182, 365, 730>, "tenor buckets out of order");
    static_assert(snicholls::all_distinct_v<'W', 'M', 'Q', 'H', 'Y', 'B'>, "a bucket code is used twice");
    //! [values]

    std::printf("  the table's rules hold at compile time\n");
    const bool ok = !snicholls::increasing_v<30, 7> && snicholls::decreasing_v<3, 3, 1>;
    std::printf(ok ? "values demo: ok\n" : "values demo: FAILED\n");
    return ok ? 0 : 1;
}
