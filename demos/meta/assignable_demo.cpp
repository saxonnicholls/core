//
//  assignable_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/meta/assignable.hpp>: a setter that refuses narrowing and numbers into strings. Exits non-zero if a value is not the one documented.
//

#include <snicholls/core/meta/assignable.hpp>

#include <cstdio>
#include <string>

namespace {

//! [assignable]
// A setter that takes anything assignable, except what would silently change the value.
struct order {
    int quantity = 0;
    std::string venue;

    template <typename E>
        requires snicholls::non_narrowing_assignable<int, E>
    void set_quantity(E&& e) { quantity = static_cast<E&&>(e); }

    template <typename E>
        requires snicholls::non_narrowing_assignable<std::string, E>
    void set_venue(E&& e) { venue = static_cast<E&&>(e); }
};
//! [assignable]

} // namespace

int main()
{
    std::printf("snicholls-core assignable demo\n");
    bool ok = true;

    //! [assignable_use]
    order o;
    o.set_quantity(100);
    o.set_venue("LSE");
    // o.set_quantity(99.5);   // does not compile: would truncate
    // o.set_venue(3);         // does not compile: would be the character '\x03'
    //! [assignable_use]

    std::printf("  %d on %s\n", o.quantity, o.venue.c_str());
    ok &= o.quantity == 100 && o.venue == "LSE";
    ok &= !snicholls::non_narrowing_assignable<int, double> && !snicholls::non_narrowing_assignable<std::string, int>;

    std::printf(ok ? "assignable demo: ok\n" : "assignable demo: FAILED\n");
    return ok ? 0 : 1;
}
