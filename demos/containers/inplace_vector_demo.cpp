//
//  inplace_vector_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/inplace_vector.hpp>: an order's legs with no allocation.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/containers/inplace_vector.hpp>

#include <cstdio>

int main()
{
    std::printf("snicholls-core inplace_vector demo\n");
    //! [inplace_vector]
    // At most four legs per order, held inside the order: no allocation on the hot path.
    struct leg {
        int instrument = 0;
        double quantity = 0;
    };
    snicholls::containers::inplace_vector<leg, 4> legs;
    legs.push_back({101, 5});
    legs.push_back({102, -5});
    const bool room = legs.try_push_back({103, 1});   // false only when full
    //! [inplace_vector]
    std::printf("  %zu legs, room %d, capacity %zu\n", legs.size(), room, legs.capacity());
    const bool ok = legs.size() == 3 && room;
    std::printf(ok ? "inplace_vector demo: ok\n" : "inplace_vector demo: FAILED\n");
    return ok ? 0 : 1;
}
