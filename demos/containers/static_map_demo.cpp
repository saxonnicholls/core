//
//  static_map_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/static_map.hpp>: currency minor units, sorted by the
//  compiler. Exits non-zero if a lookup is not the one documented.
//

#include <snicholls/core/containers/static_map.hpp>

#include <array>
#include <cstdio>
#include <string_view>
#include <utility>

int main()
{
    std::printf("snicholls-core static_map demo\n");
    using namespace std::string_view_literals;
    //! [static_map]
    // Written in any order, sorted and checked for duplicates by the compiler, found by binary search.
    constexpr snicholls::containers::static_map<std::string_view, int, 4> minor_units{
        std::array<std::pair<std::string_view, int>, 4>{{{"usd", 2}, {"jpy", 0}, {"bhd", 3}, {"eur", 2}}}};
    static_assert(minor_units.at("jpy"sv) == 0);
    //! [static_map]
    const int* usd = minor_units.find("usd"sv);
    std::printf("  usd has %d decimals\n", *usd);
    const bool ok = *usd == 2;
    std::printf(ok ? "static_map demo: ok\n" : "static_map demo: FAILED\n");
    return ok ? 0 : 1;
}
