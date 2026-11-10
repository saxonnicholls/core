//
//  perfect_map_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/perfect_map.hpp>: futures tick sizes found with no
//  probing. Exits non-zero if a lookup is not the one documented.
//

#include <snicholls/core/containers/perfect_map.hpp>

#include <array>
#include <cstdio>
#include <string_view>
#include <utility>

int main()
{
    std::printf("snicholls-core perfect_map demo\n");
    using namespace std::string_view_literals;
    //! [perfect_map]
    // The perfect hash is found by the compiler; at run time a lookup is two hashes and one compare.
    constexpr snicholls::containers::perfect_map<std::string_view, double, 4> tick_size{
        std::array<std::pair<std::string_view, double>, 4>{{{"ES", 0.25}, {"NQ", 0.25}, {"CL", 0.01}, {"GC", 0.1}}}};
    const double* cl = tick_size.find("CL"sv);
    //! [perfect_map]
    std::printf("  CL ticks in %.2f\n", *cl);
    const bool ok = *cl == 0.01 && tick_size.find("ZZ"sv) == nullptr;
    std::printf(ok ? "perfect_map demo: ok\n" : "perfect_map demo: FAILED\n");
    return ok ? 0 : 1;
}
