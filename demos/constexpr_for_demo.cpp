//
//  constexpr_for_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/constexpr_for.hpp>: walking a tuple by index, a
//  fully unrolled 3x3 matrix product, counting down with a step, and a
//  run-time range unrolled by four with a tail. Exits non-zero if any result
//  is not what it documents.
//

#include <snicholls/core/constexpr_for.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

int main()
{
    std::printf("snicholls-core constexpr_for demo\n");
    bool ok = true;

    //! [tuple]
    // The index is a type, so it can index a tuple: no plain loop can do this
    const auto row = std::make_tuple(std::string{"BTC"}, 64000.0, 3);
    std::string line;
    snicholls::constexpr_for<std::tuple_size_v<decltype(row)>>([&](auto I) {
        if constexpr (I.value > 0)
            line += ", ";
        if constexpr (std::is_same_v<std::decay_t<decltype(std::get<I.value>(row))>, std::string>)
            line += std::get<I.value>(row);
        else
            line += std::to_string(std::get<I.value>(row));
    });
    //! [tuple]
    std::printf("  tuple           %s\n", line.c_str());
    ok &= line == "BTC, 64000.000000, 3";

    // A 3x3 product with every index a constant: no loop, no run-time indexing
    constexpr std::array<std::array<int, 3>, 3> a{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
    constexpr std::array<std::array<int, 3>, 3> id{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    std::array<std::array<int, 3>, 3> c{};
    snicholls::constexpr_nest<3, 3, 3>([&](auto I, auto J, auto K) {
        c[I][J] += a[I][K] * id[K][J];
    });
    std::printf("  A x I           [%d %d %d] [%d %d %d] [%d %d %d]\n", c[0][0], c[0][1], c[0][2],
                c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2]);
    ok &= c == a;

    // Counting down by two, with the open end excluded
    std::string down;
    snicholls::constexpr_for<10, 0, -2>([&](auto I) { down += std::to_string(I.value) + ' '; });
    std::printf("  10 down by 2    %s\n", down.c_str());
    ok &= down == "10 8 6 4 2 ";

    // A run-time range, unrolled by four with a tail of three
    std::vector<int> v(11, 1);
    int sum = 0;
    snicholls::unrolled_for<4>(std::size_t{0}, v.size(), [&](std::size_t i) { sum += v[i]; });
    std::printf("  unrolled sum    %d of %zu elements\n", sum, v.size());
    ok &= sum == 11;

    std::printf(ok ? "constexpr_for demo: ok\n" : "constexpr_for demo: FAILED\n");
    return ok ? 0 : 1;
}
