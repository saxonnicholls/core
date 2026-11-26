//
//  algorithm_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/algorithm.hpp>: tables built and checked by the compiler with fill, iota, copy, reverse
//  and equal (the same code compiles as C++17), and the run-time uses: clearing a buffer, shifting a window, swapping
//  two halves. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/algorithm.hpp>

#include <array>
#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

//! [table]
// A lookup table built by the compiler: the first 16 cubes, then a copy of it reversed. C++17 or C++20 alike.
constexpr std::array<long, 16> cubes = [] {
    std::array<long, 16> t{};
    snicholls::iota(t, 0L);
    for (auto& x : t)
        x = x * x * x;
    return t;
}();

constexpr std::array<long, 16> cubes_descending = [] {
    std::array<long, 16> t{};
    snicholls::copy(cubes.begin(), cubes.end(), t.begin());
    snicholls::reverse(t.begin(), t.end());
    return t;
}();

static_assert(cubes[3] == 27 && cubes_descending[0] == 3375 && cubes_descending[15] == 0);
static_assert(snicholls::equal(cubes, std::array<long, 16>{0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728,
                                                           2197, 2744, 3375}));
//! [table]

//! [sentinel]
// A table of "no entry yet" (-1) with a few slots set, fill and fill_n at compile time.
constexpr std::array<int, 10> slots = [] {
    std::array<int, 10> t{};
    snicholls::fill(t, -1);
    snicholls::fill_n(t.begin() + 4, 3, 0);
    return t;
}();
static_assert(slots[3] == -1 && slots[4] == 0 && slots[6] == 0 && slots[7] == -1);
//! [sentinel]
} // namespace

int main()
{
    std::printf("snicholls-core algorithm demo\n");

    check(cubes[15] == 3375 && cubes_descending[1] == 2744, "the cube tables, built by the compiler");
    check(slots[5] == 0 && slots[9] == -1, "fill and fill_n: a sentinel table with three slots set");

    //! [window]
    // A sliding window of the last five prices: shift left by one with copy, append the newest at the back.
    std::array<double, 5> window{100.0, 100.5, 101.0, 100.75, 101.25};
    snicholls::copy(window.begin() + 1, window.end(), window.begin());
    window.back() = 101.5;
    // ... and the other way with copy_backward, which may overlap the end of its input.
    std::array<double, 5> older = window;
    snicholls::copy_backward(older.begin(), older.end() - 1, older.end());
    older.front() = 99.75;
    //! [window]
    check(window == std::array{100.5, 101.0, 100.75, 101.25, 101.5}, "copy shifts a window left");
    check(older == std::array{99.75, 100.5, 101.0, 100.75, 101.25}, "copy_backward shifts it right");

    //! [buffer]
    // Clearing a buffer without memset, and swapping its two halves.
    std::vector<unsigned char> buffer(32);
    snicholls::iota(buffer, static_cast<unsigned char>(0));
    snicholls::swap_ranges(buffer.begin(), buffer.begin() + 16, buffer.begin() + 16);
    const bool swapped = buffer[0] == 16 && buffer[16] == 0;
    snicholls::fill(buffer, static_cast<unsigned char>(0));
    //! [buffer]
    check(swapped, "swap_ranges exchanges the halves");
    check(snicholls::equal(buffer, std::vector<unsigned char>(32)), "fill clears it; equal checks lengths too");
    check(!snicholls::equal(buffer, std::vector<unsigned char>(31)), "a shorter range is not equal");
    check(snicholls::equal(buffer.begin(), buffer.begin() + 8, buffer.begin() + 8), "the three-iterator equal");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
