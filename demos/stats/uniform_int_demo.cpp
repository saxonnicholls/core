//
//  uniform_int_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/uniform_int.hpp>: one seed, three algorithms. core's own draw, and
//  libstdc++'s and libc++'s std::uniform_int_distribution reproduced on whatever library this demo was built
//  with. Exits non-zero if a draw is not the one documented.
//

#include <snicholls/core/stats/uniform_int.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <random>

int main()
{
    std::printf("snicholls-core uniform_int demo\n");
    namespace stats = snicholls::stats;
    bool ok = true;

    //! [uniform_int]
    // llama.cpp picks HellaSwag tasks with std::mt19937(1) and std::uniform_int_distribution, so a Linux
    // build and a Mac build pick different tasks. Name the library and the picks are the same everywhere.
    std::mt19937 linux_rng(1), mac_rng(1);
    stats::uniform_int<std::size_t, stats::draw::libstdcxx> as_on_linux(0, 99);   // GCC's library
    stats::uniform_int<std::size_t, stats::draw::libcxx> as_on_a_mac(0, 99);      // Apple's, LLVM's
    std::array<std::size_t, 6> linux_picks{}, mac_picks{};
    for (std::size_t i = 0; i < 6; ++i) {
        linux_picks[i] = as_on_linux(linux_rng);
        mac_picks[i] = as_on_a_mac(mac_rng);
    }

    // For new work: core's own draw (the default), the fastest, and also the same everywhere.
    stats::wyrand g(2026);
    stats::uniform_int<int> die(1, 6);
    const int roll = die(g);
    //! [uniform_int]

    std::printf("  libstdc++'s picks:");
    for (std::size_t v : linux_picks)
        std::printf(" %zu", v);
    std::printf("\n  libc++'s picks:   ");
    for (std::size_t v : mac_picks)
        std::printf(" %zu", v);
    std::printf("\n  core's die: %d\n", roll);

    ok &= linux_picks == std::array<std::size_t, 6>{41, 99, 72, 93, 0, 12};
    ok &= mac_picks == std::array<std::size_t, 6>{37, 12, 72, 9, 75, 5};
    ok &= roll == 6;
    ok &= die.a() == 1 && die.b() == 6 && die.min() == 1 && die.max() == 6;
    die.reset();   // nothing to reset: the draws keep no state
    ok &= stats::uniform_int<int, stats::draw::libcxx>(4, 4)(mac_rng) == 4;

    std::printf(ok ? "uniform_int demo: ok\n" : "uniform_int demo: FAILED\n");
    return ok ? 0 : 1;
}
