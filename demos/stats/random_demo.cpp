//
//  random_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/random.hpp>: a seeded generator, unbiased draws, a shuffle,
//  and one stream per thread. Exits non-zero if a draw is not the one documented.
//

#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core random demo\n");
    namespace stats = snicholls::stats;
    bool ok = true;

    //! [random]
    // The same seed gives the same draws on every platform, unlike std::uniform_int_distribution.
    stats::wyrand g(2026);
    const std::uint64_t die = stats::uniform_index(g, 6) + 1;   // 1 .. 6, unbiased
    const double u = stats::uniform_real(g);                     // [0, 1), 53 bits

    std::array<int, 5> deck{1, 2, 3, 4, 5};
    stats::shuffle(std::span<int>(deck), g);

    // One stream per worker, from one seed.
    stats::wyrand worker_3 = stats::wyrand::stream(2026, 3);
    //! [random]

    std::printf("  die %llu, u %.4f, deck %d %d %d %d %d, worker 3 draws %llu\n", static_cast<unsigned long long>(die), u, deck[0], deck[1],
                deck[2], deck[3], deck[4], static_cast<unsigned long long>(worker_3() % 1000));
    ok &= die >= 1 && die <= 6 && u >= 0 && u < 1;
    stats::wyrand again(2026);
    ok &= stats::uniform_index(again, 6) + 1 == die;

    std::printf(ok ? "random demo: ok\n" : "random demo: FAILED\n");
    return ok ? 0 : 1;
}
