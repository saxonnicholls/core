//
//  backoff_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/backoff.hpp>: the schedule a reconnecting client
//  follows, printed attempt by attempt, with the ceiling, a jittered delay and
//  the attempt count after each session. Exits non-zero if the schedule is not
//  the one documented.
//

#include <snicholls/core/backoff.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>

int main()
{
    std::printf("snicholls-core backoff demo\n");
    bool ok = true;
    using namespace std::chrono_literals;

    //! [schedule]
    constexpr snicholls::backoff_policy policy{.min = 100ms, .max = 5s, .session_grace = 30s};
    std::mt19937_64 rng{2026};   // the caller owns the randomness, the clock and the sleep

    // Six connections that drop at once, then one that stays up: the attempt count resets.
    std::uint32_t attempt = 0;
    for (int connection = 0; connection < 7; ++connection) {
        const auto ceiling = snicholls::backoff_ceiling(policy, attempt);
        const auto delay = snicholls::backoff_delay(policy, attempt, rng);   // uniform in [0, ceiling]
        const auto session = connection < 6 ? 50ms : 2min;                  // how long it then stayed up
        std::printf("  attempt %u: ceiling %5lld ms, sleep %5lld ms, session %6lld ms\n", attempt,
                    static_cast<long long>(ceiling.count()), static_cast<long long>(delay.count()),
                    static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(session).count()));
        ok &= delay >= 0ms && delay <= ceiling;
        attempt = snicholls::next_attempt(policy, attempt, session);
    }
    //! [schedule]

    ok &= attempt == 0;   // reset by the session that worked
    ok &= snicholls::backoff_ceiling(policy, 5) == 3200ms;
    ok &= snicholls::backoff_ceiling(policy, 6) == 5s;   // capped

    // The deterministic form, for a test or a printed table: the draw as a fraction.
    const auto half = snicholls::backoff_delay(policy, 3, 0.5);
    std::printf("  attempt 3 at fraction 0.5: %lld ms of %lld\n", static_cast<long long>(half.count()),
                static_cast<long long>(snicholls::backoff_ceiling(policy, 3).count()));
    ok &= half == 400ms;

    std::printf(ok ? "backoff demo: ok\n" : "backoff demo: FAILED\n");
    return ok ? 0 : 1;
}
