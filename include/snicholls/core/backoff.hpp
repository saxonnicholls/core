//
//  backoff.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The policy and the three functions are the same; core adds the deterministic form that takes the random
//  draw as a fraction, so the schedule is testable without a generator, and makes the session length generic
//  over any duration so that no clock is named here.
//

/// \file
/// \brief Retry backoff with full jitter, as a pure function of the attempt: the caller owns the clock and the sleep.
/// \ingroup core
///
/// **Overview.** When a connection drops, every client that lost it at the same moment must
/// not retry at the same moment, or the server that just came back is knocked over by the
/// herd. Full jitter answers that: the delay before attempt `n` is uniform in
/// `[0, min(max, min * 2^n)]`, so the herd spreads over the whole window while the window
/// still grows exponentially. The attempt count resets only after a session that lasted
/// `session_grace`, so a server that accepts and drops at once does not make its clients
/// spin at `min`.
///
/// \snippet backoff_demo.cpp schedule
///
/// Everything here is a function of its arguments: the caller draws the random number, reads
/// the clock and sleeps. That is what makes the schedule provable in a unit test, overflow
/// included, without breaking a database to see it.
///
/// **Relationships.** The reconnect loops in ts-moveables and in consumers' network layers; a
/// supervisor's restart policy. Group: \ref core.
///
/// **Guarantees.** `backoff_ceiling` and `next_attempt` are `constexpr` and `noexcept`, and
/// never overflow: the doubling stops at `max`, and the attempt count saturates.
/// `backoff_delay` with a fraction is `constexpr` and `noexcept` too; with a generator it is
/// as `noexcept` as the generator and performs one draw. No state, no allocation.
///
/// **Errors.** None. A fraction outside `[0, 1)` is clamped; a policy with `max < min` behaves
/// as if `max == min`.
///
/// **Performance.** A loop of at most 32 doublings and one draw. Not benchmarked: there is
/// no abstraction here to have a cost.
///
/// **Tests and demos.** `tests/backoff_test.cpp`, `demos/backoff_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<chrono>` (durations only; no clock), `<cstdint>`,
/// `<random>` (the generator concept and the uniform distribution).
///
/// **Known issues.** `std::uniform_int_distribution` is not specified bit for bit across
/// standard libraries, so the same seed gives a different schedule on libc++ and libstdc++;
/// the fraction form is the one to pin in a test.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_backoff_hpp
#define snicholls_core_backoff_hpp

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>

namespace snicholls {

    /// \brief The three numbers a backoff schedule needs.
    /// \ingroup core
    struct backoff_policy {
        /// \brief The ceiling before the first retry.
        std::chrono::milliseconds min{100};
        /// \brief The ceiling never exceeds this.
        std::chrono::milliseconds max{30'000};
        /// \brief A session at least this long resets the attempt count.
        std::chrono::milliseconds session_grace{5'000};
    };

    /// \brief The upper bound of the delay before attempt `attempt` (0-based): `min(max, min * 2^attempt)`, without overflow.
    /// \ingroup core
    /// \param policy  The policy.
    /// \param attempt The attempt, 0 for the first retry.
    /// \return The ceiling.
    [[nodiscard]] constexpr std::chrono::milliseconds backoff_ceiling(const backoff_policy& policy,
                                                                      std::uint32_t attempt) noexcept
    {
        std::chrono::milliseconds ceiling = policy.min;
        for (std::uint32_t i = 0; i < attempt && ceiling < policy.max; ++i)
            ceiling *= 2;
        return std::min(ceiling, policy.max);
    }

    /// \brief The delay before attempt `attempt`, given the random draw as a fraction: `fraction * ceiling`.
    /// \ingroup core
    /// \param policy   The policy.
    /// \param attempt  The attempt, 0 for the first retry.
    /// \param fraction A number in `[0, 1)`; clamped into that range.
    /// \return The delay, between 0 and `backoff_ceiling(policy, attempt)` inclusive.
    [[nodiscard]] constexpr std::chrono::milliseconds backoff_delay(const backoff_policy& policy, std::uint32_t attempt,
                                                                    double fraction) noexcept
    {
        const double clamped = fraction < 0.0 ? 0.0 : fraction > 1.0 ? 1.0 : fraction;
        const auto ceiling = backoff_ceiling(policy, attempt).count();
        return std::chrono::milliseconds{static_cast<std::chrono::milliseconds::rep>(clamped * static_cast<double>(ceiling))};
    }

    /// \brief The delay before attempt `attempt`, drawn uniformly from `[0, ceiling]` with `rng`: full jitter.
    /// \ingroup core
    /// \tparam Rng A `std::uniform_random_bit_generator`.
    /// \param policy  The policy.
    /// \param attempt The attempt, 0 for the first retry.
    /// \param rng     The generator; one draw is taken.
    /// \return The delay.
    template <std::uniform_random_bit_generator Rng>
    [[nodiscard]] std::chrono::milliseconds backoff_delay(const backoff_policy& policy, std::uint32_t attempt, Rng& rng)
    {
        std::uniform_int_distribution<std::chrono::milliseconds::rep> pick(0, backoff_ceiling(policy, attempt).count());
        return std::chrono::milliseconds{pick(rng)};
    }

    /// \brief The attempt count after a session that lasted `session`: 0 after a working session, else one more, saturating.
    /// \ingroup core
    /// \tparam Rep    The duration's representation.
    /// \tparam Period The duration's period.
    /// \param policy  The policy.
    /// \param attempt The attempt that produced the session.
    /// \param session How long the session stayed up.
    /// \return The next attempt count.
    template <typename Rep, typename Period>
    [[nodiscard]] constexpr std::uint32_t next_attempt(const backoff_policy& policy, std::uint32_t attempt,
                                                       std::chrono::duration<Rep, Period> session) noexcept
    {
        if (session >= policy.session_grace)
            return 0;
        return attempt == UINT32_MAX ? attempt : attempt + 1;
    }
}

#endif /* snicholls_core_backoff_hpp */
