//
//  utc_clock.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::utc_clock`, the one shape of a wall clock that core's reference implementations read, and
/// `snicholls::manual_utc_clock`, the clock that moves only when told.
/// \ingroup core
///
/// **Overview.** A component that stamps what it records (a provider's facts, a key's creation, a token's expiry, a
/// UUID's timestamp) reads a wall clock, and a test must decide what that clock says. Core's reference
/// implementations all take the clock the same way: a template parameter `Clock` modelling `utc_clock` (`c.now()`
/// returns a `utc<D>` of any precision), owned by value and reachable through `clock()`, defaulted to
/// `manual_utc_clock`. The owner stays copyable (nothing captures `this`), the clock costs no allocation and no
/// indirect call, and a test moves time with `x.clock().advance(d)`. A program's real clock is a type of its own with a
/// `now()` that reads `utc<D>::now()`, or ts-moveables' clock sources; clock sources stay above core.
///
/// \snippet time/utc_clock_demo.cpp utc_clock
///
/// **Relationships.** `time/utc.hpp` (`utc<D>`); read by `basic_uuid_generator`, `basic_memory_payment_provider`,
/// `basic_memory_security_provider`, `basic_memory_onramp_provider`, `basic_memory_key_store` and
/// `basic_memory_token_verifier` (`interfaces/*_reference.hpp`). `loop_clock` (`concepts/loop_clock.hpp`) is a
/// different thing: a loop's monotonic timer clock, which can block. Group: \ref core.
///
/// **Implementations / variants.** `manual_utc_clock`, in milliseconds. Any type with `now()` returning a `utc<D>`.
///
/// **Guarantees.** `manual_utc_clock` is `constexpr`, `noexcept`, trivially copyable and an aggregate (`{start}`); it
/// never moves backwards (`advance` ignores a negative step).
///
/// **Errors.** None.
///
/// **Performance.** A load. Not benchmarked.
///
/// **Tests and demos.** `tests/time/utc_clock_test.cpp`, `demos/time/utc_clock_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<type_traits>`, `time/utc.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0 (`manual_utc_clock` was in `interfaces/uuid_generator_reference.hpp`, which still
/// provides it).

#ifndef snicholls_core_time_utc_clock_hpp
#define snicholls_core_time_utc_clock_hpp

#include <snicholls/core/time/utc.hpp>

#include <chrono>
#include <type_traits>

namespace snicholls {

    namespace detail {
        template <typename T>
        inline constexpr bool is_utc_v = false;
        template <typename D>
        inline constexpr bool is_utc_v<utc<D>> = true;

        // The clock's time at precision D, floored: what a component that records seconds stores.
        template <typename D, typename C>
        [[nodiscard]] constexpr utc<D> clock_now(const C& c) noexcept(noexcept(c.now()))
        {
            return utc<D>(std::chrono::floor<D>(c.now().sys_time()));
        }
    } // namespace detail

    /// \brief A wall clock: `c.now()` on a const clock returns a `utc<D>` of any precision.
    /// \ingroup core
    /// \tparam C The type asked about.
    template <typename C>
    concept utc_clock = requires(const C& c) { requires detail::is_utc_v<std::remove_cvref_t<decltype(c.now())>>; };

    /// \brief A wall clock in milliseconds that moves only when told: the clock of a deterministic test or replay.
    /// \ingroup core
    struct manual_utc_clock {
        utc_ms t{};   ///< the instant it reads

        /// \brief The instant.
        /// \return `t`.
        [[nodiscard]] constexpr utc_ms now() const noexcept { return t; }
        /// \brief Moves forward by `d` (a negative `d` is ignored).
        /// \param d The step.
        constexpr void advance(std::chrono::milliseconds d) noexcept
        {
            if (d.count() > 0)
                t += d;
        }
    };

    static_assert(utc_clock<manual_utc_clock>, "snicholls::manual_utc_clock must model snicholls::utc_clock");

} // namespace snicholls

#endif // snicholls_core_time_utc_clock_hpp
