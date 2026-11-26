//
//  loop_clock_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::manual_loop_clock`: the standard-library reference loop clock; time moves only when told.
/// \ingroup core
///
/// **Overview.** What a test drives a loop's timers with: the clock starts at a given instant and moves only on
/// `advance(d)` or `advance_to(t)`, never backwards, and wakes the loop (the installed waker) each time it moves.
/// `block_for` always answers `0`, because only the program moves this clock. Never moved, it is a fixed clock.
/// ts-moveables' `manual_clock` is the same with an atomic, for a test thread driving a loop on another thread; this
/// one is single-threaded, and C++17-clean.
///
/// \snippet interfaces/loop_clock_reference_demo.cpp reference
///
/// **Relationships.** Models `loop_clock_like` (`concepts/loop_clock.hpp`); passes
/// `interfaces/loop_clock_conformance.hpp` directly and through `interfaces/loop_clock.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** Not thread-safe. `now`, `block_for` and `advance` of a clock with no waker are `noexcept`; a waker
/// that throws propagates out of `advance`.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked: a test double.
///
/// **Tests and demos.** `tests/interfaces/loop_clock_reference_test.cpp`,
/// `demos/interfaces/loop_clock_reference_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<functional>`, `<utility>`, `concepts/loop_clock.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_loop_clock_reference_hpp
#define snicholls_core_interfaces_loop_clock_reference_hpp

#include <snicholls/core/concepts/loop_clock.hpp>

#include <chrono>
#include <functional>
#include <utility>

namespace snicholls {

    /// \brief The reference loop clock: moves only on `advance` and `advance_to`, and wakes the loop when it does.
    /// \ingroup core
    class manual_loop_clock {
    public:
        /// \brief The time point.
        using time_point = loop_time_point;
        /// \brief The duration.
        using duration = std::chrono::nanoseconds;

        /// \brief A clock at the steady clock's epoch.
        manual_loop_clock() = default;
        /// \brief A clock at `start`.
        /// \param start The first instant.
        explicit manual_loop_clock(time_point start) noexcept : t_(start) {}

        /// \brief The current instant.
        /// \return It.
        [[nodiscard]] time_point now() const noexcept { return t_; }
        /// \brief Do not block: only the program moves this clock.
        /// \return 0.
        [[nodiscard]] long long block_for(time_point) const noexcept { return 0; }
        /// \brief Installs the loop's waker.
        /// \param wake The waker; empty to remove it.
        void set_waker(std::function<void()> wake) { waker_ = std::move(wake); }

        /// \brief Moves forward by `d` and wakes the loop; a negative `d` is ignored.
        /// \param d The step.
        void advance(duration d)
        {
            if (d.count() < 0)
                return;
            t_ += d;
            wake();
        }
        /// \brief Moves to `t` if it is later, and wakes the loop.
        /// \param t The instant.
        void advance_to(time_point t)
        {
            if (t_ < t)
                t_ = t;
            wake();
        }

    private:
        void wake() const
        {
            if (waker_)
                waker_();
        }

        time_point t_{};
        std::function<void()> waker_;
    };

    static_assert(is_loop_clock_like_v<manual_loop_clock>, "snicholls::manual_loop_clock must have loop_clock's members");
#if SNICHOLLS_HAS_CONCEPTS
    static_assert(loop_clock_like<manual_loop_clock>, "snicholls::manual_loop_clock must model snicholls::loop_clock_like");
#endif

} // namespace snicholls

#endif // snicholls_core_interfaces_loop_clock_reference_hpp
