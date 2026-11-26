//
//  loop_clock_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_loop_clock`: the conformance kit of the loop clock contract, C++17-clean.
/// \ingroup core
///
/// **Overview.** Drives any clock with `loop_clock_like`'s members (a clock, a `loop_clock_interface&`, an
/// `any_loop_clock`) through the contract of `concepts/loop_clock.hpp` and returns the failures as strings
/// (ts-moveables' kit style): empty is a pass. The checks: `now never goes backwards` (a thousand reads), `a due
/// deadline does not block` (`block_for` of a past instant and of now is `0`), `a wait is never below -1`, `set_waker
/// takes a callable and an empty one`. `check_loop_clock_moves(clock, step)` adds the checks of a clock that moves on
/// command: `step()` moves it forward and must call the waker once, after which `now()` is later.
///
/// \snippet interfaces/loop_clock_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/loop_clock.hpp`, `interfaces/loop_clock.hpp`; run on `manual_loop_clock` in every form
/// by `tests/interfaces/loop_clock_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** `check_loop_clock` for every clock; `check_loop_clock_moves` for manual ones.
///
/// **Guarantees.** The kit leaves no waker installed.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/loop_clock_conformance_test.cpp` (the kit catches broken clocks),
/// `demos/interfaces/loop_clock_conformance_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<functional>`, `<string>`, `<vector>`, `concepts/loop_clock.hpp`.
///
/// **Known issues.** "Never backwards" is sampled, not proven.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_loop_clock_conformance_hpp
#define snicholls_core_interfaces_loop_clock_conformance_hpp

#include <snicholls/core/concepts/loop_clock.hpp>

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace snicholls {

    /// \brief Checks `clock` against the loop clock contract.
    /// \ingroup core
    /// \tparam Clock A type with `loop_clock_like`'s members.
    /// \param clock The clock; its waker is removed on return.
    /// \return Every failed expectation, as text; empty when the clock conforms.
    template <typename Clock>
    std::vector<std::string> check_loop_clock(Clock& clock)
    {
        static_assert(is_loop_clock_like_v<Clock>, "snicholls::check_loop_clock: Clock must have loop_clock's members");
        std::vector<std::string> fail;
        const auto expect = [&](bool holds, const char* what) {
            if (!holds)
                fail.emplace_back(what);
        };

        auto last = clock.now();
        bool forwards = true;
        for (int i = 0; i < 1000; ++i) {
            const auto t = clock.now();
            forwards = forwards && !(t < last);
            last = t;
        }
        expect(forwards, "now never goes backwards");

        const auto now = clock.now();
        expect(clock.block_for(now - std::chrono::seconds{1}) == 0, "a past deadline does not block");
        expect(clock.block_for(now) == 0, "a deadline at now does not block");
        expect(clock.block_for(now + std::chrono::milliseconds{5}) >= -1, "a wait is never below -1");

        clock.set_waker([] {});
        clock.set_waker(std::function<void()>{});
        return fail;
    }

    /// \brief Checks a clock that moves on command: `step()` moves it forward, calls the waker once, and `now()` is later.
    /// \ingroup core
    /// \tparam Clock A type with `loop_clock_like`'s members.
    /// \tparam Step  A callable `void()` that moves the clock forward.
    /// \param clock The clock; its waker is removed on return.
    /// \param step  Moves it.
    /// \return Every failed expectation, as text; empty when the clock conforms.
    template <typename Clock, typename Step>
    std::vector<std::string> check_loop_clock_moves(Clock& clock, Step step)
    {
        std::vector<std::string> fail = check_loop_clock(clock);
        int woken = 0;
        clock.set_waker([&woken] { ++woken; });
        const auto before = clock.now();
        step();
        if (!(before < clock.now()))
            fail.emplace_back("a step moves now forward");
        if (woken != 1)
            fail.emplace_back("a step wakes the loop once");
        if (clock.block_for(before) != 0)
            fail.emplace_back("the passed deadline no longer blocks");
        clock.set_waker(std::function<void()>{});
        return fail;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_loop_clock_conformance_hpp
