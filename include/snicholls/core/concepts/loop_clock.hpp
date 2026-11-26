//
//  loop_clock.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The contract of ts-moveables' loop_clock (event/clock.hpp): now(), block_for() and set_waker(), which had no
//  concept twin. C++17-clean like task_pool.hpp, for ts-moveables' C++17 build; snicholls_core_cpp17_check compiles it.
//  The clocks (steady, manual with atomics, scaled replay) stay in ts-moveables.
//

/// \file
/// \brief `snicholls::loop_clock_like`: an event loop's sense of time, as a concept (C++20) and a trait (C++17).
/// \ingroup core
///
/// **Overview.** An event loop that reads time through a clock can run its timers live, in replay at 0.01x or 100x,
/// paused or single-stepped, and a test of a one-hour timeout costs no wall time. A loop clock answers:
/// - `now()`: the current instant in the loop's timeline (`std::chrono::steady_clock::time_point`), `noexcept`;
/// - `block_for(until)`: how many **real** milliseconds the poller may block while the next deadline is `until` in
///   the clock's timeline: `-1` blocks until woken (a paused clock: that deadline is not coming), `0` does not block
///   (a manual clock: only the program moves it), `noexcept`;
/// - `set_waker(f)`: the loop installs `f`, which a clock that moves on command calls when it moves, so a loop asleep
///   in the poller looks at its timers again.
///
/// The pieces, as `task_pool.hpp`: `is_loop_clock_like_v<C>` (C++17 trait over the exact member types, `const` and
/// `noexcept`), `loop_clock_like<C>` (the C++20 concept, under `SNICHOLLS_HAS_CONCEPTS`), `SNICHOLLS_LOOP_CLOCK` for a
/// template head that is constrained under C++20 and `typename` under C++17.
///
/// \snippet concepts/loop_clock_demo.cpp concept
///
/// **The contract** (`interfaces/loop_clock_conformance.hpp` checks it): `now()` never goes backwards; `block_for(t)`
/// is `0` once `t` is due, and never less than `-1`; `set_waker` accepts any callable and an empty one.
///
/// **Relationships.** The twin `interfaces/loop_clock.hpp`, the reference `interfaces/loop_clock_reference.hpp`
/// (`manual_loop_clock`), the kit `interfaces/loop_clock_conformance.hpp`; `concepts/task_pool.hpp` (the macro
/// `SNICHOLLS_HAS_CONCEPTS`). Group: \ref core.
///
/// **Implementations.** `manual_loop_clock` (single-threaded, deterministic). Above core, in ts-moveables:
/// `steady_clock_source`, `manual_clock` (atomic) and `scaled_clock` (replay).
///
/// **Guarantees.** A trait and a concept: no code.
///
/// **Errors.** None: time is always available.
///
/// **Performance.** Not applicable. (ts-moveables' loop holds no clock by default and reads `steady_clock` directly.)
///
/// **Tests and demos.** `tests/concepts/loop_clock_test.cpp`, `demos/concepts/loop_clock_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<functional>`, `<type_traits>`, `<utility>`, `concepts/task_pool.hpp`.
///
/// **Known issues.** A real sleep is the operating system's: every blocking clock lives above core.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_loop_clock_hpp
#define snicholls_core_concepts_loop_clock_hpp

#include <snicholls/core/concepts/task_pool.hpp>   // SNICHOLLS_HAS_CONCEPTS

#include <chrono>
#include <functional>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The loop's time point: `std::chrono::steady_clock`'s.
    /// \ingroup core
    using loop_time_point = std::chrono::steady_clock::time_point;

    namespace detail {
        template <typename C, typename = void>
        struct is_loop_clock_like : std::false_type {};

        template <typename C>
        struct is_loop_clock_like<C, std::void_t<decltype(std::declval<const C&>().now()),
                                                 decltype(std::declval<const C&>().block_for(std::declval<loop_time_point>())),
                                                 decltype(std::declval<C&>().set_waker(std::declval<std::function<void()>>()))>>
            : std::bool_constant<std::is_same_v<decltype(std::declval<const C&>().now()), loop_time_point> &&
                                 noexcept(std::declval<const C&>().now()) &&
                                 std::is_same_v<decltype(std::declval<const C&>().block_for(std::declval<loop_time_point>())), long long> &&
                                 noexcept(std::declval<const C&>().block_for(std::declval<loop_time_point>())) &&
                                 std::is_same_v<decltype(std::declval<C&>().set_waker(std::declval<std::function<void()>>())), void>> {};
    } // namespace detail

    /// \brief Whether `C` has a loop clock's members: `now() const noexcept`, `block_for(time_point) const noexcept`
    /// returning `long long`, `set_waker(std::function<void()>)`. The C++17 spelling of `loop_clock_like`.
    /// \ingroup core
    template <typename C>
    inline constexpr bool is_loop_clock_like_v = detail::is_loop_clock_like<std::remove_cv_t<C>>::value;

#if SNICHOLLS_HAS_CONCEPTS
    /// \brief A loop clock: `now()`, `block_for(until)` and `set_waker(f)`.
    /// \ingroup core
    template <typename C>
    concept loop_clock_like = requires(const C& c, loop_time_point t) {
        { c.now() } noexcept -> std::same_as<loop_time_point>;
        { c.block_for(t) } noexcept -> std::same_as<long long>;
    } && requires(C& c, std::function<void()> f) {
        { c.set_waker(std::move(f)) } -> std::same_as<void>;
    };

/// \brief `template <SNICHOLLS_LOOP_CLOCK Clock>`: constrained by `loop_clock_like` under C++20, `typename` under C++17.
#define SNICHOLLS_LOOP_CLOCK ::snicholls::loop_clock_like
#else
/// \brief `typename` under C++17 (see the other branch).
#define SNICHOLLS_LOOP_CLOCK typename
#endif

} // namespace snicholls

#endif // snicholls_core_concepts_loop_clock_hpp
