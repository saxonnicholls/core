//
//  task_pool.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  ts-moveables' task_pool twin (interfaces/task_pool.hpp, concepts/task_pool.hpp), moved into core word for word
//  apart from the names core's rules require (the interface is task_pool_interface, macros carry SNICHOLLS_), so
//  ts-moveables can include this and delete its own. C++17-clean, like constexpr_for.hpp: ts-moveables builds as C++17,
//  and snicholls_core_cpp17_check compiles this header as C++17.
//

/// \file
/// \brief `snicholls::task_pool_like`: somewhere to submit work, as a concept (C++20) and a trait (C++17).
/// \ingroup core
///
/// **Overview.** A task pool runs `std::function<void()>` tasks: `submit(task)` enqueues one, `wait_idle()` blocks
/// until everything submitted (including what the tasks submitted) has run, `worker_count()` is the parallelism the
/// algorithms may assume. Code that only needs "somewhere to run this" (a parallel for, a task graph, a benchmark that
/// swaps pools) depends on this contract, not on one of the pools. Given a concrete pool, a template over
/// `task_pool_like` calls it directly (inlined, devirtualised for a `final` pool); given a `task_pool_interface&` it
/// goes through the vtable; the call site picks the dispatch and one algorithm serves both.
///
/// The pieces:
/// - `is_task_pool_like_v<P>`: the members, with their exact types, `const` and `noexcept`, as a C++17 trait;
/// - `task_pool_like<P>`: the same test as a C++20 concept, when the compiler has concepts (`SNICHOLLS_HAS_CONCEPTS`);
/// - `SNICHOLLS_TASK_POOL`: `template <SNICHOLLS_TASK_POOL Pool>` is `template <task_pool_like Pool>` with concepts and
///   `template <typename Pool>` without (check `is_task_pool_like_v` in a `static_assert` then);
/// - `is_pool_task_v<F>` and `pool_task<F>`: a callable a pool can take as it is (no arguments, movable), with
///   `SNICHOLLS_POOL_TASK_TEMPLATE(F)` for a pool's `submit(F&&)` beside its `submit(task)`.
///
/// \snippet concepts/task_pool_demo.cpp concept
///
/// **The contract** (`interfaces/task_pool_conformance.hpp` checks it): every task submitted runs exactly once; when
/// `wait_idle()` returns, every task submitted before it, and every task those submitted, has run; `wait_idle()` on an
/// idle pool returns at once; `worker_count()` is at least 1 and does not throw. The concept is syntactic only: from
/// which threads `submit` may be called is the implementation's to document.
///
/// **Relationships.** The twin `interfaces/task_pool.hpp`, the reference `interfaces/task_pool_reference.hpp`
/// (`deferred_task_pool`), the kit `interfaces/task_pool_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `deferred_task_pool` (single-threaded, deterministic). Above core, in ts-moveables: the thread
/// pools (one shared queue, sharded, work-stealing, dispatch, inline).
///
/// **Guarantees.** A trait and a concept: no code, no cost.
///
/// **Errors.** None in the contract; what a throwing task does is the implementation's.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/task_pool_test.cpp`, `demos/concepts/task_pool_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<type_traits>`, `<utility>`; `<version>` and `<concepts>` when
/// available.
///
/// **Known issues.** C++17-clean by rule: no `requires` outside the `SNICHOLLS_HAS_CONCEPTS` block.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_task_pool_hpp
#define snicholls_core_concepts_task_pool_hpp

#if defined(__has_include)
#if __has_include(<version>)
#include <version>
#endif
#endif
#ifndef SNICHOLLS_HAS_CONCEPTS
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L && defined(__cpp_lib_concepts) && __cpp_lib_concepts >= 202002L
/// \brief 1 when the compiler and library have C++20 concepts (the C++20 halves of the C++17-clean contracts), else 0.
#define SNICHOLLS_HAS_CONCEPTS 1
#else
/// \brief 0: the compiler or library lacks C++20 concepts (see the other branch).
#define SNICHOLLS_HAS_CONCEPTS 0
#endif
#endif
#if SNICHOLLS_HAS_CONCEPTS
#include <concepts>
#endif

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief A task: what a pool runs (`task_pool_interface::task`).
    /// \ingroup core
    using pool_task_function = std::function<void()>;

    namespace detail {
        template <typename P, typename = void>
        struct is_task_pool_like : std::false_type {};

        template <typename P>
        struct is_task_pool_like<P, std::void_t<decltype(std::declval<P&>().submit(std::declval<pool_task_function>())),
                                                decltype(std::declval<P&>().wait_idle()),
                                                decltype(std::declval<const P&>().worker_count())>>
            : std::bool_constant<std::is_same_v<decltype(std::declval<P&>().submit(std::declval<pool_task_function>())), void> &&
                                 std::is_same_v<decltype(std::declval<P&>().wait_idle()), void> &&
                                 std::is_same_v<decltype(std::declval<const P&>().worker_count()), std::size_t> &&
                                 noexcept(std::declval<const P&>().worker_count())> {};
    } // namespace detail

    /// \brief Whether `F` can be submitted as a task: callable with no arguments (a result is discarded) and movable
    /// into the pool. The C++17 spelling of `pool_task`.
    /// \ingroup core
    template <typename F>
    inline constexpr bool is_pool_task_v = std::is_invocable_v<std::decay_t<F>&> && std::is_move_constructible_v<std::decay_t<F>> &&
                                           std::is_constructible_v<std::decay_t<F>, F>;

    /// \brief Whether `P` has a task pool's members: `submit(task)`, `wait_idle()`, `worker_count() const noexcept`
    /// returning `std::size_t`. The C++17 spelling of `task_pool_like`.
    /// \ingroup core
    template <typename P>
    inline constexpr bool is_task_pool_like_v = detail::is_task_pool_like<std::remove_cv_t<P>>::value;

#if SNICHOLLS_HAS_CONCEPTS
    /// \brief A task pool: submit a task (a lambda converts), wait until idle, the worker count without throwing.
    /// \ingroup core
    template <typename P>
    concept task_pool_like = requires(P& p, pool_task_function t) {
        { p.submit(std::move(t)) } -> std::same_as<void>;
    } && requires(P& p) {
        { p.wait_idle() } -> std::same_as<void>;
    } && requires(const P& p) {
        { p.worker_count() } noexcept -> std::same_as<std::size_t>;
    };

    /// \brief A callable a pool can take as it is, with no `std::function` around it.
    /// \ingroup core
    template <typename F>
    concept pool_task = std::invocable<std::decay_t<F>&> && std::move_constructible<std::decay_t<F>> &&
                        std::constructible_from<std::decay_t<F>, F>;

/// \brief `template <SNICHOLLS_TASK_POOL Pool>`: constrained by `task_pool_like` under C++20, `typename` under C++17.
#define SNICHOLLS_TASK_POOL ::snicholls::task_pool_like
/// \brief The template head of a pool's `submit(F&&)`: constrained by `pool_task` (C++20) or `is_pool_task_v` (C++17).
#define SNICHOLLS_POOL_TASK_TEMPLATE(F) template <::snicholls::pool_task F>
#else
/// \brief `typename` under C++17 (see the other branch).
#define SNICHOLLS_TASK_POOL typename
/// \brief The C++17 template head of a pool's `submit(F&&)`, constrained by `is_pool_task_v`.
#define SNICHOLLS_POOL_TASK_TEMPLATE(F) template <typename F, std::enable_if_t<::snicholls::is_pool_task_v<F>, int> = 0>
#endif

} // namespace snicholls

#endif // snicholls_core_concepts_task_pool_hpp
