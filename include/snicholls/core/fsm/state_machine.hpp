//
//  state_machine.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fsm`: state machines written as sml writes them (`*state<A> + event<E> [guard] / action = state<B>`),
/// with states that are values: an enumerator per state, `current()` and `restore()`, no exceptions across the boundary.
/// \ingroup core
///
/// \overview A transition table is a list of rows, each read left to right as "in this state, on this event, if this
/// guard holds, run this action and go to that state":
///
/// \snippet fsm/state_machine_hello_world_demo.cpp table
///
/// `sm<Def>` is the machine: `Def` is a class whose `operator()` returns the table, built once when the machine is
/// constructed. `process_event(e)` runs the first row that leaves the current state on `e` and whose guard holds, and
/// says whether one did. The notation, the dispatch rules and the order of entry, exit and actions are boost-ext/sml's;
/// see "Implementations and variants" for what is reproduced and what is not.
///
/// The difference from sml, and the reason this exists in core: **every state can be a value.** A state is
/// `state<T>` (a type, as sml), `"name"_s` (a string, as sml), or `enum_state<E::pending>` (an enumerator). A machine
/// whose states are the enumerators of one enumeration, or which declares `using state_enum = E;`, has `current()`
/// returning that enumerator and `restore(e)` setting it, refusing a value that is not a state of the table. So the
/// state lives as one byte on the object it describes (a transaction is `pending`, then `confirmed`), and a machine is
/// rebuilt from that byte after a restart; `fsm/persist.hpp` saves and loads it through any `key_value_store`.
///
/// \snippet fsm/state_machine_demo.cpp enum
///
/// Use it for any lifecycle (a payment, an order, a connection, a protocol handshake): the table is the specification,
/// checked by the compiler, drawn by `fsm/diagram.hpp`. Do not use it for a two-state flag (a `bool` is clearer), nor
/// for a machine whose states and transitions are only known at run time (this table is a type).
///
/// \relations `fsm/persist.hpp` (save and load through `key_value_store`), `fsm/diagram.hpp` (Mermaid, Graphviz DOT
/// and PlantUML from the table), `fsm/lifecycle.hpp` (`lifecycle_state`, the provider-neutral payment lifecycle, and its
/// reference machine). Errors are `result<V, fsm_errc>` (`result.hpp`); string states and events are
/// `basic_fixed_string` (`fixed_string.hpp`); type-level work is `meta::type_list` (`meta/type_list.hpp`); state names
/// come from `type_name` (`meta/type_name.hpp`) and `enum_name` (`meta/enums.hpp`). Group: \ref core.
///
/// \impls API and semantics modelled on boost-ext/sml (https://github.com/boost-ext/sml), BSL-1.0; this is an
/// independent C++20 implementation (concepts, class-type template arguments, fold expressions), not a copy.
/// Reproduced from sml:
/// - the table DSL: `*` (initial state, one per orthogonal region), `state<T>`, `"x"_s`, `event<E>`, `"e"_e`,
///   `[guard]`, `/ action`, `= target`, the reversed form `target <= source + event`, guard algebra (`&&`, `||`, `!`),
///   action sequences `(a, b)`, `wrap(f)` for function pointers, `eval [g] / a` inside a sequence;
/// - anonymous (eventless) transitions, internal transitions (no target), self transitions (exit and entry run),
///   `on_entry<_>` / `on_exit<_>` (and `on_entry<E>` for one triggering event, `on_entry<initial>` for the start),
///   `X` (terminate);
/// - `defer` and `process(e)` (the queues are built in, fixed capacity, no allocation: sml needs `defer_queue` and
///   `process_queue` policies), with sml's draining order: anonymous transitions, then deferred events after each
///   state change, then queued events one at a time;
/// - composite machines (a `state<Sub>` whose `Sub` returns a table), orthogonal regions, history (`(H)`);
/// - `unexpected_event<E>` / `unexpected_event<_>` rows, and `exception<E>` / `exception<_>` rows (when exceptions are
///   enabled);
/// - dependency injection: objects passed to the constructor are matched by type to the parameters of guards and
///   actions; the event is matched by its type or by a deduced (`auto`) parameter; the definition object itself is
///   available to member-function-pointer guards and actions;
/// - `process_event`, `is(...)` (one state, or one per region), `is<Sub>(...)`, `visit_current_states`,
///   `set_current_states` (with the `testing` policy), the `logger<L>` hook (`log_process_event`, `log_guard`,
///   `log_action`, `log_state_change`), `make_dispatch_table` (run-time ids to event types), and two dispatch
///   strategies (`fold_expr`, the default, and `jump_table`).
///
/// Not reproduced: `sml::thread_safe` (core has no threads by rule; lock around the machine in the caller, as
/// ts-moveables does); `dont_instantiate_statemachine_class` and passing sub-machine objects to the constructor (a
/// sub-machine definition is default-constructed); `event<_>` and `state<_>` wildcard rows other than the
/// `on_entry`, `on_exit`, `unexpected_event` and `exception` forms; sml's `sm.process_event` returning through
/// `back::process` handles and `flush_queue` (queued events are drained before `process_event` returns). The
/// examples that need other libraries are not replicated: arduino and sdl2 (hardware and SDL), dependency_injection
/// (Boost.DI) and euml_emulation (Boost.MSM's eUML notation).
///
/// \contracts
/// - **State is one small integer per region:** `std::uint8_t` up to 255 states, else `std::uint16_t`. A machine whose
///   guards and actions hold no data and that has no dependencies, queues or sub-machines is exactly that array:
///   `sizeof(sm<hello_world>) == 1`, trivially copyable (static_asserts in the test).
/// - **No exception crosses `process_event`:** it is `noexcept`. An exception from a guard or an action goes to the
///   table's `exception<E>` rows; one no row takes is caught and reported (`try_process_event` answers
///   `fsm_errc::action_failed`). The state is then as sml leaves it: the target state is current, its entry has not
///   run.
/// - **Order of a transition (sml's):** guard; the source's exit (a sub-machine's current states first, then the
///   composite's own `on_exit`); the state changes; the action; the target's entry (the composite's own `on_entry`,
///   then its sub-machine's current states). An internal transition runs only its action.
/// - **Regions:** an event goes to every region in turn; `process_event` is true if any region took it. Within a
///   region the rows are tried in table order and the first whose guard holds runs.
/// - **Composite states:** the sub-machine sees the event first; if it took it and has not terminated, the parent's
///   rows are not tried. Entering a composite resets its regions to their initial states, except a region whose
///   initial state is marked `(H)`, which resumes where it was.
/// - **Compile-time checks** (static_asserts, messages start `snicholls::fsm:`): an initial state exists; no row leaves
///   a state on an event after an unguarded row that already does; every state is reached from an initial state; no
///   row leaves `X`; with a state enumeration, every state has an enumerator. `table_diagnostics<Def>` reports the same
///   facts as values, for a test.
/// - Everything is `constexpr`: a machine can run in a constant expression. Copying a machine rebuilds its table from
///   the copied definition, so a guard that captured the definition's `this` refers to the copy's.
/// - Not synchronised: one machine, one thread at a time.
///
/// \errors `fsm_errc`. `process_event` returns `bool` as sml does; `try_process_event` returns
/// `result<void, fsm_errc>`: `no_transition` (no row leaves the current state on the event), `guard_rejected` (rows
/// exist and every guard refused), `action_failed` (a guard or action threw and no `exception` row took it),
/// `queue_full` (a `defer` or `process` found its fixed queue full; the event is dropped), `anonymous_loop` (anonymous
/// transitions ran more than `max_anonymous_steps` times in a row), `queued_event_unhandled` (an event the action
/// queued with `process` was not taken). `restore` returns `not_a_state` for an enumerator that is not a state of the
/// table. Nothing throws.
///
/// \examples Each of sml's examples is replicated in `demos/fsm/state_machine_<example>_demo.cpp`.
/// \snippet fsm/state_machine_actions_guards_demo.cpp table
/// \snippet fsm/state_machine_composite_demo.cpp composite
/// \snippet fsm/state_machine_dependencies_demo.cpp dependencies
///
/// \perf `benchmarks/fsm/state_machine_benchmark.cpp`: 4,096 random events, valid and invalid, against the same
/// machine written by hand (an enumeration, one function per event, a switch over the state), with a fairness check.
/// macOS x86-64 (32 cores, load 9), Apple Clang 17, Release, 2026-11-24, medians of 10 interleaved runs: hello_world
/// 5.39 ns per event (by hand 5.33); the payment lifecycle 7.56 ns (by hand 7.49). The loops compile to the same
/// instructions: a transition is a compare (or a bit test over the source states) and a store, and a machine with a
/// dense state enumeration numbers its states by their enumerators, so `current()` is the stored byte.
///
/// \testing `tests/fsm/state_machine_test.cpp` (sml's functional tests: transitions, guards, actions, entry and exit
/// order, anonymous, composite, orthogonal regions, history, defer and process; the compile-time checks; layout),
/// `tests/compile_fail/fsm_*.cpp` (each static_assert fires with its message, beside a control that compiles),
/// `demos/fsm/state_machine_demo.cpp` and the replicated examples `demos/fsm/state_machine_*_demo.cpp`.
///
/// \deps `<array>`, `<cstddef>`, `<cstdint>`, `<functional>`, `<span>`, `<string_view>`, `<type_traits>`,
/// `<utility>`, `<variant>`, and core's `fixed_string.hpp`, `result.hpp` (which brings `<format>`, the heaviest part),
/// `meta/callable_traits.hpp`, `meta/enums.hpp`, `meta/type_list.hpp`, `meta/type_name.hpp`.
///
/// \gotchas
/// - `state<T>` takes a type and `enum_state<V>` a value: one template cannot take both, so an enumerator is written
///   `enum_state<E::pending>` (or named once: `constexpr auto pending = enum_state<E::pending>;`).
/// - The operators `&&`, `||`, `!` and `,` are overloaded for callables in `namespace snicholls::fsm`; a table is
///   written after `using namespace snicholls::fsm;`. Two function pointers cannot be combined by an operator (the
///   language needs a class operand): `wrap` the first, as in sml.
/// - A generic (`auto`) guard or action is instantiated with the row's event to learn its parameters; on an
///   `on_entry<_>` / `on_exit<_>` row the event is only known when it runs, so such a callable may ask only for
///   dependencies that another row (or a non-generic callable) also asks for.
/// - glibc declares a function `::on_exit`, so after `using namespace snicholls::fsm;` the name `on_exit` is ambiguous
///   on Linux: write `snicholls::fsm::on_exit<_>` (sml's examples write `sml::on_exit` for the same reason), or add
///   `using snicholls::fsm::on_exit;` where the table is written.
/// - Anonymous transitions stop after `max_anonymous_steps` (1024) in a row and report `anonymous_loop`, where sml
///   would loop for ever.
/// - The constructor runs the initial states' entry actions; an exception there that no `exception` row takes is
///   discarded.
/// - MSVC ignores the standard `[[no_unique_address]]`; core spells `[[msvc::no_unique_address]]` there.
/// - Clang 18 and older (LLVM's numbering, not Apple clang's) crash compiling an action sequence that holds an `eval`
///   of the same empty action, `(a, eval [g] / a)` (sml's eval example), when the sequence's members are
///   `[[no_unique_address]]`; on those compilers the sequence's members are plain, a byte per empty action in a
///   sequence. Every other layout, and every other compiler, is unchanged.
/// - sml's examples arduino, sdl2, dependency_injection and euml_emulation are not replicated (see above).
///
/// \status Experimental, new in 0.24.0.

#ifndef snicholls_core_fsm_state_machine_hpp
#define snicholls_core_fsm_state_machine_hpp

#include <snicholls/core/fixed_string.hpp>
#include <snicholls/core/meta/callable_traits.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/meta/type_list.hpp>
#include <snicholls/core/meta/type_name.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>


/// \brief `[[no_unique_address]]` for an action sequence's members, except on Clang 18 and older, which crashes on
/// `(a, eval [g] / a)` (see Known issues).
#if defined(__clang__) && !defined(__apple_build_version__) && __clang_major__ < 19
#define SNICHOLLS_FSM_SEQ_NO_UNIQUE_ADDRESS
#else
#define SNICHOLLS_FSM_SEQ_NO_UNIQUE_ADDRESS SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS
#endif

namespace snicholls::fsm {

    /// \brief Why an event, a restore or a persisted state was refused. Never thrown.
    /// \ingroup core
    enum class fsm_errc : std::uint8_t {
        no_transition = 1,       ///< no row leaves the current state(s) on this event
        guard_rejected,          ///< rows leave the current state on this event, and every guard refused
        action_failed,           ///< a guard or an action threw, and no `exception` row took it
        queue_full,              ///< `defer` or `process` found its fixed-capacity queue full; the event was dropped
        anonymous_loop,          ///< anonymous transitions ran `max_anonymous_steps` times without settling
        queued_event_unhandled,  ///< the event was taken, but an event its action queued with `process` was not
        not_a_state,             ///< `restore` was given an enumerator that is not a state of the table
        stale,                   ///< persist: another writer saved a newer version first
        not_found,               ///< persist: nothing is saved under the key
        corrupt,                 ///< persist: the saved record is not one this library wrote
        unknown_state,           ///< persist: a saved state is not a state of this table
        shape_mismatch,          ///< persist: the record holds a different number of regions than the machine
        store_unavailable,       ///< persist: the store answered `kv_errc::unavailable`
        store_failed             ///< persist: the store failed otherwise (`kv_errc::io`, `corrupt`, `limit`, ...)
    };

    /// \brief sml's wildcard: `on_entry<_>`, `on_exit<_>`, `unexpected_event<_>`, `exception<_>`.
    /// \ingroup core
    struct _ {};

    /// \brief The event that starts a machine: `on_entry<initial>` runs only when the machine is constructed.
    /// \ingroup core
    struct initial {};

    /// \brief The event of an anonymous (eventless) row, as an action that asks for "the event" receives it.
    /// \ingroup core
    struct anonymous {};

    /// \brief The marker of a history state: `"idle"_s(H)` is an initial state that a re-entered composite resumes.
    /// \ingroup core
    struct history_t {};

    /// \brief The history marker, written as sml writes it: `state<idle>(H)`.
    /// \ingroup core
    inline constexpr history_t H{};

    /// \brief Anonymous transitions that may run in a row before `process_event` gives up with `fsm_errc::anonymous_loop`.
    /// \ingroup core
    inline constexpr std::size_t max_anonymous_steps = 1024;

    template <class Table>
    struct transition_table;

    template <class S, bool Initial = false, bool History = false>
    struct state_t;

    template <class E>
    struct event_t;

    namespace detail {

        // ---- state and event identities ------------------------------------------------------------------------

        struct terminate_state {};
        template <basic_fixed_string Name>
        struct named_state {};
        template <auto V>
        struct enumerator_state {};
        struct internal_target {};
        struct always {};
        struct none {};

        template <class T>
        struct entry_tag {};
        template <class T>
        struct exit_tag {};
        template <class T>
        struct unexpected_tag {};
        template <class T>
        struct exception_tag {};

        template <class T>
        inline constexpr bool is_entry_v = false;
        template <class T>
        inline constexpr bool is_entry_v<entry_tag<T>> = true;
        template <class T>
        inline constexpr bool is_exit_v = false;
        template <class T>
        inline constexpr bool is_exit_v<exit_tag<T>> = true;
        template <class T>
        inline constexpr bool is_unexpected_v = false;
        template <class T>
        inline constexpr bool is_unexpected_v<unexpected_tag<T>> = true;
        template <class T>
        inline constexpr bool is_exception_v = false;
        template <class T>
        inline constexpr bool is_exception_v<exception_tag<T>> = true;
        template <class T>
        inline constexpr bool is_special_event_v = is_entry_v<T> || is_exit_v<T> || is_unexpected_v<T> || is_exception_v<T>;

        // The last component of a qualified name, before any template arguments: "(anonymous namespace)::idle" -> "idle".
        constexpr std::string_view short_name(std::string_view full) noexcept
        {
            const std::size_t angle = full.find('<');
            const std::string_view head = full.substr(0, angle);
            const std::size_t colon = head.rfind("::");
            return colon == std::string_view::npos ? full : full.substr(colon + 2);
        }

        template <std::size_t N>
        constexpr std::array<char, N + 1> to_c_string(std::string_view text) noexcept
        {
            std::array<char, N + 1> out{};
            for (std::size_t i = 0; i < N; ++i)
                out[i] = text[i];
            return out;
        }

        // A name held in its own null-terminated storage, so that c_str() works on every kind of name.
        template <std::string_view const& Name>
        struct stored_name {
            static constexpr auto text = to_c_string<Name.size()>(Name);
            static constexpr std::string_view view{text.data(), Name.size()};
        };

        template <class S>
        struct type_state_name {
            static constexpr std::string_view raw = short_name(type_name<S>());
        };

        template <class S>
        struct state_info {
            static constexpr std::string_view name = stored_name<type_state_name<S>::raw>::view;
        };
        template <basic_fixed_string Name>
        struct state_info<named_state<Name>> {
            static constexpr std::string_view name{Name.c_str(), Name.size()};
        };
        template <auto V>
        struct enum_state_name {
            static constexpr std::string_view raw = enum_name<V>();
        };
        template <auto V>
        struct state_info<enumerator_state<V>> {
            static constexpr std::string_view name = stored_name<enum_state_name<V>::raw>::view;
        };
        template <>
        struct state_info<terminate_state> {
            static constexpr std::string_view name = "terminate";
        };

        template <class E>
        struct event_info {
            static constexpr std::string_view name = stored_name<type_state_name<E>::raw>::view;
        };

        // ---- what the table is made of -----------------------------------------------------------------------

        template <class T>
        inline constexpr bool is_table_v = false;
        template <class... Rs>
        inline constexpr bool is_table_v<transition_table<meta::type_list<Rs...>>> = true;

        template <class S>
        concept complete = requires { sizeof(S); };

        // A composite state: a class whose operator() returns a transition table (sml's state<sub>).
        template <class S>
        concept machine_definition = std::is_class_v<S> && complete<S> && std::is_default_constructible_v<S> &&
            requires(S& s) { requires is_table_v<std::remove_cvref_t<decltype(s())>>; };

        // Something with an operator(), template or not: &T::operator() is ambiguous in a class that adds one.
        struct call_probe {
            void operator()();
        };
        template <class T>
        struct call_probe_derived : T, call_probe {};
        template <class T>
        concept has_call_operator = std::is_class_v<T> && !std::is_final_v<T> && complete<T> &&
            !requires { &call_probe_derived<T>::operator(); };

        // ---- guard and action combinators ----------------------------------------------------------------------

        template <class G>
        struct not_ {
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS G g;
        };
        template <class A, class B>
        struct and_ {
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS A a;
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS B b;
        };
        template <class A, class B>
        struct or_ {
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS A a;
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS B b;
        };
        template <class A, class B>
        struct seq {
            // Clang 18 (LLVM 18; Apple clang 17 is LLVM 19 and unaffected) crashes compiling a sequence that holds an
            // `eval` of the same empty action, `(a, eval [g] / a)`, when both members are [[no_unique_address]]:
            // there the members are plain, which costs a byte per empty action in that sequence only.
            SNICHOLLS_FSM_SEQ_NO_UNIQUE_ADDRESS A a;
            SNICHOLLS_FSM_SEQ_NO_UNIQUE_ADDRESS B b;
        };
        struct defer_t {};
        template <class E>
        struct process_t {
            E event;
        };
        template <class G, class A>
        struct eval_t {
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS G guard;
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS A action;
        };
        template <class G>
        struct eval_guarded {
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS G guard;
            template <class A>
            constexpr eval_t<G, A> operator/(A a) const
            {
                return {guard, a};
            }
        };
        struct eval_head {
            template <class G>
            constexpr eval_guarded<G> operator[](G g) const
            {
                return {g};
            }
        };

        template <class F>
        struct wrapped {
            F f;
        };

        template <class T>
        inline constexpr bool is_combinator_v = false;
        template <class G>
        inline constexpr bool is_combinator_v<not_<G>> = true;
        template <class A, class B>
        inline constexpr bool is_combinator_v<and_<A, B>> = true;
        template <class A, class B>
        inline constexpr bool is_combinator_v<or_<A, B>> = true;
        template <class A, class B>
        inline constexpr bool is_combinator_v<seq<A, B>> = true;
        template <>
        inline constexpr bool is_combinator_v<defer_t> = true;
        template <class E>
        inline constexpr bool is_combinator_v<process_t<E>> = true;
        template <class G, class A>
        inline constexpr bool is_combinator_v<eval_t<G, A>> = true;
        template <class F>
        inline constexpr bool is_combinator_v<wrapped<F>> = true;

        template <class T>
        inline constexpr bool is_dsl_v = false;
        template <class S, bool I, bool Hs>
        inline constexpr bool is_dsl_v<state_t<S, I, Hs>> = true;
        template <class E>
        inline constexpr bool is_dsl_v<event_t<E>> = true;

        template <class T>
        concept callable_object = has_call_operator<T> && !is_dsl_v<T>;

        // An operand of the guard algebra and of action sequences.
        template <class T>
        concept operand = callable_object<T> || is_combinator_v<T> || std::is_member_function_pointer_v<T> ||
            (std::is_pointer_v<T> && std::is_function_v<std::remove_pointer_t<T>>);

    } // namespace detail

    // ---- the DSL -----------------------------------------------------------------------------------------------

    /// \brief The guard or action of a row: a callable that takes the event and/or dependencies, or a combination.
    /// \ingroup core
    template <class T>
    concept guard_or_action = detail::operand<T>;

    /// \brief One row of a transition table, built by the DSL: `src + event [guard] / action = dst`.
    /// \ingroup core
    /// \tparam Src    the source, a `state_t` (with its initial and history marks).
    /// \tparam Event  the event type, `anonymous`, or an entry, exit, unexpected or exception marker.
    /// \tparam Guard  the guard, or the internal `always`.
    /// \tparam Action the action, or the internal `none`.
    /// \tparam Dst    the target state's identity, or the internal marker of an internal transition.
    template <class Src, class Event, class Guard, class Action, class Dst>
    struct row {
        /// \brief The source `state_t`.
        using source = Src;
        /// \brief The event type.
        using event_type = Event;
        /// \brief The guard's type.
        using guard_type = Guard;
        /// \brief The action's type.
        using action_type = Action;
        /// \brief The target's identity.
        using target = Dst;
        /// \brief The guard.
        SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS Guard guard;
        /// \brief The action.
        SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS Action action;

        /// \brief `... = state<B>`: gives an internal row its target.
        /// \return the row with target `D`.
        template <class D, bool I, bool Hs>
            requires std::is_same_v<Dst, detail::internal_target>
        constexpr row<Src, Event, Guard, Action, D> operator=(state_t<D, I, Hs>) const
        {
            return {guard, action};
        }

        /// \brief `*state [guard] ...`: marks the row's source as an initial state (`[]` binds before `*`).
        /// \return the row with an initial source.
        constexpr row<state_t<typename Src::type, true, Src::history>, Event, Guard, Action, Dst> operator*() const
        {
            return {guard, action};
        }

        /// \brief `state [guard] / action`: gives an anonymous row its action.
        /// \param a the action.
        /// \return the row with action `a`.
        template <guard_or_action A>
            requires std::is_same_v<Action, detail::none> && std::is_same_v<Dst, detail::internal_target>
        constexpr row<Src, Event, Guard, A, Dst> operator/(A a) const
        {
            return {guard, a};
        }
    };

    /// \brief An event with its guard and/or action, before it is attached to a source: `event<E> [g] / a`.
    /// \ingroup core
    /// \tparam E the event.
    /// \tparam G the guard.
    /// \tparam A the action.
    template <class E, class G, class A>
    struct event_expr {
        /// \brief The guard.
        SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS G guard;
        /// \brief The action.
        SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS A action;

        /// \brief `event<E> [g] / a`.
        /// \param a the action.
        /// \return the expression with action `a`.
        template <guard_or_action A2>
            requires std::is_same_v<A, detail::none>
        constexpr event_expr<E, G, A2> operator/(A2 a) const
        {
            return {guard, a};
        }
    };

    /// \brief An event in a table: `event<E>`, or `"name"_e`. `"name"_e()` makes the event object.
    /// \ingroup core
    /// \tparam E the event type.
    template <class E>
    struct event_t {
        /// \brief The event type.
        using type = E;

        /// \brief `event<E> [guard]`.
        /// \param g the guard.
        /// \return the guarded event.
        template <guard_or_action G>
        constexpr event_expr<E, G, detail::none> operator[](G g) const
        {
            return {g, {}};
        }

        /// \brief `event<E> / action`.
        /// \param a the action.
        /// \return the event with an action.
        template <guard_or_action A>
        constexpr event_expr<E, detail::always, A> operator/(A a) const
        {
            return {{}, a};
        }

        /// \brief The event object, `E{}`: sml's `"e1"_e()`.
        /// \return `E{}`.
        constexpr E operator()() const noexcept(std::is_nothrow_default_constructible_v<E>)
            requires std::is_default_constructible_v<E>
        {
            return E{};
        }

        /// \brief The event's name: the string of `"name"_e`, or the unqualified type name.
        /// \return the name.
        [[nodiscard]] static constexpr std::string_view name() noexcept;
    };

    /// \brief A state in a table, and the value a visitor and a logger receive: `state<T>`, `"name"_s`,
    /// `enum_state<E::v>`, `X`.
    /// \ingroup core
    /// \tparam S       the state's identity.
    /// \tparam Initial marked `*` (an initial state, one per orthogonal region).
    /// \tparam History marked `(H)` (an initial state a re-entered composite resumes).
    template <class S, bool Initial, bool History>
    struct state_t {
        /// \brief The state's identity.
        using type = S;
        /// \brief Marked as an initial state.
        static constexpr bool initial = Initial;
        /// \brief Marked as a history state.
        static constexpr bool history = History;
        /// \brief A composite state: `S` returns a transition table of its own.
        static constexpr bool is_composite = detail::machine_definition<S>;

        /// \brief The state's name: the string of `"name"_s`, the enumerator's name, the unqualified type name, or
        /// `"terminate"` for `X`.
        /// \return the name; null-terminated.
        [[nodiscard]] static constexpr std::string_view name() noexcept { return detail::state_info<S>::name; }

        /// \brief The name as a C string (sml's `c_str()`).
        /// \return the name.
        [[nodiscard]] static constexpr const char* c_str() noexcept { return detail::state_info<S>::name.data(); }

        /// \brief `*state`: the initial state of a region.
        /// \return the marked state.
        constexpr state_t<S, true, History> operator*() const noexcept { return {}; }

        /// \brief `state(H)`: an initial state that a re-entered composite resumes.
        /// \return the marked state.
        constexpr state_t<S, true, true> operator()(history_t) const noexcept { return {}; }

        /// \brief `state + event<E>`.
        /// \return the row.
        template <class E>
        constexpr row<state_t, E, detail::always, detail::none, detail::internal_target> operator+(event_t<E>) const noexcept
        {
            return {};
        }

        /// \brief `state + event<E> [g] / a`.
        /// \param e the event with its guard and action.
        /// \return the row.
        template <class E, class G, class A>
        constexpr row<state_t, E, G, A, detail::internal_target> operator+(const event_expr<E, G, A>& e) const
        {
            return {e.guard, e.action};
        }

        /// \brief `state [guard]`: an anonymous row.
        /// \param g the guard.
        /// \return the row.
        template <guard_or_action G>
        constexpr row<state_t, anonymous, G, detail::none, detail::internal_target> operator[](G g) const
        {
            return {g, {}};
        }

        /// \brief `state / action`: an anonymous row.
        /// \param a the action.
        /// \return the row.
        template <guard_or_action A>
        constexpr row<state_t, anonymous, detail::always, A, detail::internal_target> operator/(A a) const
        {
            return {{}, a};
        }

        /// \brief `state = other`: an anonymous transition.
        /// \return the row.
        template <class D, bool I, bool Hs>
            requires(!std::is_same_v<D, S>)
        constexpr row<state_t, anonymous, detail::always, detail::none, D> operator=(state_t<D, I, Hs>) const noexcept
        {
            return {};
        }

        /// \brief `target <= source + event ...`: sml's reversed notation.
        /// \param r the row whose target this is.
        /// \return the row with this target.
        template <class Src, class E, class G, class A>
        constexpr row<Src, E, G, A, S> operator<=(const row<Src, E, G, A, detail::internal_target>& r) const
        {
            return {r.guard, r.action};
        }

        /// \brief `target <= source`: an anonymous transition in the reversed notation.
        /// \return the row.
        template <class D, bool I, bool Hs>
        constexpr row<state_t<D, I, Hs>, anonymous, detail::always, detail::none, S> operator<=(state_t<D, I, Hs>) const noexcept
        {
            return {};
        }

        /// \brief Two state values are equal when they name the same state.
        /// \return true when the identities are the same.
        template <class S2, bool I2, bool H2>
        friend constexpr bool operator==(state_t, state_t<S2, I2, H2>) noexcept
        {
            return std::is_same_v<S, S2>;
        }
    };

    /// \brief `state<T>`: the state whose identity is the type `T` (sml's `state<idle>`, or a composite `state<sub>`).
    /// \ingroup core
    template <class S>
    inline constexpr state_t<S> state{};

    /// \brief `enum_state<E::pending>`: the state whose identity is an enumerator.
    /// \ingroup core
    template <auto V>
        requires std::is_enum_v<decltype(V)>
    inline constexpr state_t<detail::enumerator_state<V>> enum_state{};

    /// \brief `X`: the terminal state (sml's `X`, named `"terminate"`).
    /// \ingroup core
    inline constexpr state_t<detail::terminate_state> X{};

    /// \brief `event<E>`: the event `E` in a table.
    /// \ingroup core
    template <class E>
    inline constexpr event_t<E> event{};

    /// \brief A string event: the type of `"name"_e`.
    /// \ingroup core
    /// \tparam Name the name.
    template <basic_fixed_string Name>
    struct named_event {
        /// \brief The name.
        /// \return the name.
        [[nodiscard]] static constexpr std::string_view name() noexcept { return {Name.c_str(), Name.size()}; }
    };

    /// \brief `on_entry<_>`: the entry of the row's state; `on_entry<E>` only when entered by `E`.
    /// \ingroup core
    template <class T>
    inline constexpr event_t<detail::entry_tag<T>> on_entry{};

    /// \brief `on_exit<_>`: the exit of the row's state; `on_exit<E>` only when left on `E`.
    /// \ingroup core
    template <class T>
    inline constexpr event_t<detail::exit_tag<T>> on_exit{};

    /// \brief `unexpected_event<E>`: runs when `E` arrives in a state with no row for it; `<_>` for any event.
    /// \ingroup core
    template <class T>
    inline constexpr event_t<detail::unexpected_tag<T>> unexpected_event{};

    /// \brief `exception<E>`: runs when a guard or action throws an `E`; `<_>` for any exception.
    /// \ingroup core
    template <class T>
    inline constexpr event_t<detail::exception_tag<T>> exception{};

    /// \brief `/ defer`: keep the event and offer it again after the next state change.
    /// \ingroup core
    inline constexpr detail::defer_t defer{};

    /// \brief `eval [guard] / action`, inside an action sequence: the action runs only if the guard holds.
    /// \ingroup core
    inline constexpr detail::eval_head eval{};

    /// \brief `/ process(e)`: queue `e`, processed when the current event is done.
    /// \ingroup core
    /// \param e the event to queue.
    /// \return the action.
    template <class E>
    [[nodiscard]] constexpr detail::process_t<E> process(E e)
    {
        return {std::move(e)};
    }

    /// \brief `wrap(f)`: a function or member function pointer as an operand of `&&`, `||`, `!` or `,`.
    /// \ingroup core
    /// \param f the function.
    /// \return the wrapped function.
    template <class F>
        requires std::is_member_function_pointer_v<F> || (std::is_pointer_v<F> && std::is_function_v<std::remove_pointer_t<F>>)
    [[nodiscard]] constexpr detail::wrapped<F> wrap(F f) noexcept
    {
        return {f};
    }

    /// \brief `g1 && g2`.
    /// \ingroup core
    /// \param a the first guard.
    /// \param b the second guard, called only if the first holds.
    /// \return the conjunction.
    template <guard_or_action A, guard_or_action B>
        requires(std::is_class_v<A> || std::is_class_v<B>)
    constexpr detail::and_<A, B> operator&&(A a, B b)
    {
        return {a, b};
    }

    /// \brief `g1 || g2`.
    /// \ingroup core
    /// \param a the first guard.
    /// \param b the second guard, called only if the first refuses.
    /// \return the disjunction.
    template <guard_or_action A, guard_or_action B>
        requires(std::is_class_v<A> || std::is_class_v<B>)
    constexpr detail::or_<A, B> operator||(A a, B b)
    {
        return {a, b};
    }

    /// \brief `!g`.
    /// \ingroup core
    /// \param g the guard.
    /// \return the negation.
    template <guard_or_action G>
        requires std::is_class_v<G>
    constexpr detail::not_<G> operator!(G g)
    {
        return {g};
    }

    /// \brief `(a1, a2)`: both actions, in order.
    /// \ingroup core
    /// \param a the first action.
    /// \param b the second action.
    /// \return the sequence.
    template <guard_or_action A, guard_or_action B>
        requires(std::is_class_v<A> || std::is_class_v<B>)
    constexpr detail::seq<A, B> operator,(A a, B b)
    {
        return {a, b};
    }

    /// \brief The rows of a machine, as `make_transition_table` returns them.
    /// \ingroup core
    /// \tparam Rows a `meta::type_list` of `row` types.
    template <class Rows>
    struct transition_table;

    namespace detail {
        // A guard or action that holds nothing is not stored: a fresh one is made for each call, so that two rows
        // with the same empty callable do not need two addresses, and a table of such rows takes no space at all.
        template <class T>
        inline constexpr bool stateless_v =
            std::is_empty_v<T> && std::is_default_constructible_v<T> && std::is_trivially_copyable_v<T>;

        template <std::size_t I, class R, bool G = stateless_v<typename R::guard_type>,
                  bool A = stateless_v<typename R::action_type>>
        struct row_leaf {
            typename R::guard_type guard_;
            typename R::action_type action_;
            constexpr explicit row_leaf(const R& r) : guard_(r.guard), action_(r.action) {}
            constexpr auto& guard() noexcept { return guard_; }
            constexpr auto& action() noexcept { return action_; }
        };
        template <std::size_t I, class R>
        struct row_leaf<I, R, true, false> {
            typename R::action_type action_;
            constexpr explicit row_leaf(const R& r) : action_(r.action) {}
            static constexpr auto guard() noexcept { return typename R::guard_type{}; }
            constexpr auto& action() noexcept { return action_; }
        };
        template <std::size_t I, class R>
        struct row_leaf<I, R, false, true> {
            typename R::guard_type guard_;
            constexpr explicit row_leaf(const R& r) : guard_(r.guard) {}
            constexpr auto& guard() noexcept { return guard_; }
            static constexpr auto action() noexcept { return typename R::action_type{}; }
        };
        template <std::size_t I, class R>
        struct row_leaf<I, R, true, true> {
            constexpr explicit row_leaf(const R&) noexcept {}
            static constexpr auto guard() noexcept { return typename R::guard_type{}; }
            static constexpr auto action() noexcept { return typename R::action_type{}; }
        };
        template <class Is, class... Rs>
        struct row_storage;
        template <std::size_t... Is, class... Rs>
        struct row_storage<std::index_sequence<Is...>, Rs...> : row_leaf<Is, Rs>... {
            constexpr explicit row_storage(const Rs&... rs) : row_leaf<Is, Rs>(rs)... {}
        };

        template <class T>
        inline constexpr bool is_row_v = false;
        template <class Src, class E, class G, class A, class D>
        inline constexpr bool is_row_v<row<Src, E, G, A, D>> = true;
    } // namespace detail

    /// \brief The rows of a machine, as `make_transition_table` returns them.
    /// \ingroup core
    /// \tparam Rs the rows.
    template <class... Rs>
    struct transition_table<meta::type_list<Rs...>> : detail::row_storage<std::index_sequence_for<Rs...>, Rs...> {
        /// \brief The rows, as a type list.
        using rows = meta::type_list<Rs...>;

        /// \brief The table of these rows.
        /// \param rs the rows.
        constexpr explicit transition_table(const Rs&... rs) : detail::row_storage<std::index_sequence_for<Rs...>, Rs...>(rs...) {}

        /// \brief The stored guard and action of row `I`.
        /// \return the row's storage.
        template <std::size_t I>
        constexpr auto& get() noexcept
        {
            return static_cast<detail::row_leaf<I, meta::at_t<rows, I>>&>(*this);
        }
    };

    namespace detail {

        template <class E>
        struct event_info<event_t<E>> : event_info<E> {};
        template <basic_fixed_string Name>
        struct event_info<named_event<Name>> {
            static constexpr std::string_view name{Name.c_str(), Name.size()};
        };
        template <class T>
        struct event_info<entry_tag<T>> {
            static constexpr std::string_view name = "entry";
        };
        template <class T>
        struct event_info<exit_tag<T>> {
            static constexpr std::string_view name = "exit";
        };
        template <class T>
        struct event_info<unexpected_tag<T>> {
            static constexpr std::string_view name = "unexpected";
        };
        template <class T>
        struct event_info<exception_tag<T>> {
            static constexpr std::string_view name = "exception";
        };
        template <>
        struct event_info<anonymous> {
            static constexpr std::string_view name = "";
        };
    } // namespace detail

    template <class E>
    constexpr std::string_view event_t<E>::name() noexcept
    {
        return detail::event_info<E>::name;
    }

    /// \brief The table: the rows in order, as sml's `make_transition_table`.
    /// \ingroup core
    /// \param rows the rows.
    /// \return the table.
    template <class... Rs>
    [[nodiscard]] constexpr transition_table<meta::type_list<Rs...>> make_transition_table(Rs... rows)
    {
        static_assert((detail::is_row_v<Rs> && ...),
                      "snicholls::fsm: make_transition_table takes rows: state + event [guard] / action = state");
        return transition_table<meta::type_list<Rs...>>(rows...);
    }

    // ---- policies -----------------------------------------------------------------------------------------------

    /// \brief Policy: a logger, called as sml calls one (`log_process_event`, `log_guard`, `log_action`,
    /// `log_state_change`, each a member template taking the definition type first; any may be absent).
    /// \ingroup core
    /// \tparam L the logger type; the machine holds the one passed to its constructor by reference.
    template <class L>
    struct logger {
        /// \brief The logger type.
        using type = L;
    };

    /// \brief Dispatch strategy: a fold over the states that react to the event (the default; inlines to a branch chain).
    /// \ingroup core
    struct fold_expr {};

    /// \brief Dispatch strategy: an array of handlers indexed by the current state, one per event type (sml's jump_table).
    /// \ingroup core
    struct jump_table {};

    /// \brief Policy: the dispatch strategy, `fold_expr` or `jump_table`.
    /// \ingroup core
    /// \tparam D the strategy.
    template <class D>
    struct dispatch {
        /// \brief The strategy.
        using type = D;
    };

    /// \brief Policy: the capacity of the `defer` and `process` queues (each, per machine level; default 16).
    /// \ingroup core
    /// \tparam N the capacity.
    template <std::size_t N>
    struct queue_capacity {
        /// \brief The capacity.
        static constexpr std::size_t value = N;
    };

    /// \brief Policy: allow `set_current_states`, which sets states without running actions (sml's `sml::testing`).
    /// \ingroup core
    struct testing {};

    /// \brief One region's state as persisted: a number (the enumerator's value, or the state's index when the
    /// machine has no state enumeration) and the state's name.
    /// \ingroup core
    struct wire_state {
        /// \brief The enumerator's value, or the state's index in the table.
        std::int64_t value = 0;
        /// \brief The state's name.
        std::string_view name{};
        /// \brief Equal when both parts are.
        friend constexpr bool operator==(const wire_state&, const wire_state&) = default;
    };

    namespace detail {
        template <class>
        inline constexpr bool dependent_false = false;

        struct no_logger {};

        template <class... Ps>
        struct config {
            using logger_type = no_logger;
            using dispatch_type = fold_expr;
            static constexpr std::size_t capacity = 16;
            static constexpr bool testing_enabled = false;
        };
        template <class L, class... Ps>
        struct config<logger<L>, Ps...> : config<Ps...> {
            using logger_type = L;
        };
        template <class D, class... Ps>
        struct config<dispatch<D>, Ps...> : config<Ps...> {
            static_assert(std::is_same_v<D, fold_expr> || std::is_same_v<D, jump_table>,
                          "snicholls::fsm: dispatch<D> takes fold_expr or jump_table");
            using dispatch_type = D;
        };
        template <std::size_t N, class... Ps>
        struct config<queue_capacity<N>, Ps...> : config<Ps...> {
            static_assert(N > 0, "snicholls::fsm: queue_capacity<N> needs N > 0");
            static constexpr std::size_t capacity = N;
        };
        template <class... Ps>
        struct config<testing, Ps...> : config<Ps...> {
            static constexpr bool testing_enabled = true;
        };

        template <class Def>
        using table_of_t = std::remove_cvref_t<decltype(std::declval<Def&>()())>;

        template <class T>
        inline constexpr bool is_state_v = false;
        template <class S, bool I, bool Hs>
        inline constexpr bool is_state_v<state_t<S, I, Hs>> = true;

        template <class T>
        struct sub_definition {
            using type = std::remove_cv_t<T>;
        };
        template <class S, bool I, bool Hs>
        struct sub_definition<state_t<S, I, Hs>> {
            using type = S;
        };
        template <class S, bool I, bool Hs>
        struct sub_definition<const state_t<S, I, Hs>> {
            using type = S;
        };

        // ---- actions: what they queue ----------------------------------------------------------------------

        template <class A>
        inline constexpr bool contains_defer_v = std::is_same_v<A, defer_t>;
        template <class A, class B>
        inline constexpr bool contains_defer_v<seq<A, B>> = contains_defer_v<A> || contains_defer_v<B>;
        template <class G, class A>
        inline constexpr bool contains_defer_v<eval_t<G, A>> = contains_defer_v<A>;

        template <class A>
        struct processed_events {
            using type = meta::type_list<>;
        };
        template <class E>
        struct processed_events<process_t<E>> {
            using type = meta::type_list<E>;
        };
        template <class A, class B>
        struct processed_events<seq<A, B>> {
            using type = meta::concat_t<typename processed_events<A>::type, typename processed_events<B>::type>;
        };
        template <class G, class A>
        struct processed_events<eval_t<G, A>> : processed_events<A> {};

        // ---- what a callable asks for --------------------------------------------------------------------------

        struct unknown_payload {};

        template <class F, class P>
        struct signature_of {};
        template <class F, class P>
            requires requires { typename callable_traits<F>::arguments; }
        struct signature_of<F, P> {
            using args = typename callable_traits<F>::arguments;
        };
        template <class F, class P>
            requires(!requires { typename callable_traits<F>::arguments; } && !std::is_same_v<P, unknown_payload> &&
                     requires { &F::template operator()<P>; })
        struct signature_of<F, P> {
            using args = typename callable_traits<decltype(&F::template operator()<P>)>::arguments;
        };

        template <class F, class P>
        concept has_signature = requires { typename signature_of<F, P>::args; };

        template <class F, class P>
        struct leaf_params {
            using type = meta::type_list<>;
        };
        template <class F, class P>
            requires(!std::is_invocable_v<F&> && !std::is_member_function_pointer_v<F> && has_signature<F, P>)
        struct leaf_params<F, P> {
            using type = meta::transform_t<typename signature_of<F, P>::args, std::remove_cvref_t>;
        };
        template <class F, class P>
            requires std::is_member_function_pointer_v<F>
        struct leaf_params<F, P> {
            using type = meta::push_front_t<meta::transform_t<typename callable_traits<F>::arguments, std::remove_cvref_t>,
                                            typename callable_traits<F>::class_type>;
        };

        template <class F, class P>
        struct params_of : leaf_params<F, P> {};
        template <class P>
        struct params_of<always, P> {
            using type = meta::type_list<>;
        };
        template <class P>
        struct params_of<none, P> {
            using type = meta::type_list<>;
        };
        template <class P>
        struct params_of<defer_t, P> {
            using type = meta::type_list<>;
        };
        template <class E, class P>
        struct params_of<process_t<E>, P> {
            using type = meta::type_list<>;
        };
        template <class F, class P>
        struct params_of<wrapped<F>, P> : leaf_params<F, P> {};
        template <class G, class P>
        struct params_of<not_<G>, P> : params_of<G, P> {};
        template <class A, class B, class P>
        struct params_of<and_<A, B>, P> {
            using type = meta::concat_t<typename params_of<A, P>::type, typename params_of<B, P>::type>;
        };
        template <class A, class B, class P>
        struct params_of<or_<A, B>, P> : params_of<and_<A, B>, P> {};
        template <class A, class B, class P>
        struct params_of<seq<A, B>, P> : params_of<and_<A, B>, P> {};
        template <class G, class A, class P>
        struct params_of<eval_t<G, A>, P> : params_of<and_<G, A>, P> {};

        // The object a row's callables receive as "the event", for learning their parameters.
        template <class E>
        struct payload_of {
            using type = E;
        };
        template <template <class> class Tag, class T>
            requires is_special_event_v<Tag<T>>
        struct payload_of<Tag<T>> {
            using type = std::conditional_t<std::is_same_v<T, _>, unknown_payload, T>;
        };
        template <>
        struct payload_of<exception_tag<_>> {
            using type = _;
        };

        // ---- the analysis of one table ---------------------------------------------------------------------

        template <std::size_t N>
        struct index_list {
            std::array<std::size_t, N> at{};
            std::size_t size = 0;
        };

        inline constexpr std::size_t npos = static_cast<std::size_t>(-1);

        template <class R>
        using src_of = typename R::source::type;
        template <class R>
        using targets_of = std::conditional_t<std::is_same_v<typename R::target, internal_target>, meta::type_list<>,
                                              meta::type_list<typename R::target>>;

        template <class S>
        struct enum_of_state {
            using type = void;
        };
        template <auto V>
        struct enum_of_state<enumerator_state<V>> {
            using type = decltype(V);
        };

        template <class Def>
        struct declared_enum {
            using type = void;
        };
        template <class Def>
            requires requires { typename Def::state_enum; }
        struct declared_enum<Def> {
            using type = typename Def::state_enum;
        };

        template <class States, class Declared>
        struct level_enum {
            using type = Declared;
        };
        template <class... Ss>
        struct level_enum<meta::type_list<Ss...>, void> {
            using kinds = meta::unique_t<meta::type_list<typename enum_of_state<Ss>::type...>>;
            using type = std::conditional_t<meta::size_v<kinds> == 1, meta::front_t<kinds>, void>;
        };

        // The enumerator a state maps to: itself for enum_state, by name for any other state.
        template <class S, class E>
        struct enumerator_for {
            static constexpr std::optional<E> value = enum_cast<E>(state_info<S>::name);
        };
        template <auto V, class E>
        struct enumerator_for<enumerator_state<V>, E> {
            static constexpr std::optional<E> value =
                std::is_same_v<decltype(V), E> ? std::optional<E>(static_cast<E>(V)) : std::optional<E>();
        };

        // Where a state's enumerator stands in its enumeration: the order of states in a machine with one.
        template <class S, class E>
        struct enum_rank {
            static constexpr std::size_t value =
                enumerator_for<S, E>::value ? enum_index(*enumerator_for<S, E>::value).value_or(npos) : npos;
        };
        template <class S>
        struct enum_rank<S, void> {
            static constexpr std::size_t value = 0;
        };

        template <class Def, class Rows = typename table_of_t<Def>::rows>
        struct table_info;

        template <class Def, class... Rs>
        struct table_info<Def, meta::type_list<Rs...>> {
            using rows = meta::type_list<Rs...>;
            static constexpr std::size_t row_count = sizeof...(Rs);

            // The states in order of appearance; with a state enumeration, in the enumeration's order instead, so that
            // a state's index is its enumerator's position and current() needs no table when the enumeration is dense.
            using appearance = meta::unique_t<meta::concat_t<meta::push_front_t<targets_of<Rs>, src_of<Rs>>...>>;
            using state_enum = typename level_enum<appearance, typename declared_enum<Def>::type>::type;
            template <class A, class B>
            using by_enumerator = std::bool_constant<(enum_rank<A, state_enum>::value < enum_rank<B, state_enum>::value)>;
            using states = std::conditional_t<std::is_void_v<state_enum>, appearance, meta::sort_t<appearance, by_enumerator>>;
            static constexpr std::size_t state_count = meta::size_v<states>;
            using state_id = std::conditional_t<(state_count <= 255), std::uint8_t, std::uint16_t>;
            template <class S>
            static constexpr std::size_t id_of = meta::index_of_v<states, S>;

            using initial_states = meta::unique_t<meta::concat_t<
                std::conditional_t<Rs::source::initial, meta::type_list<src_of<Rs>>, meta::type_list<>>...>>;
            static constexpr std::size_t regions = meta::size_v<initial_states>;

            template <class... Is>
            static constexpr std::array<state_id, sizeof...(Is)> ids_of(meta::type_list<Is...>) noexcept
            {
                return {static_cast<state_id>(id_of<Is>)...};
            }
            static constexpr std::array<state_id, regions> initial_ids = ids_of(initial_states{});

            template <class S>
            static constexpr bool history_of = ((std::is_same_v<src_of<Rs>, S> && Rs::source::history) || ... || false);
            template <class... Is>
            static constexpr std::array<bool, sizeof...(Is)> history_flags(meta::type_list<Is...>) noexcept
            {
                return {history_of<Is>...};
            }
            static constexpr std::array<bool, regions> history = history_flags(initial_states{});

            using events = meta::unique_t<meta::type_list<typename Rs::event_type...>>;
            template <class E>
            using is_plain_event = std::bool_constant<!std::is_same_v<E, anonymous> && !is_special_event_v<E>>;
            using plain_events = meta::filter_t<events, is_plain_event>;

            static constexpr bool has_anonymous = meta::contains_v<events, anonymous>;
            static constexpr bool has_unexpected = (is_unexpected_v<typename Rs::event_type> || ... || false);
            static constexpr bool has_exceptions = (is_exception_v<typename Rs::event_type> || ... || false);

            template <class E>
            struct exception_type {
                using type = meta::type_list<>;
            };
            template <class T>
            struct exception_type<exception_tag<T>> {
                using type = std::conditional_t<std::is_same_v<T, _>, meta::type_list<>, meta::type_list<T>>;
            };
            using exception_types = meta::unique_t<meta::concat_t<typename exception_type<typename Rs::event_type>::type...>>;

            template <class S>
            using is_composite = std::bool_constant<machine_definition<S>>;
            using composites = meta::filter_t<states, is_composite>;

            using defer_events = meta::unique_t<meta::concat_t<
                std::conditional_t<contains_defer_v<typename Rs::action_type>, meta::type_list<typename Rs::event_type>,
                                   meta::type_list<>>...>>;
            using process_events = meta::unique_t<meta::concat_t<typename processed_events<typename Rs::action_type>::type...>>;

            // Rows from state S on the row event E, in table order.
            template <class S, class E>
            static consteval index_list<row_count> match() noexcept
            {
                constexpr std::array<bool, row_count> mask{(std::is_same_v<src_of<Rs>, S> && std::is_same_v<typename Rs::event_type, E>)...};
                index_list<row_count> out;
                for (std::size_t i = 0; i < row_count; ++i)
                    if (mask[i])
                        out.at[out.size++] = i;
                return out;
            }
            template <class S, class E>
            static constexpr index_list<row_count> matching = match<S, E>();
            template <class S, class E, std::size_t... K>
            static auto as_sequence(std::index_sequence<K...>) -> std::index_sequence<matching<S, E>.at[K]...>;
            template <class S, class E>
            using rows_for = decltype(as_sequence<S, E>(std::make_index_sequence<matching<S, E>.size>{}));
            template <class S, class E>
            static constexpr bool has_rows = matching<S, E>.size != 0;

            template <class S>
            static constexpr bool has_entry = ((std::is_same_v<src_of<Rs>, S> && is_entry_v<typename Rs::event_type>) || ... || false);
            template <class S>
            static constexpr bool has_exit = ((std::is_same_v<src_of<Rs>, S> && is_exit_v<typename Rs::event_type>) || ... || false);
            template <class S>
            static constexpr bool has_unexpected_rows =
                ((std::is_same_v<src_of<Rs>, S> && is_unexpected_v<typename Rs::event_type>) || ... || false);

            // The parameters of every guard and action, less the event: the dependencies of this table.
            template <class R>
            using row_params = meta::remove_t<
                meta::concat_t<typename params_of<typename R::guard_type, typename payload_of<typename R::event_type>::type>::type,
                               typename params_of<typename R::action_type, typename payload_of<typename R::event_type>::type>::type>,
                typename payload_of<typename R::event_type>::type>;
            using params = meta::unique_t<meta::concat_t<row_params<Rs>...>>;

            // ---- the compile-time checks ----------------------------------------------------------------------

            template <class R>
            static constexpr std::size_t target_id = std::is_same_v<typename R::target, internal_target> ? npos : id_of<typename R::target>;
            static constexpr std::array<std::size_t, row_count> src_ids{id_of<src_of<Rs>>...};
            static constexpr std::array<std::size_t, row_count> dst_ids{target_id<Rs>...};
            static constexpr std::array<std::size_t, row_count> event_ids{meta::index_of_v<events, typename Rs::event_type>...};
            static constexpr std::array<bool, row_count> guarded{!std::is_same_v<typename Rs::guard_type, always>...};
            static constexpr std::array<bool, row_count> entry_row{is_entry_v<typename Rs::event_type>...};

            // A row that can never run: an earlier row leaves the same state on the same event without a guard.
            static consteval std::size_t find_shadowed() noexcept
            {
                for (std::size_t j = 0; j < row_count; ++j)
                    for (std::size_t i = 0; i < j; ++i)
                        if (src_ids[i] == src_ids[j] && event_ids[i] == event_ids[j] && !guarded[i])
                            return j;
                return npos;
            }
            static constexpr std::size_t shadowed_row = find_shadowed();

            static consteval std::array<bool, state_count> reach() noexcept
            {
                std::array<bool, state_count> seen{};
                for (const auto id : initial_ids)
                    seen[id] = true;
                for (bool grew = true; grew;) {
                    grew = false;
                    for (std::size_t i = 0; i < row_count; ++i)
                        if (dst_ids[i] != npos && seen[src_ids[i]] && !seen[dst_ids[i]])
                            grew = seen[dst_ids[i]] = true;
                }
                return seen;
            }
            static constexpr std::array<bool, state_count> reached = reach();
            static consteval std::size_t find_unreachable() noexcept
            {
                for (std::size_t s = 0; s < state_count; ++s)
                    if (!reached[s])
                        return s;
                return npos;
            }
            static constexpr std::size_t unreachable_state = find_unreachable();

            static constexpr std::size_t terminal_id = meta::index_of_v<states, terminate_state>;
            static consteval std::size_t find_exit_from_terminal() noexcept
            {
                for (std::size_t i = 0; i < row_count; ++i)
                    if (terminal_id != npos && src_ids[i] == terminal_id && !entry_row[i])
                        return i;
                return npos;
            }
            static constexpr std::size_t exit_from_terminal = find_exit_from_terminal();

            static constexpr bool special_rows_internal =
                (((!is_entry_v<typename Rs::event_type> && !is_exit_v<typename Rs::event_type>) ||
                  std::is_same_v<typename Rs::target, internal_target>) && ... && true);
            static constexpr bool defer_only_on_events =
                ((!contains_defer_v<typename Rs::action_type> || is_plain_event<typename Rs::event_type>::value) && ... && true);

            // ---- the state enumeration -------------------------------------------------------------------------

            template <class... Ss>
            static constexpr std::array<std::string_view, sizeof...(Ss)> names_of(meta::type_list<Ss...>) noexcept
            {
                return {state_info<Ss>::name...};
            }
            static constexpr std::array<std::string_view, state_count> state_names = names_of(states{});
        };

        template <class Info, class E, bool = !std::is_void_v<E>>
        struct enum_map {
            static constexpr std::size_t missing = npos;
        };
        template <class Info, class E>
        struct enum_map<Info, E, true> {
            template <class... Ss>
            static constexpr std::array<std::optional<E>, sizeof...(Ss)> of(meta::type_list<Ss...>) noexcept
            {
                return {enumerator_for<Ss, E>::value...};
            }
            static constexpr auto found = of(typename Info::states{});
            static consteval std::size_t find_missing() noexcept
            {
                for (std::size_t i = 0; i < found.size(); ++i)
                    if (!found[i])
                        return i;
                return npos;
            }
            static constexpr std::size_t missing = find_missing();
            static consteval std::array<E, Info::state_count> make_values() noexcept
            {
                std::array<E, Info::state_count> out{};
                for (std::size_t i = 0; i < found.size(); ++i)
                    out[i] = found[i].value_or(E{});
                return out;
            }
            static constexpr std::array<E, Info::state_count> values = make_values();
            // The state of each enumerator (by its position in the enumeration), or npos: lookup is two loads.
            static consteval std::array<std::size_t, enum_count_v<E>> make_ids() noexcept
            {
                std::array<std::size_t, enum_count_v<E>> out{};
                for (auto& id : out)
                    id = npos;
                for (std::size_t i = 0; i < values.size(); ++i)
                    if (const auto at = enum_index(values[i]))
                        out[*at] = i;
                return out;
            }
            static constexpr std::array<std::size_t, enum_count_v<E>> ids = make_ids();
            // Every enumerator from 0 up is a state, in order: a state's index is its enumerator's value.
            static consteval bool is_identity() noexcept
            {
                for (std::size_t i = 0; i < values.size(); ++i)
                    if (static_cast<std::size_t>(static_cast<std::underlying_type_t<E>>(values[i])) != i)
                        return false;
                return true;
            }
            static constexpr bool identity = is_identity();
            static constexpr std::size_t id_of(E e) noexcept
            {
                if constexpr (identity) {
                    const auto v = static_cast<std::size_t>(static_cast<std::underlying_type_t<E>>(e));
                    return v < values.size() ? v : npos;
                } else {
                    const auto at = enum_index(e);
                    return at ? ids[*at] : npos;
                }
            }
            static constexpr E to_enum(std::size_t id) noexcept
            {
                if constexpr (identity)
                    return static_cast<E>(static_cast<std::underlying_type_t<E>>(id));
                else
                    return values[id];
            }
        };

        // ---- the errors, named: the template argument in the compiler's note is the offending state or row ----

        template <class Src, class Event>
        struct shadowed_row_error {
            static_assert(dependent_false<Src>, "snicholls::fsm: a row can never run: an earlier row leaves the same state "
                                                "(first template argument) on the same event (second) without a guard");
            static constexpr bool ok = false;
        };
        template <class S>
        struct unreachable_state_error {
            static_assert(dependent_false<S>, "snicholls::fsm: no row reaches this state (the template argument) from an "
                                              "initial state");
            static constexpr bool ok = false;
        };
        template <class R>
        struct exit_from_terminal_error {
            static_assert(dependent_false<R>, "snicholls::fsm: a row (the template argument) leaves X, the terminal state");
            static constexpr bool ok = false;
        };
        template <class S, class E>
        struct missing_enumerator_error {
            static_assert(dependent_false<S>, "snicholls::fsm: a state (first template argument) has no enumerator of the "
                                              "machine's state enumeration (second) with its name");
            static constexpr bool ok = false;
        };
        struct no_error {
            static constexpr bool ok = true;
        };

        template <class Info, std::size_t Row = Info::shadowed_row>
        struct check_shadowed {
            using R = meta::at_t<typename Info::rows, Row>;
            using type = shadowed_row_error<src_of<R>, typename R::event_type>;
        };
        template <class Info>
        struct check_shadowed<Info, npos> {
            using type = no_error;
        };
        template <class Info, std::size_t S = Info::unreachable_state>
        struct check_reached {
            using type = unreachable_state_error<meta::at_t<typename Info::states, S>>;
        };
        template <class Info>
        struct check_reached<Info, npos> {
            using type = no_error;
        };
        template <class Info, std::size_t Row = Info::exit_from_terminal>
        struct check_terminal {
            using type = exit_from_terminal_error<meta::at_t<typename Info::rows, Row>>;
        };
        template <class Info>
        struct check_terminal<Info, npos> {
            using type = no_error;
        };
        template <class Info, class E, std::size_t S = enum_map<Info, E>::missing>
        struct check_enum {
            using type = missing_enumerator_error<meta::at_t<typename Info::states, S>, E>;
        };
        template <class Info, class E>
        struct check_enum<Info, E, npos> {
            using type = no_error;
        };

        // ---- run-time pieces -------------------------------------------------------------------------------

        // A dependency: the caller's object (an lvalue passed to the constructor) or one the machine owns.
        template <class T>
        struct dep_slot {
            T* external = nullptr;
            std::optional<T> owned{};
            constexpr T& get() noexcept { return external ? *external : *owned; }
        };

        template <class T>
        constexpr dep_slot<T> make_slot()
        {
            static_assert(std::is_default_constructible_v<T>,
                          "snicholls::fsm: a dependency that is not default-constructible must be passed to the constructor");
            return {nullptr, std::optional<T>(std::in_place)};
        }
        template <class T, class A, class... As>
            requires std::is_same_v<std::remove_cvref_t<A>, T> && std::is_lvalue_reference_v<A> &&
            (!std::is_const_v<std::remove_reference_t<A>>)
        constexpr dep_slot<T> make_slot(A&& a, As&&...)
        {
            return {&a, std::nullopt};
        }
        template <class T, class A, class... As>
            requires std::is_same_v<std::remove_cvref_t<A>, T> &&
            (!std::is_lvalue_reference_v<A> || std::is_const_v<std::remove_reference_t<A>>)
        constexpr dep_slot<T> make_slot(A&& a, As&&...)
        {
            return {nullptr, std::optional<T>(std::forward<A>(a))};
        }
        template <class T, class A, class... As>
            requires(!std::is_same_v<std::remove_cvref_t<A>, T>)
        constexpr dep_slot<T> make_slot(A&&, As&&... as)
        {
            return make_slot<T>(std::forward<As>(as)...);
        }

        template <class T>
        struct dep_leaf {
            dep_slot<T> slot;
        };

        template <class List>
        struct pool;
        template <class... Ts>
        struct pool<meta::type_list<Ts...>> : dep_leaf<Ts>... {
            using types = meta::type_list<Ts...>;
            template <class T>
            static constexpr bool has = (std::is_same_v<T, Ts> || ... || false);

            constexpr pool() : dep_leaf<Ts>{make_slot<Ts>()}... {}
            template <class... Args>
            constexpr explicit pool(Args&&... args) : dep_leaf<Ts>{make_slot<Ts>(std::forward<Args>(args)...)}...
            {
            }

            template <class T>
            constexpr T& get() noexcept
            {
                return static_cast<dep_leaf<T>&>(*this).slot.get();
            }
        };

        template <class Pool, class Logger>
        struct context {
            Pool& deps;
            Logger* log;
            std::uint8_t flags = 0;
            static constexpr std::uint8_t rejected = 1, threw = 2, looped = 4, overflowed = 8;

            template <class T>
            constexpr T& dep() noexcept
            {
                static_assert(Pool::template has<T>, "snicholls::fsm: a guard or action asks for a dependency the machine "
                                                     "does not hold (see the template argument); pass it to the constructor");
                return deps.template get<T>();
            }
        };

        template <class Def, class Ctx, class E>
        constexpr void log_process_event(Ctx& ctx, const E& e)
        {
            if constexpr (requires { ctx.log->template log_process_event<Def>(e); })
                ctx.log->template log_process_event<Def>(e);
        }
        template <class Def, class Ctx, class G, class P>
        constexpr void log_guard(Ctx& ctx, const G& g, const P& p, bool result)
        {
            if constexpr (requires { ctx.log->template log_guard<Def>(g, p, result); })
                ctx.log->template log_guard<Def>(g, p, result);
        }
        template <class Def, class Ctx, class A, class P>
        constexpr void log_action(Ctx& ctx, const A& a, const P& p)
        {
            if constexpr (requires { ctx.log->template log_action<Def>(a, p); })
                ctx.log->template log_action<Def>(a, p);
        }
        template <class Def, class Ctx, class Src, class Dst>
        constexpr void log_state_change(Ctx& ctx, const Src& src, const Dst& dst)
        {
            if constexpr (requires { ctx.log->template log_state_change<Def>(src, dst); })
                ctx.log->template log_state_change<Def>(src, dst);
        }

        // A fixed-capacity queue of events: no allocation, the order kept.
        // Tag tells the defer queue from the process queue, so that two empty ones need no address of their own.
        template <class Tag, class Events, std::size_t N>
        struct event_queue {
            constexpr bool empty() const noexcept { return true; }
        };
        template <class Tag, class... Es, std::size_t N>
            requires(sizeof...(Es) > 0)
        struct event_queue<Tag, meta::type_list<Es...>, N> {
            using item = std::variant<std::monostate, Es...>;
            std::array<item, N> items{};
            std::size_t size = 0;
            bool replaying = false;
            bool again = false;

            constexpr bool empty() const noexcept { return size == 0; }
            template <class E>
            constexpr bool push(const E& e)
            {
                if (size == N)
                    return false;
                items[size++] = item(std::in_place_type<E>, e);
                return true;
            }
            constexpr void erase(std::size_t i)
            {
                for (std::size_t k = i + 1; k < size; ++k)
                    items[k - 1] = std::move(items[k]);
                items[--size] = item{};
            }
            constexpr void clear() noexcept
            {
                for (std::size_t k = 0; k < size; ++k)
                    items[k] = item{};
                size = 0;
            }
        };

        template <class M, class Ctx>
        struct replay {
            M* machine;
            Ctx* ctx;
            constexpr bool operator()(std::monostate) const noexcept { return false; }
            template <class E>
            constexpr bool operator()(const E& e) const
            {
                return machine->guarded(e, *ctx);
            }
        };

        template <class Def, class Cfg>
        class machine;

        template <class C, class M>
        struct sub_leaf {
            M value;
        };
        template <class Cfg, class List>
        struct sub_storage;
        template <class Cfg, class... Cs>
        struct sub_storage<Cfg, meta::type_list<Cs...>> : sub_leaf<Cs, machine<Cs, Cfg>>... {};

        struct outcome {
            bool handled = false;
            bool queued_ok = true;
        };

        // One level of a machine: a table, its regions' states, its sub-machines and its queues.
        template <class Def, class Cfg>
        class machine {
        public:
            using def_type = Def;
            using table_type = table_of_t<Def>;
            using info = table_info<Def>;
            using state_id = typename info::state_id;
            using states = typename info::states;
            using composites = typename info::composites;
            using state_enum = typename info::state_enum;
            using enums = enum_map<info, state_enum>;
            static constexpr std::size_t regions = info::regions;

            static_assert(regions > 0, "snicholls::fsm: the table has no initial state: mark one with * (one per region)");
            static_assert(check_shadowed<info>::type::ok, "snicholls::fsm: a row can never run (see the note above)");
            static_assert(check_reached<info>::type::ok, "snicholls::fsm: a state is unreachable (see the note above)");
            static_assert(check_terminal<info>::type::ok, "snicholls::fsm: a row leaves X (see the note above)");
            static_assert(check_enum<info, state_enum>::type::ok, "snicholls::fsm: a state has no enumerator (see above)");
            static_assert(info::special_rows_internal, "snicholls::fsm: on_entry and on_exit rows have no target state");
            static_assert(info::defer_only_on_events, "snicholls::fsm: defer is an action of an event row");
#if !defined(__cpp_exceptions)
            static_assert(!info::has_exceptions, "snicholls::fsm: exception<E> rows need exceptions enabled");
#endif

            template <class S>
            static constexpr state_id id_of = static_cast<state_id>(info::template id_of<S>);
            template <class S>
            static constexpr bool is_composite = machine_definition<S>;

            // How many regions a snapshot holds: these and every sub-machine's.
            template <class... Cs>
            static constexpr std::size_t sub_slots(meta::type_list<Cs...>) noexcept
            {
                return (machine<Cs, Cfg>::slot_count + ... + 0);
            }
            static constexpr std::size_t slot_count = regions + sub_slots(composites{});

            // Whether this level (with its sub-machines) does anything with the plain event E.
            template <class... Cs>
            static constexpr bool subs_know(meta::type_list<Cs...>, auto tag) noexcept
            {
                using E = typename decltype(tag)::type;
                return (machine<Cs, Cfg>::template knows<E> || ... || false);
            }
            template <class E>
            struct tag_of {
                using type = E;
            };
            template <class E>
            static constexpr bool knows = meta::contains_v<typename info::plain_events, E> || subs_know(composites{}, tag_of<E>{});
            template <class... Cs>
            static constexpr bool subs_anonymous(meta::type_list<Cs...>) noexcept
            {
                return (machine<Cs, Cfg>::reacts_anonymous || ... || false);
            }
            static constexpr bool reacts_anonymous = info::has_anonymous || subs_anonymous(composites{});

            // Whether state S reacts to the row event E (a plain event, anonymous, or an exception marker).
            template <class S, class E>
            static constexpr bool reacts()
            {
                if constexpr (std::is_same_v<E, anonymous>) {
                    if constexpr (is_composite<S>)
                        return info::template has_rows<S, E> || machine<S, Cfg>::reacts_anonymous;
                    else
                        return info::template has_rows<S, E>;
                } else if constexpr (is_exception_v<E>)
                    return info::template has_rows<S, E>;
                else if constexpr (is_composite<S>)
                    return info::template has_rows<S, E> || machine<S, Cfg>::template knows<E>;
                else
                    return info::template has_rows<S, E> || info::has_unexpected;
            }
            template <class E>
            struct reacts_to {
                template <class S>
                using fn = std::bool_constant<reacts<S, E>()>;
            };
            template <class E>
            using candidates = meta::filter_t<states, reacts_to<E>::template fn>;
            template <class S>
            using enters = std::bool_constant<info::template has_entry<S> || is_composite<S>>;
            template <class S>
            using exits = std::bool_constant<info::template has_exit<S> || is_composite<S>>;
            template <class S>
            using unexpects = std::bool_constant<info::template has_unexpected_rows<S>>;

            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS Def def_;
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS table_type table_;
            std::array<state_id, regions> current_;
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS sub_storage<Cfg, composites> subs_;
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS event_queue<defer_t, typename info::defer_events, Cfg::capacity> deferred_;
            SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS event_queue<anonymous, typename info::process_events, Cfg::capacity> queued_;

            // A table that holds data (a captured `this`, a function pointer) is rebuilt from the copied definition.
            static constexpr bool rebuild = !std::is_empty_v<table_type>;

            constexpr explicit machine(Def d) : def_(std::move(d)), table_(def_()), current_(info::initial_ids) {}
            constexpr machine()
                requires std::is_default_constructible_v<Def>
                : machine(Def{})
            {
            }
            constexpr machine(const machine&)
                requires(!rebuild)
            = default;
            constexpr machine(const machine& o)
                requires rebuild
                : def_(o.def_), table_(def_()), current_(o.current_), subs_(o.subs_), deferred_(o.deferred_), queued_(o.queued_)
            {
            }
            constexpr machine(machine&&)
                requires(!rebuild)
            = default;
            constexpr machine(machine&& o)
                requires rebuild
                : def_(std::move(o.def_)), table_(def_()), current_(o.current_), subs_(std::move(o.subs_)),
                  deferred_(std::move(o.deferred_)), queued_(std::move(o.queued_))
            {
            }
            constexpr machine& operator=(const machine&)
                requires(!rebuild)
            = default;
            constexpr machine& operator=(const machine& o)
                requires rebuild
            {
                def_ = o.def_;
                current_ = o.current_;
                subs_ = o.subs_;
                deferred_ = o.deferred_;
                queued_ = o.queued_;
                return *this;
            }
            constexpr machine& operator=(machine&&)
                requires(!rebuild)
            = default;
            constexpr machine& operator=(machine&& o)
                requires rebuild
            {
                def_ = std::move(o.def_);
                current_ = o.current_;
                subs_ = std::move(o.subs_);
                deferred_ = std::move(o.deferred_);
                queued_ = std::move(o.queued_);
                return *this;
            }
            constexpr ~machine() = default;

            template <class C>
            constexpr machine<C, Cfg>& sub() noexcept
            {
                return static_cast<sub_leaf<C, machine<C, Cfg>>&>(subs_).value;
            }
            template <class C>
            constexpr const machine<C, Cfg>& sub() const noexcept
            {
                return static_cast<const sub_leaf<C, machine<C, Cfg>>&>(subs_).value;
            }

            // ---- queries -----------------------------------------------------------------------------------

            template <class Sub, class... Cs>
            static constexpr bool holds_in(meta::type_list<Cs...>) noexcept
            {
                return (machine<Cs, Cfg>::template holds<Sub> || ... || false);
            }
            template <class Sub>
            static constexpr bool holds = std::is_same_v<Sub, Def> || machine::template holds_in<Sub>(composites{});
            template <class Sub>
            struct holding {
                template <class C>
                using fn = std::bool_constant<machine<C, Cfg>::template holds<Sub>>;
            };

            // The level whose definition is Sub: this one, or one below.
            template <class Sub>
            constexpr auto& level() noexcept
            {
                static_assert(holds<Sub>, "snicholls::fsm: the machine has no sub-machine with this definition");
                if constexpr (std::is_same_v<Sub, Def>)
                    return *this;
                else
                    return sub<meta::find_if_t<composites, holding<Sub>::template fn, Def>>().template level<Sub>();
            }
            template <class Sub>
            constexpr const auto& level() const noexcept
            {
                static_assert(holds<Sub>, "snicholls::fsm: the machine has no sub-machine with this definition");
                if constexpr (std::is_same_v<Sub, Def>)
                    return *this;
                else
                    return sub<meta::find_if_t<composites, holding<Sub>::template fn, Def>>().template level<Sub>();
            }

            constexpr bool is_terminated() const noexcept
            {
                if constexpr (info::terminal_id == npos)
                    return false;
                else {
                    for (const auto s : current_)
                        if (s != info::terminal_id)
                            return false;
                    return true;
                }
            }

            template <class... Ss>
            constexpr bool is_states() const noexcept
            {
                static_assert(((meta::index_of_v<states, Ss> != npos) && ...), "snicholls::fsm: is(): not a state of this machine");
                static_assert(sizeof...(Ss) == 1 || sizeof...(Ss) == regions,
                              "snicholls::fsm: is() takes one state (any region) or one per region");
                if constexpr (sizeof...(Ss) == 1 && regions != 1) {
                    for (const auto s : current_)
                        if (((s == id_of<Ss>) && ...))
                            return true;
                    return false;
                } else {
                    std::size_t r = 0;
                    return ((current_[r++] == id_of<Ss>) && ...);
                }
            }

            template <class... Ss>
            constexpr void set_states() noexcept
            {
                static_assert(((meta::index_of_v<states, Ss> != npos) && ...), "snicholls::fsm: not a state of this machine");
                static_assert(sizeof...(Ss) == regions, "snicholls::fsm: set_current_states takes one state per region");
                std::size_t r = 0;
                ((current_[r++] = id_of<Ss>), ...);
                clear_queues();
            }

            // f(state_t<S>{}) for the current state of each region.
            template <class F>
            constexpr void visit(F&& f) const
            {
                for (std::size_t r = 0; r < regions; ++r)
                    on_current<states>(r, [&]<class S>() { f(state_t<S>{}); });
            }

            template <class List, class F>
            constexpr bool on_current(std::size_t r, F&& f) const
            {
                return [&]<class... Ss>(meta::type_list<Ss...>) {
                    return ((current_[r] == id_of<Ss> ? (f.template operator()<Ss>(), true) : false) || ... || false);
                }(List{});
            }

            constexpr state_enum current(std::size_t region) const noexcept
                requires(!std::is_void_v<state_enum>)
            {
                return enums::to_enum(current_[region]);
            }

            template <class... Es>
            constexpr result<void, fsm_errc> restore(Es... es) noexcept
                requires(!std::is_void_v<state_enum>)
            {
                static_assert(sizeof...(Es) == regions, "snicholls::fsm: restore takes one state per region");
                const std::array<std::size_t, regions> ids{enums::id_of(static_cast<state_enum>(es))...};
                for (const auto id : ids)
                    if (id == npos)
                        return err(fsm_errc::not_a_state);
                for (std::size_t r = 0; r < regions; ++r)
                    current_[r] = static_cast<state_id>(ids[r]);
                clear_queues();
                return ok();
            }

            // ---- persistence: every region, then every sub-machine's ----------------------------------------

            static constexpr std::int64_t wire_value(std::size_t id) noexcept
            {
                if constexpr (std::is_void_v<state_enum>)
                    return static_cast<std::int64_t>(id);
                else
                    return static_cast<std::int64_t>(static_cast<std::underlying_type_t<state_enum>>(enums::values[id]));
            }
            static constexpr std::size_t id_from_wire(const wire_state& w) noexcept
            {
                for (std::size_t id = 0; id < info::state_count; ++id) {
                    if constexpr (std::is_void_v<state_enum>) {
                        if (info::state_names[id] == w.name)
                            return id;
                    } else if (wire_value(id) == w.value)
                        return id;
                }
                return npos;
            }

            constexpr void write_slots(wire_state* out) const noexcept
            {
                for (std::size_t r = 0; r < regions; ++r)
                    out[r] = {wire_value(current_[r]), info::state_names[current_[r]]};
                std::size_t at = regions;
                [&]<class... Cs>(meta::type_list<Cs...>) {
                    ((sub<Cs>().write_slots(out + at), at += machine<Cs, Cfg>::slot_count), ...);
                }(composites{});
            }
            static constexpr bool valid_slots(const wire_state* in) noexcept
            {
                for (std::size_t r = 0; r < regions; ++r)
                    if (id_from_wire(in[r]) == npos)
                        return false;
                std::size_t at = regions;
                return [&]<class... Cs>(meta::type_list<Cs...>) {
                    return ((machine<Cs, Cfg>::valid_slots(in + at) && (at += machine<Cs, Cfg>::slot_count, true)) && ... && true);
                }(composites{});
            }
            constexpr void read_slots(const wire_state* in) noexcept
            {
                for (std::size_t r = 0; r < regions; ++r)
                    current_[r] = static_cast<state_id>(id_from_wire(in[r]));
                clear_queues();
                std::size_t at = regions;
                [&]<class... Cs>(meta::type_list<Cs...>) {
                    ((sub<Cs>().read_slots(in + at), at += machine<Cs, Cfg>::slot_count), ...);
                }(composites{});
            }

            // ---- processing --------------------------------------------------------------------------------

            constexpr void clear_queues() noexcept
            {
                if constexpr (!meta::empty_v<typename info::defer_events>)
                    deferred_.clear();
                if constexpr (!meta::empty_v<typename info::process_events>)
                    queued_.clear();
            }

            // Back to the initial states, except a region whose initial state is a history state.
            constexpr void reset() noexcept
            {
                for (std::size_t r = 0; r < regions; ++r)
                    if (!info::history[r])
                        current_[r] = info::initial_ids[r];
                clear_queues();
                [&]<class... Is>(meta::type_list<Is...>) {
                    (reset_if_composite<Is>(), ...);
                }(typename info::initial_states{});
            }
            template <class S>
            constexpr void reset_if_composite() noexcept
            {
                if constexpr (is_composite<S>)
                    sub<S>().reset();
            }

            template <class Ctx>
            constexpr void start(Ctx& ctx)
            {
                auto old = current_;
                enter_all(initial{}, ctx);
                static_cast<void>(run_pending(old, ctx));
            }

            template <class E, class Ctx>
            constexpr outcome process_event(const E& e, Ctx& ctx)
            {
                if constexpr (knows<E>)
                    log_process_event<Def>(ctx, e);
                auto old = current_;
                outcome o;
                o.handled = guarded(e, ctx);
                o.queued_ok = run_pending(old, ctx);
                return o;
            }

            template <class E, class Ctx>
            constexpr bool guarded(const E& e, Ctx& ctx)
            {
#if defined(__cpp_exceptions)
                if constexpr (info::has_exceptions) {
                    try {
                        return process_internal(e, ctx);
                    } catch (...) {
                        return handle_exception(ctx, typename info::exception_types{});
                    }
                }
#endif
                return process_internal(e, ctx);
            }

#if defined(__cpp_exceptions)
            template <class Ctx, class T, class... Ts>
            constexpr bool handle_exception(Ctx& ctx, meta::type_list<T, Ts...>)
            {
                try {
                    throw;
                } catch (const T& x) {
                    const bool handled = process_as<exception_tag<T>>(x, ctx);
                    if (!handled)
                        ctx.flags |= Ctx::threw;
                    return handled;
                } catch (...) {
                    return handle_exception(ctx, meta::type_list<Ts...>{});
                }
            }
            template <class Ctx>
            constexpr bool handle_exception(Ctx& ctx, meta::type_list<>)
            {
                const bool handled = process_as<exception_tag<_>>(_{}, ctx);
                if (!handled)
                    ctx.flags |= Ctx::threw;
                return handled;
            }
#endif

            template <class E, class Ctx>
            constexpr bool process_internal(const E& e, Ctx& ctx)
            {
                if constexpr (std::is_same_v<E, anonymous> || knows<E>)
                    return process_as<E>(e, ctx);
                else {
                    if constexpr (info::has_unexpected)
                        for (std::size_t r = 0; r < regions; ++r)
                            static_cast<void>(unexpected(r, e, ctx));
                    return false;
                }
            }

            template <class RowEvent, class P, class Ctx>
            constexpr bool process_as(const P& p, Ctx& ctx)
            {
                return [&]<std::size_t... R>(std::index_sequence<R...>) {
                    bool handled = false;
                    ((handled |= dispatch<RowEvent>(R, p, ctx)), ...);
                    return handled;
                }(std::make_index_sequence<regions>{});
            }

            template <class S, class RowEvent, class P, class Ctx>
            static constexpr bool jump_entry(machine& m, std::size_t r, const P& p, Ctx& ctx)
            {
                if constexpr (reacts<S, RowEvent>())
                    return m.template in_state<S, RowEvent>(r, p, ctx);
                else
                    return false;
            }
            template <class RowEvent, class P, class Ctx, class... Ss>
            static constexpr auto make_jump(meta::type_list<Ss...>) noexcept
            {
                return std::array<bool (*)(machine&, std::size_t, const P&, Ctx&), sizeof...(Ss)>{
                    &machine::template jump_entry<Ss, RowEvent, P, Ctx>...};
            }
            template <class RowEvent, class P, class Ctx>
            static constexpr auto jump = make_jump<RowEvent, P, Ctx>(states{});

            template <class RowEvent, class P, class Ctx>
            constexpr bool dispatch(std::size_t r, const P& p, Ctx& ctx)
            {
                if constexpr (std::is_same_v<typename Cfg::dispatch_type, jump_table>)
                    return jump<RowEvent, P, Ctx>[current_[r]](*this, r, p, ctx);
                else
                    return [&]<class... Ss>(meta::type_list<Ss...>) {
                        bool handled = false;
                        static_cast<void>(
                            ((current_[r] == id_of<Ss> ? (handled = in_state<Ss, RowEvent>(r, p, ctx), true) : false) || ... ||
                             false));
                        return handled;
                    }(candidates<RowEvent>{});
            }

            // The event in region r, whose current state is S.
            template <class S, class RowEvent, class P, class Ctx>
            constexpr bool in_state(std::size_t r, const P& p, Ctx& ctx)
            {
                if constexpr (is_composite<S> && std::is_same_v<RowEvent, anonymous>) {
                    auto& m = sub<S>();
                    if (!m.is_terminated())
                        return m.process_internal(p, ctx);
                    return rows<S, RowEvent>(r, p, ctx);
                } else if constexpr (is_composite<S> && !is_exception_v<RowEvent>) {
                    bool sub_handled = false;
                    if constexpr (machine<S, Cfg>::template knows<RowEvent>) {
                        const auto o = sub<S>().process_event(p, ctx);
                        sub_handled = o.handled && o.queued_ok;
                        if (sub_handled && !sub<S>().is_terminated())
                            return true;
                    }
                    return rows<S, RowEvent>(r, p, ctx) || sub_handled;
                } else if constexpr (info::template has_rows<S, RowEvent>)
                    return rows<S, RowEvent>(r, p, ctx);
                else
                    return unexpected(r, p, ctx);
            }

            template <class S, class RowEvent, class P, class Ctx>
            constexpr bool rows(std::size_t r, const P& p, Ctx& ctx)
            {
                if constexpr (!info::template has_rows<S, RowEvent>)
                    return false;
                else
                    return [&]<std::size_t... I>(std::index_sequence<I...>) {
                        const bool taken = (try_row<I>(r, p, ctx) || ...);
                        if (!taken)
                            ctx.flags |= Ctx::rejected;
                        return taken;
                    }(typename info::template rows_for<S, RowEvent>{});
            }

            template <class P, class Ctx>
            constexpr bool unexpected(std::size_t r, const P& p, Ctx& ctx)
            {
                if constexpr (info::has_unexpected)
                    on_current<meta::filter_t<states, unexpects>>(r, [&]<class S>() {
                        static_cast<void>(special_rows<S, unexpected_tag<P>, unexpected_tag<_>>(r, p, ctx));
                    });
                return false;
            }

            // Rows of a specific marker first (on_entry<E>), then the generic ones (on_entry<_>); the first that runs ends it.
            template <class S, class Specific, class Generic, class P, class Ctx>
            constexpr bool special_rows(std::size_t r, const P& p, Ctx& ctx)
            {
                const auto run = [&]<std::size_t... I>(std::index_sequence<I...>) { return (try_row<I>(r, p, ctx) || ... || false); };
                if constexpr (std::is_same_v<Specific, Generic>)
                    return run(typename info::template rows_for<S, Generic>{});
                else
                    return run(typename info::template rows_for<S, Specific>{}) || run(typename info::template rows_for<S, Generic>{});
            }

            template <class T, class Ctx>
            constexpr void enter_all(const T& trigger, Ctx& ctx)
            {
                for (std::size_t r = 0; r < regions; ++r)
                    on_current<meta::filter_t<states, enters>>(r, [&]<class S>() { enter<S>(r, trigger, ctx); });
            }
            template <class T, class Ctx>
            constexpr void leave_all(const T& trigger, Ctx& ctx)
            {
                for (std::size_t r = 0; r < regions; ++r)
                    on_current<meta::filter_t<states, exits>>(r, [&]<class S>() { leave<S>(r, trigger, ctx); });
            }

            template <class S, class T, class Ctx>
            constexpr void enter(std::size_t r, const T& trigger, Ctx& ctx)
            {
                if constexpr (info::template has_entry<S>)
                    static_cast<void>(special_rows<S, entry_tag<T>, entry_tag<_>>(r, trigger, ctx));
                if constexpr (is_composite<S>) {
                    auto& m = sub<S>();
                    auto old = m.current_;
                    m.enter_all(trigger, ctx);
                    static_cast<void>(m.run_pending(old, ctx));
                }
            }
            template <class S, class T, class Ctx>
            constexpr void leave(std::size_t r, const T& trigger, Ctx& ctx)
            {
                if constexpr (is_composite<S>)
                    sub<S>().leave_all(trigger, ctx);
                if constexpr (info::template has_exit<S>)
                    static_cast<void>(special_rows<S, exit_tag<T>, exit_tag<_>>(r, trigger, ctx));
            }

            template <std::size_t I, class P, class Ctx>
            constexpr bool try_row(std::size_t r, const P& p, Ctx& ctx)
            {
                using R = meta::at_t<typename info::rows, I>;
                auto& row = table_.template get<I>();
                decltype(auto) g = row.guard();
                if (!guard(g, p, ctx))
                    return false;
                decltype(auto) a = row.action();
                if constexpr (std::is_same_v<typename R::target, internal_target>)
                    act(a, p, ctx);
                else {
                    using Src = src_of<R>;
                    using Dst = typename R::target;
                    leave<Src>(r, p, ctx);
                    current_[r] = id_of<Dst>;
                    log_state_change<Def>(ctx, state_t<Src>{}, state_t<Dst>{});
                    if constexpr (is_composite<Dst>)
                        sub<Dst>().reset();
                    act(a, p, ctx);
                    enter<Dst>(r, p, ctx);
                }
                return true;
            }

            template <class Ctx>
            constexpr bool run_pending(std::array<state_id, regions> old, Ctx& ctx)
            {
                bool queued_ok = true;
                bool more = false;
                do {
                    bool changed = false;
                    do {
                        if constexpr (reacts_anonymous) {
                            std::size_t steps = 0;
                            while (guarded(anonymous{}, ctx))
                                if (++steps == max_anonymous_steps) {
                                    ctx.flags |= Ctx::looped;
                                    break;
                                }
                        }
                        changed = old != current_;
                        old = current_;
                    } while (replay_deferred(changed, ctx));
                    more = process_queued(queued_ok, ctx);
                } while (more);
                return queued_ok;
            }

            template <class Ctx>
            constexpr bool replay_deferred(bool changed, Ctx& ctx)
            {
                if constexpr (meta::empty_v<typename info::defer_events>)
                    return false;
                else {
                    if (!changed || deferred_.empty())
                        return false;
                    bool processed = false;
                    deferred_.replaying = true;
                    auto old = current_;
                    for (std::size_t i = 0; i < deferred_.size;) {
                        const auto item = deferred_.items[i];
                        deferred_.again = false;
                        const bool handled = std::visit(replay<machine, Ctx>{this, &ctx}, item);
                        processed |= handled;
                        if (handled && deferred_.again)
                            ++i;
                        else
                            deferred_.erase(i);
                        if constexpr (reacts_anonymous)
                            while (guarded(anonymous{}, ctx)) {
                            }
                        if (old != current_) {
                            i = 0;
                            old = current_;
                        }
                    }
                    deferred_.replaying = false;
                    return processed;
                }
            }

            template <class Ctx>
            constexpr bool process_queued(bool& queued_ok, Ctx& ctx)
            {
                if constexpr (meta::empty_v<typename info::process_events>)
                    return false;
                else {
                    if (queued_.empty())
                        return false;
                    const auto item = queued_.items[0];
                    queued_.erase(0);
                    queued_ok &= std::visit(replay<machine, Ctx>{this, &ctx}, item);
                    return true;
                }
            }

            // ---- guards and actions ------------------------------------------------------------------------

            template <class P, class Ctx>
            constexpr bool guard(always&, const P&, Ctx&) noexcept
            {
                return true;
            }
            template <class G, class P, class Ctx>
            constexpr bool guard(not_<G>& g, const P& p, Ctx& ctx)
            {
                return !guard(g.g, p, ctx);
            }
            template <class A, class B, class P, class Ctx>
            constexpr bool guard(and_<A, B>& g, const P& p, Ctx& ctx)
            {
                return guard(g.a, p, ctx) && guard(g.b, p, ctx);
            }
            template <class A, class B, class P, class Ctx>
            constexpr bool guard(or_<A, B>& g, const P& p, Ctx& ctx)
            {
                return guard(g.a, p, ctx) || guard(g.b, p, ctx);
            }
            template <class F, class P, class Ctx>
            constexpr bool guard(wrapped<F>& g, const P& p, Ctx& ctx)
            {
                const bool result = static_cast<bool>(inject(g.f, p, ctx));
                log_guard<Def>(ctx, g.f, p, result);
                return result;
            }
            template <class G, class P, class Ctx>
                requires(!is_combinator_v<G>)
            constexpr bool guard(G& g, const P& p, Ctx& ctx)
            {
                const bool result = static_cast<bool>(inject(g, p, ctx));
                log_guard<Def>(ctx, g, p, result);
                return result;
            }

            template <class P, class Ctx>
            constexpr void act(none&, const P&, Ctx&) noexcept
            {
            }
            template <class A, class B, class P, class Ctx>
            constexpr void act(seq<A, B>& a, const P& p, Ctx& ctx)
            {
                act(a.a, p, ctx);
                act(a.b, p, ctx);
            }
            template <class P, class Ctx>
            constexpr void act(defer_t&, const P& p, Ctx& ctx)
            {
                if (deferred_.replaying)
                    deferred_.again = true;
                else if (!deferred_.push(p))
                    ctx.flags |= Ctx::overflowed;
            }
            template <class E, class P, class Ctx>
            constexpr void act(process_t<E>& a, const P&, Ctx& ctx)
            {
                if (!queued_.push(a.event))
                    ctx.flags |= Ctx::overflowed;
            }
            template <class G, class A, class P, class Ctx>
            constexpr void act(eval_t<G, A>& a, const P& p, Ctx& ctx)
            {
                if (guard(a.guard, p, ctx))
                    act(a.action, p, ctx);
            }
            template <class F, class P, class Ctx>
            constexpr void act(wrapped<F>& a, const P& p, Ctx& ctx)
            {
                log_action<Def>(ctx, a.f, p);
                static_cast<void>(inject(a.f, p, ctx));
            }
            template <class A, class P, class Ctx>
                requires(!is_combinator_v<A>)
            constexpr void act(A& a, const P& p, Ctx& ctx)
            {
                log_action<Def>(ctx, a, p);
                static_cast<void>(inject(a, p, ctx));
            }

            // A callable with its arguments found by type: the event, the definition, or a dependency.
            template <class F, class P, class Ctx>
                requires std::is_invocable_v<F&>
            constexpr decltype(auto) inject(F& f, const P&, Ctx&)
            {
                return f();
            }
            template <class F, class P, class Ctx>
                requires std::is_member_function_pointer_v<F>
            constexpr decltype(auto) inject(F f, const P& p, Ctx& ctx)
            {
                using traits = callable_traits<F>;
                auto& object = resolve<typename traits::class_type>(p, ctx);
                return [&]<class... As>(meta::type_list<As...>) -> decltype(auto) {
                    return (object.*f)(resolve<As>(p, ctx)...);
                }(typename traits::arguments{});
            }
            template <class F, class P, class Ctx>
                requires(!std::is_invocable_v<F&> && !std::is_member_function_pointer_v<F>)
            constexpr decltype(auto) inject(F& f, const P& p, Ctx& ctx)
            {
                static_assert(has_signature<F, P>, "snicholls::fsm: a guard or action cannot be called with the event "
                                                   "and the machine's dependencies (see the template arguments)");
                return [&]<class... As>(meta::type_list<As...>) -> decltype(auto) {
                    return f(resolve<As>(p, ctx)...);
                }(typename signature_of<F, P>::args{});
            }

            template <class A, class P, class Ctx>
                requires(std::is_same_v<std::remove_cvref_t<A>, P> || std::is_base_of_v<std::remove_cvref_t<A>, P>)
            constexpr const P& resolve(const P& p, Ctx&) noexcept
            {
                return p;
            }
            template <class A, class P, class Ctx>
                requires(!std::is_same_v<std::remove_cvref_t<A>, P> && !std::is_base_of_v<std::remove_cvref_t<A>, P> &&
                         std::is_same_v<std::remove_cvref_t<A>, Def>)
            constexpr Def& resolve(const P&, Ctx&) noexcept
            {
                return def_;
            }
            template <class A, class P, class Ctx>
                requires(!std::is_same_v<std::remove_cvref_t<A>, P> && !std::is_base_of_v<std::remove_cvref_t<A>, P> &&
                         !std::is_same_v<std::remove_cvref_t<A>, Def>)
            constexpr auto& resolve(const P&, Ctx& ctx) noexcept
            {
                return ctx.template dep<std::remove_cvref_t<A>>();
            }
        };

        // Every dependency of a machine and its sub-machines, less the events and the definitions themselves.
        template <class Def, class Cfg>
        struct tree {
            using m = machine<Def, Cfg>;
            template <class... Cs>
            static auto sub_params(meta::type_list<Cs...>) -> meta::concat_t<typename tree<Cs, Cfg>::params...>;
            template <class... Cs>
            static auto sub_excluded(meta::type_list<Cs...>) -> meta::concat_t<typename tree<Cs, Cfg>::excluded...>;
            using params = meta::concat_t<typename m::info::params, decltype(sub_params(typename m::composites{}))>;
            using excluded = meta::concat_t<meta::type_list<Def, anonymous, initial, _, unknown_payload>,
                                            typename m::info::plain_events, typename m::info::process_events,
                                            decltype(sub_excluded(typename m::composites{}))>;
            template <class T>
            using is_excluded = std::bool_constant<meta::contains_v<excluded, T>>;
            using deps = meta::unique_t<meta::remove_if_t<params, is_excluded>>;
        };

        template <class Def, class Cfg, class L = typename Cfg::logger_type>
        struct pool_types {
            using type = meta::unique_t<meta::push_back_t<typename tree<Def, Cfg>::deps, L>>;
        };
        template <class Def, class Cfg>
        struct pool_types<Def, Cfg, no_logger> {
            using type = typename tree<Def, Cfg>::deps;
        };
    } // namespace detail

    /// \brief What the compile-time checks found in a table, as values: `sm<Def>` refuses to compile when any is set.
    /// \ingroup core
    /// \tparam Def the machine's definition.
    template <class Def>
    struct table_diagnostics {
        /// \brief The index of a row that can never run (an earlier unguarded row leaves its state on its event), or npos.
        static constexpr std::size_t shadowed_row = detail::table_info<Def>::shadowed_row;
        /// \brief The index of a state no row reaches from an initial state, or npos.
        static constexpr std::size_t unreachable_state = detail::table_info<Def>::unreachable_state;
        /// \brief The index of a row that leaves `X`, or npos.
        static constexpr std::size_t exit_from_terminal = detail::table_info<Def>::exit_from_terminal;
        /// \brief The index of a state with no enumerator in the machine's state enumeration, or npos.
        static constexpr std::size_t missing_enumerator =
            detail::enum_map<detail::table_info<Def>, typename detail::table_info<Def>::state_enum>::missing;
        /// \brief The number of initial states, which is the number of regions.
        static constexpr std::size_t regions = detail::table_info<Def>::regions;
        /// \brief The table passes every check.
        static constexpr bool ok = shadowed_row == detail::npos && unreachable_state == detail::npos &&
            exit_from_terminal == detail::npos && missing_enumerator == detail::npos && regions > 0;
        /// \brief The names of the table's states, in their order of appearance.
        static constexpr auto state_names = detail::table_info<Def>::state_names;
    };

    /// \brief The state machine: sml's `sm<Def, Policies...>`.
    /// \ingroup core
    ///
    /// \snippet fsm/state_machine_hello_world_demo.cpp use
    ///
    /// \tparam Def      the definition: a class whose `operator()` returns `make_transition_table(...)`.
    /// \tparam Policies any of `logger<L>`, `dispatch<D>`, `queue_capacity<N>`, `testing`.
    template <class Def, class... Policies>
    class sm {
        using cfg = detail::config<Policies...>;
        using impl_type = detail::machine<Def, cfg>;
        using logger_type = typename cfg::logger_type;
        using pool_type = detail::pool<typename detail::pool_types<Def, cfg>::type>;
        using context_type = detail::context<pool_type, logger_type>;

        SNICHOLLS_RESULT_NO_UNIQUE_ADDRESS pool_type deps_;
        impl_type impl_;

        template <class T>
        static constexpr bool accepted = std::is_same_v<T, Def> || pool_type::template has<T>;

        template <class A, class... As>
        static constexpr Def pick_definition(A& a, As&... as)
        {
            if constexpr (std::is_same_v<std::remove_cvref_t<A>, Def>)
                return Def(a);
            else
                return pick_definition(as...);
        }
        static constexpr Def pick_definition()
        {
            static_assert(std::is_default_constructible_v<Def>,
                          "snicholls::fsm: a definition that is not default-constructible must be passed to the constructor");
            return Def{};
        }

        constexpr context_type context() noexcept
        {
            if constexpr (std::is_same_v<logger_type, detail::no_logger>)
                return {deps_, nullptr};
            else
                return {deps_, &deps_.template get<logger_type>()};
        }

        template <class E>
        constexpr detail::outcome run(const E& e, context_type& ctx) noexcept
        {
#if defined(__cpp_exceptions)
            try {
                return impl_.process_event(e, ctx);
            } catch (...) {
                ctx.flags |= context_type::threw;
                return {};
            }
#else
            return impl_.process_event(e, ctx);
#endif
        }

        constexpr void start() noexcept
        {
            auto ctx = context();
#if defined(__cpp_exceptions)
            try {
                impl_.start(ctx);
            } catch (...) {
            }
#else
            impl_.start(ctx);
#endif
        }

        template <class Sub>
        using level_def = typename detail::sub_definition<Sub>::type;

    public:
        /// \brief The definition.
        using definition = Def;
        /// \brief The machine's state enumeration, or `void`: `Def::state_enum`, or the enumeration of every state.
        using state_enum = typename impl_type::state_enum;
        /// \brief The plain events of the table, as a type list.
        using events = typename impl_type::info::plain_events;
        /// \brief The number of orthogonal regions.
        static constexpr std::size_t regions = impl_type::regions;
        /// \brief The number of regions a snapshot holds: this machine's and every sub-machine's.
        static constexpr std::size_t slot_count = impl_type::slot_count;

        /// \brief A machine with a default-constructed definition and owned, default-constructed dependencies; the initial
        /// states' entry actions run.
        constexpr sm()
            requires std::is_default_constructible_v<Def>
            : deps_(), impl_(Def{})
        {
            start();
        }

        /// \brief A machine with its definition and dependencies, matched by type (order does not matter): an lvalue is
        /// held by reference, an rvalue or a const lvalue is copied into the machine. The initial entry actions run.
        /// \param args the definition object and/or dependencies.
        template <class... Args>
            requires(sizeof...(Args) > 0 && !(std::is_same_v<std::remove_cvref_t<Args>, sm> || ...))
        constexpr explicit(sizeof...(Args) == 1) sm(Args&&... args)
            : deps_(std::forward<Args>(args)...), impl_(pick_definition(args...))
        {
            static_assert((accepted<std::remove_cvref_t<Args>> && ...),
                          "snicholls::fsm: a constructor argument is neither the definition nor a dependency of a guard or action");
            start();
        }

        /// \brief Processes one event: the first row that leaves a current state on it and whose guard holds runs, in
        /// every region; then anonymous transitions, deferred and queued events.
        /// \param e the event.
        /// \return true if a row took the event (and every event its actions queued); false otherwise. Never throws.
        template <class E>
        constexpr bool process_event(const E& e) noexcept
        {
            auto ctx = context();
            const auto o = run(e, ctx);
            return o.handled && o.queued_ok;
        }

        /// \brief `process_event` with the reason for a refusal.
        /// \param e the event.
        /// \return nothing, or `fsm_errc::no_transition`, `guard_rejected`, `action_failed`, `queue_full`,
        /// `anonymous_loop` or `queued_event_unhandled`.
        template <class E>
        constexpr result<void, fsm_errc> try_process_event(const E& e) noexcept
        {
            auto ctx = context();
            const auto o = run(e, ctx);
            if (ctx.flags & context_type::threw)
                return err(fsm_errc::action_failed);
            if (ctx.flags & context_type::overflowed)
                return err(fsm_errc::queue_full);
            if (ctx.flags & context_type::looped)
                return err(fsm_errc::anonymous_loop);
            if (!o.handled)
                return err((ctx.flags & context_type::rejected) ? fsm_errc::guard_rejected : fsm_errc::no_transition);
            if (!o.queued_ok)
                return err(fsm_errc::queued_event_unhandled);
            return ok();
        }

        /// \brief `is(s)`: some region is in `s`; `is(s1, s2, ...)`: region i is in si. `is<Sub>(...)` asks the
        /// sub-machine `Sub` (a definition, or `decltype(state<Sub>)`).
        /// \return whether the machine is in those states.
        template <class Sub = Def, class... Ts>
            requires(detail::is_state_v<Ts> && ...)
        [[nodiscard]] constexpr bool is(Ts...) const noexcept
        {
            return impl_.template level<level_def<Sub>>().template is_states<typename Ts::type...>();
        }

        /// \brief Whether every region of the machine (or of `Sub`) is in `X`.
        /// \return true when terminated.
        template <class Sub = Def>
        [[nodiscard]] constexpr bool is_terminated() const noexcept
        {
            return impl_.template level<level_def<Sub>>().is_terminated();
        }

        /// \brief Calls `f(state_t<S>{})` with the current state of each region, in region order (of `Sub`, if given).
        /// \param f the visitor; overload it on `state_t<Sub>` to descend into a composite.
        template <class Sub = Def, class F>
        constexpr void visit_current_states(F&& f) const
        {
            impl_.template level<level_def<Sub>>().visit(std::forward<F>(f));
        }

        /// \brief The current state of a region, as the machine's state enumeration (of `Sub`, if given).
        /// \param region the region, 0 for a machine with one.
        /// \return the enumerator.
        template <class Sub = Def>
        [[nodiscard]] constexpr auto current(std::size_t region = 0) const noexcept
            requires(!std::is_void_v<typename detail::machine<level_def<Sub>, cfg>::state_enum>)
        {
            return impl_.template level<level_def<Sub>>().current(region);
        }

        /// \brief Sets the state of every region (of `Sub`, if given) from enumerators, as after a restart: no entry or
        /// exit action runs, and the queues are emptied.
        /// \param es one enumerator per region.
        /// \return nothing, or `fsm_errc::not_a_state` (and nothing changed) when one is not a state of the table.
        template <class Sub = Def, class... Es>
            requires(sizeof...(Es) > 0 && (std::is_enum_v<Es> && ...))
        constexpr result<void, fsm_errc> restore(Es... es) noexcept
        {
            return impl_.template level<level_def<Sub>>().restore(es...);
        }

        /// \brief Every region's state, this machine's then each sub-machine's in table order, as numbers and names.
        /// \return the snapshot.
        [[nodiscard]] constexpr std::array<wire_state, slot_count> snapshot() const noexcept
        {
            std::array<wire_state, slot_count> out{};
            impl_.write_slots(out.data());
            return out;
        }

        /// \brief Sets every region from a snapshot, no action running: by enumerator value where the level has a state
        /// enumeration, by name otherwise.
        /// \param slots the snapshot.
        /// \return nothing, or `fsm_errc::shape_mismatch` (wrong size) or `fsm_errc::unknown_state`; nothing changed then.
        constexpr result<void, fsm_errc> restore_snapshot(std::span<const wire_state> slots) noexcept
        {
            if (slots.size() != slot_count)
                return err(fsm_errc::shape_mismatch);
            if (!impl_type::valid_slots(slots.data()))
                return err(fsm_errc::unknown_state);
            impl_.read_slots(slots.data());
            return ok();
        }

        /// \brief Sets the states of every region (of `Sub`, if given) without running an action: sml's `testing` policy.
        template <class Sub = Def, class... Ts>
            requires(cfg::testing_enabled && (detail::is_state_v<Ts> && ...))
        constexpr void set_current_states(Ts...) noexcept
        {
            impl_.template level<level_def<Sub>>().template set_states<typename Ts::type...>();
        }

        /// \brief The definition object the table was built from.
        /// \return the definition.
        [[nodiscard]] constexpr const Def& definition_object() const noexcept { return impl_.def_; }
    };

    /// \brief `sm machine{[] { return make_transition_table(...); }}`: the definition deduced, as sml's in-place `sm`.
    /// \ingroup core
    template <class Def, class... Deps>
        requires detail::machine_definition<Def>
    sm(Def, Deps&&...) -> sm<Def>;

    namespace detail {
        template <class E>
        concept with_id = requires { E::id; };
    } // namespace detail

    /// \brief sml's `make_dispatch_table`: a function from a run-time id to the event type whose static `id` it is.
    /// \ingroup core
    ///
    /// Each event type of the table with a `static constexpr` integer `id` in `[Min, Max]` is constructed from the run-time
    /// event (or default-constructed if it cannot be) and processed.
    /// \tparam Runtime the run-time event.
    /// \tparam Min     the smallest id.
    /// \tparam Max     the largest id.
    /// \param machine the machine; it must outlive the function.
    /// \return a callable `(const Runtime&, int id) -> bool`, false for an id no event has.
    template <class Runtime, int Min, int Max, class SM>
    [[nodiscard]] constexpr auto make_dispatch_table(SM& machine) noexcept
    {
        static_assert(Min <= Max, "snicholls::fsm: make_dispatch_table needs Min <= Max");
        using handler = bool (*)(SM&, const Runtime&);
        constexpr auto handlers = []<class... Es>(meta::type_list<Es...>) {
            std::array<handler, static_cast<std::size_t>(Max - Min + 1)> out{};
            const auto add = [&]<class E>() {
                if constexpr (detail::with_id<E>) {
                    if constexpr (E::id >= Min && E::id <= Max)
                        out[static_cast<std::size_t>(E::id - Min)] = [](SM& m, const Runtime& r) {
                            if constexpr (std::is_constructible_v<E, const Runtime&>)
                                return m.process_event(E(r));
                            else
                                return m.process_event(E{});
                        };
                }
            };
            (add.template operator()<Es>(), ...);
            return out;
        }(typename SM::events{});
        return [&machine, handlers](const Runtime& event, int id) noexcept {
            if (id < Min || id > Max)
                return false;
            const auto h = handlers[static_cast<std::size_t>(id - Min)];
            return h != nullptr && h(machine, event);
        };
    }

    namespace literals {
        /// \brief `"idle"_s`: a state named by a string.
        /// \ingroup core
        /// \return the state.
        template <basic_fixed_string Name>
        constexpr state_t<detail::named_state<Name>> operator""_s() noexcept
        {
            return {};
        }

        /// \brief `"e1"_e`: an event named by a string; `"e1"_e()` is the event object.
        /// \ingroup core
        /// \return the event.
        template <basic_fixed_string Name>
        constexpr event_t<named_event<Name>> operator""_e() noexcept
        {
            return {};
        }
    } // namespace literals
    using namespace literals;

} // namespace snicholls::fsm

#endif /* snicholls_core_fsm_state_machine_hpp */
