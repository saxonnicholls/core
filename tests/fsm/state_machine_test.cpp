//
//  state_machine_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  fsm::sm on one level: the DSL, transitions (external, internal, self, anonymous), guards and their algebra,
//  actions and sequences, entry and exit order, dependencies, unexpected events and exceptions, enumerator states
//  with current() and restore(), the compile-time checks as table_diagnostics, the layout contract, constexpr use and
//  noexcept. Cases follow boost-ext/sml's test/ft (transitions.cpp, guards.cpp, states.cpp, events.cpp,
//  unexpected_events.cpp, dependencies.cpp). Composite machines, regions, history, defer and process are in
//  state_machine_hierarchy_test.cpp.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls::fsm;
using snicholls::fsm::on_exit;   // glibc declares ::on_exit; sml's examples write sml::on_exit for the same reason

struct e1 {};
struct e2 {};
struct e3 {};
struct e4 {};
struct with_value {
    int value = 0;
};

// ---- the layout contract ---------------------------------------------------------------------------------------

struct minimal {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"a"_s + event<e1> = "b"_s, "b"_s + event<e2> = X);
    }
};
static_assert(sizeof(sm<minimal>) == 1, "one byte: the state of the one region");
static_assert(std::is_trivially_copyable_v<sm<minimal>>);
static_assert(std::is_nothrow_move_constructible_v<sm<minimal>>);
static_assert(noexcept(std::declval<sm<minimal>&>().process_event(e1{})));
static_assert(noexcept(std::declval<sm<minimal>&>().try_process_event(e1{})));
static_assert(noexcept(std::declval<const sm<minimal>&>().is(X)));
static_assert(std::is_void_v<sm<minimal>::state_enum>);
static_assert(sm<minimal>::regions == 1 && sm<minimal>::slot_count == 1);

struct two_regions {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"a"_s + event<e1> = "b"_s, *"c"_s + event<e1> = "d"_s);
    }
};
static_assert(sizeof(sm<two_regions>) == 2, "one byte per region");

// More than 255 states widen the state to 16 bits: tested on the trait, not by writing 256 rows.
static_assert(std::is_same_v<snicholls::fsm::detail::table_info<minimal>::state_id, std::uint8_t>);

// A machine runs in a constant expression.
static_assert([] {
    sm<minimal> m;
    const bool first = m.process_event(e1{});
    const bool second = m.process_event(e2{});
    return first && second && m.is(X) && !m.process_event(e1{});
}());

// ---- transitions --------------------------------------------------------------------------------------------------

TEST(FsmTransitions, ExternalTransitionMovesToTheTarget)
{
    sm<minimal> m;
    EXPECT_TRUE(m.is("a"_s));
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.is("b"_s));
    EXPECT_FALSE(m.is("a"_s));
}

TEST(FsmTransitions, AnEventWithNoRowIsRefusedAndChangesNothing)
{
    sm<minimal> m;
    EXPECT_FALSE(m.process_event(e2{}));
    EXPECT_TRUE(m.is("a"_s));
    const auto why = m.try_process_event(e2{});
    ASSERT_TRUE(why.is_err());
    EXPECT_EQ(why.unwrap_err_unchecked(), fsm_errc::no_transition);
    // an event the table does not know at all
    EXPECT_EQ(m.try_process_event(e4{}).unwrap_err_unchecked(), fsm_errc::no_transition);
    EXPECT_TRUE(m.try_process_event(e1{}).is_ok());
}

struct ordering {
    std::vector<std::string>* log = nullptr;
    auto operator()() const
    {
        const auto note = [this](const char* what) { return [this, what] { log->push_back(what); }; };
        return make_transition_table(
            *"idle"_s + event<e1> / note("e1 action") = "s1"_s,
             "s1"_s + on_entry<_> / note("s1 entry"),
             "s1"_s + on_exit<_> / note("s1 exit"),
             "s1"_s + event<e2> / note("internal"),
             "s1"_s + event<e3> / note("self") = "s1"_s,
             "s1"_s + event<e4> / note("leave") = X);
    }
};

TEST(FsmTransitions, EntryActionExitRunInSmlsOrder)
{
    std::vector<std::string> log;
    sm<ordering> m{ordering{&log}};
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_EQ(log, (std::vector<std::string>{"e1 action", "s1 entry"}));
    log.clear();
    EXPECT_TRUE(m.process_event(e2{}));   // internal: no exit, no entry
    EXPECT_EQ(log, (std::vector<std::string>{"internal"}));
    log.clear();
    EXPECT_TRUE(m.process_event(e3{}));   // self: exit, action, entry
    EXPECT_EQ(log, (std::vector<std::string>{"s1 exit", "self", "s1 entry"}));
    log.clear();
    EXPECT_TRUE(m.process_event(e4{}));
    EXPECT_EQ(log, (std::vector<std::string>{"s1 exit", "leave"}));
    EXPECT_TRUE(m.is(X));
}

struct anonymous_chain {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(
            *"a"_s = "b"_s,
             "b"_s = "c"_s,
             "c"_s + event<e1> = "d"_s,
             "d"_s / [] {} = X);
    }
};

TEST(FsmTransitions, SubsequentAnonymousTransitionsRunToAStableState)
{
    sm<anonymous_chain> m;   // a -> b -> c at construction
    EXPECT_TRUE(m.is("c"_s));
    EXPECT_TRUE(m.process_event(e1{}));   // d, then at once X
    EXPECT_TRUE(m.is(X));
}

struct guarded_anonymous {
    auto operator()() const
    {
        return make_transition_table(
            *"idle"_s [([](const int& n) { return n > 0; })] = "go"_s,
             "idle"_s + event<e1> / [](int& n) { ++n; },
             "go"_s + event<e2> = X);
    }
};

TEST(FsmTransitions, AGuardedAnonymousTransitionWaitsForItsGuard)
{
    int n = 0;
    sm<guarded_anonymous> m{n};
    EXPECT_TRUE(m.is("idle"_s));
    EXPECT_TRUE(m.process_event(e1{}));   // n = 1, then the anonymous transition fires
    EXPECT_TRUE(m.is("go"_s));
}

struct forever {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"spin"_s + event<e1> = "loop"_s, "loop"_s / [] {});
    }
};

TEST(FsmTransitions, AnAnonymousLoopStopsAndIsReported)
{
    sm<forever> m;
    const auto why = m.try_process_event(e1{});
    ASSERT_TRUE(why.is_err());
    EXPECT_EQ(why.unwrap_err_unchecked(), fsm_errc::anonymous_loop);
    EXPECT_TRUE(m.is("loop"_s));
}

struct overloads {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(
            *"idle"_s + event<with_value> [([](const with_value& v) { return v.value == 1; })] = "one"_s,
             "idle"_s + event<with_value> [([](const with_value& v) { return v.value == 2; })] = "two"_s,
             "idle"_s + event<with_value> = "other"_s);
    }
};

TEST(FsmTransitions, RowsAreTriedInTableOrderAndTheFirstWhoseGuardHoldsRuns)
{
    sm<overloads> one;
    EXPECT_TRUE(one.process_event(with_value{1}));
    EXPECT_TRUE(one.is("one"_s));
    sm<overloads> two;
    EXPECT_TRUE(two.process_event(with_value{2}));
    EXPECT_TRUE(two.is("two"_s));
    sm<overloads> other;
    EXPECT_TRUE(other.process_event(with_value{7}));
    EXPECT_TRUE(other.is("other"_s));
}

struct reversed_notation {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table("b"_s <= *"a"_s + event<e1>, X <= "b"_s + event<e2> / [] {}, "c"_s <= "b"_s + event<e3>);
    }
};

TEST(FsmTransitions, TheReversedNotationBuildsTheSameRows)
{
    sm<reversed_notation> m;
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.is("b"_s));
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_TRUE(m.is(X));
}

// ---- guards -------------------------------------------------------------------------------------------------------

struct short_circuit {
    int true_calls = 0;
    int false_calls = 0;
    auto operator()()
    {
        auto yes = [this] { ++true_calls; return true; };
        auto no = [this] { ++false_calls; return false; };
        return make_transition_table(
            *"s1"_s + event<e1> [yes || yes] = "s2"_s,
             "s2"_s + event<e1> [no && yes] = "s3"_s,
             "s2"_s + event<e2> [!no] = "s3"_s);
    }
};

TEST(FsmGuards, AndAndOrShortCircuit)
{
    sm<short_circuit> m;
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_EQ(m.definition_object().true_calls, 1);
    EXPECT_EQ(m.definition_object().false_calls, 0);
    EXPECT_FALSE(m.process_event(e1{}));
    EXPECT_EQ(m.definition_object().true_calls, 1);
    EXPECT_EQ(m.definition_object().false_calls, 1);
    EXPECT_TRUE(m.is("s2"_s));
    EXPECT_EQ(m.try_process_event(e1{}).unwrap_err_unchecked(), fsm_errc::guard_rejected);
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_TRUE(m.is("s3"_s));
}

TEST(FsmGuards, CopyingAMachineRebuildsItsTableAroundTheCopiedDefinition)
{
    sm<short_circuit> m;
    sm<short_circuit> copy = m;
    EXPECT_TRUE(copy.process_event(e1{}));
    EXPECT_EQ(copy.definition_object().true_calls, 1);   // the copy's own definition, not m's
    EXPECT_EQ(m.definition_object().true_calls, 0);
    m = copy;
    EXPECT_TRUE(m.is("s2"_s));
    EXPECT_EQ(m.definition_object().true_calls, 1);
}

bool free_guard(int i) { return i == 42; }

struct functions {
    bool member_guard(int i) const { return i == 42; }
    void member_action(int& i) { i += 1; }
    auto operator()()
    {
        return make_transition_table(
            *"a"_s + event<e1> [&free_guard] = "b"_s,
             "b"_s + event<e1> [&functions::member_guard] / &functions::member_action = "c"_s,
             "c"_s + event<e1> [wrap(&free_guard) && wrap(&free_guard)] = X);
    }
};

TEST(FsmGuards, FreeAndMemberFunctionsWorkAsGuardsAndActions)
{
    int dep = 42;
    sm<functions> m{dep};
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_EQ(dep, 43);
    EXPECT_FALSE(m.process_event(e1{}));   // 43 is not 42
    dep = 42;
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.is(X));
}

// ---- actions and dependencies -------------------------------------------------------------------------------------

struct injected {
    auto operator()() const
    {
        const auto set = [](std::string& s, const with_value& e, int& count) {
            s += std::to_string(e.value);
            ++count;
        };
        const auto generic = [](const auto& e, std::string& s) { s += std::to_string(e.value * 10); };
        return make_transition_table(*"a"_s + event<with_value> / (set, generic) = "b"_s,
                                     "b"_s + event<with_value> / [](int count, const with_value&) { static_cast<void>(count); });
    }
};

TEST(FsmDependencies, ParametersAreMatchedByTypeInAnyOrder)
{
    std::string text;
    int count = 0;
    sm<injected> m{count, text};   // any order
    EXPECT_TRUE(m.process_event(with_value{4}));
    EXPECT_EQ(text, "440");
    EXPECT_EQ(count, 1);
    EXPECT_TRUE(m.process_event(with_value{5}));   // a by-value dependency is a copy
}

TEST(FsmDependencies, ADependencyNotPassedIsOwnedAndDefaultConstructed)
{
    sm<injected> m;
    EXPECT_TRUE(m.process_event(with_value{1}));
    EXPECT_TRUE(m.is("b"_s));
    const int seven = 7;
    sm<injected> copied{seven};   // a const lvalue is copied in, not referred to
    EXPECT_TRUE(copied.process_event(with_value{1}));
}

// ---- unexpected events and exceptions -----------------------------------------------------------------------------

struct unexpected_rows {
    std::vector<std::string>* log = nullptr;
    auto operator()() const
    {
        return make_transition_table(
            *"idle"_s + event<e1> = "s1"_s,
             "idle"_s + unexpected_event<e2> / [this] { log->push_back("e2 unexpected"); },
             "idle"_s + unexpected_event<_> / [this](const auto&) { log->push_back("something unexpected"); },
             "s1"_s + unexpected_event<_> / [this] { log->push_back("unexpected in s1"); } = X);
    }
};

TEST(FsmUnexpected, SpecificThenGenericRowsTakeEventsWithNoRow)
{
    std::vector<std::string> log;
    sm<unexpected_rows> m{unexpected_rows{&log}};
    EXPECT_FALSE(m.process_event(e2{}));   // unexpected is still false, as in sml
    EXPECT_FALSE(m.process_event(e3{}));
    EXPECT_EQ(log, (std::vector<std::string>{"e2 unexpected", "something unexpected"}));
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_FALSE(m.process_event(e4{}));   // not an event of the table at all
    EXPECT_TRUE(m.is(X));
}

#if defined(__cpp_exceptions)
struct throwing {
    std::vector<std::string>* log = nullptr;
    auto operator()() const
    {
        return make_transition_table(
            *"idle"_s + event<e1> / [] { throw std::runtime_error("e1"); },
             "idle"_s + event<e2> / [] { throw 42; },
             "idle"_s + event<e3> [([]() -> bool { throw std::logic_error("guard"); })] = "never"_s,
             "idle"_s + exception<std::runtime_error> / [this](const std::runtime_error& e) { log->push_back(e.what()); },
             "idle"_s + exception<_> / [this] { log->push_back("other"); },
             "never"_s + event<e4> = X);
    }
};

TEST(FsmExceptions, ExceptionRowsTakeWhatGuardsAndActionsThrow)
{
    std::vector<std::string> log;
    sm<throwing> m{throwing{&log}};
    EXPECT_TRUE(m.process_event(e1{}));   // taken by exception<runtime_error>
    EXPECT_TRUE(m.process_event(e2{}));   // taken by exception<_>
    EXPECT_TRUE(m.process_event(e3{}));   // a guard that throws: logic_error is not runtime_error
    EXPECT_EQ(log, (std::vector<std::string>{"e1", "other", "other"}));
    EXPECT_TRUE(m.is("idle"_s));
}

struct throwing_unhandled {
    constexpr auto operator()() const
    {
        return make_transition_table(*"idle"_s + event<e1> / [] { throw std::bad_alloc(); } = "next"_s, "next"_s + event<e2> = X);
    }
};

TEST(FsmExceptions, NoExceptionCrossesProcessEvent)
{
    sm<throwing_unhandled> m;
    EXPECT_NO_THROW(static_cast<void>(m.process_event(e1{})));
    sm<throwing_unhandled> n;
    const auto why = n.try_process_event(e1{});
    ASSERT_TRUE(why.is_err());
    EXPECT_EQ(why.unwrap_err_unchecked(), fsm_errc::action_failed);
    EXPECT_TRUE(n.is("next"_s));   // as sml: the state changed before the action ran
}
#endif

// ---- states as values ---------------------------------------------------------------------------------------------

enum class tx : std::uint8_t { requested, pending, confirmed, failed, unused };

struct tx_machine {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(
            *enum_state<tx::requested> + event<e1> = enum_state<tx::pending>,
             enum_state<tx::pending> + event<e2> = enum_state<tx::confirmed>,
             enum_state<tx::pending> + event<e3> = enum_state<tx::failed>);
    }
};
static_assert(std::is_same_v<sm<tx_machine>::state_enum, tx>);
static_assert(sizeof(sm<tx_machine>) == 1);

TEST(FsmEnumStates, CurrentIsTheEnumerator)
{
    sm<tx_machine> m;
    EXPECT_EQ(m.current(), tx::requested);
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_EQ(m.current(), tx::pending);
    EXPECT_TRUE(m.is(enum_state<tx::pending>));
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_EQ(m.current(0), tx::confirmed);
}

TEST(FsmEnumStates, RestoreSetsTheStateAndRefusesAnEnumeratorTheTableDoesNotUse)
{
    sm<tx_machine> m;
    EXPECT_TRUE(m.restore(tx::pending).is_ok());
    EXPECT_EQ(m.current(), tx::pending);
    const auto unused = m.restore(tx::unused);
    ASSERT_TRUE(unused.is_err());
    EXPECT_EQ(unused.unwrap_err_unchecked(), fsm_errc::not_a_state);
    EXPECT_EQ(m.restore(static_cast<tx>(200)).unwrap_err_unchecked(), fsm_errc::not_a_state);
    EXPECT_EQ(m.current(), tx::pending);   // unchanged
    EXPECT_TRUE(m.process_event(e3{}));
    EXPECT_EQ(m.current(), tx::failed);
}

TEST(FsmEnumStates, RestoreRunsNoEntryAction)
{
    std::vector<std::string> log;
    sm<ordering> m{ordering{&log}};
    log.clear();
    EXPECT_TRUE(m.restore_snapshot(std::array<wire_state, 1>{{{0, "s1"}}}).is_ok());
    EXPECT_TRUE(log.empty());
    EXPECT_TRUE(m.is("s1"_s));
}

// Type and string states map to a declared enumeration by name.
enum class door : std::uint8_t { open, closed, locked };
struct open {};
struct door_machine {
    using state_enum = door;
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(
            *state<open> + event<e1> = "closed"_s,
             "closed"_s + event<e1> = state<open>,
             "closed"_s + event<e2> = enum_state<door::locked>,
             enum_state<door::locked> + event<e2> = "closed"_s);
    }
};

TEST(FsmEnumStates, ADeclaredEnumerationMapsTypeAndStringStatesByName)
{
    sm<door_machine> m;
    EXPECT_EQ(m.current(), door::open);
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_EQ(m.current(), door::closed);
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_EQ(m.current(), door::locked);
    EXPECT_TRUE(m.restore(door::open).is_ok());
    EXPECT_TRUE(m.is(state<open>));
}

TEST(FsmSnapshot, ValuesAreEnumeratorsOrIndicesAndNamesAreTheStatesNames)
{
    sm<tx_machine> m;
    EXPECT_TRUE(m.process_event(e1{}));
    const auto enum_snap = m.snapshot();
    EXPECT_EQ(enum_snap[0], (wire_state{1, "pending"}));

    sm<minimal> n;
    EXPECT_TRUE(n.process_event(e1{}));
    EXPECT_EQ(n.snapshot()[0], (wire_state{1, "b"}));

    sm<tx_machine> back;
    EXPECT_TRUE(back.restore_snapshot(enum_snap).is_ok());
    EXPECT_EQ(back.current(), tx::pending);
    // an enumeration level reads the value, not the name
    EXPECT_TRUE(back.restore_snapshot(std::array<wire_state, 1>{{{3, "renamed"}}}).is_ok());
    EXPECT_EQ(back.current(), tx::failed);
    EXPECT_EQ(back.restore_snapshot(std::array<wire_state, 1>{{{4, "unused"}}}).unwrap_err_unchecked(),
              fsm_errc::unknown_state);
    EXPECT_EQ(back.restore_snapshot(std::array<wire_state, 2>{}).unwrap_err_unchecked(), fsm_errc::shape_mismatch);
    EXPECT_EQ(back.current(), tx::failed);
    // a level without an enumeration reads the name
    EXPECT_EQ(n.restore_snapshot(std::array<wire_state, 1>{{{0, "zzz"}}}).unwrap_err_unchecked(), fsm_errc::unknown_state);
    EXPECT_TRUE(n.restore_snapshot(std::array<wire_state, 1>{{{99, "a"}}}).is_ok());
    EXPECT_TRUE(n.is("a"_s));
}

TEST(FsmSnapshot, CheckSnapshotAlsoComparesNamesWhereTheValueDecides)
{
    using enum_sm = sm<tx_machine>;
    using name_sm = sm<minimal>;
    static_assert(enum_sm::check_snapshot(std::array<wire_state, 1>{{{1, "pending"}}}).is_ok());
    static_assert(enum_sm::check_snapshot(std::array<wire_state, 1>{{{1, ""}}}).is_ok());   // no name: by value
    static_assert(enum_sm::check_snapshot(std::array<wire_state, 1>{{{3, "renamed"}}}).unwrap_err_unchecked() ==
                  fsm_errc::state_name_mismatch);
    static_assert(enum_sm::check_snapshot(std::array<wire_state, 1>{{{4, "unused"}}}).unwrap_err_unchecked() ==
                  fsm_errc::unknown_state);
    static_assert(enum_sm::check_snapshot(std::array<wire_state, 2>{}).unwrap_err_unchecked() == fsm_errc::shape_mismatch);
    static_assert(noexcept(enum_sm::check_snapshot(std::array<wire_state, 1>{})));
    // a level without an enumeration decides by name, which therefore always agrees
    static_assert(name_sm::check_snapshot(std::array<wire_state, 1>{{{99, "a"}}}).is_ok());
    static_assert(name_sm::check_snapshot(std::array<wire_state, 1>{{{0, "zzz"}}}).unwrap_err_unchecked() ==
                  fsm_errc::unknown_state);
    // and it changes nothing: a machine's own snapshot always passes
    enum_sm m;
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(enum_sm::check_snapshot(m.snapshot()).is_ok());
    EXPECT_EQ(m.current(), tx::pending);
}

// ---- names --------------------------------------------------------------------------------------------------------

struct named_type {};
static_assert(state<named_type>.name() == "named_type");
static_assert(decltype("fin wait 1"_s)::name() == "fin wait 1");
static_assert(enum_state<tx::pending>.name() == "pending");
static_assert(X.name() == "terminate");
static_assert(std::string_view(X.c_str()) == "terminate");
static_assert(std::string_view(enum_state<tx::pending>.c_str()) == "pending");
static_assert(event<e1>.name() == "e1");
static_assert("e3"_e.name() == "e3");
static_assert(state<named_type> == state<named_type> && !(state<named_type> == X));
static_assert((*state<named_type>).initial && !state<named_type>.initial);
static_assert(state<named_type>(H).history);

TEST(FsmNames, VisitCurrentStatesHandsEachRegionItsState)
{
    sm<two_regions> m;
    std::string seen;
    m.visit_current_states([&](auto s) { seen += std::string(s.name()) + ","; });
    EXPECT_EQ(seen, "a,c,");
    EXPECT_TRUE(m.process_event(e1{}));
    seen.clear();
    m.visit_current_states([&](auto s) { seen += std::string(s.name()) + ","; });
    EXPECT_EQ(seen, "b,d,");
}

// ---- the compile-time checks, as values ---------------------------------------------------------------------------

struct shadowed {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"a"_s + event<e1> = "b"_s, "a"_s + event<e1> = "c"_s, "c"_s + event<e2> = "a"_s);
    }
};
static_assert(table_diagnostics<shadowed>::shadowed_row == 1);
static_assert(!table_diagnostics<shadowed>::ok);

struct shadowed_by_unguarded_then_guarded {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"a"_s + event<e1> = "b"_s, "a"_s + event<e1> [([] { return true; })] = "b"_s);
    }
};
static_assert(table_diagnostics<shadowed_by_unguarded_then_guarded>::shadowed_row == 1);

struct guarded_first {   // a guarded row before an unguarded one is the "otherwise" form: allowed
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"a"_s + event<e1> [([] { return false; })] = "b"_s, "a"_s + event<e1> = "c"_s);
    }
};
static_assert(table_diagnostics<guarded_first>::ok);

struct unreachable {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"a"_s + event<e1> = "b"_s, "island"_s + event<e1> = "b"_s);
    }
};
static_assert(table_diagnostics<unreachable>::state_names[table_diagnostics<unreachable>::unreachable_state] == "island");

struct leaves_x {
    constexpr auto operator()() const noexcept { return make_transition_table(*"a"_s + event<e1> = X, X + event<e1> = "a"_s); }
};
static_assert(table_diagnostics<leaves_x>::exit_from_terminal == 1);

struct x_with_entry {   // an entry action on X is not an exit from it
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"a"_s + event<e1> = X, X + on_entry<_> / [] {});
    }
};
static_assert(table_diagnostics<x_with_entry>::ok);

enum class few : std::uint8_t { alpha, beta };
struct missing_enumerator {
    using state_enum = few;
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"alpha"_s + event<e1> = "gamma"_s);
    }
};
static_assert(table_diagnostics<missing_enumerator>::missing_enumerator == 1);

struct no_initial {
    constexpr auto operator()() const noexcept { return make_transition_table("a"_s + event<e1> = "b"_s); }
};
static_assert(table_diagnostics<no_initial>::regions == 0 && !table_diagnostics<no_initial>::ok);

static_assert(table_diagnostics<minimal>::ok && table_diagnostics<tx_machine>::ok && table_diagnostics<door_machine>::ok);

// ---- the testing policy and the logger ----------------------------------------------------------------------------

TEST(FsmTesting, SetCurrentStatesRunsNoActionAndClearsNothingElse)
{
    std::vector<std::string> log;
    sm<ordering, snicholls::fsm::testing> m{ordering{&log}};
    m.set_current_states("s1"_s);
    EXPECT_TRUE(log.empty());
    EXPECT_TRUE(m.process_event(e4{}));
    EXPECT_TRUE(m.is(X));
}

struct counting_logger {
    int events = 0, guards = 0, actions = 0, changes = 0;
    template <class SM, class E>
    void log_process_event(const E&) { ++events; }
    template <class SM, class G, class E>
    void log_guard(const G&, const E&, bool) { ++guards; }
    template <class SM, class A, class E>
    void log_action(const A&, const E&) { ++actions; }
    template <class SM, class S, class D>
    void log_state_change(const S&, const D&) { ++changes; }
};

struct partial_logger {   // any hook may be missing
    int changes = 0;
    template <class SM, class S, class D>
    void log_state_change(const S&, const D&) { ++changes; }
};

TEST(FsmLogger, EveryHookIsCalledAndMissingOnesAreSkipped)
{
    counting_logger full;
    sm<short_circuit, logger<counting_logger>> m{full};
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_EQ(full.events, 1);
    EXPECT_EQ(full.guards, 1);   // the first || operand held
    EXPECT_EQ(full.changes, 1);
    partial_logger part;
    std::vector<std::string> log;
    sm<ordering, logger<partial_logger>> n{part, ordering{&log}};
    EXPECT_TRUE(n.process_event(e1{}));
    EXPECT_EQ(part.changes, 1);
}

// ---- in place and dispatch strategies -----------------------------------------------------------------------------

TEST(FsmDispatch, JumpTableAndFoldAgree)
{
    sm<overloads> fold;
    sm<overloads, dispatch<jump_table>> table;
    for (int v : {2, 9}) {
        sm<overloads> f;
        sm<overloads, dispatch<jump_table>> t;
        EXPECT_EQ(f.process_event(with_value{v}), t.process_event(with_value{v}));
        EXPECT_EQ(f.snapshot(), t.snapshot());
    }
    EXPECT_EQ(fold.snapshot(), table.snapshot());
}

struct runtime {
    int id;
};
struct numbered {
    static constexpr int id = 3;
};
struct numbered_table {
    constexpr auto operator()() const noexcept { return make_transition_table(*"a"_s + event<numbered> = X); }
};

TEST(FsmDispatch, MakeDispatchTableRefusesIdsWithNoEvent)
{
    sm<numbered_table> m;
    auto dispatch_event = make_dispatch_table<runtime, 1, 4>(m);
    EXPECT_FALSE(dispatch_event(runtime{2}, 2));
    EXPECT_FALSE(dispatch_event(runtime{5}, 5));
    EXPECT_TRUE(dispatch_event(runtime{3}, 3));
    EXPECT_TRUE(m.is(X));
}

} // namespace
