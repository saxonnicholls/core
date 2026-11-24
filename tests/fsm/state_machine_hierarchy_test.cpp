//
//  state_machine_hierarchy_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  fsm::sm across levels and regions, a companion of state_machine_test.cpp: composite machines (the sub-machine
//  first, entry and exit order across levels, re-entry resets, completion by an anonymous row), orthogonal regions,
//  history (whole and per region), defer (order, replay after a change, the dropped event) and process (queued
//  events, the result when one is not taken, the fixed queue's overflow). Cases follow boost-ext/sml's test/ft
//  (composite.cpp, orthogonal_regions.cpp, history.cpp, actions_defer.cpp, actions_process.cpp).
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using namespace snicholls::fsm;
using snicholls::fsm::on_exit;   // glibc declares ::on_exit; sml's examples write sml::on_exit for the same reason

struct e1 {};
struct e2 {};
struct e3 {};
struct e4 {};
struct e5 {};

using calls = std::vector<std::string>;

// ---- composite ----------------------------------------------------------------------------------------------------

constexpr auto ss1_1 = state<class ss1_1_t>;
constexpr auto ss1_2 = state<class ss1_2_t>;
constexpr auto ss2_1 = state<class ss2_1_t>;
constexpr auto ss2_2 = state<class ss2_2_t>;

struct sub1 {
    auto operator()() const
    {
        return make_transition_table(
            *ss1_1 + event<e1> = ss1_2,
             ss1_1 + on_entry<_> / [](calls& c) { c.push_back("ss1_1 entry"); },
             ss1_2 + on_entry<_> / [](calls& c) { c.push_back("ss1_2 entry"); },
             ss1_1 + on_exit<_> / [](calls& c) { c.push_back("ss1_1 exit"); },
             ss1_2 + on_exit<_> / [](calls& c) { c.push_back("ss1_2 exit"); });
    }
};
struct sub2 {
    auto operator()() const
    {
        return make_transition_table(
            *ss2_1 + event<e1> = ss2_2,
             ss2_1 + on_entry<_> / [](calls& c) { c.push_back("ss2_1 entry"); },
             ss2_2 + on_entry<_> / [](calls& c) { c.push_back("ss2_2 entry"); },
             ss2_1 + on_exit<_> / [](calls& c) { c.push_back("ss2_1 exit"); },
             ss2_2 + on_exit<_> / [](calls& c) { c.push_back("ss2_2 exit"); });
    }
};
struct top {
    auto operator()() const
    {
        return make_transition_table(
            *state<sub1> + event<e2> = state<sub2>,
             state<sub2> + event<e3> = state<sub1>,
             state<sub1> + on_entry<_> / [](calls& c) { c.push_back("sub1 entry"); },
             state<sub2> + on_entry<_> / [](calls& c) { c.push_back("sub2 entry"); },
             state<sub1> + on_exit<_> / [](calls& c) { c.push_back("sub1 exit"); },
             state<sub2> + on_exit<_> / [](calls& c) { c.push_back("sub2 exit"); });
    }
};

TEST(FsmComposite, EntryAndExitRunOuterInOnEntryAndInnerOutOnExit)
{
    calls c;
    sm<top> m{c};
    EXPECT_EQ(c, (calls{"sub1 entry", "ss1_1 entry"}));
    EXPECT_TRUE(m.is(state<sub1>));
    c.clear();
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_EQ(c, (calls{"ss1_1 exit", "sub1 exit", "sub2 entry", "ss2_1 entry"}));
    EXPECT_TRUE(m.is(state<sub2>));
}

TEST(FsmComposite, TheSubMachineSeesTheEventFirst)
{
    calls c;
    sm<top> m{c};
    c.clear();
    EXPECT_TRUE(m.process_event(e1{}));   // only the sub-machine has e1
    EXPECT_TRUE(m.is<sub1>(ss1_2));
    EXPECT_EQ(c, (calls{"ss1_1 exit", "ss1_2 entry"}));
}

TEST(FsmComposite, ReEnteringACompositeWithoutHistoryStartsItAgain)
{
    calls c;
    sm<top> m{c};
    EXPECT_TRUE(m.process_event(e1{}));   // sub1 in ss1_2
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_TRUE(m.process_event(e3{}));   // back to sub1: reset to ss1_1
    EXPECT_TRUE(m.is<sub1>(ss1_1));
    EXPECT_TRUE(m.is<decltype(state<sub2>)>(ss2_1));
}

struct inner_same {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"idle"_s + event<e1> = "s1"_s, "s1"_s + event<e2> = "s2"_s);
    }
};
struct outer_same {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"idle"_s + event<e1> = state<inner_same>, state<inner_same> + event<e2> = X);
    }
};

TEST(FsmComposite, AnEventTheSubMachineTakesDoesNotReachTheParent)
{
    sm<outer_same> m;
    EXPECT_TRUE(m.process_event(e1{}));   // the parent takes e1: the sub-machine is not current yet
    EXPECT_TRUE(m.is<inner_same>("idle"_s));
    EXPECT_TRUE(m.process_event(e1{}));   // now the sub-machine takes e1
    EXPECT_TRUE(m.is<inner_same>("s1"_s));
    EXPECT_TRUE(m.process_event(e2{}));   // the sub-machine takes e2 too, and has not terminated
    EXPECT_TRUE(m.is(state<inner_same>));
    EXPECT_TRUE(m.is<inner_same>("s2"_s));
    EXPECT_TRUE(m.process_event(e2{}));   // the sub-machine has no e2 now: the parent takes it
    EXPECT_TRUE(m.is(X));
}

struct finishing {
    constexpr auto operator()() const noexcept { return make_transition_table(*"working"_s + event<e1> = X); }
};
struct completion {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*state<finishing> = "done"_s, "done"_s + event<e2> = X);
    }
};

TEST(FsmComposite, AnAnonymousRowFromACompositeRunsWhenItTerminates)
{
    sm<completion> m;
    EXPECT_TRUE(m.is(state<finishing>));   // not terminated: the anonymous row waits
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.is("done"_s));
    EXPECT_TRUE(m.is_terminated<finishing>());
}

struct deepest {
    constexpr auto operator()() const noexcept { return make_transition_table(*"d1"_s + event<e1> = "d2"_s); }
};
struct middle {
    constexpr auto operator()() const noexcept { return make_transition_table(*state<deepest> + event<e2> = "m2"_s); }
};
struct outermost {
    constexpr auto operator()() const noexcept { return make_transition_table(*state<middle> + event<e3> = X); }
};

TEST(FsmComposite, LevelsNestAndAreFoundByTheirDefinition)
{
    sm<outermost> m;
    static_assert(sm<outermost>::slot_count == 3);
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.is<deepest>("d2"_s));
    EXPECT_TRUE(m.is<middle>(state<deepest>));
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_TRUE(m.is<middle>("m2"_s));
    const auto snap = m.snapshot();
    EXPECT_EQ(snap[0].name, "middle");
    EXPECT_EQ(snap[1].name, "m2");
    EXPECT_EQ(snap[2].name, "d2");
    sm<outermost> back;
    EXPECT_TRUE(back.restore_snapshot(snap).is_ok());
    EXPECT_TRUE(back.is<middle>("m2"_s) && back.is<deepest>("d2"_s));
    EXPECT_TRUE(back.process_event(e3{}));
    EXPECT_TRUE(back.is(X));
}

// ---- orthogonal regions -------------------------------------------------------------------------------------------

struct regions {
    auto operator()() const
    {
        return make_transition_table(
            *"a1"_s + event<e1> = "a2"_s,
             "a2"_s + on_entry<_> / [](calls& c) { c.push_back("a2 entry"); },
            *"b1"_s + event<e1> = "b2"_s,
             "b2"_s + on_entry<_> / [](calls& c) { c.push_back("b2 entry"); },
             "b2"_s + event<e2> = "b3"_s);
    }
};

TEST(FsmRegions, AnEventIsConsumedByEveryRegionInOrder)
{
    calls c;
    sm<regions> m{c};
    static_assert(sm<regions>::regions == 2);
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.is("a2"_s, "b2"_s));
    EXPECT_EQ(c, (calls{"a2 entry", "b2 entry"}));
    EXPECT_TRUE(m.process_event(e2{}));   // only the second region takes e2: still true
    EXPECT_TRUE(m.is("a2"_s, "b3"_s));
    EXPECT_TRUE(m.is("b3"_s));            // one state: any region
    EXPECT_FALSE(m.is("b2"_s));
}

// ---- history ------------------------------------------------------------------------------------------------------

struct history_sub {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(
            "idle"_s(H) + event<e1> = "s1"_s,   // (H) makes it initial
             "s1"_s + event<e2> = "s2"_s,
            *"idle2"_s + event<e2> = "s3"_s);   // a region without history
    }
};
struct history_top {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(
            *"idle"_s + event<e1> = state<history_sub>,
             state<history_sub> + event<e3> = "s1"_s,
             "s1"_s + event<e4> = state<history_sub>,
             state<history_sub> + event<e5> = X);
    }
};

TEST(FsmHistory, AHistoryRegionResumesAndTheOtherRestarts)
{
    sm<history_top> m;
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.is<history_sub>("idle"_s, "idle2"_s));
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_TRUE(m.is<history_sub>("s2"_s, "s3"_s));
    EXPECT_TRUE(m.process_event(e3{}));
    EXPECT_TRUE(m.process_event(e4{}));   // re-entered
    EXPECT_TRUE(m.is<history_sub>("s2"_s, "idle2"_s));
    EXPECT_TRUE(m.process_event(e5{}));
    EXPECT_TRUE(m.is(X));
}

// ---- defer --------------------------------------------------------------------------------------------------------

constexpr auto state1 = state<class state1_t>;
constexpr auto state2 = state<class state2_t>;
constexpr auto state3 = state<class state3_t>;
constexpr auto state4 = state<class state4_t>;
constexpr auto state5 = state<class state5_t>;
constexpr auto state6 = state<class state6_t>;

struct defer_order {
    auto operator()() const
    {
        return make_transition_table(
            *state1 + event<e1> / defer,
             state1 + event<e2> / defer,
             state1 + event<e3> / defer,
             state1 + event<e4> = state2,
             state2 + event<e1> / defer,
             state2 + event<e3> / defer,
             state2 + event<e2> = state3,
             state3 + event<e1> = state4,
             state3 + event<e3> = state5,
             state4 + event<e4> = state6,
             state6 + event<e3> / [](calls& c) { c.push_back("e3 should have been dropped"); });
    }
};

TEST(FsmDefer, DeferredEventsReplayInOrderAfterEachChangeAndUntakenOnesAreDropped)
{
    calls c;
    sm<defer_order> m{c};
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_TRUE(m.process_event(e2{}));
    EXPECT_TRUE(m.process_event(e3{}));
    EXPECT_TRUE(m.process_event(e4{}));   // state2, e2 replayed -> state3, e1 -> state4, e3 dropped
    EXPECT_TRUE(m.is(state4));
    EXPECT_TRUE(m.process_event(e4{}));
    EXPECT_TRUE(m.is(state6));
    EXPECT_TRUE(c.empty());
}

struct defer_small {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"wait"_s + event<e1> / defer, "wait"_s + event<e2> = "go"_s, "go"_s + event<e1> = X);
    }
};

TEST(FsmDefer, AFullQueueDropsTheEventAndSaysSo)
{
    sm<defer_small, queue_capacity<2>> m;
    EXPECT_TRUE(m.try_process_event(e1{}).is_ok());
    EXPECT_TRUE(m.try_process_event(e1{}).is_ok());
    const auto full = m.try_process_event(e1{});
    ASSERT_TRUE(full.is_err());
    EXPECT_EQ(full.unwrap_err_unchecked(), fsm_errc::queue_full);
    EXPECT_TRUE(m.process_event(e2{}));   // go, then the first deferred e1 -> X; the second is dropped
    EXPECT_TRUE(m.is(X));
}

TEST(FsmDefer, RestoreEmptiesTheQueues)
{
    sm<defer_small, snicholls::fsm::testing> m;
    EXPECT_TRUE(m.process_event(e1{}));
    m.set_current_states("wait"_s);
    EXPECT_TRUE(m.process_event(e2{}));   // nothing deferred any more
    EXPECT_TRUE(m.is("go"_s));
}

// ---- process ------------------------------------------------------------------------------------------------------

struct processing {
    auto operator()() const
    {
        return make_transition_table(
            *"idle"_s + event<e1> / (process(e2{}), process(e3{}), [](calls& c) { c.push_back("e1"); }) = "s1"_s,
             "s1"_s + event<e2> / [](calls& c) { c.push_back("e2"); } = "s2"_s,
             "s2"_s + event<e3> / [](calls& c) { c.push_back("e3"); } = "s3"_s,
             "s3"_s + on_entry<_> / process(e4{}),
             "s3"_s + event<e4> = X);
    }
};

TEST(FsmProcess, QueuedEventsRunInOrderAfterTheCurrentOne)
{
    calls c;
    sm<processing> m{c};
    EXPECT_TRUE(m.process_event(e1{}));
    EXPECT_EQ(c, (calls{"e1", "e2", "e3"}));
    EXPECT_TRUE(m.is(X));   // e4, queued by s3's entry, ran too
}

struct process_refused {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"idle"_s + event<e1> / process(e5{}) = "s1"_s, "s1"_s + event<e2> = X);
    }
};

TEST(FsmProcess, AQueuedEventNobodyTakesMakesTheResultFalse)
{
    sm<process_refused> m;
    EXPECT_FALSE(m.process_event(e1{}));
    EXPECT_TRUE(m.is("s1"_s));
    sm<process_refused> n;
    EXPECT_EQ(n.try_process_event(e1{}).unwrap_err_unchecked(), fsm_errc::queued_event_unhandled);
}

} // namespace
