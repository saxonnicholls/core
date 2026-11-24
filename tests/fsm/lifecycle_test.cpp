//
//  lifecycle_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  fsm::lifecycle_state and the reference lifecycle: the stable values, the presentation (one attribute set per
//  enumerator, the colours core's rules give), is_final, and every row of the table, refused events included. The
//  machine is checked at compile time too.
//

#include <snicholls/core/fsm/lifecycle.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

namespace {

using namespace snicholls::fsm;
using namespace snicholls::fsm::lifecycle_event;
using s = lifecycle_state;

// The values are part of the wire format: they never change.
static_assert(static_cast<std::uint8_t>(s::requested) == 0 && static_cast<std::uint8_t>(s::pending) == 1 &&
              static_cast<std::uint8_t>(s::confirmed) == 2 && static_cast<std::uint8_t>(s::settled) == 3 &&
              static_cast<std::uint8_t>(s::failed) == 4 && static_cast<std::uint8_t>(s::canceled) == 5 &&
              static_cast<std::uint8_t>(s::expired) == 6 && static_cast<std::uint8_t>(s::reversed) == 7);
static_assert(snicholls::enum_count_v<lifecycle_state> == 8);
static_assert(snicholls::enum_name(s::confirmed) == "confirmed");

static_assert(attributes(s::settled).get<snicholls::colour>() == snicholls::colours::green);
static_assert(attributes(s::failed).get<snicholls::colour>() == snicholls::colours::red);
static_assert(attributes(s::pending).get<snicholls::colour>() == snicholls::colours::grey);
static_assert(attributes(s::requested).get<snicholls::colour>() == snicholls::colours::grey);
static_assert(attributes(s::confirmed).get<snicholls::label>().text == "Confirmed");
static_assert(snicholls::has_attributes<lifecycle_state>);
static_assert(noexcept(attributes(s::settled)) && noexcept(is_final(s::settled)));

static_assert(!is_final(s::requested) && !is_final(s::pending) && !is_final(s::confirmed) && !is_final(s::settled));
static_assert(is_final(s::failed) && is_final(s::canceled) && is_final(s::expired) && is_final(s::reversed));

static_assert(std::is_same_v<sm<lifecycle>::state_enum, lifecycle_state>);
static_assert(sizeof(sm<lifecycle>) == 1 && std::is_trivially_copyable_v<sm<lifecycle>>);
static_assert(table_diagnostics<lifecycle>::ok);

// pending -> confirmed -> settled, at compile time
static_assert([] {
    sm<lifecycle> m;
    const bool a = m.process_event(submit{});
    const bool b = m.process_event(confirm{});
    const bool c = m.process_event(settle{});
    return a && b && c && m.current() == s::settled;
}());

template <class... Es>
lifecycle_state after(const Es&... es)
{
    sm<lifecycle> m;
    const bool all = (m.process_event(es) && ...);
    EXPECT_TRUE(all);
    return m.current();
}

TEST(FsmLifecycle, EveryRowLeadsWhereTheTableSays)
{
    EXPECT_EQ(after(), s::requested);
    EXPECT_EQ(after(submit{}), s::pending);
    EXPECT_EQ(after(cancel{}), s::canceled);
    EXPECT_EQ(after(fail{}), s::failed);
    EXPECT_EQ(after(submit{}, confirm{}), s::confirmed);
    EXPECT_EQ(after(submit{}, fail{}), s::failed);
    EXPECT_EQ(after(submit{}, cancel{}), s::canceled);
    EXPECT_EQ(after(submit{}, expire{}), s::expired);
    EXPECT_EQ(after(submit{}, confirm{}, settle{}), s::settled);
    EXPECT_EQ(after(submit{}, confirm{}, fail{}), s::failed);
    EXPECT_EQ(after(submit{}, confirm{}, reverse{}), s::reversed);
    EXPECT_EQ(after(submit{}, confirm{}, settle{}, reverse{}), s::reversed);
}

TEST(FsmLifecycle, FinalStatesTakeNoEvent)
{
    for (const auto final_state : {s::failed, s::canceled, s::expired, s::reversed}) {
        sm<lifecycle> m;
        ASSERT_TRUE(m.restore(final_state).is_ok());
        EXPECT_FALSE(m.process_event(submit{}));
        EXPECT_FALSE(m.process_event(confirm{}));
        EXPECT_FALSE(m.process_event(settle{}));
        EXPECT_FALSE(m.process_event(fail{}));
        EXPECT_FALSE(m.process_event(cancel{}));
        EXPECT_FALSE(m.process_event(expire{}));
        EXPECT_FALSE(m.process_event(reverse{}));
        EXPECT_EQ(m.current(), final_state);
    }
}

TEST(FsmLifecycle, OutOfOrderEventsAreRefusedAsValues)
{
    sm<lifecycle> m;
    EXPECT_EQ(m.try_process_event(settle{}).unwrap_err_unchecked(), fsm_errc::no_transition);   // not yet confirmed
    EXPECT_EQ(m.try_process_event(reverse{}).unwrap_err_unchecked(), fsm_errc::no_transition);
    EXPECT_TRUE(m.process_event(submit{}));
    EXPECT_FALSE(m.process_event(submit{}));   // a duplicate webhook changes nothing
    EXPECT_EQ(m.current(), s::pending);
}

TEST(FsmLifecycle, EveryEnumeratorIsAStateSoEveryRestoreSucceeds)
{
    for (const auto v : snicholls::enum_values_v<lifecycle_state>) {
        sm<lifecycle> m;
        EXPECT_TRUE(m.restore(v).is_ok());
        EXPECT_EQ(m.current(), v);
    }
    sm<lifecycle> m;
    EXPECT_EQ(m.restore(static_cast<lifecycle_state>(8)).unwrap_err_unchecked(), fsm_errc::not_a_state);
}

} // namespace
