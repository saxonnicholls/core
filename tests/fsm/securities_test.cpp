//
//  securities_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  fsm/securities.hpp: the trade, dealing, corporate-action and repo lifecycles. The stable values, one attribute set
//  per enumerator, is_final, every row of every table, refused events, every enumerator restorable, the tables checked
//  at compile time, and each machine saved and loaded through the reference key-value store.
//

#include <snicholls/core/fsm/persist.hpp>
#include <snicholls/core/fsm/securities.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/meta/enums.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <type_traits>

namespace {

using namespace snicholls::fsm;

static_assert(static_cast<int>(trade_state::placed) == 0 && static_cast<int>(trade_state::cancelled) == 5);
static_assert(static_cast<int>(dealing_state::requested) == 0 && static_cast<int>(dealing_state::settled) == 4);
static_assert(static_cast<int>(action_state::announced) == 0 && static_cast<int>(action_state::cancelled) == 4);
static_assert(static_cast<int>(repo_state::open) == 0 && static_cast<int>(repo_state::closed) == 2);
static_assert(snicholls::enum_count_v<trade_state> == 6 && snicholls::enum_count_v<dealing_state> == 5 &&
              snicholls::enum_count_v<action_state> == 5 && snicholls::enum_count_v<repo_state> == 3);

static_assert(table_diagnostics<trade_lifecycle>::ok && table_diagnostics<dealing_lifecycle>::ok &&
              table_diagnostics<corporate_action_lifecycle>::ok && table_diagnostics<repo_lifecycle>::ok);
static_assert(sizeof(sm<trade_lifecycle>) == 1 && std::is_trivially_copyable_v<sm<repo_lifecycle>>);
static_assert(std::is_same_v<sm<dealing_lifecycle>::state_enum, dealing_state>);

static_assert(attributes(trade_state::settled).get<snicholls::colour>() == snicholls::colours::green);
static_assert(attributes(trade_state::failed).get<snicholls::colour>() == snicholls::colours::red);
static_assert(attributes(dealing_state::issued).get<snicholls::label>().text == "Issued");
static_assert(attributes(action_state::paid).get<snicholls::colour>() == snicholls::colours::green);
static_assert(attributes(repo_state::accruing).get<snicholls::tooltip>().text == "Interest is accruing");
static_assert(snicholls::has_attributes<trade_state> && snicholls::has_attributes<dealing_state> &&
              snicholls::has_attributes<action_state> && snicholls::has_attributes<repo_state>);

static_assert(is_final(trade_state::settled) && is_final(trade_state::failed) && is_final(trade_state::cancelled) &&
              !is_final(trade_state::placed) && !is_final(trade_state::filled) && !is_final(trade_state::settling));
static_assert(is_final(dealing_state::cancelled) && is_final(dealing_state::settled) && !is_final(dealing_state::issued));
static_assert(is_final(action_state::paid) && is_final(action_state::cancelled) && !is_final(action_state::record));
static_assert(is_final(repo_state::closed) && !is_final(repo_state::open) && !is_final(repo_state::accruing));
static_assert(noexcept(is_final(repo_state::open)) && noexcept(attributes(repo_state::open)));

// a trade, at compile time
static_assert([] {
    sm<trade_lifecycle> m;
    const bool a = m.process_event(trade_event::fill{}), b = m.process_event(trade_event::instruct{}),
               c = m.process_event(trade_event::settle{});
    return a && b && c && m.current() == trade_state::settled;
}());

template <class Def, class... Es>
typename Def::state_enum after(const Es&... es)
{
    sm<Def> m;
    const bool all = (m.process_event(es) && ...);
    EXPECT_TRUE(all);
    return m.current();
}

TEST(FsmSecurities, EveryTradeRowLeadsWhereTheTableSays)
{
    using namespace trade_event;
    using s = trade_state;
    EXPECT_EQ(after<trade_lifecycle>(), s::placed);
    EXPECT_EQ(after<trade_lifecycle>(fill{}), s::filled);
    EXPECT_EQ(after<trade_lifecycle>(cancel{}), s::cancelled);
    EXPECT_EQ(after<trade_lifecycle>(fill{}, instruct{}), s::settling);
    EXPECT_EQ(after<trade_lifecycle>(fill{}, instruct{}, settle{}), s::settled);
    EXPECT_EQ(after<trade_lifecycle>(fill{}, instruct{}, fail{}), s::failed);
    sm<trade_lifecycle> m;
    EXPECT_FALSE(m.process_event(settle{})) << "nothing settles before it is filled";
    EXPECT_TRUE(m.process_event(fill{}));
    EXPECT_FALSE(m.process_event(cancel{})) << "a filled trade is not cancelled";
    EXPECT_FALSE(m.process_event(fill{})) << "a duplicate fill changes nothing";
    EXPECT_EQ(m.current(), s::filled);
}

TEST(FsmSecurities, EveryDealingRowLeadsWhereTheTableSays)
{
    using namespace dealing_event;
    using s = dealing_state;
    EXPECT_EQ(after<dealing_lifecycle>(price{}), s::priced);
    EXPECT_EQ(after<dealing_lifecycle>(cancel{}), s::cancelled);
    EXPECT_EQ(after<dealing_lifecycle>(price{}, cancel{}), s::cancelled);
    EXPECT_EQ(after<dealing_lifecycle>(price{}, issue{}), s::issued);
    EXPECT_EQ(after<dealing_lifecycle>(price{}, issue{}, settle{}), s::settled);
    sm<dealing_lifecycle> m;
    EXPECT_FALSE(m.process_event(issue{})) << "no units before the price";
    EXPECT_TRUE(m.process_event(price{}) && m.process_event(issue{}));
    EXPECT_FALSE(m.process_event(cancel{})) << "issued units are not cancelled by withdrawing the request";
}

TEST(FsmSecurities, EveryCorporateActionRowLeadsWhereTheTableSays)
{
    using namespace action_event;
    using s = action_state;
    EXPECT_EQ(after<corporate_action_lifecycle>(go_ex{}), s::ex);
    EXPECT_EQ(after<corporate_action_lifecycle>(withdraw{}), s::cancelled);
    EXPECT_EQ(after<corporate_action_lifecycle>(go_ex{}, withdraw{}), s::cancelled);
    EXPECT_EQ(after<corporate_action_lifecycle>(go_ex{}, record_date{}), s::record);
    EXPECT_EQ(after<corporate_action_lifecycle>(go_ex{}, record_date{}, pay{}), s::paid);
    sm<corporate_action_lifecycle> m;
    EXPECT_FALSE(m.process_event(pay{})) << "nothing is paid before the record date";
    EXPECT_TRUE(m.process_event(go_ex{}) && m.process_event(record_date{}));
    EXPECT_FALSE(m.process_event(withdraw{})) << "after the record date the register is entitled";
}

TEST(FsmSecurities, EveryRepoRowLeadsWhereTheTableSays)
{
    using namespace repo_event;
    using s = repo_state;
    EXPECT_EQ(after<repo_lifecycle>(accrue{}), s::accruing);
    EXPECT_EQ(after<repo_lifecycle>(accrue{}, accrue{}, accrue{}), s::accruing);
    EXPECT_EQ(after<repo_lifecycle>(repurchase{}), s::closed);
    EXPECT_EQ(after<repo_lifecycle>(accrue{}, repurchase{}), s::closed);
    sm<repo_lifecycle> m;
    ASSERT_TRUE(m.restore(s::closed).is_ok());
    EXPECT_FALSE(m.process_event(accrue{})) << "a closed repo accrues nothing";
}

template <class Def>
void every_enumerator_restores()
{
    for (const auto v : snicholls::enum_values_v<typename Def::state_enum>) {
        sm<Def> m;
        EXPECT_TRUE(m.restore(v).is_ok());
        EXPECT_EQ(m.current(), v);
    }
    sm<Def> m;
    EXPECT_EQ(m.restore(static_cast<typename Def::state_enum>(200)).unwrap_err_unchecked(), fsm_errc::not_a_state);
}

TEST(FsmSecurities, EveryEnumeratorIsAStateSoEveryRestoreSucceeds)
{
    every_enumerator_restores<trade_lifecycle>();
    every_enumerator_restores<dealing_lifecycle>();
    every_enumerator_restores<corporate_action_lifecycle>();
    every_enumerator_restores<repo_lifecycle>();
}

TEST(FsmSecurities, EachMachineSavesAndLoads)
{
    snicholls::reference_key_value_store store;
    sm<trade_lifecycle> trade;
    ASSERT_TRUE(trade.process_event(trade_event::fill{}));
    ASSERT_TRUE(save(store, "trade/1", trade, 0).is_ok());
    sm<dealing_lifecycle> dealing;
    ASSERT_TRUE(dealing.process_event(dealing_event::price{}));
    ASSERT_TRUE(save(store, "dealing/1", dealing, 0).is_ok());
    sm<corporate_action_lifecycle> action;
    ASSERT_TRUE(action.process_event(action_event::go_ex{}) && action.process_event(action_event::record_date{}));
    ASSERT_TRUE(save(store, "action/1", action, 0).is_ok());
    sm<repo_lifecycle> repo;
    ASSERT_TRUE(repo.process_event(repo_event::accrue{}));
    ASSERT_TRUE(save(store, "repo/1", repo, 0).is_ok());

    sm<trade_lifecycle> t;
    sm<dealing_lifecycle> d;
    sm<corporate_action_lifecycle> a;
    sm<repo_lifecycle> r;
    EXPECT_EQ(*load(store, "trade/1", t), 1u);
    EXPECT_EQ(t.current(), trade_state::filled);
    EXPECT_EQ(*load(store, "dealing/1", d), 1u);
    EXPECT_EQ(d.current(), dealing_state::priced);
    EXPECT_EQ(*load(store, "action/1", a), 1u);
    EXPECT_EQ(a.current(), action_state::record);
    EXPECT_EQ(*load(store, "repo/1", r), 1u);
    EXPECT_EQ(r.current(), repo_state::accruing);
    EXPECT_EQ(save(store, "trade/1", trade, 0).unwrap_err_unchecked(), fsm_errc::stale);
}

} // namespace
