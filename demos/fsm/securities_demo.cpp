//
//  securities_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/securities.hpp>: a share trade placed, filled, settling and settled, with its state on
//  the order and saved through the reference key-value store across a restart; a fund application and a redemption;
//  a dividend from announcement to payment; a repo that accrues and closes; the failures and cancellations; each
//  machine drawn as Mermaid. Exits non-zero if a state is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/fsm/diagram.hpp>
#include <snicholls/core/fsm/persist.hpp>
#include <snicholls/core/fsm/securities.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/meta/enums.hpp>

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

struct order {
    long quantity = 0;
    snicholls::fsm::trade_state state = snicholls::fsm::trade_state::placed;   // the state, on the object
};

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm securities demo\n");
    bool ok = true;

    //! [trade]
    sm<trade_lifecycle> machine;   // placed
    order o{1'000};
    ok &= machine.process_event(trade_event::fill{});       // a buy fact: the trade date
    ok &= machine.process_event(trade_event::instruct{});   // the settlement instruction is in
    o.state = machine.current();
    snicholls::println(o.state);                              // Settling, in its colour
    //! [trade]
    ok &= o.state == trade_state::settling;

    //! [persist]
    snicholls::reference_key_value_store store;
    ok &= save(store, "order/2026-11-26/0001", machine, 0).is_ok();   // version 1
    sm<trade_lifecycle> after_restart;                                // another process
    const auto version = load(store, "order/2026-11-26/0001", after_restart);
    ok &= version.is_ok() && *version == 1 && after_restart.current() == trade_state::settling;
    ok &= after_restart.process_event(trade_event::settle{});         // a settle fact, T+2
    ok &= save(store, "order/2026-11-26/0001", after_restart, *version).is_ok();
    //! [persist]
    ok &= is_final(after_restart.current()) && after_restart.current() == trade_state::settled;
    snicholls::println(after_restart.current());

    sm<trade_lifecycle> failing, withdrawn;
    ok &= failing.process_event(trade_event::fill{}) && failing.process_event(trade_event::instruct{}) &&
          failing.process_event(trade_event::fail{}) && failing.current() == trade_state::failed;
    ok &= withdrawn.process_event(trade_event::cancel{}) && withdrawn.current() == trade_state::cancelled;
    ok &= !withdrawn.process_event(trade_event::fill{});   // nothing leaves a final state
    snicholls::println(failing.current());

    //! [dealing]
    sm<dealing_lifecycle> application;   // requested before the cut-off
    ok &= application.process_event(dealing_event::price{});    // the NAV is struck
    ok &= application.process_event(dealing_event::issue{});    // units issued on the register
    ok &= application.process_event(dealing_event::settle{});   // the cash settled
    sm<dealing_lifecycle> redemption;
    ok &= redemption.process_event(dealing_event::price{}) && redemption.process_event(dealing_event::cancel{});
    //! [dealing]
    ok &= application.current() == dealing_state::settled && redemption.current() == dealing_state::cancelled;
    snicholls::println(application.current());

    sm<corporate_action_lifecycle> dividend;
    ok &= dividend.process_event(action_event::go_ex{}) && dividend.process_event(action_event::record_date{}) &&
          dividend.process_event(action_event::pay{}) && dividend.current() == action_state::paid;
    sm<corporate_action_lifecycle> withdrawn_offer;
    ok &= withdrawn_offer.process_event(action_event::withdraw{}) && withdrawn_offer.current() == action_state::cancelled;
    snicholls::println(dividend.current());

    sm<repo_lifecycle> repo;
    for (int day = 0; day < 30; ++day)
        ok &= repo.process_event(repo_event::accrue{});
    ok &= repo.current() == repo_state::accruing && repo.process_event(repo_event::repurchase{});
    ok &= is_final(repo.current()) && !is_final(repo_state::open);
    snicholls::println(repo.current());

    for (const auto s : snicholls::enum_values_v<dealing_state>)
        std::printf("  %-9.*s %06x\n", static_cast<int>(attributes(s).get<snicholls::label>().text.size()),
                    attributes(s).get<snicholls::label>().text.data(),
                    static_cast<unsigned>(attributes(s).get<snicholls::colour>().rgb));
    const std::string diagram = to_mermaid<trade_lifecycle>();
    std::printf("%s", diagram.c_str());
    ok &= diagram.find("settling --> settled") != std::string::npos;
    std::printf("%s", to_mermaid<repo_lifecycle>().c_str());

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
