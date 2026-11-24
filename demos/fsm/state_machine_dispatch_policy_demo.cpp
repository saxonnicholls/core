//
//  state_machine_dispatch_policy_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/dispatch_policy.cpp (BSL-1.0): the same connection machine under each dispatch
//  strategy (core has fold_expr, the default, and jump_table; sml also has branch_stm and switch_stm, which the
//  compiler makes of the fold). sml's thread_safe policy has no counterpart: core has no threads by rule, so a
//  machine is single-threaded and the caller locks around it (ts-moveables' strands and mutexes are the tools).
//  Exits non-zero if the two strategies do not run the same actions to the same state.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

namespace {

//! [policy]
struct connect {};
struct established {};
struct ping {};
struct disconnect {};
struct timeout {};

struct dispatch_policy {
    auto operator()() const
    {
        const auto establish = [](std::string& log) { log += "establish "; };
        const auto close = [](std::string& log) { log += "close "; };
        const auto is_valid = [](const auto&) { return true; };
        const auto reset_timeout = [](std::string& log) { log += "reset_timeout "; };

        using namespace snicholls::fsm;
        return make_transition_table(
            *"Disconnected"_s + event<connect> / establish   = "Connecting"_s,
             "Connecting"_s   + event<established>            = "Connected"_s,
             "Connected"_s    + event<ping> [is_valid] / reset_timeout,
             "Connected"_s    + event<timeout> / establish    = "Connecting"_s,
             "Connected"_s    + event<disconnect> / close     = "Disconnected"_s);
    }
};
//! [policy]

template <class Machine>
bool drive(Machine& machine)
{
    using namespace snicholls::fsm;
    bool ok = machine.process_event(connect{});
    ok &= machine.process_event(established{});
    ok &= machine.process_event(ping{});
    ok &= machine.process_event(disconnect{});
    ok &= machine.process_event(connect{});
    ok &= machine.process_event(established{});
    ok &= machine.process_event(ping{});
    return ok && machine.is("Connected"_s);
}

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's dispatch_policy\n");
    std::string folded;
    std::string jumped;

    //! [strategies]
    sm<dispatch_policy> by_fold{folded};                                // fold_expr: a branch chain, inlined
    sm<dispatch_policy, dispatch<jump_table>> by_table{jumped};          // jump_table: one indirect call
    //! [strategies]

    bool ok = drive(by_fold) && drive(by_table);
    std::printf("  fold_expr:  %s\n  jump_table: %s\n", folded.c_str(), jumped.c_str());
    ok &= folded == jumped && folded == "establish reset_timeout close establish reset_timeout ";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
