//
//  state_machine_states_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/states.cpp (BSL-1.0): states as forward-declared classes (state<class idle>)
//  and as strings, with entry and exit actions. Exits non-zero if the machine does not end in X with the entry and
//  exit of s1 run once each.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

namespace {

std::string trace;

//! [states]
struct e1 {};
struct e2 {};
struct e3 {};

struct states {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        const auto idle = state<class idle>;   // a state is any type, complete or not
        return make_transition_table(
            *idle + event<e1> = "s1"_s,
             "s1"_s + on_entry<_> / [] { trace += "s1 on entry; "; },
             "s1"_s + snicholls::fsm::on_exit<_> / [] { trace += "s1 on exit; "; },
             "s1"_s + event<e2> = state<class s2>,
             state<class s2> + event<e3> = X);
    }
};
//! [states]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's states\n");
    sm<states> machine;
    bool ok = machine.process_event(e1{});
    ok &= machine.process_event(e2{});
    ok &= machine.process_event(e3{});
    ok &= machine.is(X);
    std::printf("  %s\n", trace.c_str());
    ok &= trace == "s1 on entry; s1 on exit; ";
    machine.visit_current_states([](auto s) { std::printf("  now in %s\n", s.c_str()); });
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
