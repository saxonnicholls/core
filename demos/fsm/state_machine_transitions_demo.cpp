//
//  state_machine_transitions_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/transitions.cpp (BSL-1.0): an anonymous transition at start, an internal
//  transition (no exit, no entry), a self transition (exit and entry run), and an external one to X. Exits non-zero
//  if the actions do not run in sml's order.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

namespace {

std::string trace;
void say(const char* what)
{
    std::printf("  %s\n", what);
    trace += what;
    trace += "; ";
}

//! [transitions]
struct e1 {};
struct e2 {};
struct e3 {};

struct transitions {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s                   / [] { say("anonymous transition"); } = "s1"_s,
             "s1"_s + event<e1>        / [] { say("internal transition"); },
             "s1"_s + event<e2>        / [] { say("self transition"); } = "s1"_s,
             "s1"_s + on_entry<_>      / [] { say("s1 entry"); },
             "s1"_s + snicholls::fsm::on_exit<_>       / [] { say("s1 exit"); },
             "s1"_s + event<e3>        / [] { say("external transition"); } = X);
    }
};
//! [transitions]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's transitions\n");
    sm<transitions> machine;   // the anonymous transition runs at once
    bool ok = machine.is("s1"_s);
    ok &= machine.process_event(e1{});
    ok &= machine.process_event(e2{});
    ok &= machine.process_event(e3{});
    ok &= machine.is(X);
    ok &= trace == "anonymous transition; s1 entry; internal transition; s1 exit; self transition; s1 entry; s1 exit; "
                   "external transition; ";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
