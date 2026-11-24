//
//  state_machine_composite_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/composite.cpp (BSL-1.0): a sub-machine as a state of its parent, entered on e2,
//  run to X by e3 and e4, and left on e5; the sub-machine's state queried with is<Sub>. Exits non-zero if a state is
//  not the one sml's example asserts.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

namespace {

std::string trace;

//! [composite]
struct e1 {};
struct e2 {};
struct e3 {};
struct e4 {};
struct e5 {};

struct sub {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s + event<e3> / [] { trace += "in sub sm; "; } = "s1"_s,
             "s1"_s + event<e4> / [] { trace += "finish sub sm; "; } = X);
    }
};

struct composite {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s + event<e1> = "s1"_s,
             "s1"_s + event<e2> / [] { trace += "enter sub sm; "; } = state<sub>,
             state<sub> + event<e5> / [] { trace += "exit sub sm; "; } = X);
    }
};
//! [composite]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's composite\n");
    bool ok = true;

    sm<composite> machine;
    ok &= machine.is("idle"_s);
    ok &= machine.is<decltype(state<sub>)>("idle"_s);

    ok &= machine.process_event(e1{});
    ok &= machine.is("s1"_s);
    ok &= machine.is<decltype(state<sub>)>("idle"_s);

    ok &= machine.process_event(e2{});   // enter sub sm
    ok &= machine.is(state<sub>);
    ok &= machine.is<sub>("idle"_s);     // the definition type names the sub-machine too

    ok &= machine.process_event(e3{});   // in sub sm
    ok &= machine.is(state<sub>);
    ok &= machine.is<sub>("s1"_s);

    ok &= machine.process_event(e4{});   // finish sub sm
    ok &= machine.is(state<sub>);
    ok &= machine.is<sub>(X);

    ok &= machine.process_event(e5{});   // exit sub sm
    ok &= machine.is(X);
    ok &= machine.is<sub>(X);

    std::printf("  %s\n", trace.c_str());
    ok &= trace == "enter sub sm; in sub sm; finish sub sm; exit sub sm; ";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
