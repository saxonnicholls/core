//
//  state_machine_history_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/history.cpp (BSL-1.0): a sub-machine whose initial state is a history state,
//  written in the reversed notation (target <= source + event); left on e3 and re-entered on e4, it resumes in s1
//  instead of starting again. Exits non-zero if it does not.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

namespace {

std::string trace;

//! [history]
struct sub {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            "s1"_s <= "idle"_s(H) + "e1"_e / [] { trace += "in sub; "; },
            X      <= "s1"_s      + "e2"_e / [] { trace += "in sub again; "; });
    }
};

struct history {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            state<sub> <= *"idle"_s  + "e1"_e / [] { trace += "enter sub; "; },
            "s1"_s     <= state<sub> + "e3"_e / [] { trace += "exit sub; "; },
            state<sub> <= "s1"_s     + "e4"_e / [] { trace += "enter sub again; "; });
    }
};
//! [history]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's history\n");
    sm<history> machine;
    bool ok = machine.process_event("e1"_e());   // enter sub
    ok &= machine.process_event("e1"_e());       // in sub
    ok &= machine.is<sub>("s1"_s);
    ok &= machine.process_event("e3"_e());       // exit sub
    ok &= machine.process_event("e4"_e());       // enter sub again: s1, as it was left
    ok &= machine.is<sub>("s1"_s);
    ok &= machine.process_event("e2"_e());       // in sub again
    ok &= machine.is<sub>(X);
    std::printf("  %s\n", trace.c_str());
    ok &= trace == "enter sub; in sub; exit sub; enter sub again; in sub again; ";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
