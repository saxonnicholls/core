//
//  state_machine_error_handling_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/error_handling.cpp (BSL-1.0): three regions, one whose actions throw, one that
//  takes the exceptions (exception<std::runtime_error>, then exception<_> for anything else), and one that takes
//  unexpected events (unexpected_event<some_event>, then unexpected_event<_>). No exception leaves process_event.
//  Exits non-zero if a region is not where sml's example asserts.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

std::string trace;
void say(const char* what)
{
    std::printf("  %s\n", what);
    trace += what;
    trace += "; ";
}

//! [errors]
struct some_event {};

struct error_handling {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *("idle"_s) + "event1"_e / [] { throw std::runtime_error{"error"}; },
              "idle"_s  + "event2"_e / [] { throw 0; },

            *("exceptions handling"_s) + exception<std::runtime_error> / [] { say("exception caught"); },
              "exceptions handling"_s  + exception<_> / [] { say("generic exception caught"); } = X,

            *("unexpected events handling"_s) + unexpected_event<some_event> / [] { say("unexpected event 'some_event'"); },
              "unexpected events handling"_s  + unexpected_event<_> / [] { say("generic unexpected event"); } = X);
    }
};
//! [errors]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's error_handling\n");
    sm<error_handling> machine;
    bool ok = true;

    static_cast<void>(machine.process_event("event1"_e()));   // throws runtime_error, taken by region 2
    ok &= machine.is("idle"_s, "exceptions handling"_s, "unexpected events handling"_s);

    static_cast<void>(machine.process_event("event2"_e()));   // throws 0, taken by exception<_>
    ok &= machine.is("idle"_s, X, "unexpected events handling"_s);

    static_cast<void>(machine.process_event(some_event{}));   // unexpected event
    ok &= machine.is("idle"_s, X, "unexpected events handling"_s);

    static_cast<void>(machine.process_event(int{}));          // unexpected, any event
    ok &= machine.is("idle"_s, X, X);
    std::printf("  regions: %s\n", ok ? "as sml asserts" : "NOT as sml asserts");

    // An exception no row takes is a value too: event1 now finds no exception row in X
    const auto why = machine.try_process_event("event1"_e());
    ok &= why.is_err() && why.unwrap_err_unchecked() == fsm_errc::action_failed;
    std::printf("  event1 with nobody left to catch it: %s\n", ok ? "action_failed" : "WRONG");

    ok &= trace == "exception caught; generic exception caught; unexpected event 'some_event'; generic unexpected event; ";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
