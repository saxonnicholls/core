//
//  state_machine_defer_and_process_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/defer_and_process.cpp (BSL-1.0): e1 is deferred in idle and replayed after the
//  state changes; process(e) queues an event that runs once the current one is done. sml needs defer_queue and
//  process_queue policies; core's queues are built in, fixed-capacity and allocation-free. Exits non-zero if a result
//  is not the one sml's example asserts.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

namespace {

//! [defer]
struct e1 {};
struct e2 {};
struct e3 {};
struct e4 {};

struct defer_and_process {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s + event<e1> / defer,
             "idle"_s + event<e2> = "s1"_s,
             "s1"_s   + event<e1> / process(e2{}) = "s2"_s,
             "s2"_s   + event<e3> / process(e4{}),
             "s2"_s   + event<e4> = X);
    }
};
//! [defer]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's defer_and_process\n");
    sm<defer_and_process> machine;
    bool ok = machine.is("idle"_s);

    bool handled = machine.process_event(e1{});   // deferred
    ok &= handled && machine.is("idle"_s);

    // idle -> s1, then the deferred e1 takes s1 -> s2 and queues e2, which s2 does not take: false
    handled = machine.process_event(e2{});
    ok &= !handled && machine.is("s2"_s);
    std::printf("  e2: %s, now in s2\n", handled ? "true" : "false");

    handled = machine.process_event(e3{});   // queues e4, which takes s2 -> X
    ok &= handled && machine.is(X);

    // The same refusal, with its reason
    sm<defer_and_process> again;
    static_cast<void>(again.process_event(e1{}));
    const auto why = again.try_process_event(e2{});
    ok &= why.is_err() && why.unwrap_err_unchecked() == fsm_errc::queued_event_unhandled;
    const auto name = snicholls::enum_name(why.unwrap_err_unchecked());
    std::printf("  try_process_event(e2): %.*s\n", static_cast<int>(name.size()), name.data());

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
