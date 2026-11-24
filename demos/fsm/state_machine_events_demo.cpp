//
//  state_machine_events_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/events.cpp (BSL-1.0): events as types, an event variable, a string event
//  ("e3"_e, made with "e3"_e()), and an event carrying data read by a guard and an action. Exits non-zero if the
//  machine does not reach X with the data sml's example asserts.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

namespace {

int seen = 0;

//! [events]
struct e1 {};
struct e2 {
    bool value = true;
};
constexpr auto event2 = snicholls::fsm::event<e2>;
struct e4 {
    int value = 0;
};

struct events {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        auto guard = [](const e2& e) { return e.value; };
        return make_transition_table(
            *"idle"_s + event<e1> = "s1"_s,
             "s1"_s + event2 [guard] = "s2"_s,
             "s2"_s + "e3"_e = "s3"_s,
             "s3"_s + event<e4> / [](const auto& e) { seen = e.value; } = X);
    }
};
//! [events]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's events\n");
    sm<events> machine;
    bool ok = machine.process_event(e1{});
    ok &= !machine.process_event(e2{false});   // the guard refuses
    ok &= machine.try_process_event(e2{false}).unwrap_err_unchecked() == fsm_errc::guard_rejected;
    ok &= machine.process_event(e2{});
    ok &= machine.process_event("e3"_e());
    ok &= machine.process_event(e4{42});
    ok &= machine.is(X) && seen == 42;
    std::printf("  e4 carried %d; events are named %s, %s, %s\n", seen, event<e1>.name().data(), "e3"_e.name().data(),
                event<e4>.name().data());
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
