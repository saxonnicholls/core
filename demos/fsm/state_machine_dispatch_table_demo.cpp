//
//  state_machine_dispatch_table_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/dispatch_table.cpp (BSL-1.0): events that arrive as a run-time id are mapped to
//  their types by each type's static id, the event built from the run-time one when it can be. Exits non-zero if the
//  machine does not reach X, or an unknown id is not refused.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

namespace {

//! [dispatch_table]
struct runtime_event {
    int id = 0;
};
struct event1 {
    static constexpr auto id = 1;
    explicit event1(const runtime_event&) {}
};
struct event2 {
    static constexpr auto id = 2;
};

struct dispatch_table {
    auto operator()() const noexcept
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s + event<event1> = "s1"_s,
             "s1"_s + event<event2> = X);
    }
};
//! [dispatch_table]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's dispatch_table\n");
    sm<dispatch_table> machine;
    auto dispatch_event = make_dispatch_table<runtime_event, 1 /*min*/, 5 /*max*/>(machine);
    bool ok = true;
    {
        runtime_event event{1};
        ok &= dispatch_event(event, event.id);
    }
    ok &= machine.is("s1"_s);
    ok &= !dispatch_event(runtime_event{4}, 4);   // in range, but no event has id 4
    ok &= !dispatch_event(runtime_event{9}, 9);   // out of range
    {
        runtime_event event{2};
        ok &= dispatch_event(event, event.id);
    }
    ok &= machine.is(X);
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
