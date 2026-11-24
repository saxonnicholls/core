//
//  state_machine_orthogonal_regions_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/orthogonal_regions.cpp (BSL-1.0): two initial states make two regions, each
//  event goes to both, and is(a, b) checks one state per region. Exits non-zero if a state is not the one sml's
//  example asserts.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

namespace {

//! [regions]
struct e1 {};
struct e2 {};
struct e3 {};

struct orthogonal_regions {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s + event<e1> = "s1"_s,
             "s1"_s + event<e2> = X,

            *"idle2"_s + event<e2> = "s2"_s,
             "s2"_s + event<e3> = X);
    }
};
//! [regions]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's orthogonal_regions\n");
    sm<orthogonal_regions> machine;
    static_assert(sm<orthogonal_regions>::regions == 2);
    bool ok = machine.is("idle"_s, "idle2"_s);
    ok &= machine.process_event(e1{});
    ok &= machine.is("s1"_s, "idle2"_s);
    ok &= machine.process_event(e2{});   // both regions take e2
    ok &= machine.is(X, "s2"_s);
    ok &= machine.process_event(e3{});
    ok &= machine.is(X, X);
    ok &= machine.is_terminated();
    machine.visit_current_states([](auto s) { std::printf("  region in %s\n", s.c_str()); });
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
