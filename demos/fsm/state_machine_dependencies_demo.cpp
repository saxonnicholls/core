//
//  state_machine_dependencies_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/dependencies.cpp (BSL-1.0): a dependency passed to the constructor reaches the
//  guards and actions that name its type, in any parameter position, beside the event. Exits non-zero if the machine
//  does not move as sml's example asserts.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

namespace {

//! [dependencies]
struct e1 {
    int i = 0;
};
struct dependency {
    int i = 0;
};

struct dependencies {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        const auto guard = [](const dependency& d) { return !d.i; };                          // more could follow
        const auto action = [](dependency& d, const auto& event) { d.i = event.i + 42; };    // order does not matter
        return make_transition_table(
            *"idle"_s + event<e1> [guard] / action = "s1"_s,
             "s1"_s   + event<e1> [([](const auto& event, const dependency& d) { return d.i == event.i; })] = X);
    }
};
//! [dependencies]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's dependencies\n");
    dependency d;
    sm<dependencies> machine{d};   // every dependency, in any order, held by reference
    bool ok = machine.is("idle"_s);

    ok &= machine.process_event(e1{0});
    ok &= machine.is("s1"_s) && d.i == 42;   // the action set the caller's dependency

    ok &= !machine.process_event(e1{0});
    ok &= machine.is("s1"_s);                // 42 != 0

    ok &= machine.process_event(e1{42});
    ok &= machine.is(X);
    std::printf("  dependency.i = %d\n", d.i);
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
