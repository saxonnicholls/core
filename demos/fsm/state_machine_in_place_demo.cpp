//
//  state_machine_in_place_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/in_place.cpp (BSL-1.0): a machine whose definition is a lambda written where
//  the machine is declared, its type deduced. Exits non-zero if the action does not run or the machine does not end
//  in X.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's in_place\n");
    int actions = 0;

    //! [in_place]
    struct start {};
    sm machine{[] {
        using namespace snicholls::fsm;
        return make_transition_table(*"idle"_s + event<start> / [](int& n) { ++n; } = X);
    }};
    //! [in_place]

    sm counted{[] {
                   using namespace snicholls::fsm;
                   return make_transition_table(*"idle"_s + event<start> / [](int& n) { ++n; } = X);
               },
               actions};   // with a dependency, the definition is named once more
    bool ok = machine.process_event(start{}) && machine.is(X);
    ok &= counted.process_event(start{}) && counted.is(X) && actions == 1;
    std::printf("  action ran %d time(s) on the caller's counter\n", actions);
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
