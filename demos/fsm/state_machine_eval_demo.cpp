//
//  state_machine_eval_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/eval.cpp (BSL-1.0): eval [guard] / action inside an action sequence runs the
//  action only when the guard holds. Exits non-zero if the counter is not 3.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

namespace {

//! [eval]
struct e1 {};

struct eval_example {
    auto operator()() const
    {
        const auto guard = [] { return true; };
        const auto never = [] { return false; };
        const auto action = [](int& a) { ++a; };
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s + event<e1> [guard] / (action, eval [guard] / action, eval [never] / action, action) = X);
    }
};
//! [eval]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's eval\n");
    int a{};
    sm<eval_example> machine{a};
    bool ok = machine.process_event(e1{});
    ok &= a == 3 && machine.is(X);
    std::printf("  a = %d\n", a);
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
