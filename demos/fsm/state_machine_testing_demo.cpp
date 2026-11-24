//
//  state_machine_testing_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/testing.cpp (BSL-1.0): the testing policy sets the machine straight into the
//  state under test, and a fake dependency observes the action. Exits non-zero if the transition under test does not
//  run.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

namespace {

//! [testing]
struct e1 {};
struct e2 {};
struct e3 {};

struct data {
    int value = 0;
};

struct testing {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        const auto guard = [](data& d) { return !d.value; };
        const auto action = [](data& d) { d.value = 42; };
        return make_transition_table(
            *"idle"_s + event<e1> = "s1"_s,
             "s1"_s + event<e2> = "s2"_s,
             "s2"_s + event<e3> [guard] / action = X);   // the transition under test
    }
};
//! [testing]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's testing\n");
    data fake_data{0};
    sm<::testing, snicholls::fsm::testing> machine{fake_data};
    machine.set_current_states("s2"_s);   // no action runs
    bool ok = machine.process_event(e3{});
    ok &= machine.is(X);
    ok &= fake_data.value == 42;
    std::printf("  fake_data.value = %d\n", fake_data.value);
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
