//
//  state_machine_visitor_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/visitor.cpp (BSL-1.0): visit_current_states hands each region's state to a
//  visitor, which descends into a composite by an overload on its state. Exits non-zero if the names are not sml's.
//

#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/overloaded.hpp>

#include <cstdio>
#include <string>

namespace {

struct e1 {};
struct e2 {};
struct e3 {};
struct e4 {};
struct e5 {};

struct sub {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s + event<e3> = "sub1"_s,
             "sub1"_s + event<e4> = X);
    }
};

struct composite {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s   + event<e1> = "s1"_s,
             "s1"_s     + event<e2> = state<sub>,
             state<sub> + event<e5> = X);
    }
};

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's visitor\n");
    sm<composite> machine;
    std::string seen;

    //! [visitor]
    const auto state_name = [&](auto state) { seen += state.name(); };
    const auto visit = [&] {
        machine.visit_current_states(snicholls::overloaded{
            [&](state_t<sub>) {   // a composite: its name, then its own states
                seen += "sub:";
                machine.visit_current_states<sub>([&](auto s) { state_name(s); });
            },
            [&](auto s) { state_name(s); }});
        seen += '\n';
    };
    //! [visitor]

    bool ok = machine.process_event(e1{});
    visit();   // s1
    ok &= machine.process_event(e2{});
    visit();   // sub:idle
    ok &= machine.process_event(e3{});
    visit();   // sub:sub1
    ok &= machine.process_event(e4{});
    visit();   // sub:terminate
    ok &= machine.process_event(e5{});
    visit();   // terminate
    std::printf("%s", seen.c_str());
    ok &= seen == "s1\nsub:idle\nsub:sub1\nsub:terminate\nterminate\n";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
