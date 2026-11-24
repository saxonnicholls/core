//
//  state_machine_plant_uml_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/plant_uml.cpp (BSL-1.0): the table drawn as PlantUML, with guards combined by
//  !, && and ||, an action sequence, entry and exit actions and a sub-machine of two regions; and, beside it, the same
//  table as Mermaid and Graphviz DOT (fsm/diagram.hpp). sml's example walks its internal types to print; here the
//  diagram comes from the table. Exits non-zero if the PlantUML is not the expected text.
//

#include <snicholls/core/fsm/diagram.hpp>
#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

//! [plant_uml]
struct e1 {};
struct e2 {};
struct e3 {};
struct e4 {};
struct e5 {};

struct guard {
    bool operator()() const { return true; }
};
struct guard2 {
    bool operator()() const { return true; }
};
struct action {
    void operator()() {}
};
struct action2 {
    void operator()() {}
};
struct on_s1_entry {
    void operator()() {}
};
struct on_s2_exit {
    void operator()() {}
};

struct sub_machine {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"orth1"_s = X,
            *"orth2"_s = X);
    }
};

struct plant_uml {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"idle"_s + event<e1> = "s1"_s,
             "s1"_s + event<e2> [!guard{} && guard2{}] / action{} = "s2"_s,
             "s1"_s + on_entry<_> / on_s1_entry{},
             "s2"_s + event<e3> [guard{} || guard2{}] = "s1"_s,
             "s2"_s + snicholls::fsm::on_exit<_> / on_s2_exit{},
             "s2"_s + event<e4> / action{} = state<sub_machine>,
             state<sub_machine> + event<e5> / (action{}, action2{}) = X);
    }
};
//! [plant_uml]

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's plant_uml\n");
    const std::string uml = to_plantuml<plant_uml>();
    std::printf("%s\n", uml.c_str());
    const bool ok = uml == "@startuml\n"
                           "\n"
                           "[*] --> idle\n"
                           "idle --> s1 : e1\n"
                           "s1 --> s2 : e2 [!guard && guard2] / action\n"
                           "s1 : entry / on_s1_entry\n"
                           "s2 --> s1 : e3 [guard || guard2]\n"
                           "s2 : exit / on_s2_exit\n"
                           "state sub_machine {\n"
                           "    [*] --> orth1\n"
                           "    orth1 --> [*]\n"
                           "    --\n"
                           "    [*] --> orth2\n"
                           "    orth2 --> [*]\n"
                           "}\n"
                           "s2 --> sub_machine : e4 / action\n"
                           "sub_machine --> [*] : e5 / (action, action2)\n"
                           "\n"
                           "@enduml\n";
    std::printf("%s\n", to_mermaid<plant_uml>().c_str());
    std::printf("%s\n", to_dot<plant_uml>().c_str());
    sm<plant_uml> machine;   // and it runs: e1, e2 is refused by !guard, e3 is not an event of s1
    bool runs = machine.process_event(e1{}) && !machine.process_event(e2{}) && machine.is("s1"_s);
    std::printf("%s\n", ok && runs ? "ok" : "FAILED");
    return ok && runs ? 0 : 1;
}
