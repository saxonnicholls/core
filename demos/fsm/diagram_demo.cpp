//
//  diagram_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/diagram.hpp>: the reference payment lifecycle drawn as Mermaid, Graphviz DOT and
//  PlantUML, straight from its table, and a machine with two regions drawn as Mermaid. Exits non-zero if the Mermaid
//  diagram is not the documented text.
//

#include <snicholls/core/fsm/diagram.hpp>
#include <snicholls/core/fsm/lifecycle.hpp>

#include <cstdio>
#include <string>

namespace {

struct tick {};
struct tock {};

struct clock_and_alarm {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"stopped"_s + event<tick> = "running"_s,
             "running"_s + event<tock> = "stopped"_s,
            *"quiet"_s   + event<tock> = "ringing"_s);
    }
};

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm diagram demo\n");

    //! [mermaid]
    const std::string mermaid = to_mermaid<lifecycle>();   // or to_mermaid(machine)
    //! [mermaid]
    std::printf("%s\n", mermaid.c_str());
    bool ok = mermaid == "stateDiagram-v2\n"
                         "    [*] --> requested\n"
                         "    requested --> pending : submit\n"
                         "    requested --> canceled : cancel\n"
                         "    requested --> failed : fail\n"
                         "    pending --> confirmed : confirm\n"
                         "    pending --> failed : fail\n"
                         "    pending --> canceled : cancel\n"
                         "    pending --> expired : expire\n"
                         "    confirmed --> settled : settle\n"
                         "    confirmed --> failed : fail\n"
                         "    confirmed --> reversed : reverse\n"
                         "    settled --> reversed : reverse\n";

    sm<lifecycle> machine;
    ok &= to_dot(machine) == to_dot<lifecycle>() && to_plantuml(machine) == to_plantuml<lifecycle>();
    std::printf("%s\n%s\n", to_dot<lifecycle>().c_str(), to_plantuml<lifecycle>().c_str());

    const std::string regions = to_mermaid<clock_and_alarm>();
    std::printf("%s\n", regions.c_str());
    ok &= regions.find("state clock_and_alarm {") != std::string::npos && regions.find("        --\n") != std::string::npos;

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
