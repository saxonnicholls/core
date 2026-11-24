//
//  diagram_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  fsm::to_mermaid, to_dot and to_plantuml: every kind of row (initial, external, internal, anonymous, entry and
//  exit, to X), guard algebra and action sequences in labels, names that need an identifier, composites and
//  orthogonal regions, and the sm-argument overloads giving the same text.
//

#include <snicholls/core/fsm/diagram.hpp>
#include <snicholls/core/fsm/lifecycle.hpp>

#include <gtest/gtest.h>

#include <string>

namespace {

using namespace snicholls::fsm;

struct e1 {};
struct e2 {};
struct ok_guard {
    bool operator()() const { return true; }
};
struct act {
    void operator()() const {}
};

struct sample {
    auto operator()() const
    {
        return make_transition_table(
            *"fin wait"_s + event<e1> [ok_guard{} && !ok_guard{}] / (act{}, act{}) = "done"_s,
             "fin wait"_s + event<e2> / act{},
             "fin wait"_s + on_entry<_> / act{},
             "done"_s = X);
    }
};

TEST(FsmDiagram, PlantumlDrawsEveryKindOfRow)
{
    EXPECT_EQ(to_plantuml<sample>(), "@startuml\n\n"
                                     "state \"fin wait\" as fin_wait\n"
                                     "[*] --> fin_wait\n"
                                     "fin_wait --> done : e1 [ok_guard && !ok_guard] / (act, act)\n"
                                     "fin_wait : e2 / act\n"
                                     "fin_wait : entry / act\n"
                                     "done --> [*]\n"
                                     "\n@enduml\n");
}

TEST(FsmDiagram, MermaidDeclaresNamesThatAreNotIdentifiers)
{
    const std::string m = to_mermaid<sample>();
    EXPECT_EQ(m, "stateDiagram-v2\n"
                 "    state \"fin wait\" as fin_wait\n"
                 "    [*] --> fin_wait\n"
                 "    fin_wait --> done : e1 [ok_guard && !ok_guard] / (act, act)\n"
                 "    fin_wait : e2 / act\n"
                 "    fin_wait : entry / act\n"
                 "    done --> [*]\n");
}

TEST(FsmDiagram, DotHasAStartPointAndATerminalNode)
{
    const std::string d = to_dot<sample>();
    EXPECT_NE(d.find("digraph sample {"), std::string::npos);
    EXPECT_NE(d.find("fin_wait [label=\"fin wait\"];"), std::string::npos);
    EXPECT_NE(d.find("start0 -> fin_wait;"), std::string::npos);
    EXPECT_NE(d.find("done -> terminate [label=\"\"];"), std::string::npos);
    EXPECT_NE(d.find("fin_wait -> fin_wait [label=\"e2 / act\", style=dashed];"), std::string::npos);
}

struct inner {
    constexpr auto operator()() const noexcept { return make_transition_table(*"a"_s + event<e1> = X, *"b"_s = X); }
};
struct outer {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*"idle"_s + event<e1> = state<inner>, state<inner> + event<e2> = X);
    }
};

TEST(FsmDiagram, CompositesNestAndRegionsAreSeparated)
{
    const std::string p = to_plantuml<outer>();
    EXPECT_NE(p.find("state inner {\n    [*] --> a\n    a --> [*] : e1\n    --\n    [*] --> b\n    b --> [*]\n}\n"), std::string::npos);
    const std::string m = to_mermaid<outer>();
    EXPECT_NE(m.find("    state inner {\n"), std::string::npos);
    EXPECT_NE(m.find("state \"a\" as inner_a"), std::string::npos);   // Mermaid's names are global: prefixed
    const std::string d = to_dot<outer>();
    EXPECT_NE(d.find("subgraph cluster_inner_ {"), std::string::npos);
    EXPECT_NE(d.find("idle -> inner_start [lhead=cluster_inner_, label=\"e1\"];"), std::string::npos);
}

TEST(FsmDiagram, TheMachineOverloadsGiveTheSameText)
{
    sm<lifecycle> m;
    EXPECT_EQ(to_mermaid(m), to_mermaid<lifecycle>());
    EXPECT_EQ(to_dot(m), to_dot<lifecycle>());
    EXPECT_EQ(to_plantuml(m), to_plantuml<lifecycle>());
    EXPECT_EQ(to_mermaid<sm<lifecycle>>(), to_mermaid<lifecycle>());
    EXPECT_NE(to_plantuml<lifecycle>().find("confirmed --> settled : settle\n"), std::string::npos);
}

TEST(FsmDiagram, LambdasAreDrawnAsLambdaAndQueuesByTheirEvent)
{
    struct lambdas {
        auto operator()() const
        {
            return make_transition_table(*"a"_s + event<e1> [([] { return true; })] / process(e2{}) = "b"_s,
                                         "b"_s + event<e2> / defer, "b"_s + event<e1> = X);
        }
    };
    const std::string p = to_plantuml<lambdas>();
    EXPECT_NE(p.find("a --> b : e1 [lambda] / process(e2)\n"), std::string::npos);
    EXPECT_NE(p.find("b : e2 / defer\n"), std::string::npos);
}

} // namespace
