//
//  state_machine_actions_guards_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/actions_guards.cpp (BSL-1.0): guards and actions as lambdas, function objects,
//  wrapped free functions and member functions of the definition; guard algebra (&&, ||, !) and action sequences;
//  an int dependency passed to the constructor. Exits non-zero if the calls are not the ones sml's example makes.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

namespace {

std::string trace;
void note(const char* what) { trace += what; trace += ' '; }

//! [table]
struct e1 {};
struct e2 {};
struct e3 {};
struct e4 {};
struct e5 {};

bool guard2_impl(int i);

struct actions_guards {
    using self = actions_guards;

    auto operator()()
    {
        using namespace snicholls::fsm;

        auto guard1 = [] { note("guard1"); return true; };
        auto guard2 = wrap(&guard2_impl);
        auto action1 = [](const auto& e) { note(event<std::remove_cvref_t<decltype(e)>>.name().data()); };

        struct action2 {
            void operator()(int i) { note(i == 42 ? "action2" : "action2-wrong"); }
        };

        return make_transition_table(
            *"idle"_s + event<e1> = "s1"_s,
             "s1"_s + event<e2> [guard1] / action1 = "s2"_s,
             "s2"_s + event<e3> [guard1 && ![] { return false; }] / (action1, action2{}) = "s3"_s,
             "s3"_s + event<e4> [!guard1 || guard2] / (action1, [] { note("action3"); }) = "s4"_s,
             "s3"_s + event<e4> [guard1] / ([] { note("action4"); }, [this] { action4(); }) = "s5"_s,
             "s5"_s + event<e5> [&self::guard3] / (wrap(&self::action4), &self::action5) = X);
    }

    bool guard3(int i) const noexcept
    {
        note(i == 42 ? "guard3" : "guard3-wrong");
        return true;
    }
    void action4() const { note("action4"); }
    void action5(int i, const e5&) { note(i == 42 ? "action5" : "action5-wrong"); }
};

bool guard2_impl(int i)
{
    note(i == 42 ? "guard2" : "guard2-wrong");
    return false;
}
//! [table]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's actions_guards\n");

    //! [use]
    actions_guards definition{};
    sm<actions_guards> machine{definition, 42};   // the definition and an int dependency, in any order
    bool ok = machine.process_event(e1{});
    ok &= machine.process_event(e2{});
    ok &= machine.process_event(e3{});
    ok &= machine.process_event(e4{});
    ok &= machine.process_event(e5{});
    ok &= machine.is(X);
    //! [use]

    std::printf("  calls: %s\n", trace.c_str());
    // action1 prints the event's name; guard2 refuses, so e4 takes the second s3 row
    ok &= trace == "guard1 e2 guard1 e3 action2 guard1 guard2 guard1 action4 action4 guard3 action4 action5 ";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
