//
//  state_machine_logging_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/logging.cpp (BSL-1.0): a logger policy whose four hooks see every event, guard,
//  action and state change, with the names of the types involved. Exits non-zero if the log is not sml's.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

namespace {

//! [logger]
struct my_logger {
    std::string log;
    void line(std::string_view a, std::string_view b, std::string_view c = {}, std::string_view d = {})
    {
        std::string text = "[" + std::string(a) + "][" + std::string(b) + "] " + std::string(c);
        if (!d.empty())
            text += " " + std::string(d);
        std::printf("  %s\n", text.c_str());
        log += text + "\n";
    }

    template <class SM, class Event>
    void log_process_event(const Event&)
    {
        line(snicholls::type_name<SM>(), "process_event", snicholls::type_name<Event>());
    }
    template <class SM, class Guard, class Event>
    void log_guard(const Guard&, const Event&, bool result)
    {
        line(snicholls::type_name<SM>(), "guard", snicholls::type_name<Guard>(),
             std::string(snicholls::type_name<Event>()) + (result ? " [OK]" : " [Reject]"));
    }
    template <class SM, class Action, class Event>
    void log_action(const Action&, const Event&)
    {
        line(snicholls::type_name<SM>(), "action", snicholls::type_name<Action>(), snicholls::type_name<Event>());
    }
    template <class SM, class Src, class Dst>
    void log_state_change(const Src& src, const Dst& dst)
    {
        line(snicholls::type_name<SM>(), "transition", std::string(src.c_str()) + " -> " + dst.c_str());
    }
};
//! [logger]

} // namespace

struct e1 {};
struct e2 {};
struct guard {
    bool operator()() const { return true; }
};
struct action {
    void operator()() {}
};

struct logging {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(*"idle"_s + event<e1> [guard{} && guard{}] / action{} = "s1"_s);
    }
};

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's logging\n");
    //! [use]
    my_logger logger;
    sm<logging, snicholls::fsm::logger<my_logger>> machine{logger};   // held by reference
    bool ok = machine.process_event(e1{});
    ok &= !machine.process_event(e2{});   // not an event of the table: nothing to log
    //! [use]
    ok &= logger.log == "[logging][process_event] e1\n"
                        "[logging][guard] guard e1 [OK]\n"
                        "[logging][guard] guard e1 [OK]\n"
                        "[logging][transition] idle -> s1\n"
                        "[logging][action] action e1\n";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
