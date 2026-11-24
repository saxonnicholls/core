//
//  state_machine_hello_world_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/hello_world.cpp (BSL-1.0): the closing half of a TCP connection, string
//  states, guards and actions, and X when the connection is released. Exits non-zero if a state is not the one sml's
//  example asserts.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>

namespace {

std::string trace;

//! [table]
struct release {};
struct ack {};
struct fin {};
struct timeout {};

const auto is_ack_valid = [](const ack&) { return true; };
const auto is_fin_valid = [](const fin&) { return true; };
const auto send_fin = [] { trace += "send_fin "; };
const auto send_ack = [] { trace += "send_ack "; };

struct hello_world {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"established"_s + event<release> / send_fin = "fin wait 1"_s,
             "fin wait 1"_s  + event<ack> [is_ack_valid] = "fin wait 2"_s,
             "fin wait 2"_s  + event<fin> [is_fin_valid] / send_ack = "timed wait"_s,
             "timed wait"_s  + event<timeout> / send_ack = X);
    }
};
//! [table]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's hello_world\n");
    bool ok = true;

    //! [use]
    sm<hello_world> machine;
    ok &= machine.is("established"_s);

    ok &= machine.process_event(release{});
    ok &= machine.is("fin wait 1"_s);

    ok &= machine.process_event(ack{});
    ok &= machine.is("fin wait 2"_s);

    ok &= machine.process_event(fin{});
    ok &= machine.is("timed wait"_s);

    ok &= machine.process_event(timeout{});
    ok &= machine.is(X);   // released
    //! [use]

    std::printf("  actions: %s\n", trace.c_str());
    ok &= trace == "send_fin send_ack send_ack ";
    ok &= !machine.process_event(timeout{});   // nothing leaves X
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
