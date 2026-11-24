//
//  state_machine_data_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/data.cpp (BSL-1.0): data per state (Connected and Interrupted carry an id),
//  shared data in the definition (the address), and member functions of the definition as actions. Exits non-zero if
//  the lines printed are not sml's.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>
#include <string>
#include <utility>

namespace {

std::string trace;

//! [data]
struct connect {
    int id{};
};
struct disconnect {};
struct interrupt {};

struct Disconnected {};
struct Connected {
    int id{};   // per-state data
};
struct Interrupted {
    int id{};   // per-state data
};

class data {
    using Self = data;

public:
    explicit data(std::string address) : address{std::move(address)} {}

    auto operator()()
    {
        using namespace snicholls::fsm;
        const auto set = [](const auto& event, Connected& state) { state.id = event.id; };
        const auto update = [](Connected& src_state, Interrupted& dst_state) { dst_state.id = src_state.id; };
        return make_transition_table(
            *state<Disconnected> + event<connect>    / (set, &Self::print)    = state<Connected>,
             state<Connected>    + event<interrupt>  / (update, &Self::print) = state<Interrupted>,
             state<Interrupted>  + event<connect>    / (set, &Self::print)    = state<Connected>,
             state<Connected>    + event<disconnect> / (&Self::print)         = X);
    }

private:
    void print(Connected& state) { trace += address + ':' + std::to_string(state.id) + '\n'; }

    std::string address{};   // data shared between states
};
//! [data]

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: sml's data\n");
    data d{std::string{"127.0.0.1"}};
    sm<data> machine{d, Connected{1}};   // the definition, and the initial data of a state (owned by the machine)
    bool ok = machine.process_event(connect{1024});
    ok &= machine.process_event(interrupt{});
    ok &= machine.process_event(connect{1025});
    ok &= machine.process_event(disconnect{});
    ok &= machine.is(X);
    std::printf("%s", trace.c_str());
    ok &= trace == "127.0.0.1:1024\n127.0.0.1:1024\n127.0.0.1:1025\n127.0.0.1:1025\n";
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
