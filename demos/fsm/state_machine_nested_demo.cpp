//
//  state_machine_nested_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Replicates boost-ext/sml's example/nested.cpp (BSL-1.0): a machine declared inside a class template, with its
//  events and definition private to the class. sml skips this example on GCC; core builds it on every compiler.
//  Exits non-zero if the machine does not reach X.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdio>

namespace {

//! [nested]
template <class = class Dummy>   // a template, as in sml's example, to delay the point of instantiation
class top {
    struct e1 {};

    struct nested {
        auto operator()() const noexcept
        {
            using namespace snicholls::fsm;
            return make_transition_table(*"idle"_s + event<e1> = X);
        }
    };

public:
    bool process()
    {
        return machine.process_event(e1{}) && machine.is(snicholls::fsm::X);
    }

private:
    snicholls::fsm::sm<nested> machine{};
};
//! [nested]

} // namespace

int main()
{
    std::printf("snicholls-core fsm: sml's nested\n");
    top<> machine{};
    const bool ok = machine.process();
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
