//
//  fsm_unreachable_state.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: fsm::sm over a state that no row reaches from an initial state.
//  Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed in
//  tests/CMakeLists.txt; without it, the positive control (a valid table) must compile.
//

#include <snicholls/core/fsm/state_machine.hpp>

struct e1 {};

struct table {
    constexpr auto operator()() const noexcept
    {
        using namespace snicholls::fsm;
#ifdef SNICHOLLS_COMPILE_FAIL
        return make_transition_table(*"a"_s + event<e1> = "b"_s, "island"_s + event<e1> = "b"_s);
#else
        return make_transition_table(*"a"_s + event<e1> = "b"_s, "b"_s + event<e1> = "a"_s);
#endif
    }
};

int main()
{
    snicholls::fsm::sm<table> machine;
    return machine.process_event(e1{}) ? 0 : 1;
}
