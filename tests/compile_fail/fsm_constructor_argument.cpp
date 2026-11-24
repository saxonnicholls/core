//
//  fsm_constructor_argument.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: fsm::sm given a constructor argument that no guard or action asks for.
//  Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed in
//  tests/CMakeLists.txt; without it, the positive control (the dependency the action takes) must compile.
//

#include <snicholls/core/fsm/state_machine.hpp>

struct e1 {};

struct table {
    constexpr auto operator()() const noexcept
    {
        using namespace snicholls::fsm;
        return make_transition_table(*"a"_s + event<e1> / [](int& n) { ++n; } = "b"_s);
    }
};

int main()
{
#ifdef SNICHOLLS_COMPILE_FAIL
    double unused = 0;
    snicholls::fsm::sm<table> machine{unused};
#else
    int count = 0;
    snicholls::fsm::sm<table> machine{count};
#endif
    return machine.process_event(e1{}) ? 0 : 1;
}
