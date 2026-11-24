//
//  fsm_missing_enumerator.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: fsm::sm over a state with no enumerator in the declared state enumeration.
//  Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed in
//  tests/CMakeLists.txt; without it, the positive control (a valid table) must compile.
//

#include <snicholls/core/fsm/state_machine.hpp>

struct e1 {};

enum class door : unsigned char { open, closed };
struct table {
    using state_enum = door;
    constexpr auto operator()() const noexcept
    {
        using namespace snicholls::fsm;
#ifdef SNICHOLLS_COMPILE_FAIL
        return make_transition_table(*"open"_s + event<e1> = "ajar"_s);
#else
        return make_transition_table(*"open"_s + event<e1> = "closed"_s);
#endif
    }
};

int main()
{
    snicholls::fsm::sm<table> machine;
    return machine.process_event(e1{}) ? 0 : 1;
}
