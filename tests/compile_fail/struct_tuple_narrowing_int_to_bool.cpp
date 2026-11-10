//
//  struct_tuple_narrowing_int_to_bool.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: from_tuple with a narrowing element, int -> bool: brace-init narrows (a bool source is the control).
//  Built with SNICHOLLS_COMPILE_FAIL it must fail to compile; without it, it
//  is the positive control and must compile.
//

#include <snicholls/core/reflect/struct_tuple.hpp>

#include <tuple>

struct target { bool value; };

int main()
{
#ifdef SNICHOLLS_COMPILE_FAIL
    const auto t = snicholls::from_tuple<target>(std::tuple<int>{1});
#else
    const auto t = snicholls::from_tuple<target>(std::tuple<bool>{true});
#endif
    return t.value ? 0 : 1;
}
