//
//  struct_tuple_narrowing_int_to_double.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: from_tuple with a narrowing element, int -> double: brace-init narrows (a float source is the control).
//  Built with SNICHOLLS_COMPILE_FAIL it must fail to compile; without it, it
//  is the positive control and must compile.
//

#include <snicholls/core/reflect/struct_tuple.hpp>

#include <tuple>

struct target { double value; };

int main()
{
#ifdef SNICHOLLS_COMPILE_FAIL
    const auto t = snicholls::from_tuple<target>(std::tuple<int>{1});
#else
    const auto t = snicholls::from_tuple<target>(std::tuple<float>{1.0f});
#endif
    return t.value ? 0 : 1;
}
