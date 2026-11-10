//
//  struct_tuple_size_mismatch.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: from_tuple with a tuple whose size is not the struct's
//  member count. Built with SNICHOLLS_COMPILE_FAIL it must fail to compile;
//  without it, it is the positive control and must compile.
//

#include <snicholls/core/reflect/struct_tuple.hpp>

#include <tuple>

struct pair_of_ints { int a; int b; };

int main()
{
#ifdef SNICHOLLS_COMPILE_FAIL
    const auto p = snicholls::from_tuple<pair_of_ints>(std::tuple<int>{1});
#else
    const auto p = snicholls::from_tuple<pair_of_ints>(std::tuple<int, int>{1, 2});
#endif
    return p.a;
}
