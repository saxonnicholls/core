//
//  member_count_over_limit.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: a struct with more members than SNICHOLLS_MAX_STRUCT_MEMBERS
//  (lowered to 4 here). Built with SNICHOLLS_COMPILE_FAIL it must fail to
//  compile; without it (4 members, at the limit), it is the positive control
//  and must compile.
//

#define SNICHOLLS_MAX_STRUCT_MEMBERS 4
#include <snicholls/core/reflect/member_count.hpp>

#ifdef SNICHOLLS_COMPILE_FAIL
struct wide { int a, b, c, d, e; };
#else
struct wide { int a, b, c, d; };
#endif

int main()
{
    return static_cast<int>(snicholls::member_count_v<wide>) - 4;
}
