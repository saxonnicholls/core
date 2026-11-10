//
//  member_count_reference_member.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: a struct with a reference member. member_count, and so
//  tie / to_tuple / from_tuple, reject it. Built with SNICHOLLS_COMPILE_FAIL it
//  must fail to compile; without it (the same struct holding a pointer), it is
//  the positive control and must compile.
//

#include <snicholls/core/reflect/member_count.hpp>

#ifdef SNICHOLLS_COMPILE_FAIL
struct holder { int& ref; };
#else
struct holder { int* ref; };
#endif

int main()
{
    return static_cast<int>(snicholls::member_count_v<holder>) - 1;
}
