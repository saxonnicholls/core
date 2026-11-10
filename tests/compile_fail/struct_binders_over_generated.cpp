//
//  struct_binders_over_generated.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: SNICHOLLS_MAX_STRUCT_MEMBERS raised above the 256 members
//  the binders are generated for. Built with SNICHOLLS_COMPILE_FAIL it must fail
//  to compile; without it (the generated 256), it is the positive control and
//  must compile.
//

#ifdef SNICHOLLS_COMPILE_FAIL
#define SNICHOLLS_MAX_STRUCT_MEMBERS 300
#else
#define SNICHOLLS_MAX_STRUCT_MEMBERS 256
#endif
#include <snicholls/core/reflect/struct_tuple.hpp>

struct point { int x; int y; };

int main()
{
    point p{1, 2};
    return snicholls::to_tuple(p) == std::tuple<int, int>{1, 2} ? 0 : 1;
}
