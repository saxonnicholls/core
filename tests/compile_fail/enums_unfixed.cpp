//
//  enums_unfixed.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: reflecting an unscoped enumeration without a fixed underlying type.
//  Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed in
//  tests/CMakeLists.txt; without it, the positive control (a fixed type) must compile.
//

#include <snicholls/core/meta/enums.hpp>

#ifdef SNICHOLLS_COMPILE_FAIL
enum colour { red, green };
#else
enum colour : int { red, green };
#endif

int main() { return snicholls::detail::enum_table<colour>::count() == 2 ? 0 : 1; }
