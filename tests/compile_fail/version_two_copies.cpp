//
//  version_two_copies.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: two versions of core in one translation unit. Another copy's version.hpp has already defined
//  the version macros (here written out, as a second copy at another version would leave them); this copy's header
//  shares the guard name, so without the two-versions check it would be skipped silently. Built with
//  SNICHOLLS_COMPILE_FAIL it must fail with the message listed in tests/CMakeLists.txt; without it, the positive
//  control (the same version twice) must compile.
//

#ifdef SNICHOLLS_COMPILE_FAIL
#define SNICHOLLS_CORE_VERSION_MAJOR 0
#define SNICHOLLS_CORE_VERSION_MINOR 1
#define SNICHOLLS_CORE_VERSION_PATCH 0
#else
#include <snicholls/core/version.hpp>
#endif

#include <snicholls/core/version.hpp>

int main() { return 0; }
