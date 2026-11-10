//
//  thread_apply_mixed_void.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: an overload set where some calls return void and some do not. Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed
//  in tests/CMakeLists.txt; without it, the positive control must compile.
//

#include <snicholls/core/thread_apply.hpp>

#include <snicholls/core/overloaded.hpp>

#include <tuple>

int main()
{
#ifdef SNICHOLLS_COMPILE_FAIL
    snicholls::thread_apply(snicholls::overloaded{[](int) {}, [](double) { return 1; }}, std::tuple{1, 2.0});
#else
    snicholls::thread_apply(snicholls::overloaded{[](int) {}, [](double) {}}, std::tuple{1, 2.0});
#endif
    return 0;
}
