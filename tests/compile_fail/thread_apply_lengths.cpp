//
//  thread_apply_lengths.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: threading over tuples of different lengths. Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed
//  in tests/CMakeLists.txt; without it, the positive control must compile.
//

#include <snicholls/core/thread_apply.hpp>

#include <tuple>

int main()
{
    constexpr auto add = [](int a, int b) { return a + b; };
#ifdef SNICHOLLS_COMPILE_FAIL
    const auto sums = snicholls::thread_apply(add, std::tuple{1, 2}, std::tuple{3});
#else
    const auto sums = snicholls::thread_apply(add, std::tuple{1, 2}, std::tuple{3, 4});
#endif
    return std::get<0>(sums) == 4 ? 0 : 1;
}
